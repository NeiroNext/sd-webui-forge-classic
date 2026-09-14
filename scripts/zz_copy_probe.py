"""Who allocates the device-to-device copies of a forward. Set ZZ_COPY_PROBE=1.

The live profile shows ~220 ms a step in Memcpy DtoD plus elementwise copy kernels - pure movement,
no arithmetic. This attributes each one to the line that asked for it, so the question becomes
whether it can be avoided rather than whether it can be made faster.
"""

import os
import sys

if os.environ.get("ZZ_COPY_PROBE"):
    import torch

    MIN_BYTES = int(os.environ.get("ZZ_COPY_MIN", str(1 << 20)))
    OUT = os.environ.get("ZZ_COPY_OUT", "logs/copy_sites.txt")
    FIRST, LAST = 2, 5          # record forwards 2..4
    S = {"n": 0, "on": False, "done": False, "sites": {}}
    HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def site():
        f = sys._getframe(2)
        for _ in range(12):     # climb out of torch's own wrappers to the first frame of the tree
            name = f.f_code.co_filename
            if name.startswith(HERE) and "zz_copy_probe" not in name:
                return f"{os.path.relpath(name, HERE)}:{f.f_lineno} {f.f_code.co_name}"
            if f.f_back is None:
                break
            f = f.f_back
        return "outside the tree"

    def note(op, t, made):
        if not S["on"] or not isinstance(made, torch.Tensor) or not made.is_cuda:
            return
        if made.data_ptr() == t.data_ptr():
            return              # contiguous() on a contiguous tensor returns self, no copy
        n = made.numel() * made.element_size()
        if n < MIN_BYTES:
            return
        key = (op, site())
        c, b = S["sites"].get(key, (0, 0))
        S["sites"][key] = (c + 1, b + n)

    _contig, _to, _clone, _cat = (torch.Tensor.contiguous, torch.Tensor.to,
                                  torch.Tensor.clone, torch.cat)

    def contiguous(self, *a, **k):
        r = _contig(self, *a, **k)
        note("contiguous", self, r)
        return r

    def to(self, *a, **k):
        r = _to(self, *a, **k)
        note("to", self, r)
        return r

    def clone(self, *a, **k):
        r = _clone(self, *a, **k)
        note("clone", self, r)
        return r

    def cat(tensors, *a, **k):
        r = _cat(tensors, *a, **k)
        seq = list(tensors)
        if seq:
            note("cat", seq[0], r)
        return r

    torch.Tensor.contiguous, torch.Tensor.to, torch.Tensor.clone, torch.cat = contiguous, to, clone, cat

    def dump():
        rows = sorted(S["sites"].items(), key=lambda kv: -kv[1][1])
        span = LAST - FIRST
        with open(OUT, "w", encoding="utf-8") as f:
            tot_c = sum(v[0] for v in S["sites"].values()) / span
            tot_b = sum(v[1] for v in S["sites"].values()) / span / 2**20
            f.write(f"[COPY] device copies of one forward above {MIN_BYTES >> 20} MB: "
                    f"{tot_c:.0f} calls, {tot_b:.0f} MB\n")
            for (op, where), (c, b) in rows[:25]:
                f.write(f"[COPY] {b / span / 2**20:8.1f} MB {c / span:6.1f}x  {op:<11} {where}\n")
        print(open(OUT, encoding="utf-8").read(), flush=True)

    def on_call(*_):
        S["n"] += 1
        if S["done"]:
            return
        if S["n"] == FIRST:
            S["on"] = True
        elif S["n"] == LAST:
            S["on"] = False
            S["done"] = True
            dump()

    from modules import script_callbacks

    script_callbacks.on_cfg_denoiser(on_call)
    print("[COPY] attributing device copies", flush=True)
