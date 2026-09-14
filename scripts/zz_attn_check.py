"""int8 attention error on REAL activations, after RoPE. Set ZZ_ATTN_CHECK=1.

Patches both attention call sites of the DiTs we care about: backend.nn.flux (Flux AND Chroma, which
imports it) and backend.nn.lumina (NextDiT, i.e. Z-Image). Shapes are recorded for every call even
when the kernel cannot take them, because knowing the head dimension is half the question.
"""

import json
import os

if os.environ.get("ZZ_ATTN_CHECK"):
    import torch

    OUT = os.environ.get("ZZ_ATTN_OUT", "logs/attn_rows.jsonl")
    LIMIT = int(os.environ.get("ZZ_ATTN_LIMIT", "57"))
    STATE = {"n": 0, "off": False, "rows": [], "shapes": {}}

    def qdq(x):
        """per-row absmax quantise and back, exactly what the kernel does to Q and K"""
        s = x.abs().amax(dim=-1, keepdim=True).float().clamp_min(1e-8) / 127.0
        return (torch.round(x.float() / s).clamp_(-127, 127) * s).to(x.dtype)

    def summarise():
        with open(OUT, "w", encoding="utf-8") as fh:
            for r in STATE["rows"]:
                fh.write(json.dumps(r) + "\n")
        print(f"[ATTN] --- {STATE['n']} calls seen ---", flush=True)
        for key, n in sorted(STATE["shapes"].items()):
            print(f"[ATTN] shape {key}: {n} calls", flush=True)
        if not STATE["rows"]:
            print("[ATTN] no call the kernel could take", flush=True)
            return
        ker = sorted(r["kernel"] for r in STATE["rows"])
        flo = sorted(r["floor"] for r in STATE["rows"])
        n = len(ker)
        print(f"[ATTN] measured on {n} calls", flush=True)
        print(f"[ATTN] kernel      mean {sum(ker)/n:.3e}  median {ker[n//2]:.3e}  worst {ker[-1]:.3e}", flush=True)
        print(f"[ATTN] quant floor mean {sum(flo)/n:.3e}  median {flo[n//2]:.3e}  worst {flo[-1]:.3e}", flush=True)
        print(f"[ATTN] kernel adds over the floor: mean {sum(a - b for a, b in zip(ker, flo))/n:+.3e}", flush=True)
        w = max(STATE["rows"], key=lambda r: r["kernel"])
        print(f"[ATTN] worst call #{w['i']} T={w['T']}: kernel {w['kernel']:.3e} floor {w['floor']:.3e} "
              f"outlier q/k {w['outlier_q']:.0f}/{w['outlier_k']:.0f}", flush=True)

    def wrap(orig, tag):
        def probe(q, k, v, heads, mask=None, **kw):
            ref = orig(q, k, v, heads, mask=mask, **kw)
            if STATE["off"]:
                return ref
            STATE["n"] += 1
            key = (f"{tag} heads={q.shape[1] if q.dim() == 4 else '?'} dim={q.shape[-1]} "
                   f"tokens={q.shape[-2]} {str(q.dtype).replace('torch.', '')}"
                   f"{' masked' if mask is not None else ''}")
            STATE["shapes"][key] = STATE["shapes"].get(key, 0) + 1
            takeable = (mask is None and q.dim() == 4 and q.shape[0] == 1
                        and q.shape[-1] == 128 and q.dtype == torch.float16)
            if takeable:
                try:
                    import torch.nn.functional as F

                    from backend import attention_int8 as ai

                    qq, kk, vv = (t[0].contiguous() for t in (q, k, v))
                    H, T, D = qq.shape
                    got = ai.attention(qq, kk, vv)
                    torch.cuda.synchronize()

                    # head by head: at the hires token count a whole-tensor fp32 copy is a gigabyte
                    rr = ref[0].reshape(T, H, D)
                    num_k = num_f = den = 0.0
                    oq = ok = 0.0
                    for hh in range(H):
                        rh = rr[:, hh].float()
                        den += float((rh * rh).sum())
                        d = got[hh].float() - rh
                        num_k += float((d * d).sum())

                        qh, kh = qq[hh], kk[hh]
                        s_q = qh.abs().amax(-1, keepdim=True).float().clamp_min(1e-8) / 127.0
                        s_k = kh.abs().amax(-1, keepdim=True).float().clamp_min(1e-8) / 127.0
                        qd = (torch.round(qh.float() / s_q).clamp_(-127, 127) * s_q).half()
                        kd = (torch.round(kh.float() / s_k).clamp_(-127, 127) * s_k).half()
                        fh = F.scaled_dot_product_attention(qd[None, None], kd[None, None],
                                                            vv[hh][None, None])[0, 0].float()
                        d = fh - rh
                        num_f += float((d * d).sum())
                        del rh, d, qd, kd, fh, s_q, s_k

                        a = qh.abs().float()
                        oq += float((a.amax(-1) / a.mean(-1).clamp_min(1e-8)).median()) / H
                        a = kh.abs().float()
                        ok += float((a.amax(-1) / a.mean(-1).clamp_min(1e-8)).median()) / H
                        del a
                    del got, rr

                    e_kernel = (num_k / den) ** 0.5
                    e_floor = (num_f / den) ** 0.5
                    STATE["rows"].append({"i": STATE["n"], "tag": tag, "T": T, "H": H, "kernel": e_kernel,
                                          "floor": e_floor, "outlier_q": oq, "outlier_k": ok})
                    print(f"[ATTN] {STATE['n']:>3} {tag} T={T} H={H}  kernel {e_kernel:.3e}  "
                          f"floor {e_floor:.3e}  outlier q/k {oq:.0f}/{ok:.0f}", flush=True)
                except Exception as e:
                    print(f"[ATTN] call {STATE['n']} ({tag}) failed: {type(e).__name__}: {e}", flush=True)

            if STATE["n"] >= LIMIT:
                STATE["off"] = True
                summarise()
            return ref

        return probe

    for modname, tag in (("backend.nn.flux", "flux"), ("backend.nn.lumina", "nextdit")):
        try:
            mod = __import__(modname, fromlist=["attention_function"])
            mod.attention_function = wrap(mod.attention_function, tag)
            print(f"[ATTN] probing {modname}", flush=True)
        except Exception as e:
            print(f"[ATTN] cannot probe {modname}: {e}", flush=True)
