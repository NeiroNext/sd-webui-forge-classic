"""Actually run Flux attention through the int8 kernel, to see it in pixels. Set ZZ_ATTN_USE=1."""

import os

if os.environ.get("ZZ_ATTN_USE"):
    import torch

    from backend.nn import flux as nnflux

    orig = nnflux.attention_function
    STATE = {"ours": 0, "fallback": 0}

    def use(q, k, v, heads, mask=None, **kw):
        if mask is None and q.dim() == 4 and q.shape[0] == 1 and q.shape[-1] == 128 and q.dtype == torch.float16:
            try:
                from backend import attention_int8 as ai

                out = ai.attention(q[0].contiguous(), k[0].contiguous(), v[0].contiguous())
                STATE["ours"] += 1
                H, T, D = out.shape
                return out.transpose(0, 1).reshape(1, T, H * D)
            except Exception as e:
                print(f"[ATTNUSE] fell back: {type(e).__name__}: {e}", flush=True)
        STATE["fallback"] += 1
        return orig(q, k, v, heads, mask=mask, **kw)

    nnflux.attention_function = use

    from modules import script_callbacks

    def report(*_):
        print(f"[ATTNUSE] int8 attention {STATE['ours']} calls, fallback {STATE['fallback']}", flush=True)

    script_callbacks.on_image_saved(report)
    print("[ATTNUSE] Flux attention runs on the int8 kernel", flush=True)
