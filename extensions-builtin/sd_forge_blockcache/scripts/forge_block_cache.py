import os

import gradio as gr

from modules import scripts, shared
from modules.ui_components import InputAccordion

# Threshold file used by logs/fbcache_sweep.sh; only honoured when the launcher sets FBCACHE_SWEEP=1
SWEEP_FILE = "logs/fbcache_threshold.txt"


def _sweep_override():
    if os.environ.get("FBCACHE_SWEEP") != "1":
        return None
    try:
        with open(SWEEP_FILE) as f:
            parts = f.read().split()
        threshold = float(parts[0])
        if threshold <= 0:
            return None
        return threshold, int(parts[1]) if len(parts) > 1 else 3
    except Exception:
        return None


class BlockCacheForForge(scripts.Script):
    sorting_priority = 19
    hook = None

    def title(self):
        return "Block Cache"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with InputAccordion(False, label=self.title()) as enabled:
            gr.Markdown("Skip transformer blocks on steps that barely change the image. DiT models only (Flux, Chroma, Z-Image).")
            threshold = gr.Slider(minimum=0.0, maximum=0.5, value=0.12, step=0.01, label="Threshold", info="higher = more skipping = faster, less detail ; 0.08-0.12 is a sane range")
            with gr.Row():
                warmup = gr.Number(value=2, precision=0, minimum=0, label="Uncached starting steps")
                max_consecutive = gr.Number(value=3, precision=0, minimum=0, label="Max consecutive cached steps", info="0 = unlimited")
            keep_last = gr.Checkbox(True, label="Never cache the last step")

        return [enabled, threshold, warmup, max_consecutive, keep_last]

    def process_before_every_sampling(self, p, enabled: bool, threshold: float, warmup: int, max_consecutive: int, keep_last: bool, **kwargs):
        if BlockCacheForForge.hook is not None:
            BlockCacheForForge.hook.remove()
            BlockCacheForForge.hook = None

        self.state = None

        if (override := _sweep_override()) is not None:
            enabled, threshold, max_consecutive = True, override[0], override[1]

        if not enabled or threshold <= 0:
            return

        unet = p.sd_model.forge_objects.unet.clone()
        dm = unet.model.diffusion_model
        if not hasattr(dm, "double_blocks") or not hasattr(dm, "final_layer"):
            print("[BlockCache] not a supported DiT model; skipped")
            return

        warmup = max(int(warmup), 0)
        max_consecutive = max(int(max_consecutive), 0)
        st = {"prev": None, "residual": None, "base": None, "skip": False, "calls": 0, "skips": 0, "consecutive": 0}
        self.state = st

        def cacheable() -> bool:
            step, total = shared.state.sampling_step, shared.state.sampling_steps
            if step < warmup:
                return False
            if keep_last and total and step >= total - 1:
                return False
            return not (max_consecutive and st["consecutive"] >= max_consecutive)

        def first_block(a, extra):
            # the blocks update img in place ("img += ..."), so the input and the block output
            # have to be copied before anything downstream writes into that storage
            before = a["img"].clone()
            out = extra["original_block"](a)
            residual = out["img"] - before

            prev = st["prev"]
            skip = False
            if prev is not None and st["residual"] is not None and prev.shape == residual.shape and cacheable():
                relative = ((residual - prev).abs().mean() / prev.abs().mean().clamp(min=1e-6)).item()
                skip = relative < threshold

            st["prev"] = residual
            st["base"] = out["img"].clone()
            st["skip"] = skip
            st["consecutive"] = st["consecutive"] + 1 if skip else 0
            st["calls"] += 1
            st["skips"] += int(skip)
            return out

        def passthrough_double(a, extra):
            if st["skip"]:
                return {"img": a["img"], "txt": a["txt"]}
            return extra["original_block"](a)

        def passthrough_single(a, extra):
            if st["skip"]:
                return {"img": a["img"]}
            return extra["original_block"](a)

        unet.set_model_patch_replace(first_block, "dit", "double_block", 0)
        for i in range(1, len(dm.double_blocks)):
            unet.set_model_patch_replace(passthrough_double, "dit", "double_block", i)
        for i in range(len(dm.single_blocks)):
            unet.set_model_patch_replace(passthrough_single, "dit", "single_block", i)

        def before_final_layer(module, inputs):
            image, base = inputs[0], st["base"]
            if base is None or base.shape != image.shape:
                return None
            if not st["skip"]:
                st["residual"] = image - base
            elif st["residual"] is not None:
                return (image + st["residual"],) + tuple(inputs[1:])
            return None

        BlockCacheForForge.hook = dm.final_layer.register_forward_pre_hook(before_final_layer)
        p.sd_model.forge_objects.unet = unet
        p.extra_generation_params["Block Cache"] = f"{threshold} / {warmup} / {max_consecutive} / {'keep last' if keep_last else 'cache last'}"

    def postprocess(self, p, processed, *args):
        if (st := getattr(self, "state", None)) is not None:
            print(f"[BlockCache] skipped {st['skips']} of {st['calls']} model calls")
        if BlockCacheForForge.hook is not None:
            BlockCacheForForge.hook.remove()
            BlockCacheForForge.hook = None
