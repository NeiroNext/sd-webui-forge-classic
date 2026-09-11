# PROTOTYPE: W8A8 int8 GEMM for GGUF Linear layers via torch._int_mm (dp4a on Pascal). Inert unless ZZ_INT8 is set.
# ZZ_INT8=1 int8 path ; ZZ_INT8=check run both, print the int8 error per layer shape, return stock
# ZZ_INT8_CACHE=1 quantize each weight once and keep it in pinned CPU memory ; ZZ_INT8_PROF=1 per-phase cuda-event timing
import os
from collections import defaultdict

import torch

import backend.operations as ops
from backend.operations import main_stream_worker, weights_manual_cast
from backend.operations_gguf import dequantize_tensor
from modules import script_callbacks

MODE = os.environ.get("ZZ_INT8", "")
if MODE == "0":
    MODE = ""
from backend.operations_nf4 import dequantize_nf4

Linear = ops.ForgeOperationsGGUF.Linear
PlainLinear = ops.ForgeOperations.Linear  # fp16 / fp8-storage checkpoints
NF4Linear = ops.ForgeOperationsNF4.Linear

STATS = {"int8": 0, "fp16": 0, "failed": 0, "step": 0}
ERR = defaultdict(list)  # (N, K) -> relative errors seen in check mode
PROF = os.environ.get("ZZ_INT8_PROF", "0") == "1"
CACHE = os.environ.get("ZZ_INT8_CACHE", "0") == "1"  # keep the int8 weight in pinned CPU memory after the first call
from backend import memory_management
TIMES = defaultdict(float)  # phase -> ms accumulated over the current call
EXCLUDE_K = {int(k) for k in os.environ.get("ZZ_INT8_EXCLUDE_K", "").split(",") if k}  # skip layers by input width


class _Phase:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        if PROF:
            self.e0 = torch.cuda.Event(enable_timing=True)
            self.e0.record()

    def __exit__(self, *a):
        if PROF:
            e1 = torch.cuda.Event(enable_timing=True)
            e1.record()
            e1.synchronize()
            TIMES[self.name] += self.e0.elapsed_time(e1)


def int8_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    n, k = weight.shape

    with _Phase("weight quant"):
        s_w = weight.abs().amax(dim=1).float().clamp_min_(1e-8) / 127.0  # per output channel
        w8 = torch.round(weight.float() * (1.0 / s_w)[:, None]).clamp_(-127, 127).to(torch.int8)
    with _Phase("weight transpose"):
        w8 = w8.t().contiguous()  # [K, N]

    with _Phase("act quant"):
        s_x = x2.abs().amax(dim=1).float().clamp_min_(1e-8) / 127.0  # per token
        x8 = torch.round(x2.float() * (1.0 / s_x)[:, None]).clamp_(-127, 127).to(torch.int8)

    with _Phase("int_mm"):
        acc = torch._int_mm(x8, w8)
    with _Phase("rescale"):
        out = rescale(acc, s_x, s_w, bias, x.dtype)
    return out.reshape(*shape[:-1], n)


def rescale(acc, s_x, s_w, bias, dtype):
    out = acc * s_x[:, None]  # int32 * fp32 promotes to fp32 in one pass; the int32 sum itself is exact
    out.mul_(s_w[None, :])
    out = out.to(dtype)
    if bias is not None:
        out += bias
    return out


def quantize_weight(weight):
    s_w = weight.abs().amax(dim=1).float().clamp_min_(1e-8) / 127.0  # per output channel
    w8 = torch.round(weight.float() * (1.0 / s_w)[:, None]).clamp_(-127, 127).to(torch.int8).t().contiguous()  # [K, N]
    return w8, s_w


def int8_linear_cached(self, x):
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    with _Phase("weight upload"):
        w8 = self._w8.to(x.device, non_blocking=True)
        s_w = self._s_w
    with _Phase("bias"):
        bias = self.bias
        if bias is not None:
            bias = memory_management.cast_to_device(dequantize_tensor(bias), x.device, x.dtype)
    with _Phase("act quant"):
        s_x = x2.abs().amax(dim=1).float().clamp_min_(1e-8) / 127.0
        x8 = torch.round(x2.float() * (1.0 / s_x)[:, None]).clamp_(-127, 127).to(torch.int8)
    with _Phase("int_mm"):
        acc = torch._int_mm(x8, w8)
    with _Phase("rescale"):
        out = rescale(acc, s_x, s_w, bias, x.dtype)
    return out.reshape(*shape[:-1], w8.shape[1])


def eligible(x: torch.Tensor, weight: torch.Tensor, layer=None) -> bool:
    m = x.numel() // x.shape[-1]
    n, k = weight.shape
    if layer is not None and (len(layer.weight_function) > 0 or not getattr(layer, "_int8_ok", False)):  # LoRA or not the DiT
        return False
    return x.device.type == "cuda" and weight.ndim == 2 and m > 16 and k % 8 == 0 and n % 8 == 0 and k not in EXCLUDE_K


def _body(self, x, weight, bias):
    if not eligible(x, weight, self):
        STATS["fp16"] += 1
        with _Phase("fp16 linear (ineligible)"):
            return torch.nn.functional.linear(x, weight, bias)

    if MODE == "check":
        ref = torch.nn.functional.linear(x, weight, bias)
        if STATS["step"] <= 1:
            out = int8_linear(x, weight, bias)
            err = ((out.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-8)).item()
            ERR[tuple(weight.shape)].append(err)
        STATS["fp16"] += 1
        return ref

    try:
        out = int8_linear(x, weight, bias)
        STATS["int8"] += 1
        if CACHE:
            w8, s_w = quantize_weight(weight)
            self._w8 = w8.to("cpu", non_blocking=False).pin_memory()
            self._s_w = s_w
        return out
    except RuntimeError as e:
        STATS["failed"] += 1
        if STATS["failed"] == 1:
            print(f"[INT8] _int_mm failed for weight {tuple(weight.shape)}: {str(e).splitlines()[0]}", flush=True)
        return torch.nn.functional.linear(x, weight, bias)


def _cached_ok(self, x):
    return MODE == "1" and CACHE and getattr(self, "_w8", None) is not None and len(self.weight_function) == 0 and x.numel() // x.shape[-1] > 16


def forward(self, x):  # ForgeOperationsGGUF.Linear
    if self.bias is not None and self.bias.dtype != x.dtype:
        self.bias = ops.utils.tensor2parameter(dequantize_tensor(self.bias).to(x.dtype))
    if self.weight is not None and self.weight.dtype != x.dtype and getattr(self.weight, "gguf_cls", None) is None:
        self.weight = ops.utils.tensor2parameter(self.weight.to(x.dtype))
    if _cached_ok(self, x):
        STATS["int8"] += 1
        return int8_linear_cached(self, x)
    weight, bias, signal = weights_manual_cast(self, x, weight_fn=dequantize_tensor, skip_bias_dtype=True)
    with main_stream_worker(weight, bias, signal):
        return _body(self, x, weight, bias)


def forward_plain(self, x):  # ForgeOperations.Linear
    if not self.parameters_manual_cast:
        weight, bias = ops.get_weight_and_bias(self)
        return torch.nn.functional.linear(x, weight, bias)
    if _cached_ok(self, x):
        STATS["int8"] += 1
        return int8_linear_cached(self, x)
    weight, bias, signal = weights_manual_cast(self, x)
    with main_stream_worker(weight, bias, signal):
        return _body(self, x, weight, bias)


def forward_nf4(self, x):  # ForgeOperationsNF4.Linear
    if _cached_ok(self, x):
        STATS["int8"] += 1
        return int8_linear_cached(self, x)
    weight, bias, signal = weights_manual_cast(self, x, weight_fn=dequantize_nf4)
    with main_stream_worker(weight, bias, signal):
        return _body(self, x, weight, bias)


def _on_model_loaded(sd_model):
    n = 0
    try:
        dit = sd_model.forge_objects.unet.model.diffusion_model
    except AttributeError:
        return
    for m in dit.modules():
        if isinstance(m, (Linear, PlainLinear, NF4Linear)):
            m._int8_ok = True
            n += 1
    print(f"[INT8] marked {n} Linear layers of {type(dit).__name__}", flush=True)


# a merged LoRA replaces the layer's weight (set_attr in patch_weight_to_device) and unpatch_model restores it:
# both invalidate the cached int8 weight, the next forward requantises from whatever weight is current
from backend.patcher.base import ModelPatcher

_orig_patch_weight = ModelPatcher.patch_weight_to_device
_orig_unpatch = ModelPatcher.unpatch_model


def _patch_weight_to_device(self, key, *args, **kwargs):
    try:
        module = ops.utils.get_attr(self.model, key.rsplit(".", 1)[0])
        if hasattr(module, "_w8"):
            del module._w8, module._s_w
            STATS["invalidated"] = STATS.get("invalidated", 0) + 1
    except AttributeError:
        pass
    return _orig_patch_weight(self, key, *args, **kwargs)


def _unpatch_model(self, *args, **kwargs):
    n = 0
    for key in list(self.backup.keys()):  # only the layers whose weight was actually merged and is about to be restored
        try:
            m = ops.utils.get_attr(self.model, key.rsplit(".", 1)[0])
        except AttributeError:
            continue
        if hasattr(m, "_w8"):
            del m._w8, m._s_w
            n += 1
    if n:
        print(f"[INT8] unpatch_model: dropped {n} cached int8 weights", flush=True)
    return _orig_unpatch(self, *args, **kwargs)


if MODE:
    ModelPatcher.patch_weight_to_device = _patch_weight_to_device
    ModelPatcher.unpatch_model = _unpatch_model
    Linear.forward = forward
    PlainLinear.forward = forward_plain
    NF4Linear.forward = forward_nf4
    script_callbacks.on_model_loaded(_on_model_loaded)
    print(f"[INT8] mode={MODE} cache={CACHE} exclude_k={sorted(EXCLUDE_K)}", flush=True)


def _on_denoiser(params):  # fires BEFORE the model call of each step, so the counts belong to the previous call
    s = STATS["step"]  # state.sampling_step lags one call behind, count ourselves
    if 1 <= s <= 2:
        print(f"[INT8] call {s - 1}: int8 {STATS['int8']} fp16 {STATS['fp16']} failed {STATS['failed']} invalidated {STATS.get('invalidated', 0)}", flush=True)
        if PROF:
            for name, ms in sorted(TIMES.items(), key=lambda kv: -kv[1]):
                print(f"[INT8]   {name:26s} {ms:8.1f} ms", flush=True)
            TIMES.clear()
        if MODE == "check" and s == 1:
            for shape, errs in sorted(ERR.items(), key=lambda kv: -max(kv[1])):
                print(f"[INT8]   weight {shape}: n={len(errs)} rel.err mean {sum(errs)/len(errs):.2e} max {max(errs):.2e}", flush=True)
    STATS["step"] = s + 1
    STATS["int8"] = STATS["fp16"] = STATS["failed"] = 0


script_callbacks.on_cfg_denoiser(_on_denoiser)


def _on_denoised(params):
    if MODE == "check" and STATS["step"] == 1:
        for shape, errs in sorted(ERR.items(), key=lambda kv: -max(kv[1])):
            print(f"[INT8]   weight {shape}: n={len(errs)} rel.err mean {sum(errs)/len(errs):.2e} max {max(errs):.2e}", flush=True)


script_callbacks.on_cfg_denoised(_on_denoised)
