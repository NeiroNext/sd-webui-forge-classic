"""
int8 GEMM for the Linear layers of a DiT: `torch._int_mm` runs on dp4a / IMMA hardware that fp16 cannot use on
Pascal (4.5x the fp32 rate on a GTX 1070). Weights are quantised once at load, per output channel; activations
per token at every call; a LoRA is added as a low-rank side branch instead of being merged into the weight.
"""

import json
import logging
import os
import struct
from functools import partial

import torch

from backend import memory_management
from backend.logging import setup_logger

logger = logging.getLogger("int8")
setup_logger(logger)

# models whose Linear GEMMs are large enough for the unfused quantise / rescale passes to pay for themselves
# (measured on a GTX 1070: Flux 1.9x, Chroma 2.8x, Z-Image 2.7x; SDXL and Wan 1.3B gain nothing)
INT8_MODELS = ("IntegratedFluxTransformer2DModel", "IntegratedChromaTransformer2DModel", "NextDiT")
MIN_ROWS = 17  # torch._int_mm needs M > 16: a modulation GEMV is padded up to it, that is cheaper than a dequantise
CACHE_VERSION = "1"  # bump when the blob layout or the quantisation changes


class ParameterInt8(torch.nn.Parameter):
    """One flat uint8 blob: the int8 weight transposed to [K, N], then the fp32 per-output-channel scale [N]"""

    def __new__(cls, data, *, real_shape=None, computation_dtype=None, requires_grad=False):
        return super().__new__(cls, data, requires_grad=False)

    def __init__(self, data, *, real_shape=None, computation_dtype=None, requires_grad=False):
        super().__init__()
        if real_shape is not None:
            self.real_shape = torch.Size(real_shape)
            self.computation_dtype = computation_dtype or torch.float16
            self.arena: torch.Tensor | None = None  # the buffer this weight is a slice of, see memory_management.pin_memory

    @property
    def shape(self):
        return self.real_shape

    def copy_with_data(self, data):
        new = ParameterInt8(data, real_shape=self.real_shape, computation_dtype=self.computation_dtype)
        new.arena = self.arena if data.data_ptr() == self.data.data_ptr() else None  # `.to(cpu)` hands back the same slice
        return new

    def detach(self):  # torch.nn.Parameter(p) keeps the subclass only if detach() returns it
        return self.copy_with_data(self.data.detach())

    def to(self, *args, **kwargs):  # the blob never changes dtype, only device
        kwargs.pop("dtype", None)
        args = tuple(a for a in args if not isinstance(a, torch.dtype))
        return self.copy_with_data(self.data.to(*args, **kwargs))

    def pin_memory(self, device=None):
        return self.copy_with_data(torch.Tensor.pin_memory(self, device=device))

    def w8(self) -> torch.Tensor:
        n, k = self.real_shape
        return self.data[: k * n].view(torch.int8).view(k, n)

    def scale(self) -> torch.Tensor:
        n, k = self.real_shape
        return self.data[k * n :].view(torch.float32)

    def dequantize(self, dtype=None) -> torch.Tensor:
        return (self.w8().t().to(torch.float32) * self.scale()[:, None]).to(dtype or self.computation_dtype)


def blob_size(n: int, k: int) -> int:
    return k * n + 4 * n


def quantize(weight: torch.Tensor, computation_dtype=torch.float16, out: torch.Tensor = None) -> ParameterInt8:
    w = weight.detach().to(torch.float32)
    s = w.abs().amax(dim=1).clamp_min_(1e-8) / 127.0
    q = torch.round(w * (1.0 / s)[:, None]).clamp_(-127, 127).to(torch.int8).t().contiguous()
    blob = torch.cat([q.view(-1).view(torch.uint8), s.view(torch.uint8)])
    if out is None:
        blob = blob.to("cpu")
    else:
        out.copy_(blob)
        blob = out
    return ParameterInt8(blob, real_shape=weight.shape, computation_dtype=computation_dtype)


def dequantize_source(weight: torch.Tensor, device: torch.device) -> torch.Tensor:
    if type(weight).__name__ == "ParameterNF4":
        from backend.operations_nf4 import dequantize_nf4

        return dequantize_nf4(weight.to(device))
    if getattr(weight, "gguf_cls", None) is not None:
        from backend.loader_gguf import dequantize

        return dequantize(weight.to(device), torch.float16)
    return weight.to(device=device, dtype=torch.float16)


def int_mm_works(device: torch.device) -> bool:
    """dp4a / IMMA: present since Pascal, but not on every device torch runs on"""
    try:
        a = torch.zeros(MIN_ROWS, 8, dtype=torch.int8, device=device)
        torch._int_mm(a, a.new_zeros(8, 8))
        return True
    except Exception as e:
        logger.warning(f"int8 GEMM unavailable on {device}: {e}")
        return False


def quantize_model(model: torch.nn.Module, source: str = None) -> int:
    name = type(model).__name__
    if name not in INT8_MODELS:
        logger.info(f"Not quantising {name}: its Linear layers are too small to gain from int8")
        return 0

    device = memory_management.get_torch_device()
    if not int_mm_works(device):
        return 0

    # only the Linear classes whose forward we hook: a mixed-precision checkpoint builds its own
    from backend.operations import ForgeOperations, ForgeOperationsGGUF, ForgeOperationsNF4

    supported = (ForgeOperations.Linear, ForgeOperationsGGUF.Linear, ForgeOperationsNF4.Linear)

    todo, skipped = [], 0
    for module_name, module in model.named_modules():
        if not isinstance(module, supported) or module.weight is None:
            continue
        n, k = module.weight.shape if module.weight.ndim == 2 else (0, 0)
        if n == 0 or n % 8 or k % 8:
            skipped += 1
            continue
        todo.append((module_name, module, (n, k)))

    if source is not None and os.path.isfile(source):
        try:
            path = cache_path(source)
            if not cache_is_valid(path, source, name, todo):
                build_cache(path, source, name, todo)
            load_cache(path, todo)
            logger.info(f"Loaded {len(todo)} int8 Linear layers of {name} from {os.path.basename(path)} ({skipped} skipped)")
            return len(todo)
        except Exception as e:
            logger.warning(f"int8 cache unavailable ({e}); quantising into RAM instead")

    quantize_into_ram(todo)
    logger.info(f"Quantised {len(todo)} Linear layers of {name} to int8 ({skipped} skipped)")
    return len(todo)


def install(module, weight: ParameterInt8):
    module.weight = weight
    module.convert_weight = convert_weight
    module.set_weight = partial(set_weight, module)


def quantize_into_ram(todo: list):
    from backend.operations_nf4 import ARENA_BYTES

    device = memory_management.get_torch_device()
    # a weight is dequantised on the GPU and the int8 blob kept on the CPU; the memory manager moves it back
    # the Windows allocator rounds ~50 MB blocks up by half: pack the blobs into a few big arenas instead
    arena, at = None, 0
    for _, module, (n, k) in todo:
        size = blob_size(n, k)
        if arena is None or at + size > arena.numel():
            arena, at = torch.empty(max(size, ARENA_BYTES), dtype=torch.uint8), 0
        weight = quantize(dequantize_source(module.weight, device), getattr(module.weight, "computation_dtype", torch.float16), out=arena[at : at + size])
        weight.arena = arena
        at += size
        install(module, weight)


# region disk cache: a safetensors file of blobs, written once weight by weight and mapped read-only afterwards,
# so neither building nor using it ever holds the int8 model in RAM


def cache_path(source: str) -> str:
    stem = os.path.splitext(os.path.basename(source))[0]
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(source))), "int8", f"{stem}.int8.safetensors")


def cache_tags(source: str, model_name: str) -> dict[str, str]:
    st = os.stat(source)
    return {"int8_version": CACHE_VERSION, "model": model_name, "source": os.path.basename(source), "source_size": str(st.st_size), "source_mtime": str(int(st.st_mtime))}


def cache_is_valid(path: str, source: str, model_name: str, todo: list) -> bool:
    if not os.path.isfile(path):
        return False
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    meta = header.pop("__metadata__", {})
    if any(meta.get(k) != v for k, v in cache_tags(source, model_name).items()):
        return False
    shapes = json.loads(meta.get("shapes", "{}"))
    return all(shapes.get(name) == [n, k] and header.get(name, {}).get("shape") == [blob_size(n, k)] for name, _, (n, k) in todo)


def build_cache(path: str, source: str, model_name: str, todo: list):
    device = memory_management.get_torch_device()
    header, at = {}, 0
    for name, _, (n, k) in todo:
        size = blob_size(n, k)
        header[name] = {"dtype": "U8", "shape": [size], "data_offsets": [at, at + size]}
        at += size
    header["__metadata__"] = dict(cache_tags(source, model_name), shapes=json.dumps({name: [n, k] for name, _, (n, k) in todo}))
    head = json.dumps(header).encode("utf-8")
    head += b" " * (-len(head) % 8)

    logger.info(f"Building the int8 cache {os.path.basename(path)} ({at / 2**30:.1f} GB)")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(head)))
        f.write(head)
        for _, module, _ in todo:
            f.write(quantize(dequantize_source(module.weight, device)).data.numpy().tobytes())
    os.replace(tmp, path)


def load_cache(path: str, todo: list):
    from backend.utils import map_safetensors

    sd, meta = map_safetensors(path, torch.device("cpu"))
    for name, module, (n, k) in todo:
        install(module, ParameterInt8(sd[name], real_shape=(n, k), computation_dtype=getattr(module.weight, "computation_dtype", torch.float16)))


# region patcher hooks: a merged LoRA dequantises, merges in fp16 and requantises


def convert_weight(weight: ParameterInt8, inplace=False) -> torch.Tensor:
    return weight.dequantize()


def set_weight(layer, out_weight: torch.Tensor, inplace_update=False, seed=None, return_weight=False):
    if return_weight:
        return out_weight
    layer.weight = quantize(out_weight, layer.weight.computation_dtype).to(out_weight.device)
    layer.__dict__.pop("_int8_lora", None)


# region LoRA side branch


def lora_entries(layer) -> list | None:
    """(strength, adapter, offset) for every plain up/down LoRA on the layer; None if anything needs the merge path"""
    entries = []
    for fn in layer.weight_function:
        if hasattr(fn, "patches"):  # LowVramPatch
            patches = fn.patches.get(fn.key, [])
        elif hasattr(fn, "patch"):  # OnlineLoRAPatch
            patches = fn.patch
        else:
            return None
        for strength, v, strength_model, offset, function in patches:
            weights = getattr(v, "weights", None)
            if getattr(v, "name", None) != "lora" or weights is None or function is not None or strength_model != 1.0:
                return None
            if weights[3] is not None or weights[4] is not None or weights[5] is not None:  # mid / dora / reshape
                return None
            entries.append((strength, v, offset))
    return entries


def lora_branch(layer, x2: torch.Tensor, out: torch.Tensor, entries: list) -> None:
    """out += strength * alpha/r * (x @ down^T) @ up^T for each LoRA, the same delta merge_lora_to_weight adds"""
    sig = tuple((id(v), float(strength)) for strength, v, _ in entries)
    cached = layer.__dict__.get("_int8_lora")
    if cached is None or cached[0] != sig:
        mats = []
        for strength, v, offset in entries:
            up, down, alpha = v.weights[0], v.weights[1], v.weights[2]
            scale = strength * (alpha / down.shape[0] if alpha is not None else 1.0)
            down_t = down.flatten(start_dim=1).to(device=x2.device, dtype=x2.dtype).t().contiguous()
            up_t = (up.flatten(start_dim=1).to(device=x2.device, dtype=torch.float32) * scale).to(x2.dtype).t().contiguous()
            mats.append((down_t, up_t, offset))
        layer._int8_lora = cached = (sig, mats)
    for down_t, up_t, offset in cached[1]:
        if offset is None:
            out += (x2 @ down_t) @ up_t
        elif offset[0] == 0:  # a LoRA trained on one part of a fused weight (q / k / v of a qkv)
            out[:, offset[1] : offset[1] + offset[2]] += (x2 @ down_t) @ up_t
        else:
            out += (x2[:, offset[1] : offset[1] + offset[2]] @ down_t) @ up_t


# region forward


def int8_linear(x: torch.Tensor, weight: ParameterInt8, bias: torch.Tensor, layer, lora: list) -> torch.Tensor:
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    rows = x2.shape[0]
    if rows < MIN_ROWS:
        x2 = torch.nn.functional.pad(x2, (0, 0, 0, MIN_ROWS - rows))
    s_x = x2.abs().amax(dim=1).float().clamp_min_(1e-8) / 127.0
    x8 = torch.round(x2.float() * (1.0 / s_x)[:, None]).clamp_(-127, 127).to(torch.int8)
    out = torch._int_mm(x8, weight.w8()) * s_x[:, None]  # int32 accumulate is exact, the scaling must stay fp32
    out.mul_(weight.scale()[None, :])
    out = out.to(x.dtype)
    if rows < MIN_ROWS:
        out, x2 = out[:rows], x2[:rows]
    if bias is not None:
        out += bias
    if lora:
        lora_branch(layer, x2, out, lora)
    return out.reshape(*shape[:-1], out.shape[-1])


def forward(layer, x: torch.Tensor) -> torch.Tensor:
    from backend.operations import main_stream_worker, weights_manual_cast

    lora = lora_entries(layer) if layer.weight_function else []
    functions, layer.weight_function = layer.weight_function, []  # the int8 blob goes through the cast untouched
    try:
        weight, bias, signal = weights_manual_cast(layer, x, skip_weight_dtype=True)
    finally:
        layer.weight_function = functions

    with main_stream_worker(weight, bias, signal):
        if lora is not None:
            return int8_linear(x, weight, bias, layer, lora)
        w = weight.dequantize(x.dtype)  # a patch the side branch cannot express: fp16 with the merge
        for f in functions:
            w = f(w)
        return torch.nn.functional.linear(x, w, bias)
