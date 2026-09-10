import json

import torch

# Dependency-free loader for the bitsandbytes 4-bit checkpoints (`flux1-dev-bnb-nf4` and friends).
# Layout per weight, as written by `QuantState.as_dict(packed=True)`:
#   <name>.weight                                   uint8 [numel/2, 1]  two 4-bit codes per byte, high nibble first
#   <name>.weight.absmax                            float32 [numel/blocksize]     (uint8 codes when nested)
#   <name>.weight.quant_map                         float32 [16]                  the codebook
#   <name>.weight.quant_state.bitsandbytes__nf4     uint8   JSON: blocksize, dtype, shape, quant_type (+ nested_*)
#   <name>.weight.nested_absmax / nested_quant_map  float32                       only when nested (double quant)
# Nibble order and blockwise math follow bitsandbytes `csrc/kernels.cu` (kDequantizeBlockwise) and
# `backends/default/ops.py` (_dequantize_4bit_compute).

QUANT_STATE_KEYS = ("quant_state.bitsandbytes__nf4", "quant_state.bitsandbytes__fp4")


def packed_size(packed_bytes: int, absmax_bytes: int, nested_absmax_bytes: int = 0) -> int:
    """Bytes `load_nf4_parameter` needs for one weight; each float32 section starts at a multiple of 4"""
    size = -packed_bytes % 4 + packed_bytes + absmax_bytes
    if nested_absmax_bytes:
        size += -size % 4 + nested_absmax_bytes
    return size


class ParameterNF4(torch.nn.Parameter):
    """
    One uint8 buffer holding the packed weights followed by the absmax bytes,
    so that moving, pinning and memory accounting see a single tensor (like ParameterGGUF)
    """

    def __init__(self, torch_tensor, *, real_shape=None, blocksize=64, code=None, packed_bytes=0, nested=None, no_init=False):
        super().__init__()
        if no_init:
            return

        self.real_shape: torch.Size = torch.Size(real_shape)
        self.blocksize: int = blocksize
        self.code: torch.Tensor = code  # float32 [16]
        self.packed_bytes: int = packed_bytes
        self.nested: dict | None = nested  # {"code": float32 [256], "blocksize": int, "offset": float, "absmax_bytes": int}
        self.computation_dtype = torch.float16
        self.quant_type = "nf4"
        self.arena: torch.Tensor | None = None  # the buffer this weight is a slice of, see memory_management.pin_memory

    def __new__(cls, torch_tensor, *, real_shape=None, blocksize=64, code=None, packed_bytes=0, nested=None, no_init=False):
        return super().__new__(cls, torch_tensor, requires_grad=False)

    @property
    def shape(self):
        return self.real_shape

    def copy_with_data(self, data):
        new = ParameterNF4(data, no_init=True)
        new.real_shape = self.real_shape
        new.blocksize = self.blocksize
        new.code = self.code.to(data.device)
        new.packed_bytes = self.packed_bytes
        new.nested = None if self.nested is None else {**self.nested, "code": self.nested["code"].to(data.device)}
        new.computation_dtype = self.computation_dtype
        new.quant_type = self.quant_type
        new.arena = self.arena if data.data_ptr() == self.data.data_ptr() else None  # `.to(cpu)` hands back the same slice
        return new

    def to(self, *args, **kwargs):
        # the packed bytes never change dtype; only the device (and copy / non_blocking) matter
        device, _, non_blocking, _ = torch._C._nn._parse_to(*args, **{k: v for k, v in kwargs.items() if k != "copy"})
        return self.copy_with_data(self.data.to(device=device, non_blocking=non_blocking, copy=kwargs.get("copy", False)))

    def pin_memory(self, device=None):
        return self.copy_with_data(torch.Tensor.pin_memory(self, device=device))


def load_nf4_parameter(state_dict: dict, key: str, device: torch.device, computation_dtype: torch.dtype, consume: bool = False, out: torch.Tensor = None) -> ParameterNF4 | None:
    """
    Build a ParameterNF4 from the `<key>` and `<key>.*` entries of a bitsandbytes checkpoint; None if `<key>` is not 4-bit.
    With `consume` the entries are dropped from `state_dict` as they are read, so each layer is freed as soon as it is packed.
    """
    qs_key = next((f"{key}.{k}" for k in QUANT_STATE_KEYS if f"{key}.{k}" in state_dict), None)
    if qs_key is None:
        return None

    take = state_dict.pop if consume else state_dict.__getitem__

    meta = json.loads(bytes(take(qs_key).tolist()).decode())
    packed = take(key).reshape(-1)
    absmax = take(f"{key}.absmax")
    nested = None

    def aligned(*parts):  # float32 sections must start at a multiple of 4 bytes to be viewed back
        parts = [part.reshape(-1).view(torch.uint8) for part in parts]
        placed, pos = [], 0
        for part in parts:
            pos += -pos % 4
            placed.append((pos, part))
            pos += part.numel()

        buffer = torch.empty(pos, dtype=torch.uint8, device=parts[0].device) if out is None else out
        for at, part in placed:
            buffer[at : at + part.numel()] = part
        return buffer

    if "nested_absmax" in meta or f"{key}.nested_absmax" in state_dict:
        # double quantization: absmax itself is stored as uint8 codes, scaled blockwise by nested_absmax
        nested = {
            "code": take(f"{key}.nested_quant_map").to(torch.float32),
            "blocksize": int(meta["nested_blocksize"]),
            "offset": float(meta["nested_offset"]),
            "absmax_bytes": absmax.numel(),
        }
        buffer = aligned(packed, absmax, take(f"{key}.nested_absmax").to(torch.float32))
    else:
        buffer = aligned(packed, absmax.to(torch.float32))

    param = ParameterNF4(
        buffer.to(device),
        real_shape=meta["shape"],
        blocksize=int(meta["blocksize"]),
        code=take(f"{key}.quant_map").to(device=device, dtype=torch.float32),
        packed_bytes=packed.numel(),
        nested=nested,
    )
    param.computation_dtype = computation_dtype
    param.quant_type = qs_key.rsplit("bitsandbytes__", 1)[1]  # once packed, the quant state keys are gone
    return param


def pack_4bit_parameters(state_dict: dict, device: torch.device = None) -> None:
    """
    Replace the several tensors bitsandbytes stores per weight with one ParameterNF4 each, in place.

    Done before `Module.load_state_dict`, which hands every submodule its own copy of the dict: those copies
    keep the raw tensors alive, so packing during the load holds the whole component in memory next to the
    buffers being built. Consuming the entries here frees each layer as soon as it is packed.
    """
    device = device or torch.device("cpu")
    for key in [k for k in list(state_dict) if any(f"{k}.{q}" in state_dict for q in QUANT_STATE_KEYS)]:
        if (param := load_nf4_parameter(state_dict, key, device, torch.float16, consume=True)) is not None:
            state_dict[key] = param


def with_4bit_shapes(state_dict: dict) -> dict:
    """
    For architecture detection: packed 4-bit weights have shape [numel/2, 1], so return a shallow copy
    where each of them is replaced by an empty meta tensor of the real shape (from the quant state)
    """
    keys = [k for k in state_dict if any(k.endswith(q) for q in QUANT_STATE_KEYS)]
    if not keys:
        return state_dict

    sd = dict(state_dict)
    for k in keys:
        meta = json.loads(bytes(state_dict[k].tolist()).decode())
        sd[k.split(".weight.")[0] + ".weight"] = torch.empty(meta["shape"], dtype=torch.bfloat16, device="meta")
    return sd


def _blockwise(values: torch.Tensor, absmax: torch.Tensor, blocksize: int) -> torch.Tensor:
    full = (values.numel() // blocksize) * blocksize
    out = values[:full].view(-1, blocksize) * absmax[: full // blocksize].view(-1, 1)
    if full < values.numel():
        out = torch.cat([out.reshape(-1), values[full:] * absmax[full // blocksize]])
    return out.reshape(-1)


def dequantize_nf4(weight: torch.Tensor) -> torch.Tensor:
    if not isinstance(weight, ParameterNF4):
        return weight

    data = weight.data
    dtype = weight.computation_dtype
    packed = data[: weight.packed_bytes]

    align = lambda n: (n + 3) // 4 * 4
    if weight.nested is None:
        absmax = data[align(weight.packed_bytes) :].view(torch.float32)
    else:
        start = align(weight.packed_bytes)
        n = weight.nested["absmax_bytes"]
        codes = data[start : start + n]
        absmax2 = data[align(start + n) :].view(torch.float32)
        absmax = _blockwise(weight.nested["code"][codes.long()], absmax2, weight.nested["blocksize"]) + weight.nested["offset"]

    # one 256-entry table gives both nibbles of a byte at once: [256, 2] = (high, low)
    # bf16 has too few mantissa bits for the blockwise multiply (~4e-3 relative error), do it in fp32 like bitsandbytes;
    # fp16 stays fp16, the error is 2.4e-4 (measured) and it halves the traffic on GPUs without fast fp32
    math_dtype = torch.float32 if dtype == torch.bfloat16 else dtype
    code = weight.code.to(math_dtype)
    byte = torch.arange(256, device=data.device)
    table = torch.stack([code[byte >> 4], code[byte & 0x0F]], dim=1)
    values = table.index_select(0, packed.int()).view(-1)[: weight.real_shape.numel()]

    return _blockwise(values, absmax.to(math_dtype), weight.blocksize).reshape(weight.real_shape).to(dtype)
