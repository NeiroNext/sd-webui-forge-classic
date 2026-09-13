"""
Fused device kernels for the two unfused passes around the int8 GEMM. Measured on a GTX 1070, one Flux
forward at 768x1024: the quantise chain costs 2192 ms of an 8310 ms step against 2623 for `_int_mm`
itself, because every stage of `round(x.float() * inv).clamp_().to(int8)` and of the int32 -> fp16
rescale writes a full temporary. One kernel each moves the minimum instead.

Compiled at runtime with NVRTC, so no host compiler and no CUDA toolkit are needed - only the driver.
The arithmetic is the same operation in the same precision as the PyTorch path, so the output is
bit-identical; anything that fails here falls back to that path.
"""

import hashlib
import logging
import os

import torch

from backend.logging import setup_logger

logger = logging.getLogger("int8")
setup_logger(logger)

ENABLED = os.environ.get("INT8_FUSED", "1") != "0"

SOURCE = r"""
typedef unsigned short half_t;

// fp16 through inline PTX: the cuda_fp16.h header is not available to NVRTC without a toolkit, and
// cvt.rn is the round-to-nearest-even that torch's own half conversion uses
__device__ __forceinline__ float h2f(half_t h) { float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f; }
__device__ __forceinline__ half_t f2h(float f) { half_t h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f)); return h; }

#define TPB 256
#define NWARP (TPB / 32)

__device__ __forceinline__ float block_max(float v) {
    __shared__ float red[NWARP];
    #pragma unroll
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = red[0];
        #pragma unroll
        for (int i = 1; i < NWARP; ++i) m = fmaxf(m, red[i]);
        red[0] = m;
    }
    __syncthreads();
    return red[0];
}

// x [rows, K] half -> out [rows, K] int8 and scale [rows] float, one block per row, K % 8 == 0
extern "C" __global__ void quant_rows(const uint4* __restrict__ x, uint2* __restrict__ out,
                                      float* __restrict__ scale, int K8) {
    const long long off = (long long)blockIdx.x * K8;
    const uint4* xr = x + off;
    uint2* orow = out + off;

    float m = 0.0f;
    for (int i = threadIdx.x; i < K8; i += TPB) {
        uint4 p = xr[i];
        const half_t* h = (const half_t*)&p;
        #pragma unroll
        for (int t = 0; t < 8; ++t) m = fmaxf(m, fabsf(h2f(h[t])));
    }
    m = block_max(m);

    // torch turns a division by a constant into a multiply by its reciprocal; do the same or the
    // scale comes out one ulp apart
    const float s = fmaxf(m, 1e-8f) * (float)(1.0 / 127.0);
    if (threadIdx.x == 0) scale[blockIdx.x] = s;
    const float inv = 1.0f / s;

    for (int i = threadIdx.x; i < K8; i += TPB) {
        uint4 p = xr[i];
        const half_t* h = (const half_t*)&p;
        uint2 q;
        signed char* o = (signed char*)&q;
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            float v = rintf(h2f(h[t]) * inv);
            o[t] = (signed char)fminf(fmaxf(v, -127.0f), 127.0f);
        }
        orow[i] = q;
    }
}

// acc [rows, N] int32 -> out [rows, N] half, scaled by the row and the output channel, N % 4 == 0
extern "C" __global__ void rescale_rows(const int* __restrict__ acc, const float* __restrict__ sx,
                                        const float* __restrict__ sw, const half_t* __restrict__ bias,
                                        half_t* __restrict__ out, int N, int has_bias) {
    const long long off = (long long)blockIdx.y * N;
    const float s = sx[blockIdx.y];
    const int4* a4 = (const int4*)(acc + off);
    uint2* o2 = (uint2*)(out + off);
    const int N4 = N >> 2;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N4; i += gridDim.x * blockDim.x) {
        const int4 p = a4[i];
        const int pv[4] = {p.x, p.y, p.z, p.w};
        const int base = i << 2;
        uint2 q;
        half_t* oh = (half_t*)&q;
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            float v = (float)pv[t] * s;   // the two scales are applied one after the other, as the
            v = v * sw[base + t];         // PyTorch path does, so the rounding matches
            half_t h = f2h(v);
            if (has_bias) h = f2h(h2f(h) + h2f(bias[base + t]));
            oh[t] = h;
        }
        o2[i] = q;
    }
}
"""

TPB = 256
_state = {"ready": None, "quant": None, "rescale": None, "args": {}}


def _cubin(arch: str) -> bytes:
    from cuda.bindings import nvrtc

    digest = hashlib.sha256(SOURCE.encode()).hexdigest()[:16]
    path = os.path.join(torch.hub.get_dir(), "int8_kernels", f"{digest}_{arch}.cubin")
    if os.path.isfile(path):
        with open(path, "rb") as f:
            return f.read()

    err, prog = nvrtc.nvrtcCreateProgram(SOURCE.encode(), b"int8.cu", 0, [], [])
    assert err == nvrtc.nvrtcResult.NVRTC_SUCCESS, err
    opts = [f"--gpu-architecture={arch}".encode(), b"--std=c++17"]
    if nvrtc.nvrtcCompileProgram(prog, len(opts), opts)[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        log = bytearray(nvrtc.nvrtcGetProgramLogSize(prog)[1])
        nvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError(log.decode(errors="replace").strip())
    blob = bytearray(nvrtc.nvrtcGetCUBINSize(prog)[1])
    nvrtc.nvrtcGetCUBIN(prog, blob)
    nvrtc.nvrtcDestroyProgram(prog)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(blob)
    os.replace(tmp, path)
    return bytes(blob)


def _build() -> bool:
    import numpy as np
    from cuda.bindings import driver as cu

    major, minor = torch.cuda.get_device_capability()
    cubin = _cubin(f"sm_{major}{minor}")

    torch.zeros(1, device="cuda")  # the primary context must be current before the driver API is used
    err, mod = cu.cuModuleLoadData(cubin)
    assert err == cu.CUresult.CUDA_SUCCESS, err
    for key, name in (("quant", b"quant_rows"), ("rescale", b"rescale_rows")):
        err, fn = cu.cuModuleGetFunction(mod, name)
        assert err == cu.CUresult.CUDA_SUCCESS, err
        _state[key] = fn
    _state["module"] = mod  # keep the module alive, the functions point into it

    # the driver copies the argument values at launch, so one scratch buffer per kernel is enough
    for key, spec in (("quant", "QQQi"), ("rescale", "QQQQQii")):
        bufs = [np.zeros(1, dtype=np.uint64 if c == "Q" else np.int32) for c in spec]
        ptrs = np.array([b.ctypes.data for b in bufs], dtype=np.uint64)
        _state["args"][key] = (bufs, ptrs, ptrs.ctypes.data)
    _state["np"] = np
    _state["cu"] = cu
    return True


def available() -> bool:
    if _state["ready"] is None:
        _state["ready"] = False
        if ENABLED and torch.cuda.is_available():
            try:
                _state["ready"] = _build()
                logger.info("Using fused int8 quantise / rescale kernels")
            except ImportError:  # the launcher installs it for --int8-linear / --int8-cache, the dropdown cannot
                logger.warning("Fused int8 kernels need `pip install cuda-python`; using the PyTorch path")
            except Exception as e:
                logger.warning(f"Fused int8 kernels unavailable ({type(e).__name__}: {e}); using the PyTorch path")
    return _state["ready"]


def _launch(key: str, grid: tuple, *values):
    bufs, _, addr = _state["args"][key]
    for b, v in zip(bufs, values):
        b[0] = v
    cu = _state["cu"]
    err = cu.cuLaunchKernel(_state[key], grid[0], grid[1], 1, TPB, 1, 1, 0,
                            torch.cuda.current_stream().cuda_stream, addr, 0)[0]
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuLaunchKernel: {err}")


def quantize_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row absmax scale and the int8 codes, identical to round(x.float() / s).clamp_(-127, 127)"""
    rows, k = x.shape
    if available() and x.dtype == torch.float16 and x.is_contiguous() and k % 8 == 0 and rows > 0:
        out = torch.empty(rows, k, dtype=torch.int8, device=x.device)
        scale = torch.empty(rows, dtype=torch.float32, device=x.device)
        _launch("quant", (rows, 1), x.data_ptr(), out.data_ptr(), scale.data_ptr(), k // 8)
        return out, scale

    s = torch.linalg.vector_norm(x, torch.inf, dim=1, dtype=torch.float32).clamp_min_(1e-8) / 127.0
    return torch.round(x.float() * (1.0 / s)[:, None]).clamp_(-127, 127).to(torch.int8), s


def rescale(acc: torch.Tensor, s_x: torch.Tensor, s_w: torch.Tensor, bias: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """int32 accumulator to `dtype`, scaled by row and by output channel, with the bias added"""
    rows, n = acc.shape
    if available() and dtype == torch.float16 and acc.is_contiguous() and n % 4 == 0 and rows > 0 and (bias is None or bias.dtype == torch.float16):
        out = torch.empty(rows, n, dtype=dtype, device=acc.device)
        blocks = min((n // 4 + TPB - 1) // TPB, 1024)
        _launch("rescale", (blocks, rows), acc.data_ptr(), s_x.data_ptr(), s_w.data_ptr(),
                0 if bias is None else bias.data_ptr(), out.data_ptr(), n, 0 if bias is None else 1)
        return out

    ob = acc * s_x[:rows, None]
    ob.mul_(s_w[None, :])
    ob = ob.to(dtype)
    if bias is not None:
        ob += bias
    return ob
