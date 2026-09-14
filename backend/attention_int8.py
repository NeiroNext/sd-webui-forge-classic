"""
Flash-style attention with the QK^T product in int8, for the 128-wide heads of Flux / Chroma / Z-Image.

`torch._int_mm` on QK^T is only 1.59x over fp16 because the int32 score matrix has to be written out,
and a tiled attention in plain PyTorch is 1.5-2x SLOWER than the fused fp16 kernel. Inside a kernel
the int32 accumulator never leaves the registers. What makes it win is not the int8 though: dp4a
costs nothing measurable here, and the time goes into feeding the multipliers, so each thread owns
several query rows and every element of V read from shared memory feeds that many multiplies. That
costs no extra registers - a block always holds BQ*DD accumulators, only their shape changes.

Measured on a GTX 1070 against the fused fp16 kernel that runs otherwise, at the real shapes:
Chroma x1.40 and x1.22, Flux x1.22, Z-Image x1.21 and x1.18, x1.30 at the hires token count. A step
is not all attention, so end to end that is low-to-mid single digits - Flux and Z-Image ~6-7%, Chroma
4.7%, and hires only ~2%, where the isolated ratio is highest: the hires pass streams the same weights
and its Linear layers grow with the tokens too, so attention weighs less there than the shape alone
suggests. Hires does NOT page with this on, which was the one real risk (38.7 vs 42.4 s/it worst
step, against the >60 that means the driver is paging).

The error is the cost of quantising Q and K and nothing else - over one step of each model the kernel
lands within 1e-05 of quantise-and-back through the stock kernel - and Q and K carry no outliers
after RoPE, so a per-row absmax scale needs no rotation. Against its own stock output a finished
image is 43.9 dB on Flux, 32.4 on hires, 31.5 on Chroma and 30.7 on Z-Image, the last two lower
because CFG 4 and an 8-step schedule both amplify a perturbation into a small geometric shift.

Compiled at runtime with NVRTC, so no host compiler and no CUDA toolkit are needed, only the driver.
Anything this cannot take - a mask, a custom scale, GQA, another head width, another dtype, a batch -
goes to the implementation it wraps, and so does everything if the kernel fails to build or to run.
"""

import hashlib
import logging
import os

import torch

from backend.logging import setup_logger

logger = logging.getLogger("int8")
setup_logger(logger)

ENABLED = os.environ.get("INT8_ATTN", "1") != "0"

# the winning point of a sweep over the tile sizes, the block size, the shared padding AND the number
# of query rows per thread (320 configurations): best per rows-per-thread 1 -> 0.76 of the fp16
# kernel, 2 -> 1.10, 4 -> 1.30, 8 -> 1.29, 16 -> 1.16, 32 -> 0.81
BQ, BK, NTH, LC, KPAD, VPAD, DD = 64, 32, 128, 8, 2, 4, 128
RR = BQ // (NTH // LC)
SMEM = ((BQ * (DD // 4 + KPAD) + BK * (DD // 4 + KPAD)) * 4
        + (BQ * BK + BQ + BK + BQ + BQ) * 4 + BK * (DD + VPAD) * 2)

SOURCE = r"""
typedef unsigned short half_t;
__device__ __forceinline__ float h2f(half_t h) { float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f; }
__device__ __forceinline__ half_t f2h(float f) { half_t h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f)); return h; }
__device__ __forceinline__ int dp4(int a, int b, int c) {
    int d; asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); return d;
}

#define BQ __BQ__
#define BK __BK__
#define NTH __NTH__
#define LC __LC__
#define DD 128
#define DW (DD/4)
#define QSTR (DW+__KPAD__)
#define KSTR (DW+__KPAD__)
#define VSTR (DD+__VPAD__)
#define NR (NTH/LC)
#define RR (BQ/NR)
#define CPT (DD/LC)
#define CP2 (CPT/2)
#define JPT (BK/LC)
#define NEG -1e30f

extern "C" __global__ void attn_int8(const int* __restrict__ q8, const float* __restrict__ sq,
                                     const int* __restrict__ k8, const float* __restrict__ sk,
                                     const half_t* __restrict__ v, half_t* __restrict__ out,
                                     int T, float scale, int vhs, int vls) {
    extern __shared__ int smem[];
    int* qs = smem;
    int* ks = qs + BQ * QSTR;
    float* ss = (float*)(ks + BK * KSTR);
    float* sqs = ss + BQ * BK;
    float* sks = sqs + BQ;
    float* ms = sks + BK;
    float* ls = ms + BQ;
    half_t* vs = (half_t*)(ls + BQ);

    const int h = blockIdx.y;
    const int q0 = blockIdx.x * BQ;
    const int tid = threadIdx.x;
    const long long hoff = (long long)h * T;

    for (int idx = tid; idx < BQ * DW; idx += NTH) {
        const int r = idx / DW, w = idx - r * DW;
        qs[r * QSTR + w] = (q0 + r < T) ? q8[(hoff + q0 + r) * DW + w] : 0;
    }
    for (int r = tid; r < BQ; r += NTH) { sqs[r] = (q0 + r < T) ? sq[hoff + q0 + r] : 0.0f; ms[r] = NEG; ls[r] = 0.0f; }

    const int rg = tid / LC;      /* row group: owns rows rg, rg+NR, rg+2*NR, ... */
    const int g = tid % LC;       /* lane: owns columns g, g+LC, ... and keys g, g+LC, ... */

    float acc[RR][2 * CP2];
    #pragma unroll
    for (int i = 0; i < RR; ++i)
        #pragma unroll
        for (int u = 0; u < 2 * CP2; ++u) acc[i][u] = 0.0f;
    __syncthreads();

    for (int k0 = 0; k0 < T; k0 += BK) {
        for (int idx = tid; idx < BK * DW; idx += NTH) {
            const int r = idx / DW, w = idx - r * DW;
            ks[r * KSTR + w] = (k0 + r < T) ? k8[(hoff + k0 + r) * DW + w] : 0;
        }
        /* V may be a strided slice of the fused qkv, so it is addressed by its own two strides */
        for (int idx = tid; idx < BK * DD; idx += NTH) {
            const int r = idx / DD, c = idx - r * DD;
            vs[r * VSTR + c] = (k0 + r < T)
                ? v[(long long)h * vhs + (long long)(k0 + r) * vls + c] : (half_t)0;
        }
        for (int r = tid; r < BK; r += NTH) sks[r] = (k0 + r < T) ? sk[hoff + k0 + r] : 0.0f;
        __syncthreads();

        int ai[RR][JPT];
        #pragma unroll
        for (int i = 0; i < RR; ++i)
            #pragma unroll
            for (int u = 0; u < JPT; ++u) ai[i][u] = 0;

        for (int w = 0; w < DW; ++w) {
            int qv[RR], kv[JPT];
            #pragma unroll
            for (int i = 0; i < RR; ++i) qv[i] = qs[(rg + NR * i) * QSTR + w];
            #pragma unroll
            for (int u = 0; u < JPT; ++u) kv[u] = ks[(g + LC * u) * KSTR + w];
            #pragma unroll
            for (int i = 0; i < RR; ++i)
                #pragma unroll
                for (int u = 0; u < JPT; ++u) ai[i][u] = dp4(qv[i], kv[u], ai[i][u]);
        }

        #pragma unroll
        for (int i = 0; i < RR; ++i) {
            const int row = rg + NR * i;
            const float rs = sqs[row] * scale;
            float pv[JPT];
            float mx = NEG;
            #pragma unroll
            for (int u = 0; u < JPT; ++u) {
                const int key = g + LC * u;
                pv[u] = (k0 + key < T) ? (float)ai[i][u] * rs * sks[key] : NEG;
                mx = pv[u] > mx ? pv[u] : mx;
            }
            /* every lane of the group must reach the shuffle, so it cannot be called inside a branch */
            #pragma unroll
            for (int m = 1; m < LC; m <<= 1) { float o = __shfl_xor_sync(0xffffffffu, mx, m); mx = o > mx ? o : mx; }

            const float mold = ms[row];
            const float mnew = mx > mold ? mx : mold;
            const float corr = __expf(mold - mnew);

            float sum = 0.0f;
            #pragma unroll
            for (int u = 0; u < JPT; ++u) {
                const float e = __expf(pv[u] - mnew);
                ss[row * BK + g + LC * u] = e;
                sum += e;
            }
            #pragma unroll
            for (int m = 1; m < LC; m <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, m);

            if (g == 0) { ms[row] = mnew; ls[row] = ls[row] * corr + sum; }
            #pragma unroll
            for (int u = 0; u < 2 * CP2; ++u) acc[i][u] *= corr;
        }
        __syncthreads();

        for (int j = 0; j < BK; ++j) {
            const unsigned* vr = (const unsigned*)(vs + j * VSTR) + g;
            unsigned wv[CP2];
            #pragma unroll
            for (int u = 0; u < CP2; ++u) wv[u] = vr[LC * u];
            #pragma unroll
            for (int i = 0; i < RR; ++i) {
                const float p = ss[(rg + NR * i) * BK + j];
                #pragma unroll
                for (int u = 0; u < CP2; ++u) {
                    acc[i][2 * u] += p * h2f((half_t)(wv[u] & 0xffffu));
                    acc[i][2 * u + 1] += p * h2f((half_t)(wv[u] >> 16));
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < RR; ++i) {
        const int row = rg + NR * i;
        if (q0 + row < T) {
            const float inv = 1.0f / ls[row];
            unsigned* o = (unsigned*)(out + (hoff + q0 + row) * DD) + g;
            #pragma unroll
            for (int u = 0; u < CP2; ++u)
                o[LC * u] = (unsigned)f2h(acc[i][2 * u] * inv) | ((unsigned)f2h(acc[i][2 * u + 1] * inv) << 16);
        }
    }
}
""".replace("__BQ__", str(BQ)).replace("__BK__", str(BK)).replace("__NTH__", str(NTH)) \
   .replace("__LC__", str(LC)).replace("__KPAD__", str(KPAD)).replace("__VPAD__", str(VPAD))

_state = {"ready": None, "fn": None}


def _cubin(arch: str) -> bytes:
    from cuda.bindings import nvrtc

    digest = hashlib.sha256(SOURCE.encode()).hexdigest()[:16]
    path = os.path.join(torch.hub.get_dir(), "int8_kernels", f"attn_{digest}_{arch}.cubin")
    if os.path.isfile(path):
        with open(path, "rb") as f:
            return f.read()

    err, prog = nvrtc.nvrtcCreateProgram(SOURCE.encode(), b"attn.cu", 0, [], [])
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
    if (major, minor) < (6, 1):  # dp4a
        return False
    cubin = _cubin(f"sm_{major}{minor}")

    torch.zeros(1, device="cuda")  # the primary context must be current before the driver API is used
    err, mod = cu.cuModuleLoadData(cubin)
    assert err == cu.CUresult.CUDA_SUCCESS, err
    err, fn = cu.cuModuleGetFunction(mod, b"attn_int8")
    assert err == cu.CUresult.CUDA_SUCCESS, err

    bufs = [np.zeros(1, dtype=d) for d in
            (np.uint64,) * 6 + (np.int32, np.float32, np.int32, np.int32)]
    ptrs = np.array([b.ctypes.data for b in bufs], dtype=np.uint64)
    _state.update(fn=fn, module=mod, cu=cu, bufs=bufs, addr=ptrs.ctypes.data, ptrs=ptrs)
    return True


def available() -> bool:
    if _state["ready"] is None:
        _state["ready"] = False
        if ENABLED and torch.cuda.is_available():
            try:
                _state["ready"] = _build()
            except ImportError:
                logger.warning("int8 attention needs `pip install cuda-python`; using the stock attention")
            except Exception as e:
                logger.warning(f"int8 attention unavailable ({type(e).__name__}: {e}); using the stock attention")
    return _state["ready"]


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """q, k, v: [heads, tokens, 128] fp16 with the last axis contiguous, returns a contiguous one"""
    from backend import int8_kernels

    H, T, _ = q.shape
    q8, s_q = int8_kernels.quantize_heads(q)
    k8, s_k = int8_kernels.quantize_heads(k)
    if v.stride(2) != 1:
        v = v.contiguous()
    out = torch.empty(H, T, DD, dtype=q.dtype, device=q.device)

    for b, val in zip(_state["bufs"], (q8.data_ptr(), s_q.data_ptr(), k8.data_ptr(), s_k.data_ptr(),
                                       v.data_ptr(), out.data_ptr(), T, DD ** -0.5,
                                       v.stride(0), v.stride(1))):
        b[0] = val
    cu = _state["cu"]
    err = cu.cuLaunchKernel(_state["fn"], (T + BQ - 1) // BQ, H, 1, NTH, 1, 1, SMEM,
                            torch.cuda.current_stream().cuda_stream, _state["addr"], 0)[0]
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuLaunchKernel: {err}")
    return out


def wrap(fallback):
    """Return an attention function that takes what the kernel can and hands everything else on."""

    def attention_int8(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
        if (_state["ready"] and skip_reshape and mask is None and q.dtype is torch.float16 and q.is_cuda
                and kwargs.get("scale") is None and not kwargs.get("enable_gqa")
                and q.shape[0] == 1 and q.shape[1] == heads and q.shape[3] == DD
                and k.shape == q.shape and v.shape == q.shape and q.stride(3) == 1):
            try:
                out = attention(q[0], k[0], v[0])
                if skip_output_reshape:
                    return out.unsqueeze(0)
                return out.transpose(0, 1).reshape(1, q.shape[2], heads * DD)
            except Exception as e:
                _state["ready"] = False
                logger.warning(f"int8 attention failed ({type(e).__name__}: {e}); using the stock attention from now on")
        return fallback(q, k, v, heads, mask, attn_precision, skip_reshape, skip_output_reshape, **kwargs)

    return attention_int8


def self_test(T: int = 3584, H: int = 24):
    import torch.nn.functional as F

    from backend.attention import attention_pytorch

    if not available():
        print("int8 attention is not available here")
        return None

    torch.manual_seed(0)
    q, k, v = (torch.randn(1, H, T, DD, device="cuda", dtype=torch.float16) * 0.3 for _ in range(3))
    ref = F.scaled_dot_product_attention(q, k, v)
    fn = wrap(attention_pytorch)
    for keep in (True, False):
        got = fn(q, k, v, H, skip_reshape=True, skip_output_reshape=keep)
        torch.cuda.synchronize()
        r = ref if keep else ref.transpose(1, 2).reshape(1, T, H * DD)
        err = (got.float() - r.float()).norm() / r.float().norm()
        print(f"skip_output_reshape={keep}: shape {tuple(got.shape)}, relative error vs SDPA {err:.3e}")

    # everything the kernel must refuse; each has to come back from the fallback with the right shape
    for name, call in (
        ("a mask", lambda: fn(q, k, v, H, mask=torch.zeros(T, T, device="cuda", dtype=torch.float16), skip_reshape=True)),
        ("a custom scale", lambda: fn(q, k, v, H, skip_reshape=True, scale=0.5)),
        ("bf16", lambda: fn(*(t.bfloat16() for t in (q, k, v)), H, skip_reshape=True)),
        ("head dim 64", lambda: fn(*(t[..., :64].contiguous() for t in (q, k, v)), H, skip_reshape=True)),
        ("batch 2", lambda: fn(*(t.repeat(2, 1, 1, 1) for t in (q, k, v)), H, skip_reshape=True)),
        ("no skip_reshape", lambda: fn(*(t.transpose(1, 2).reshape(1, T, H * DD) for t in (q, k, v)), H)),
    ):
        print(f"refused {name}: fell back, shape {tuple(call().shape)}")
    print(f"shared memory {SMEM} B, {RR} query rows per thread")
    return True
