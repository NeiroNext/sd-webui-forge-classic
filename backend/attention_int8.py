"""
EXPERIMENT (branch int8-attn): a flash-style attention kernel with the QK^T product in int8.

Why a kernel at all: `torch._int_mm` on QK^T is only 1.59x over fp16 because the int32 score matrix
has to be written out (1.2 GB per call), and a PyTorch-level tiled attention is 1.5-2x SLOWER than
the fused fp16 SDPA. Inside a kernel the int32 accumulator never leaves the registers, so the dp4a
rate is the only thing that matters. Compiled with NVRTC like backend/int8_kernels.py - no toolkit.

What made it finally beat SDPA is NOT the int8: an ablation (scratchpad/attn_ablate.py) showed dp4a
costs nothing at all here, and 71% of the time went into the V path - staging it, converting it from
fp16 and multiplying it - because one thread owned one query row, so every element loaded from
shared fed exactly one multiply. RR rows per thread make the same load feed RR multiplies at no
extra register cost: a block always holds BQ*DD accumulators, and only their shape changes.

Not wired into the model; `self_test()` is the only entry point.
"""

import logging
import os

import torch

from backend.logging import setup_logger

logger = logging.getLogger("int8")
setup_logger(logger)

# the winning point of scratchpad/attn_tune2.py, which swept the rows-per-thread axis as well
BQ, BK, NTH, LC, KPAD, VPAD, DD = 64, 32, 128, 8, 2, 4, 128
RR = BQ // (NTH // LC)

SOURCE = r"""
typedef unsigned short half_t;
__device__ __forceinline__ float h2f(half_t h) { float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f; }
__device__ __forceinline__ half_t f2h(float f) { half_t h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f)); return h; }
__device__ __forceinline__ int dp4(int a, int b, int c) {
    int d; asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); return d;
}

#define BQ 64
#define BK 32
#define NTH 128
#define LC 8
#define DD 128
#define DW (DD/4)
#define QSTR (DW+2)
#define KSTR (DW+2)
#define VSTR (DD+4)
#define NR (NTH/LC)
#define RR (BQ/NR)
#define CPT (DD/LC)
#define CP2 (CPT/2)
#define JPT (BK/LC)
#define NEG -1e30f

extern "C" __global__ void attn_int8(const int* __restrict__ q8, const float* __restrict__ sq,
                                     const int* __restrict__ k8, const float* __restrict__ sk,
                                     const half_t* __restrict__ v, half_t* __restrict__ out,
                                     int T, float scale) {
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
        for (int idx = tid; idx < BK * DD; idx += NTH) {
            const int r = idx / DD, c = idx - r * DD;
            vs[r * VSTR + c] = (k0 + r < T) ? v[(hoff + k0 + r) * DD + c] : (half_t)0;
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
"""

SMEM = ((BQ * (DD // 4 + KPAD) + BK * (DD // 4 + KPAD)) * 4
        + (BQ * BK + BQ + BK + BQ + BQ) * 4 + BK * (DD + VPAD) * 2)
_state = {"fn": None, "args": None}


def _build():
    import numpy as np
    from cuda.bindings import driver as cu
    from cuda.bindings import nvrtc

    major, minor = torch.cuda.get_device_capability()
    err, prog = nvrtc.nvrtcCreateProgram(SOURCE.encode(), b"attn.cu", 0, [], [])
    assert err == nvrtc.nvrtcResult.NVRTC_SUCCESS, err
    opts = [f"--gpu-architecture=sm_{major}{minor}".encode(), b"--std=c++17"]
    if nvrtc.nvrtcCompileProgram(prog, len(opts), opts)[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        log = bytearray(nvrtc.nvrtcGetProgramLogSize(prog)[1])
        nvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError(log.decode(errors="replace").strip())
    blob = bytearray(nvrtc.nvrtcGetCUBINSize(prog)[1])
    nvrtc.nvrtcGetCUBIN(prog, blob)
    nvrtc.nvrtcDestroyProgram(prog)

    torch.zeros(1, device="cuda")
    err, mod = cu.cuModuleLoadData(bytes(blob))
    assert err == cu.CUresult.CUDA_SUCCESS, err
    err, fn = cu.cuModuleGetFunction(mod, b"attn_int8")
    assert err == cu.CUresult.CUDA_SUCCESS, err

    bufs = [np.zeros(1, dtype=d) for d in
            (np.uint64, np.uint64, np.uint64, np.uint64, np.uint64, np.uint64, np.int32, np.float32)]
    ptrs = np.array([b.ctypes.data for b in bufs], dtype=np.uint64)
    _state.update(fn=fn, module=mod, cu=cu, bufs=bufs, addr=ptrs.ctypes.data, ptrs=ptrs)
    return True


def ready() -> bool:
    if _state["fn"] is None:
        _build()
    return True


def quantize_rows(x: torch.Tensor):
    from backend import int8_kernels

    h, t, d = x.shape
    q, s = int8_kernels.quantize_rows(x.reshape(h * t, d))
    return q, s.reshape(h, t)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """q, k, v: [heads, tokens, 128] fp16"""
    ready()
    H, T, D = q.shape
    assert D == DD and q.dtype == torch.float16
    q8, sq = quantize_rows(q)
    k8, sk = quantize_rows(k)
    v = v.contiguous()
    out = torch.empty_like(q)

    for b, val in zip(_state["bufs"], (q8.data_ptr(), sq.data_ptr(), k8.data_ptr(), sk.data_ptr(),
                                       v.data_ptr(), out.data_ptr(), T, D ** -0.5)):
        b[0] = val
    cu = _state["cu"]
    err = cu.cuLaunchKernel(_state["fn"], (T + BQ - 1) // BQ, H, 1, NTH, 1, 1, SMEM,
                            torch.cuda.current_stream().cuda_stream, _state["addr"], 0)[0]
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuLaunchKernel: {err}")
    return out


def self_test(T: int = 3584, H: int = 24):
    import torch.nn.functional as F

    torch.manual_seed(0)
    q, k, v = (torch.randn(H, T, DD, device="cuda", dtype=torch.float16) * 0.3 for _ in range(3))
    ref = F.scaled_dot_product_attention(q[None], k[None], v[None])[0]
    got = attention(q, k, v)
    torch.cuda.synchronize()
    err = (got.float() - ref.float()).norm() / ref.float().norm()
    print(f"shared memory {SMEM} B, grid {(T + BQ - 1) // BQ}x{H}, {RR} query rows per thread, "
          f"relative error vs SDPA {err:.3e}")
    return err
