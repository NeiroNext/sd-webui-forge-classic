"""
EXPERIMENT (branch int8-attn): a flash-style attention kernel with the QK^T product in int8.

Why a kernel at all: `torch._int_mm` on QK^T is only 1.59x over fp16 because the int32 score matrix
has to be written out (1.2 GB per call), and a PyTorch-level tiled attention is 1.5-2x SLOWER than
the fused fp16 SDPA. Inside a kernel the int32 accumulator never leaves the registers, so the dp4a
rate is the only thing that matters. Compiled with NVRTC like backend/int8_kernels.py - no toolkit.

Not wired into the model; `self_test()` is the entry point.
"""

import logging
import os

import torch

from backend.logging import setup_logger

logger = logging.getLogger("int8")
setup_logger(logger)

BQ, BK, DD = 32, 32, 128
NTH = 128

SOURCE = r"""
typedef unsigned short half_t;
__device__ __forceinline__ float h2f(half_t h) { float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f; }
__device__ __forceinline__ half_t f2h(float f) { half_t h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f)); return h; }
__device__ __forceinline__ int dp4(int a, int b, int c) {
    int d; asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); return d;
}

#define BQ 32
#define BK 32
#define DD 128
#define DW (DD/4)
#define QSTR (DW+1)
#define KSTR (DW+1)
#define VSTR (DD+8)
#define NTH 128
#define JPT (BK/4)   /* score columns per thread */
#define NEG -1e30f

// q8/k8 are the int8 rows reinterpreted as int32 words; sq/sk are the per-row scales
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
    for (int r = tid; r < BQ; r += NTH) {
        sqs[r] = (q0 + r < T) ? sq[hoff + q0 + r] : 0.0f;
        ms[r] = NEG;
        ls[r] = 0.0f;
    }

    float acc[32];
    #pragma unroll
    for (int u = 0; u < 32; ++u) acc[u] = 0.0f;

    const int row = tid >> 2;          // 0..BQ-1, the query this thread works on
    const int grp = tid & 3;           // 0..3
    const int jc = grp * JPT;          // its slice of the score tile
    const int pc = grp;                // interleaved columns: grp, grp+4, ... so the four
                                       // lanes of a row never land in the same shared bank
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

        int ai[JPT];
        #pragma unroll
        for (int u = 0; u < JPT; ++u) ai[u] = 0;
        for (int w = 0; w < DW; ++w) {
            const int qv = qs[row * QSTR + w];
            #pragma unroll
            for (int u = 0; u < JPT; ++u) ai[u] = dp4(qv, ks[(jc + u) * KSTR + w], ai[u]);
        }

        const float rs = sqs[row] * scale;
        float mx = NEG;
        float pv[JPT];
        #pragma unroll
        for (int u = 0; u < JPT; ++u) {
            pv[u] = (k0 + jc + u < T) ? (float)ai[u] * rs * sks[jc + u] : NEG;
            mx = pv[u] > mx ? pv[u] : mx;
        }
        // every lane of the group of four must reach the shuffle, so no calling it inside a branch
        float o1 = __shfl_xor_sync(0xffffffffu, mx, 1);
        mx = o1 > mx ? o1 : mx;
        float o2 = __shfl_xor_sync(0xffffffffu, mx, 2);
        mx = o2 > mx ? o2 : mx;

        const float mold = ms[row];
        const float mnew = mx > mold ? mx : mold;
        const float corr = __expf(mold - mnew);

        float sum = 0.0f;
        #pragma unroll
        for (int u = 0; u < JPT; ++u) {
            const float e = __expf(pv[u] - mnew);
            ss[row * BK + jc + u] = e;
            sum += e;
        }
        sum += __shfl_xor_sync(0xffffffffu, sum, 1);
        sum += __shfl_xor_sync(0xffffffffu, sum, 2);

        if (grp == 0) {
            ms[row] = mnew;
            ls[row] = ls[row] * corr + sum;
        }
        #pragma unroll
        for (int u = 0; u < 32; ++u) acc[u] *= corr;
        __syncthreads();

        for (int j = 0; j < BK; ++j) {
            const float p = ss[row * BK + j];
            // one 32-bit load per two columns: half the shared traffic, and the four lanes of a
            // row read four consecutive words, so no bank conflict
            const unsigned* vr = (const unsigned*)(vs + j * VSTR) + pc;
            #pragma unroll
            for (int u = 0; u < 16; ++u) {
                const unsigned w = vr[4 * u];
                acc[2 * u] += p * h2f((half_t)(w & 0xffffu));
                acc[2 * u + 1] += p * h2f((half_t)(w >> 16));
            }
        }
        __syncthreads();
    }

    if (q0 + row < T) {
        const float inv = 1.0f / ls[row];
        unsigned* o = (unsigned*)(out + (hoff + q0 + row) * DD) + pc;
        #pragma unroll
        for (int u = 0; u < 16; ++u)
            o[4 * u] = (unsigned)f2h(acc[2 * u] * inv) | ((unsigned)f2h(acc[2 * u + 1] * inv) << 16);
    }
}
"""

SMEM = (BQ * (DD // 4 + 1) + BK * (DD // 4 + 1)) * 4 + (BQ * BK + BQ + BK + BQ + BQ) * 4 + BK * (DD + 8) * 2
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
    s = x.abs().amax(dim=2).float().clamp_min(1e-8) / 127.0
    q = torch.round(x.float() / s[..., None]).clamp_(-127, 127).to(torch.int8)
    return q.contiguous(), s.contiguous()


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
    print(f"shared memory {SMEM} B, grid {(T + BQ - 1) // BQ}x{H}, relative error vs SDPA {err:.3e}")
    return err
