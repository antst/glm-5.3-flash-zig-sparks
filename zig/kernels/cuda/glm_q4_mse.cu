#include "glm_pdl.cuh"
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

__device__ __forceinline__ float sum32(float v) {
    for (int d = 16; d; d >>= 1) v = __fadd_rn(v, __shfl_down_sync(0xffffffff, v, d));
    return __shfl_sync(0xffffffff, v, 0);
}

__device__ __forceinline__ float code(float v, float s, float b) {
    float z = __fdiv_rn(__fsub_rn(v, b), s);
    z = fminf(15.f, fmaxf(0.f, z));
    return float(__float2int_rn(z));
}

// One warp owns a complete group; words use eight low-to-high nibbles as qmm.quantize4_mse_raw.
extern "C" __global__ void tf_glm_q4_mse_v1(
    const __nv_bfloat16 *w, int64_t groups, uint32_t *words,
    __nv_bfloat16 *scales, __nv_bfloat16 *biases, uint32_t *invalid) {
    glm_pdl::enter();
    int lane = threadIdx.x & 31;
    int64_t g = int64_t(blockIdx.x) * 4 + threadIdx.x / 32;
    if (g >= groups) return;
    float x = __bfloat162float(w[g * 64 + 2 * lane]);
    float y = __bfloat162float(w[g * 64 + 2 * lane + 1]);
    bool bad = !isfinite(x) || !isfinite(y);
    if (__any_sync(0xffffffff, bad)) {
        if (!lane) atomicOr(invalid, 1u);
        x = y = 0.f;
    }
    float lo = fminf(x, y), hi = fmaxf(x, y);
    for (int d = 16; d; d >>= 1) {
        lo = fminf(lo, __shfl_down_sync(0xffffffff, lo, d));
        hi = fmaxf(hi, __shfl_down_sync(0xffffffff, hi, d));
    }
    lo = __shfl_sync(0xffffffff, lo, 0);
    hi = __shfl_sync(0xffffffff, hi, 0);
    constexpr float clips[16] = {1.f, .97f, .94f, .91f, .88f, .85f, .82f, .79f,
                                 .76f, .73f, .70f, .67f, .64f, .61f, .58f, .55f};
    float best = INFINITY, bs = 0.f, bb = 0.f;
    for (int i = 0; i < 16; ++i) {
        float l = __fmul_rn(lo, clips[i]), h = __fmul_rn(hi, clips[i]);
        float s = __bfloat162float(__float2bfloat16_rn(fmaxf(__fdiv_rn(__fsub_rn(h, l), 15.f), 1e-8f)));
        float b = __bfloat162float(__float2bfloat16_rn(l));
        if (!isfinite(s) || !isfinite(b) || !(s > 0.f)) {
            if (!lane) atomicOr(invalid, 2u);
            s = 1.f;
            b = 0.f;
        }
        float ex = __fsub_rn(__fadd_rn(__fmul_rn(code(x, s, b), s), b), x);
        float ey = __fsub_rn(__fadd_rn(__fmul_rn(code(y, s, b), s), b), y);
        float err = sum32(__fadd_rn(__fmul_rn(ex, ex), __fmul_rn(ey, ey)));
        if (!isfinite(err)) {
            if (!lane) atomicOr(invalid, 4u);
        }
        if (i == 0 || err < best) {
            best = err;
            bs = s;
            bb = b;
        }
    }
    uint32_t q = uint32_t(code(x, bs, bb)) | (uint32_t(code(y, bs, bb)) << 4);
    uint32_t q1 = __shfl_down_sync(0xffffffff, q, 1);
    uint32_t q2 = __shfl_down_sync(0xffffffff, q, 2);
    uint32_t q3 = __shfl_down_sync(0xffffffff, q, 3);
    if (!(lane & 3)) words[g * 8 + lane / 4] = q | (q1 << 8) | (q2 << 16) | (q3 << 24);
    if (!lane) {
        scales[g] = __float2bfloat16_rn(bs);
        biases[g] = __float2bfloat16_rn(bb);
    }
}
