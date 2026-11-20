#include "glm_pdl.cuh"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <math.h>

extern "C" __global__ void tf_glm_head_fp8_quant_g64_v1(
    const __nv_bfloat16 *src, int64_t groups, uint8_t *codes, float *scales, uint32_t *invalid) {
    glm_pdl::enter();
    int lane = threadIdx.x & 31;
    int64_t g = int64_t(blockIdx.x) * 4 + threadIdx.x / 32;
    if (g >= groups) return;
    float values[2];
    float amax = 0.f;
    bool bad = false;
    for (int i = 0; i < 2; ++i) {
        values[i] = __bfloat162float(src[g * 64 + i * 32 + lane]);
        bad |= !isfinite(values[i]);
        amax = fmaxf(amax, fabsf(values[i]));
    }
    if (__any_sync(0xffffffff, bad)) {
        if (!lane) atomicOr(invalid, 1u);
        for (int i = 0; i < 2; ++i) values[i] = 0.f;
        amax = 0.f;
    }
    for (int d = 16; d; d >>= 1) amax = fmaxf(amax, __shfl_down_sync(0xffffffff, amax, d));
    amax = __shfl_sync(0xffffffff, amax, 0);
    float scale = fmaxf(__fmul_rn(amax, __uint_as_float(0x3b124925u)), 1e-30f);
    if (!lane) scales[g] = scale;
    for (int i = 0; i < 2; ++i) {
        float q = fminf(448.f, fmaxf(-448.f, __fdiv_rn(values[i], scale)));
        codes[g * 64 + i * 32 + lane] = __nv_cvt_float_to_fp8(q, __NV_SATFINITE, __NV_E4M3);
    }
}

// The FP8 lane uses one independently computed FP32 scale per output and 64 inputs.
extern "C" __global__ void tf_glm_head_fp8_scales_g64_v1(
    const float *src, float *out, int n, int k, int npad) {
    glm_pdl::enter();
    int64_t at = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    int kg = k / 64;
    int64_t total = int64_t(npad) * kg;
    if (at >= total) return;
    int col = at % 64;
    int g = (at / 64) % kg;
    int row = (at / (64 * kg)) * 64 + col;
    out[at] = row < n ? src[int64_t(row) * kg + g] : 1.f;
}
