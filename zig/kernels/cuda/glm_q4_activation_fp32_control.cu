#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

__device__ __forceinline__ float control_float_identity(float x) { return x; }

__device__ __forceinline__ float act_value(uint16_t bits) {
    return __bfloat162float(__ushort_as_bfloat16(bits));
}

__device__ __forceinline__ float act_code(float v, float scale, float bias) {
    float z = __fdiv_rn(__fsub_rn(v, bias), scale);
    return float(__float2int_rn(fminf(15.f, fmaxf(0.f, z))));
}

__device__ float act_error(const float *row, const double *importance, float scale, float bias) {
    float error = 0.;
    for (int j = 0; j < 64; ++j) {
        float q = act_code(row[j], scale, bias);
        float delta = __fsub_rn(__fadd_rn(__fmul_rn(q, float(scale)), float(bias)), float(row[j]));
        error = __fadd_rn(error, __fmul_rn(importance[j], __fmul_rn(delta, delta)));
    }
    return error;
}

__device__ __forceinline__ void act_neighbors(uint16_t center, uint16_t *out) {
    out[0] = center;
    out[1] = (center & 0x7fff) == 0 ? uint16_t(0x8001) : uint16_t(center - 1);
    out[2] = (center & 0x7fff) == 0 ? uint16_t(1) : uint16_t(center == 65535 ? center : center + 1);
}

// Status: invalid bitmask, changed groups, zero-moment groups, maximum rounds used.
extern "C" __global__ void __launch_bounds__(128) tf_glm_q4_activation_fp32_control_v1(
    const __nv_bfloat16 *weights, const double *importance, int64_t groups, int64_t column_groups,
    uint32_t *words, __nv_bfloat16 *scales, __nv_bfloat16 *biases, uint32_t *status) {
    int64_t group = int64_t(blockIdx.x) * 128 + threadIdx.x;
    if (group >= groups) return;
    if (column_groups <= 0) { atomicOr(status, 1u); return; }
    const double *moments = importance + (group % column_groups) * 64;
    float row[64];
    float total = 0.;
    for (int j = 0; j < 64; ++j) {
        row[j] = __bfloat162float(weights[group * 64 + j]);
        if (!isfinite(row[j]) || !isfinite(moments[j]) || moments[j] < 0.) {
            atomicOr(status, 2u); return;
        }
        total = __fadd_rn(total, moments[j]);
    }
    uint16_t scale = __bfloat16_as_ushort(scales[group]);
    uint16_t bias = __bfloat16_as_ushort(biases[group]);
    if (!(act_value(scale) > 0.f) || !isfinite(act_value(scale)) || !isfinite(act_value(bias))) {
        atomicOr(status, 4u); return;
    }
    if (total == 0.) { atomicAdd(status + 2, 1u); return; }
    float error = act_error(row, moments, act_value(scale), act_value(bias));
    if (!isfinite(total) || !isfinite(error)) { atomicOr(status, 8u); return; }
    const uint16_t original_scale = scale, original_bias = bias;
    unsigned rounds = 0;
    for (int iteration = 0; iteration < 2; ++iteration) {
        rounds = unsigned(iteration + 1);
        const uint16_t before_scale = scale, before_bias = bias;
        float sq = 0., sx = 0., sqq = 0., sqx = 0.;
        for (int j = 0; j < 64; ++j) {
            float q = act_code(row[j], act_value(scale), act_value(bias));
            sq = __fadd_rn(sq, __fmul_rn(moments[j], q));
            sx = __fadd_rn(sx, __fmul_rn(moments[j], float(row[j])));
            sqq = __fadd_rn(sqq, __fmul_rn(moments[j], __fmul_rn(q, q)));
            sqx = __fadd_rn(sqx, __fmul_rn(moments[j], __fmul_rn(q, float(row[j]))));
        }
        float denominator = __fsub_rn(__fmul_rn(total, sqq), __fmul_rn(sq, sq));
        if (!(denominator > 0.) || !isfinite(denominator)) break;
        float fitted_scale = __fdiv_rn(__fsub_rn(__fmul_rn(total, sqx), __fmul_rn(sq, sx)), denominator);
        float fitted_bias = __fdiv_rn(__fsub_rn(sx, __fmul_rn(fitted_scale, sq)), total);
        if (!(fitted_scale > 0.) || !isfinite(fitted_scale) || !isfinite(fitted_bias)) break;
        uint16_t ss[3], bs[3];
        act_neighbors(__bfloat16_as_ushort(__float2bfloat16_rn(control_float_identity(fitted_scale))), ss);
        act_neighbors(__bfloat16_as_ushort(__float2bfloat16_rn(control_float_identity(fitted_bias))), bs);
        for (int i = 0; i < 3; ++i) {
            if (!(act_value(ss[i]) > 0.f) || !isfinite(act_value(ss[i]))) continue;
            for (int j = 0; j < 3; ++j) {
                if (!isfinite(act_value(bs[j]))) continue;
                float candidate = act_error(row, moments, act_value(ss[i]), act_value(bs[j]));
                if (candidate < error) { scale = ss[i]; bias = bs[j]; error = candidate; }
            }
        }
        if (scale == before_scale && bias == before_bias) break;
    }
    atomicMax(status + 3, rounds);
    if (scale == original_scale && bias == original_bias) return;
    for (int j = 0; j < 8; ++j) {
        uint32_t packed = 0;
        for (int i = 0; i < 8; ++i)
            packed |= uint32_t(act_code(row[j * 8 + i], act_value(scale), act_value(bias))) << (4 * i);
        words[group * 8 + j] = packed;
    }
    scales[group] = __ushort_as_bfloat16(scale);
    biases[group] = __ushort_as_bfloat16(bias);
    atomicAdd(status + 1, 1u);
}
