#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>
#include <type_traits>

namespace glm_q8_correlated {
constexpr unsigned mask = 0xffffffffu;
__device__ __forceinline__ int triangle(int i, int j) {
    if (i > j) { int t = i; i = j; j = t; }
    return i * (129 - i) / 2 + j - i;
}
template<class Real> __device__ Real sum(Real x) {
    for (int d = 16; d > 0; d >>= 1) x += __shfl_down_sync(mask, x, d);
    return x;
}
template<class Real> __device__ Real gradient(
    const double *g, Real ridge, int i, Real d0, Real d1) {
    Real result = 0;
    #pragma unroll 1
    for (int j = 0; j < 64; ++j) {
        Real dj = __shfl_sync(mask, j < 32 ? d0 : d1, j & 31);
        Real h = Real(g[triangle(i, j)]) + (i == j ? ridge : Real(0));
        result += h * dj;
    }
    return result;
}
// Four output groups share one input Gram tile. A warp owns all64 coordinates;
// the64 ascending updates are uniform warp control, without cross-group races.
template<bool Reduced, bool Skip> __device__ void run(
    const __nv_bfloat16 *weights, const double *covariance, int64_t groups, int64_t column_groups,
    uint32_t *words, const __half *scales, uint32_t *status, double *errors) {
    using Real = typename std::conditional<Reduced, float, double>::type;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (column_groups <= 0 || groups <= 0 || groups % column_groups != 0 || blockIdx.x >= column_groups) {
        if (tid == 0) atomicOr(status, 1u);
        return;
    }
    __shared__ double g[2080];
    __shared__ double damp;
    __shared__ unsigned bad;
    __shared__ bool zero;
    if (tid == 0) bad = 0;
    __syncthreads();
    for (int i = tid; i < 2080; i += 128) {
        double v = covariance[int64_t(blockIdx.x) * 2080 + i];
        g[i] = v;
        if (!isfinite(v)) atomicOr(&bad, 2u);
    }
    __syncthreads();
    if (tid == 0) {
        double trace = 0;
        for (int i = 0; i < 64; ++i) {
            double v = g[triangle(i, i)];
            if (v < 0) bad |= 2u;
            trace += v;
        }
        if (!isfinite(trace)) bad |= 2u;
        zero = trace == 0;
        if (zero) for (int i = 0; i < 2080; ++i) if (g[i] != 0.) bad |= 2u;
        damp = 1.e-4 * trace / 64.;
    }
    __syncthreads();
    if (bad) { if (tid == 0) atomicOr(status, bad); return; }
    const int64_t row = int64_t(blockIdx.y) * 4 + warp;
    const int64_t group = row * column_groups + blockIdx.x;
    if (group >= groups) return;  // Entire last warp, after the final block barrier.
    Real scale = Real(__half2float(scales[group]));
    Real w0 = Real(__bfloat162float(weights[group * 64 + lane]));
    Real w1 = Real(__bfloat162float(weights[group * 64 + lane + 32]));
    int q0 = int((words[group * 16 + lane / 4] >> (8 * (lane & 3))) & 255u) - 128;
    int q1 = int((words[group * 16 + 8 + lane / 4] >> (8 * (lane & 3))) & 255u) - 128;
    const int old0 = q0, old1 = q1;
    unsigned invalid = __ballot_sync(mask, !isfinite(scale) || scale < 0 || !isfinite(w0) ||
        !isfinite(w1) || q0 < -127 || q1 < -127);
    if (invalid) { if (lane == 0) atomicOr(status, 4u); return; }
    if (zero) {
        if (lane == 0) {
            if (errors) errors[group * 2] = errors[group * 2 + 1] = 0.;
            atomicAdd(status + 2, 1u);
        }
        return;
    }
    const Real ridge = Real(damp);
    Real d0 = Real(q0) * scale - w0, d1 = Real(q1) * scale - w1;
    Real r0 = gradient(g, ridge, lane, d0, d1);
    Real r1 = gradient(g, ridge, lane + 32, d0, d1);
    Real before = sum(d0 * r0 + d1 * r1);
    before = __shfl_sync(mask, before, 0);
    if (!isfinite(before) || before < 0) { if (lane == 0) atomicOr(status, 8u); return; }
    if (scale == 0) {
        if (lane == 0 && errors) errors[group * 2] = errors[group * 2 + 1] = double(before);
        return;
    }
    if (!Skip) for (int sweep = 0; sweep < 2; ++sweep) {
        #pragma unroll 1
        for (int i = 0; i < 64; ++i) {
            int qi = __shfl_sync(mask, i < 32 ? q0 : q1, i & 31);
            Real ri = __shfl_sync(mask, i < 32 ? r0 : r1, i & 31);
            Real diagonal = Real(g[triangle(i, i)]) + ridge;
            Real proposed = Real(qi) - ri / (scale * diagonal);
            if (!isfinite(proposed)) { if (lane == 0) atomicOr(status, 8u); return; }
            proposed = fmin(Real(127), fmax(Real(-127), proposed));
            int next = Reduced ? __float2int_rn(float(proposed)) : __double2int_rn(double(proposed));
            Real delta = Real(next - qi) * scale;
            Real change = Real(2) * delta * ri + delta * delta * diagonal;
            if (!(change < 0)) continue;
            if (lane == (i & 31)) { if (i < 32) q0 = next; else q1 = next; }
            r0 += (Real(g[triangle(lane, i)]) + (lane == i ? ridge : Real(0))) * delta;
            r1 += (Real(g[triangle(lane + 32, i)]) + (lane + 32 == i ? ridge : Real(0))) * delta;
        }
    }
    d0 = Real(q0) * scale - w0; d1 = Real(q1) * scale - w1;
    r0 = gradient(g, ridge, lane, d0, d1);
    r1 = gradient(g, ridge, lane + 32, d0, d1);
    Real after = sum(d0 * r0 + d1 * r1);
    after = __shfl_sync(mask, after, 0);
    if (!isfinite(after) || after < 0) { if (lane == 0) atomicOr(status, 8u); return; }
    unsigned changed = unsigned(q0 != old0) + unsigned(q1 != old1);
    for (int shift = 16; shift > 0; shift >>= 1) changed += __shfl_down_sync(mask, changed, shift);
    changed = __shfl_sync(mask, changed, 0);
    bool improved = after < before && changed != 0;
    if (lane == 0) {
        if (errors) { errors[group * 2] = double(before); errors[group * 2 + 1] = double(improved ? after : before); }
        atomicMax(status + 3, 2u);
    }
    if (!improved) return;
    // Every lane participates in each shuffle; only word owners write after.
    int first = lane & ~3;
    uint32_t a = uint32_t(q0 + 128), b = uint32_t(q1 + 128);
    uint32_t p0 = uint32_t(__shfl_sync(mask, a, first));
    uint32_t p1 = uint32_t(__shfl_sync(mask, b, first));
    for (int j = 1; j < 4; ++j) {
        p0 |= __shfl_sync(mask, a, first + j) << (8 * j);
        p1 |= __shfl_sync(mask, b, first + j) << (8 * j);
    }
    if ((lane & 3) == 0) {
        words[group * 16 + lane / 4] = p0;
        words[group * 16 + 8 + lane / 4] = p1;
    }
    if (lane == 0) atomicAdd(status + 1, 1u);
}
} // namespace glm_q8_correlated
