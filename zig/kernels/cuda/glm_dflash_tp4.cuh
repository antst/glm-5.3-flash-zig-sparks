#include "glm_pdl.cuh"
// Private unqualified TP4 operators; context rings retain their full eight-KV-head row layout.
namespace {
constexpr int TP_QH = 8;
constexpr int TP_KVW = 2 * HD;
}

extern "C" __global__ void df_block_qkv_tp4_v1(const __nv_bfloat16* __restrict__ qkv, int ldq, const __nv_bfloat16* __restrict__ qn,
                                        const __nv_bfloat16* __restrict__ kn, const int* __restrict__ pos, float eps, float theta,
                                        __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ kb, __nv_bfloat16* __restrict__ vb) {
    glm_pdl::enter();
    __shared__ float red[2];
    const int r = blockIdx.x, h = blockIdx.y;
    const __nv_bfloat16* row = qkv + (long long)r * ldq;
    if (h < TP_QH) {
        normRope(row + h * HD, qn, eps, pos[r], theta, q + (long long)r * TP_QH * HD + h * HD, red);
    } else {
        const int kh = h - TP_QH;
        normRope(row + TP_QH * HD + kh * HD, kn, eps, pos[r], theta, kb + (long long)r * TP_KVW + kh * HD, red);
        const __nv_bfloat16* v = row + TP_QH * HD + TP_KVW + kh * HD;
        __nv_bfloat16* vo = vb + (long long)r * TP_KVW + kh * HD;
        vo[threadIdx.x] = v[threadIdx.x];
        vo[threadIdx.x + 64] = v[threadIdx.x + 64];
    }
}

extern "C" __global__ void __launch_bounds__(256) df_attn_part_tp4_v1(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kb,
        const __nv_bfloat16* __restrict__ vb, const uint64_t* __restrict__ rings, const long long* __restrict__ starts,
        const long long* __restrict__ froms, int layer, float scale, float* __restrict__ pm, float* __restrict__ pl,
        float* __restrict__ pacc, int kh0) {
    glm_pdl::enter();
    const int s = blockIdx.x, kh = blockIdx.y, c = blockIdx.z;
    const int j = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = s * BLOCK + j;
    const long long start = starts[s], p = start + j;
    long long lo = p - (WIN - 1);
    if (lo < froms[s]) lo = froms[s];
    if (lo < 0) lo = 0;
    const long long n_ctx = start > lo ? start - lo : 0;
    const long long T = n_ctx + BLOCK, L = (T + CH - 1) / CH;
    const long long n0 = (long long)c * L, n1 = n0 + L < T ? n0 + L : T;
    float qv[4][4], m[4], l[4], acc[4][4];
    for (int h = 0; h < 4; ++h) {
        const __nv_bfloat16* qh = q + (long long)r * TP_QH * HD + (kh * 4 + h) * HD + lane * 4;
        for (int i = 0; i < 4; ++i) qv[h][i] = bf(qh[i]) * scale;
        m[h] = -INFINITY;
        l[h] = 0.f;
        for (int i = 0; i < 4; ++i) acc[h][i] = 0.f;
    }
    const __nv_bfloat16* K = ringK(rings[s], layer);
    const __nv_bfloat16* V = ringV(rings[s], layer);
    for (long long n = n0; n < n1; ++n) {
        const __nv_bfloat16 *kr, *vr;
        if (n < n_ctx) {
            const int slot = (int)((lo + n) % WIN);
            kr = K + (long long)slot * KVW + (kh0 + kh) * HD + lane * 4;
            vr = V + (long long)slot * KVW + (kh0 + kh) * HD + lane * 4;
        } else {
            const long long rb = (long long)s * BLOCK + (n - n_ctx);
            kr = kb + rb * TP_KVW + kh * HD + lane * 4;
            vr = vb + rb * TP_KVW + kh * HD + lane * 4;
        }
        float kf[4], vf[4];
        for (int i = 0; i < 4; ++i) {
            kf[i] = bf(kr[i]);
            vf[i] = bf(vr[i]);
        }
        for (int h = 0; h < 4; ++h) {
            float d = qv[h][0] * kf[0] + qv[h][1] * kf[1] + qv[h][2] * kf[2] + qv[h][3] * kf[3];
            for (int off = 16; off > 0; off >>= 1) d += __shfl_xor_sync(0xffffffffu, d, off);
            const float mn = fmaxf(m[h], d);
            const float corr = __expf(m[h] - mn), pe = __expf(d - mn);
            l[h] = l[h] * corr + pe;
            for (int i = 0; i < 4; ++i) acc[h][i] = acc[h][i] * corr + pe * vf[i];
            m[h] = mn;
        }
    }
    for (int h = 0; h < 4; ++h) {
        const long long at = ((long long)r * TP_QH + kh * 4 + h) * CH + c;
        float* pa = pacc + at * HD + lane * 4;
        for (int i = 0; i < 4; ++i) pa[i] = acc[h][i];
        if (lane == 0) {
            pm[at] = m[h];   // -inf with l = 0 for a chunk past the row's keys: the merge weighs it 0
            pl[at] = l[h];
        }
    }
}

extern "C" __global__ void df_attn_merge_tp4_v1(const float* __restrict__ pm, const float* __restrict__ pl, const float* __restrict__ pacc,
                                         __nv_bfloat16* __restrict__ o) {
    glm_pdl::enter();
    const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const long long base = ((long long)r * TP_QH + h) * CH;
    float m = -INFINITY;
    for (int c = 0; c < CH; ++c) m = fmaxf(m, pm[base + c]);
    float l = 0.f, a = 0.f;
    for (int c = 0; c < CH; ++c) {
        const float w = __expf(pm[base + c] - m);
        l += pl[base + c] * w;
        a += pacc[(base + c) * HD + d] * w;
    }
    o[(long long)r * TP_QH * HD + h * HD + d] = tobf(a / l);
}
