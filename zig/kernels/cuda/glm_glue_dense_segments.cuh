// Dense nonpaged shared windows: reuse each sequence's chunk math with a row descriptor.
#pragma once
#include "glm_pdl.cuh"
#include "glm_glue_attn.cuh"

namespace glm_dense_segments {
struct Row {
  unsigned long long cache, scales, position;
  int row0, chunks;
};
static_assert(sizeof(Row) == 32, "dense row ABI");
static_assert(alignof(Row) == 8, "dense row alignment");

template <bool FP8>
__device__ __forceinline__ void chunks(const Row* table, const __nv_bfloat16* qa, float* po,
                                     float* pm, float* pl, long long rows, long long heads, float scale) {
  const int r = blockIdx.x;
  if (r >= rows) return;
  const Row d = table[r];
  if (blockIdx.z >= d.chunks) return;
  // The original body uses global r; subtract the segment's first row for its causal position.
  const int origin = *reinterpret_cast<const int*>(d.position) - d.row0;
  glm_attn::chunks_wide<16, FP8>(qa, reinterpret_cast<const __nv_bfloat16*>(d.cache), &origin,
      nullptr, nullptr, po, pm, pl, rows, heads, 0, scale, 0, nullptr, 0,
      reinterpret_cast<const float*>(d.scales));
}
}  // namespace glm_dense_segments

extern "C" __global__ void __launch_bounds__(256)
tf_glm_dense_seg_chunks_16_v1(const glm_dense_segments::Row* table, const __nv_bfloat16* qa,
                             float* po, float* pm, float* pl, long long rows, long long heads, float scale) {
    glm_pdl::enter();
  glm_dense_segments::chunks<false>(table, qa, po, pm, pl, rows, heads, scale);
}

extern "C" __global__ void __launch_bounds__(256)
tf_glm_dense_seg_chunks_16_fp8_v1(const glm_dense_segments::Row* table, const __nv_bfloat16* qa,
                                 float* po, float* pm, float* pl, long long rows, long long heads, float scale) {
    glm_pdl::enter();
  glm_dense_segments::chunks<true>(table, qa, po, pm, pl, rows, heads, scale);
}

// Same merge arithmetic as tf_glm_merge; only the partial row stride and per-row chunk count are selected here.
extern "C" __global__ void __launch_bounds__(128)
tf_glm_dense_seg_merge_v1(const glm_dense_segments::Row* table, const float* po, const float* pm,
                         const float* pl, __nv_bfloat16* out, long long rows, long long heads) {
    glm_pdl::enter();
  constexpr int LW = glm_attn::LW;
  const int64_t r = blockIdx.x, h = blockIdx.y;
  const int nch = table[r].chunks;
  const int k0 = threadIdx.x * 4;
  float m = -INFINITY, l = 0.0f, o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int c = 0; c < nch; ++c) {
    const int64_t base = (c * rows + r) * heads + h;
    const float cm = pm[base], cl = pl[base];
    const bool active = cl > 0.0f;
    const float nm = active ? fmaxf(m, cm) : m;
    const float b = active ? glm_triton::exp_approx(cm - nm) : 0.0f;
    if (c == 0) {
      for (int j = 0; j < 4; ++j) o[j] = __fmaf_rn(b, po[base * LW + k0 + j], 0.0f);
      l = __fmaf_rn(b, cl, 0.0f);
    } else {
      const float a = active ? (m == -INFINITY ? 0.0f : glm_triton::exp_approx(m - nm)) : 1.0f;
      for (int j = 0; j < 4; ++j) o[j] = __fmaf_rn(o[j], a, __fmul_rn(b, po[base * LW + k0 + j]));
      l = __fmaf_rn(l, a, __fmul_rn(b, cl));
    }
    m = nm;
  }
  for (int j = 0; j < 4; ++j) out[(r * heads + h) * LW + k0 + j] = __float2bfloat16_rn(glm_triton::div_full(o[j], l));
}
