#pragma once
#include "glm_pdl.cuh"

namespace glm_kda_segments {
struct Segment {
  unsigned long long p, conv, state_in, state_out, k, v, g, b;
  int row0, rows;
};
static_assert(sizeof(Segment) == 72, "KDA segment ABI");
static_assert(alignof(Segment) == 8, "KDA segment alignment");
}  // namespace glm_kda_segments

extern "C" __global__ void __launch_bounds__(512)
tf_glm_kda_seg_prep_v1(const glm_kda_segments::Segment* table, int count, int H,
                      int p_stride, int b_off, const __nv_bfloat16* A, int a_stride,
                      const float* cw, const float* a_log, const float* dt_bias, float lower, float* q) {
    glm_pdl::enter();
  if (blockIdx.z >= count) return;
  const auto d = table[blockIdx.z];
  if (blockIdx.x >= d.rows) return;  // uniform CTA gate before the original body's barriers
  prep_kernel_body(H, reinterpret_cast<const __nv_bfloat16*>(d.p), p_stride, b_off,
      A + (size_t)d.row0 * a_stride, a_stride, reinterpret_cast<const __nv_bfloat16*>(d.conv),
      cw, a_log, dt_bias, lower, q + (size_t)d.row0 * H * DK,
      reinterpret_cast<float*>(d.k), reinterpret_cast<__nv_bfloat16*>(d.v),
      reinterpret_cast<float*>(d.g), reinterpret_cast<float*>(d.b));
}

extern "C" __global__ void __launch_bounds__(128)
tf_glm_kda_seg_step_w4_v1(const glm_kda_segments::Segment* table, int count, int H,
                      const float* q, __nv_bfloat16* y) {
    glm_pdl::enter();
  if (blockIdx.z >= count) return;
  const auto d = table[blockIdx.z];
  step_kernel_body<4, 16>(H, reinterpret_cast<const float*>(d.state_in),
      q + (size_t)d.row0 * H * DK, reinterpret_cast<const float*>(d.k),
      reinterpret_cast<const __nv_bfloat16*>(d.v), reinterpret_cast<const float*>(d.g),
      reinterpret_cast<const float*>(d.b), d.rows, y + (size_t)d.row0 * H * DV,
      reinterpret_cast<float*>(d.state_out));
}

extern "C" __global__ void __launch_bounds__(128)
tf_glm_kda_seg_out_v1(const glm_kda_segments::Segment* table, int count, int H,
                     const __nv_bfloat16* y, const __nv_bfloat16* G, int g_stride,
                     const __nv_bfloat16* norm_w, float eps, __nv_bfloat16* out) {
    glm_pdl::enter();
  if (blockIdx.z >= count) return;
  const auto d = table[blockIdx.z];
  if (blockIdx.x >= d.rows) return;
  out_kernel_body(H, y + (size_t)d.row0 * H * DV, G + (size_t)d.row0 * g_stride,
      g_stride, norm_w, eps, out + (size_t)d.row0 * H * DV);
}
