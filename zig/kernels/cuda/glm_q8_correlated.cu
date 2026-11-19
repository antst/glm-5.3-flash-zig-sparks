#include "glm_q8_correlated.cuh"
extern "C" __global__ void __launch_bounds__(128) tf_glm_q8_correlated_v1(
    const __nv_bfloat16 *w, const double *g, int64_t groups, int64_t columns,
    uint32_t *words, const __half *scales, uint32_t *status, double *errors) {
    glm_q8_correlated::run<false, false>(w, g, groups, columns, words, scales, status, errors);
}
