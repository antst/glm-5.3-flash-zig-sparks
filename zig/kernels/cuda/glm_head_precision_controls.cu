#include "glm_pdl.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
// Deliberately reduced sums, adapting contract_controls.cu's tf_ctl_q35_dense_half witness.
extern "C" __global__ void tf_head_half_control(const __nv_bfloat16* x, const __nv_bfloat16* w,
                                               float* out, int count, int k) {
    glm_pdl::enter();
    int at = blockIdx.x * blockDim.x + threadIdx.x;
    if (at >= count) return;
    __half sum = __float2half_rn(0.f);
    for (int i = 0; i < k; ++i)
        sum = __hadd(sum, __hmul(__float2half_rn(__bfloat162float(x[i])),
                                __float2half_rn(__bfloat162float(w[i]))));
    out[at] = __half2float(sum);
}
extern "C" __global__ void tf_head_bfloat_control(const __nv_bfloat16* x, const __nv_bfloat16* w,
                                                 float* out, int count, int k) {
    glm_pdl::enter();
    int at = blockIdx.x * blockDim.x + threadIdx.x;
    if (at >= count) return;
    __nv_bfloat16 sum = __float2bfloat16_rn(0.f);
    for (int i = 0; i < k; ++i) sum = __hadd(sum, __hmul(x[i], w[i]));
    out[at] = __bfloat162float(sum);
}
