#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "exl3_decode.cuh"
#include "exl3_plan.cuh"

// Transposed fp32 prompt accumulators in the shared EXL3 module; decode kernels above stay unchanged.
#include "exl3_prompt.cuh"
using namespace tf_exl3_prompt;
extern "C" __global__ void tf_exl3_prompt_rotate(const __nv_bfloat16* __restrict__ x, int x_stride, const half* __restrict__ suh,
        half* __restrict__ out, int K) {
    const int row = blockIdx.x, blk = blockIdx.y;
    const int lane = threadIdx.x;
    const half* sh = suh + blk * 128 + 4 * lane;
    const __nv_bfloat16* xr = x + (size_t)row * x_stride + blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = __bfloat162float(xr[j]) * __half2float(sh[j]);
    fwht128(v, lane);
    half* o = out + (size_t)row * K + blk * 128 + 4 * lane;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

extern "C" __global__ void __launch_bounds__(256, 2) tf_exl3_prompt_gu(const half* __restrict__ Xg, const half* __restrict__ Xu,
        const uint64_t* __restrict__ Tg, const uint64_t* __restrict__ Tu, const int* __restrict__ items, const int* __restrict__ counts,
        const int* __restrict__ members, const half* __restrict__ svh_g, const half* __restrict__ svh_u, const half* __restrict__ suh_d,
        half* __restrict__ xd, int K, int N, int xdiv, float limit, int act_mode, int skip) {

    const int item = blockIdx.y;
    if (item >= counts[0]) return;
    const int e = items[3 * item];
    if (e == skip) return;

    __shared__ __align__(16) unsigned char smem_raw[tsmem_bytes()];   // one buffer for either instantiation
    __shared__ int rows_sh[TPM];
    if (xdiv > 1) tgateup<TPM>(Xg, Xu, Tg, Tu, items, counts, members, svh_g, svh_u, suh_d, xd, K, N, xdiv, limit, act_mode, smem_raw, rows_sh);
    else tgateup<2 * TPM>(Xg, Xu, Tg, Tu, items, counts, members, svh_g, svh_u, suh_d, xd, K, N, 1, limit, act_mode, smem_raw, rows_sh);
}

// Block (256 model-width columns, item): y[row][slot][:] = the down epilogue of the item's pairs (this rank's share).
extern "C" __global__ void __launch_bounds__(256, 2) tf_exl3_prompt_down(const half* __restrict__ Xd, const uint64_t* __restrict__ Td,
        const int* __restrict__ items, const int* __restrict__ counts, const int* __restrict__ members, const half* __restrict__ svh_d,
        float* __restrict__ y, int K, int D, int skip) {

    __shared__ __align__(16) unsigned char smem_raw[tsmem_bytes()];
    __shared__ int rows_sh[TPM];
    const int cb = blockIdx.x, item = blockIdx.y;
    if (item >= counts[0]) return;
    const int e = items[3 * item];
    if (e == skip) return;
    const int first = items[3 * item + 1], cnt = items[3 * item + 2];
    if (threadIdx.x < TPM) rows_sh[threadIdx.x] = (int)threadIdx.x < cnt ? members[first + threadIdx.x] : -1;
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float acc[TMT][TNB][4];
    tmain<1, TPM>(Xd, Xd, Td, Td, rows_sh, cnt, e, K, D >> 4, cb, 1, reinterpret_cast<half*>(smem_raw), acc);
    float* Ed = reinterpret_cast<float*>(smem_raw);
    for (int hf = 0; hf < 2 && hf * TPH < cnt; ++hf) {
        if (hf) __syncthreads();
        tspill(Ed, DN_ES, warp * TMT * 16, acc, hf);
        __syncthreads();
        for (int i = warp; i < TPH && hf * TPH + i < cnt; i += TNW) {
            const int p = rows_sh[hf * TPH + i];
#pragma unroll
            for (int blk = 0; blk < 2; ++blk) {
                const int n = cb * 256 + blk * 128 + 4 * lane;
                float v[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) v[j] = Ed[(size_t)i * DN_ES + blk * 128 + 4 * lane + j];
                fwht128(v, lane);
                float* o = y + (size_t)p * D + n;
#pragma unroll
                for (int j = 0; j < 4; ++j) o[j] = v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * D + n + j]);
            }
        }
    }
}
