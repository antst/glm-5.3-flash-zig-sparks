#include "glm_pdl.cuh"
// GLM TP4 decode: exact grouped EXL3 MCG4 tile order with the single-split down epilogue in the same CTA.
#include "exl3_grouped.cuh"

#if __CUDA_ARCH__ >= 1200
constexpr int LOAD = tf_exl3::VECTOR;
#else
constexpr int LOAD = tf_exl3::PLAIN;
#endif

struct DownArgs {
    tf_exl3::GroupedArgs group;
    const half* scale;
    float* out;
    int output_slots;
};

extern "C" __global__ void tf_glm_exl3_mcg4_gateup_shared(tf_exl3::GroupedArgs args) {
    glm_pdl::enter();
    tf_exl3::grouped_body<1, 8, 4, LOAD, false, 8>(args, blockIdx.x, blockIdx.y, blockIdx.z);
}

extern "C" __global__ void tf_glm_exl3_mcg4_down(DownArgs args) {
    glm_pdl::enter();
    tf_exl3::grouped_body<1, 8, 4, LOAD, true>(args.group, blockIdx.x, blockIdx.y, blockIdx.z, args.scale, args.out, args.output_slots);
}
