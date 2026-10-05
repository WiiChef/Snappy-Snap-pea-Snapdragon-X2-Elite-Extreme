// snap-pea additions to ggml/src/ggml-opencl/kernels/gemv_noshuffle_q8_0_f32.cl (qualcomm/llama.cpp @ fb62b8a).
// Extracted from the patch series in ../patches/live; hunks are separated by '// ----'.

// ----
// ============================================================================
// Multi-column (2..8) int8 split-K GEMV for speculative-verify batches on small-M
// weights (M <= 1024), where the cooperative-K GEMM fills only M/256 work-groups.
// Same transposed weight layout as the base kernel; activations quantized per
// 32-block by kernel_q8mc_quant_act (qa: [ncol][K] int8, da: [ncol][K/32] float).
// Each (K slice, subgroup) reduces a disjoint set of blocks into
// partial[kslice][c][row]; kernel_gemv_splitk_reduce_f32 sums the slices over
// M * ncol outputs. No subgroup builtins here.
// ============================================================================
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#define NS_MC_MAX 8
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
__kernel void kernel_gemv_noshuffle_q8_0_q8a_dp4a_mc_splitk(
        __read_only  image1d_buffer_t src0_q,
        global half  * src0_d,
        global const uint  * qa,
        global const float * da,
        global float * partial,                 // [ksplit][ncol][M]
        int ne00,
        int ne01,
        int ncol)
{
    const uint groupId = get_local_id(1);
    const uint nsg     = get_local_size(1);
    const uint ksplit  = get_num_groups(1);
    const uint kslice  = get_group_id(1);
    const uint gid     = get_global_id(0);
    const uint lid     = get_local_id(0);

    const uint K = ne00;
    const uint M = ne01;
    const uint nbk   = K / QK8_0;
    const uint gid_s = min(gid, M - 1);
    const uint BS    = 8 * M;

    float total[NS_MC_MAX];
    #pragma unroll
    for (int c = 0; c < NS_MC_MAX; ++c) total[c] = 0.0f;

    #pragma unroll 1
    for (uint k = kslice * nsg + groupId; k < nbk; k += ksplit * nsg) {
        const uint wb = gid_s + k * BS;
        const uint w0 = read_imageui(src0_q, wb + M * 0).x;
        const uint w1 = read_imageui(src0_q, wb + M * 1).x;
        const uint w2 = read_imageui(src0_q, wb + M * 2).x;
        const uint w3 = read_imageui(src0_q, wb + M * 3).x;
        const uint w4 = read_imageui(src0_q, wb + M * 4).x;
        const uint w5 = read_imageui(src0_q, wb + M * 5).x;
        const uint w6 = read_imageui(src0_q, wb + M * 6).x;
        const uint w7 = read_imageui(src0_q, wb + M * 7).x;
        const float dw = convert_float(src0_d[gid_s + k * M]);
        #pragma unroll
        for (int c = 0; c < NS_MC_MAX; ++c) {
            if (c < ncol) {
                const uint8 a = vload8(c * nbk + k, qa);
                int acc = dot_acc_sat_4x8packed_ss_int(w0, a.s0, 0);
                acc = dot_acc_sat_4x8packed_ss_int(w1, a.s1, acc);
                acc = dot_acc_sat_4x8packed_ss_int(w2, a.s2, acc);
                acc = dot_acc_sat_4x8packed_ss_int(w3, a.s3, acc);
                acc = dot_acc_sat_4x8packed_ss_int(w4, a.s4, acc);
                acc = dot_acc_sat_4x8packed_ss_int(w5, a.s5, acc);
                acc = dot_acc_sat_4x8packed_ss_int(w6, a.s6, acc);
                acc = dot_acc_sat_4x8packed_ss_int(w7, a.s7, acc);
                total[c] = mad((float) acc, dw * da[c * nbk + k], total[c]);
            }
        }
    }

    __local float red[64 * 7 * NS_MC_MAX];
    if (groupId > 0) {
        #pragma unroll
        for (int c = 0; c < NS_MC_MAX; ++c) {
            if (c < ncol) red[(64 * (groupId - 1) + lid) * NS_MC_MAX + c] = total[c];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (groupId == 0 && gid < M) {
        #pragma unroll
        for (int c = 0; c < NS_MC_MAX; ++c) {
            if (c < ncol) {
                float v = total[c];
                for (uint i = 0; i + 1 < nsg; ++i) v += red[(64 * i + lid) * NS_MC_MAX + c];
                partial[((ulong) kslice * ncol + c) * M + gid] = v;
            }
        }
    }
}
#endif // cl_khr_integer_dot_product

