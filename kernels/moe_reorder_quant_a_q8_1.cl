// snap-pea additions to ggml/src/ggml-opencl/kernels/moe_reorder_quant_a_q8_1.cl (qualcomm/llama.cpp @ fb62b8a).
// Extracted from the patch series in ../patches/live; hunks are separated by '// ----'.

// ----

// Same as above, for the ragged dp4a GEMM, which skips padded token slots: padded slots are left untouched.
__kernel void kernel_moe_reorder_quant_a_q8_1_ragged(
        __global const float  * src,        // original activations (offset applied)
        __global const uint   * router,     // post-router indices [tok_slots]
        __global       char   * qa,
        __global       half   * da,
        __global       half   * sa,
        __global const int    * total_tiles,
        uint  K,
        ushort map_ratio,
        uint  tile_size,
        uint  n_kblocks                      // K / 32
) {
    const uint blk = get_global_id(0);       // 32-block along K
    const uint tok = get_global_id(1);       // token slot (post_router_idx)

    if (blk >= n_kblocks || tok >= (uint)total_tiles[0] * tile_size) {
        return;
    }

    const uint out_base = tok * K + blk * 32;
    const uint bidx     = tok * n_kblocks + blk;

    const uint router_idx = router[tok];
    // ragged GEMM never reads padded slots: skip the zero-fill (small speculative-verify batches are mostly padding)
    if (router_idx == 0xFFFFFFFF) {
        return;
    }

    float v[32];
    float amax = 0.0f;
    if (router_idx == 0xFFFFFFFF) {
        #pragma unroll
        for (int i = 0; i < 32; ++i) v[i] = 0.0f;
    } else {
        const uint act_idx = router_idx / map_ratio;
        const uint in_base = act_idx * K + blk * 32;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            v[i] = src[in_base + i];
            amax = fmax(amax, fabs(v[i]));
        }
    }

    const float d  = amax / 127.0f;
    const float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;

    int sum = 0;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        const int q = (int)rint(v[i] * id);
        qa[out_base + i] = (char)q;
        sum += q;
    }

    da[bidx] = (half)d;
    sa[bidx] = (half)(d * (float)sum);
}
