// snap-pea additions to ggml/src/ggml-opencl/kernels/mul_mv_q8_0_f32_flat.cl (qualcomm/llama.cpp opencl/x2-unified-everything @ 8085b4e).
// Extracted from the patch series in ../patches/live; hunks are separated by '// ----'.

// ----

#ifndef N_COLS_Q8_0_MC
#define N_COLS_Q8_0_MC 4
#endif

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mv_q8_0_f32_flat_mc(
    global char * src0_q,
    global half * src0_d,
    global char * src1,
    ulong         offset1,
    global char * dst,
    ulong         offsetd,
    int           ne00,
    int           ne01,
    ulong         nb01,
    ulong         nb02,
    ulong         nb03,
    int           ne12,
    ulong         nb11,
    ulong         nb12,
    ulong         nb13,
    int           ne0,
    int           ne1,
    int           r2,
    int           r3
) {
    src1 = (global char*)((global char*)src1 + offset1);
    dst  = (global char*)((global char*)dst  + offsetd);

    int nb = ne00/QK8_0;

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = get_group_id(2);

    int first_row = (r0*N_SG_Q8_0 + get_sub_group_id()) * N_R0_Q8_0;
    first_row = min(first_row, max(ne01 - N_R0_Q8_0, 0));
    int first_col = r1*N_COLS_Q8_0_MC;

    uint i12 = im%ne12;
    uint i13 = im/ne12;

    ulong offset_src1_base = i12*nb12 + i13*nb13;
    ulong offset_src0_base = first_row*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;

    global char * ax0, * ax1, * ax2, * ax3;
    global half * ad0, * ad1, * ad2, * ad3;
    ulong offset_src0;

    offset_src0 = offset_src0_base + 0*nb01;
    offset_src0 = offset_src0/34;
    ax0 = (global char *) ((global char *) src0_q + offset_src0*sizeof(char)*QK8_0);
    ad0 = (global half *) ((global char *) src0_d + offset_src0*sizeof(half));

    offset_src0 = offset_src0_base + 1*nb01;
    offset_src0 = offset_src0/34;
    ax1 = (global char *) ((global char *) src0_q + offset_src0*sizeof(char)*QK8_0);
    ad1 = (global half *) ((global char *) src0_d + offset_src0*sizeof(half));

    offset_src0 = offset_src0_base + 2*nb01;
    offset_src0 = offset_src0/34;
    ax2 = (global char *) ((global char *) src0_q + offset_src0*sizeof(char)*QK8_0);
    ad2 = (global half *) ((global char *) src0_d + offset_src0*sizeof(half));

    offset_src0 = offset_src0_base + 3*nb01;
    offset_src0 = offset_src0/34;
    ax3 = (global char *) ((global char *) src0_q + offset_src0*sizeof(char)*QK8_0);
    ad3 = (global half *) ((global char *) src0_d + offset_src0*sizeof(half));

    const short ix = get_sub_group_local_id()/4;
    const short il = get_sub_group_local_id()%4;

    float4 sumf[N_COLS_Q8_0_MC];
    for (int c = 0; c < N_COLS_Q8_0_MC; c++) {
        sumf[c] = (float4)(0.f);
    }

    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH/4) {
        float8 qv0 = convert_float8(vload8(0, ax0 + ib*QK8_0 + il*NB_Q8_0));
        float8 qv1 = convert_float8(vload8(0, ax1 + ib*QK8_0 + il*NB_Q8_0));
        float8 qv2 = convert_float8(vload8(0, ax2 + ib*QK8_0 + il*NB_Q8_0));
        float8 qv3 = convert_float8(vload8(0, ax3 + ib*QK8_0 + il*NB_Q8_0));
        float d0 = ad0[ib];
        float d1 = ad1[ib];
        float d2 = ad2[ib];
        float d3 = ad3[ib];

        for (int c = 0; c < N_COLS_Q8_0_MC; c++) {
            const int col = first_col + c;
            if (col < ne1) {
                global float * y = (global float *)(src1 + offset_src1_base + col*nb11);
                float8 yl = vload8(0, y + ib*QK8_0 + il*NB_Q8_0);

                float sumq0 = 0.f;
                sumq0 += qv0.s0*yl.s0;
                sumq0 += qv0.s1*yl.s1;
                sumq0 += qv0.s2*yl.s2;
                sumq0 += qv0.s3*yl.s3;
                sumq0 += qv0.s4*yl.s4;
                sumq0 += qv0.s5*yl.s5;
                sumq0 += qv0.s6*yl.s6;
                sumq0 += qv0.s7*yl.s7;

                float sumq1 = 0.f;
                sumq1 += qv1.s0*yl.s0;
                sumq1 += qv1.s1*yl.s1;
                sumq1 += qv1.s2*yl.s2;
                sumq1 += qv1.s3*yl.s3;
                sumq1 += qv1.s4*yl.s4;
                sumq1 += qv1.s5*yl.s5;
                sumq1 += qv1.s6*yl.s6;
                sumq1 += qv1.s7*yl.s7;

                float sumq2 = 0.f;
                sumq2 += qv2.s0*yl.s0;
                sumq2 += qv2.s1*yl.s1;
                sumq2 += qv2.s2*yl.s2;
                sumq2 += qv2.s3*yl.s3;
                sumq2 += qv2.s4*yl.s4;
                sumq2 += qv2.s5*yl.s5;
                sumq2 += qv2.s6*yl.s6;
                sumq2 += qv2.s7*yl.s7;

                float sumq3 = 0.f;
                sumq3 += qv3.s0*yl.s0;
                sumq3 += qv3.s1*yl.s1;
                sumq3 += qv3.s2*yl.s2;
                sumq3 += qv3.s3*yl.s3;
                sumq3 += qv3.s4*yl.s4;
                sumq3 += qv3.s5*yl.s5;
                sumq3 += qv3.s6*yl.s6;
                sumq3 += qv3.s7*yl.s7;

                sumf[c] += (float4)(sumq0*d0, sumq1*d1, sumq2*d2, sumq3*d3);
            }
        }
    }

    for (int c = 0; c < N_COLS_Q8_0_MC; c++) {
        float4 tot = (float4)(
            sub_group_reduce_add(sumf[c].s0),
            sub_group_reduce_add(sumf[c].s1),
            sub_group_reduce_add(sumf[c].s2),
            sub_group_reduce_add(sumf[c].s3)
        );

        if (get_sub_group_local_id() == 0 && first_col + c < ne1) {
            global float * dst_f32 = (global float *) dst + (ulong)im*ne0*ne1 + (ulong)(first_col + c)*ne0;
            if (first_row + 0 < ne01) dst_f32[first_row + 0] = tot.s0;
            if (first_row + 1 < ne01) dst_f32[first_row + 1] = tot.s1;
            if (first_row + 2 < ne01) dst_f32[first_row + 2] = tot.s2;
            if (first_row + 3 < ne01) dst_f32[first_row + 3] = tot.s3;
        }
    }
}

// ============================================================================
// int8 (dp4a) multi-column verify GEMV for flat Q8_0 weights (MTP / speculative
// verify, 2..8 columns). The float multi-column kernel above is ALU-bound on the
// 248k-row LM head; here activations are quantized per 32-value block to int8
// (kernel_q8mc_quant_act) and each row block is a pair of packed int8 dot
// products. Same tiling as the float kernel: 4 rows per subgroup, lanes split
// as 16 (blocks) x 4 (8-byte chunks).
// ============================================================================
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
// MC_NCOL (compile-time width 2..8) sizes the accumulators for the actual verify width and folds the
// column bound; without it the kernels take ncol at run time and reserve 8 columns.
#ifdef MC_NCOL
#define Q8MC_MAXC MC_NCOL
#define Q8MC_NCOL MC_NCOL
#else
#define Q8MC_MAXC 8
#define Q8MC_NCOL ncol
#endif
#ifndef MC_R0
#define MC_R0 4
#endif

kernel void kernel_q8mc_quant_act(
    global const char * src1, ulong offset1, ulong nb11,
    int K, int ncol,
    global char  * qa,    // [ncol][K]
    global float * da)    // [ncol][K/32]
{
    const int nbk = K / QK8_0;
    const int gid = get_global_id(0);
    if (gid >= ncol * nbk) return;
    const int col = gid / nbk;
    const int b   = gid - col * nbk;
    global const float * x = (global const float *)(src1 + offset1 + col*nb11) + b*QK8_0;
    float v[QK8_0];
    float amax = 0.0f;
    for (int i = 0; i < QK8_0; i++) { v[i] = x[i]; amax = fmax(amax, fabs(v[i])); }
    const float d  = amax / 127.0f;
    const float id = amax > 0.0f ? 127.0f / amax : 0.0f;
    global char * q = qa + (ulong)col*K + b*QK8_0;
    for (int i = 0; i < QK8_0; i++) q[i] = (char) rint(v[i] * id);
    da[col*nbk + b] = d;
}

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mv_q8_0_q8a_dp4a_mc(
    global char  * src0_q,
    global half  * src0_d,
    global char  * qa,
    global float * da,
    global char  * dst,
    ulong          offsetd,
    int            ne00,
    int            ne01,
    ulong          nb01,
    int            ne0,
    int            ncol)
{
    dst = (global char*)((global char*)dst + offsetd);
    const int nb = ne00/QK8_0;
    const int r0 = get_group_id(0);

    int first_row = (r0*N_SG_Q8_0 + get_sub_group_id()) * MC_R0;
    first_row = min(first_row, max(ne01 - MC_R0, 0));

    global char * ax[MC_R0];
    global half * ad[MC_R0];
    for (int r = 0; r < MC_R0; r++) {
        const ulong off = ((ulong)(first_row + r)*nb01)/34;
        ax[r] = src0_q + off*QK8_0;
        ad[r] = (global half *)((global char *)src0_d + off*sizeof(half));
    }

    const short ix = get_sub_group_local_id()/4;
    const short il = get_sub_group_local_id()%4;

    float sumf[Q8MC_MAXC][MC_R0];
    for (int c = 0; c < Q8MC_MAXC; c++)
        for (int r = 0; r < MC_R0; r++) sumf[c][r] = 0.0f;

    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH/4) {
        uint2 w[MC_R0];
        float dw[MC_R0];
        for (int r = 0; r < MC_R0; r++) {
            w[r]  = as_uint2(vload8(0, ax[r] + ib*QK8_0 + il*NB_Q8_0));
            dw[r] = ad[r][ib];
        }
        #pragma unroll
        for (int c = 0; c < Q8MC_MAXC; c++) {
            if (c >= Q8MC_NCOL) break;
            const uint2 a  = as_uint2(vload8(0, qa + (ulong)c*ne00 + ib*QK8_0 + il*NB_Q8_0));
            const float dc = da[c*nb + ib];
            for (int r = 0; r < MC_R0; r++) {
                int s = dot_acc_sat_4x8packed_ss_int(w[r].x, a.x, 0);
                s     = dot_acc_sat_4x8packed_ss_int(w[r].y, a.y, s);
                sumf[c][r] += (float)s * (dw[r] * dc);
            }
        }
    }

    #pragma unroll
    for (int c = 0; c < Q8MC_MAXC; c++) {
        if (c >= Q8MC_NCOL) break;
        global float * dst_f32 = (global float *) dst + (ulong)c*ne0;
        for (int r = 0; r < MC_R0; r++) {
            const float tot = sub_group_reduce_add(sumf[c][r]);
            if (get_sub_group_local_id() == 0 && first_row + r < ne01) dst_f32[first_row + r] = tot;
        }
    }
}

// Same as kernel_mul_mv_q8_0_q8a_dp4a_mc, but the quantized activations (ncol * K bytes) are
// staged once per work-group in local memory and shared by its MC_NSG subgroups. With 5 columns
// a subgroup otherwise reads 10 KB of activations for 8 KB of weights (lm_head: 248320 rows).
#define Q8MC_LDS_BYTES 16384
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mv_q8_0_q8a_dp4a_mc_lds(
    global char  * src0_q,
    global half  * src0_d,
    global char  * qa,
    global float * da,
    global char  * dst,
    ulong          offsetd,
    int            ne00,
    int            ne01,
    ulong          nb01,
    int            ne0,
    int            ncol)
{
    local uint  qa_l[Q8MC_LDS_BYTES / 4];
    local float da_l[Q8MC_LDS_BYTES / QK8_0];

    dst = (global char*)((global char*)dst + offsetd);
    const int nb = ne00/QK8_0;

    const int lsz  = get_local_size(0) * get_local_size(1);
    const int lidf = get_local_id(1) * get_local_size(0) + get_local_id(0);
    for (int i = lidf; i < ncol * ne00 / 4; i += lsz) qa_l[i] = ((global const uint *) qa)[i];
    for (int i = lidf; i < ncol * nb; i += lsz)       da_l[i] = da[i];
    barrier(CLK_LOCAL_MEM_FENCE);

    int first_row = (get_group_id(0) * get_local_size(1) + get_local_id(1)) * MC_R0;
    first_row = min(first_row, max(ne01 - MC_R0, 0));

    global char * ax[MC_R0];
    global half * ad[MC_R0];
    for (int r = 0; r < MC_R0; r++) {
        const ulong off = ((ulong)(first_row + r)*nb01)/34;
        ax[r] = src0_q + off*QK8_0;
        ad[r] = (global half *)((global char *)src0_d + off*sizeof(half));
    }

    const short ix = get_sub_group_local_id()/4;
    const short il = get_sub_group_local_id()%4;

    float sumf[Q8MC_MAXC][MC_R0];
    for (int c = 0; c < Q8MC_MAXC; c++)
        for (int r = 0; r < MC_R0; r++) sumf[c][r] = 0.0f;

    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH/4) {
        uint2 w[MC_R0];
        float dw[MC_R0];
        for (int r = 0; r < MC_R0; r++) {
            w[r]  = as_uint2(vload8(0, ax[r] + ib*QK8_0 + il*NB_Q8_0));
            dw[r] = ad[r][ib];
        }
        #pragma unroll
        for (int c = 0; c < Q8MC_MAXC; c++) {
            if (c >= Q8MC_NCOL) break;
            const uint2 a  = vload2(0, qa_l + (c*ne00 + ib*QK8_0 + il*NB_Q8_0) / 4);
            const float dc = da_l[c*nb + ib];
            for (int r = 0; r < MC_R0; r++) {
                int s = dot_acc_sat_4x8packed_ss_int(w[r].x, a.x, 0);
                s     = dot_acc_sat_4x8packed_ss_int(w[r].y, a.y, s);
                sumf[c][r] += (float)s * (dw[r] * dc);
            }
        }
    }

    #pragma unroll
    for (int c = 0; c < Q8MC_MAXC; c++) {
        if (c >= Q8MC_NCOL) break;
        global float * dst_f32 = (global float *) dst + (ulong)c*ne0;
        for (int r = 0; r < MC_R0; r++) {
            const float tot = sub_group_reduce_add(sumf[c][r]);
            if (get_sub_group_local_id() == 0 && first_row + r < ne01) dst_f32[first_row + r] = tot;
        }
    }
}

// Flash-decoding split for q8_0 KV, dk = dv = 256, GQA 8 (Qwen3.5/3.6 MoE full-attention layers), n_q = 1.
// Drop-in for flash_attn_f32_q8_0_q1_vec_mq_split (same arguments, same partial records, same merge).
// That kernel reduces every KV row across the subgroup once per head (8 serial reductions per row);
// here a lane owns a whole KV row for QK (8 heads x 64 dp4a against Q requantized to int8 in local
// memory), softmax statistics are reduced once per 64-row tile, and for PV a lane owns 4 of the 256
// output dims. One 64-lane subgroup per (kv head, split).
#define FAD_DK   256
#define FAD_G    8
#define FAD_NB   (FAD_DK / 32)
#define FAD_BLK  34
#define FAD_MINIT (-3.0e38f)

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_fa_q8_d256_g8_dec(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    const float scale,
    const int n_q,
    const int n_kv,
    const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const float max_bias,
    const float m0,
    const float m1,
    const int n_head_log2,
    const float logit_softcap,
    const int n_head_kv,
    const global void * mask_void,
    const ulong mask_offset,
    const ulong mask_nb1,
    const ulong mask_nb2,
    const ulong mask_nb3,
    const int mask_ne2,
    const int mask_ne3,
    global float * partial_void,
    const int n_splits,
    const int kv_per_split
) {
    const int lid         = get_local_id(0);
    const int hb_idx      = get_global_id(1);
    const int batch_idx   = hb_idx / n_head_kv;
    const int head_kv_idx = hb_idx % n_head_kv;
    const int split_q     = get_global_id(2);
    const int split_idx   = split_q % n_splits;
    const int q_idx       = split_q / n_splits;

    const int kv_start = split_idx * kv_per_split;
    const int kv_end   = min(kv_start + kv_per_split, n_kv);
    const ulong rec_stride = (ulong) (2 + FAD_DK);

    if (kv_start >= kv_end) {
        if (lid < FAD_G) {
            const int head_idx = head_kv_idx * FAD_G + lid;
            global float * rec = partial_void +
                ((((ulong) batch_idx * n_head + head_idx) * n_q + q_idx) * n_splits + split_idx) * rec_stride;
            rec[0] = FAD_MINIT;
            rec[1] = 0.0f;
        }
        return;
    }

    // Q -> int8: lane = (head, block), 8 x 8 = 64.
    local uint  qp[FAD_G * FAD_NB * 8];
    local float qd[FAD_G * FAD_NB];
    local float P[64 * FAD_G];
    {
        const int h = lid / FAD_NB;
        const int b = lid % FAD_NB;
        const global float * qr = (const global float *) ((const global char *) q_void + q_offset +
            batch_idx * q_nb3 + (head_kv_idx * FAD_G + h) * q_nb2 + (ulong) q_idx * q_nb1) + b * 32;
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) amax = fmax(amax, fabs(qr[i]));
        const float d  = amax / 127.0f;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;
        for (int w = 0; w < 8; ++w) {
            const char4 c = convert_char4_sat_rte(vload4(w, qr) * id);
            qp[lid * 8 + w] = as_uint(c);
        }
        qd[lid] = d;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const global char * k_base = (const global char *) k_void + k_offset + batch_idx * k_nb3 + head_kv_idx * k_nb2;
    const global char * v_base = (const global char *) v_void + v_offset + batch_idx * v_nb3 + head_kv_idx * v_nb2;

    const global half * mask_row[FAD_G];
    {
        const global char * mb = mask_void == NULL ? NULL :
            (const global char *) mask_void + mask_offset + (batch_idx % mask_ne3) * mask_nb3 + (ulong) q_idx * mask_nb1;
        for (int h = 0; h < FAD_G; ++h) {
            mask_row[h] = mb == NULL ? NULL :
                (const global half *) (mb + ((head_kv_idx * FAD_G + h) % mask_ne2) * mask_nb2);
        }
    }

    // PV ownership: dims 4*lid .. 4*lid+3 = block lid/8, quartet lid%8.
    const int vb = lid / 8;
    const int vj = lid % 8;

    float  m[FAD_G], l[FAD_G];
    float4 o[FAD_G];
    for (int h = 0; h < FAD_G; ++h) { m[h] = FAD_MINIT; l[h] = 0.0f; o[h] = (float4)(0.0f); }

    for (int t0 = kv_start; t0 < kv_end; t0 += 64) {
        const int  r     = t0 + lid;
        const bool valid = r < kv_end;

        float s[FAD_G];
        for (int h = 0; h < FAD_G; ++h) s[h] = 0.0f;
        if (valid) {
            const global char * kr = k_base + (ulong) r * k_nb1;
            for (int b = 0; b < FAD_NB; ++b) {
                const global char * kb = kr + b * FAD_BLK;
                const float kd = vload_half(0, (const global half *) kb);
                const uint4 ka = as_uint4(vload16(0, (const global uchar *) (kb + 2)));
                const uint4 kc = as_uint4(vload16(1, (const global uchar *) (kb + 2)));
                for (int h = 0; h < FAD_G; ++h) {
                    const uint8 q8 = vload8(0, qp + (h * FAD_NB + b) * 8);
                    int acc = dot_acc_sat_4x8packed_ss_int(q8.s0, ka.x, 0);
                    acc = dot_acc_sat_4x8packed_ss_int(q8.s1, ka.y, acc);
                    acc = dot_acc_sat_4x8packed_ss_int(q8.s2, ka.z, acc);
                    acc = dot_acc_sat_4x8packed_ss_int(q8.s3, ka.w, acc);
                    acc = dot_acc_sat_4x8packed_ss_int(q8.s4, kc.x, acc);
                    acc = dot_acc_sat_4x8packed_ss_int(q8.s5, kc.y, acc);
                    acc = dot_acc_sat_4x8packed_ss_int(q8.s6, kc.z, acc);
                    acc = dot_acc_sat_4x8packed_ss_int(q8.s7, kc.w, acc);
                    s[h] = mad((float) acc, kd * qd[h * FAD_NB + b], s[h]);
                }
            }
            for (int h = 0; h < FAD_G; ++h) {
                s[h] *= scale;
                if (mask_row[h] != NULL) s[h] += (float) mask_row[h][r];
            }
        } else {
            for (int h = 0; h < FAD_G; ++h) s[h] = FAD_MINIT;
        }

        for (int h = 0; h < FAD_G; ++h) {
            const float tmax  = sub_group_reduce_max(s[h]);
            const float m_new = fmax(m[h], tmax);
            const float alpha = native_exp(m[h] - m_new);
            const float p     = valid ? native_exp(s[h] - m_new) : 0.0f;
            l[h] = l[h] * alpha + sub_group_reduce_add(p);
            m[h] = m_new;
            o[h] *= alpha;
            P[lid * FAD_G + h] = p;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        const int nr = min(64, kv_end - t0);
        const global char * vrow = v_base + (ulong) t0 * v_nb1 + vb * FAD_BLK;
#ifdef FAD_PV_UNROLL
        #pragma unroll FAD_PV_UNROLL
#endif
        for (int i = 0; i < nr; ++i) {
            const global char * vblk = vrow + (ulong) i * v_nb1;
            const float  vd = vload_half(0, (const global half *) vblk);
            const float4 vv = convert_float4(vload4(vj, vblk + 2)) * vd;
            const float4 p0 = vload4(0, P + i * FAD_G);
            const float4 p1 = vload4(1, P + i * FAD_G);
            o[0] = mad((float4)(p0.s0), vv, o[0]);
            o[1] = mad((float4)(p0.s1), vv, o[1]);
            o[2] = mad((float4)(p0.s2), vv, o[2]);
            o[3] = mad((float4)(p0.s3), vv, o[3]);
            o[4] = mad((float4)(p1.s0), vv, o[4]);
            o[5] = mad((float4)(p1.s1), vv, o[5]);
            o[6] = mad((float4)(p1.s2), vv, o[6]);
            o[7] = mad((float4)(p1.s3), vv, o[7]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int h = 0; h < FAD_G; ++h) {
        const int head_idx = head_kv_idx * FAD_G + h;
        global float * rec = partial_void +
            ((((ulong) batch_idx * n_head + head_idx) * n_q + q_idx) * n_splits + split_idx) * rec_stride;
        if (lid == 0) { rec[0] = m[h]; rec[1] = l[h]; }
        vstore4(o[h], lid, rec + 2);
    }
}
// f16-KV twin of kernel_fa_q8_d256_g8_dec (same arguments, records and merge): used for the MTP
// draft context, whose KV stays f16. QK: a lane owns a KV row, float math against Q in local memory.
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_fa_f16_d256_g8_dec(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    const float scale,
    const int n_q,
    const int n_kv,
    const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const float max_bias,
    const float m0,
    const float m1,
    const int n_head_log2,
    const float logit_softcap,
    const int n_head_kv,
    const global void * mask_void,
    const ulong mask_offset,
    const ulong mask_nb1,
    const ulong mask_nb2,
    const ulong mask_nb3,
    const int mask_ne2,
    const int mask_ne3,
    global float * partial_void,
    const int n_splits,
    const int kv_per_split
) {
    const int lid         = get_local_id(0);
    const int hb_idx      = get_global_id(1);
    const int batch_idx   = hb_idx / n_head_kv;
    const int head_kv_idx = hb_idx % n_head_kv;
    const int split_q     = get_global_id(2);
    const int split_idx   = split_q % n_splits;
    const int q_idx       = split_q / n_splits;

    const int kv_start = split_idx * kv_per_split;
    const int kv_end   = min(kv_start + kv_per_split, n_kv);
    const ulong rec_stride = (ulong) (2 + FAD_DK);

    if (kv_start >= kv_end) {
        if (lid < FAD_G) {
            const int head_idx = head_kv_idx * FAD_G + lid;
            global float * rec = partial_void +
                ((((ulong) batch_idx * n_head + head_idx) * n_q + q_idx) * n_splits + split_idx) * rec_stride;
            rec[0] = FAD_MINIT;
            rec[1] = 0.0f;
        }
        return;
    }

    // Q (8 heads x 256 floats) in local memory, read as float8 broadcasts
    local float qf[FAD_G * FAD_DK];
    local float P[64 * FAD_G];
    for (int i = lid; i < FAD_G * FAD_DK; i += 64) {
        const int h = i / FAD_DK;
        const int d = i % FAD_DK;
        const global float * qr = (const global float *) ((const global char *) q_void + q_offset +
            batch_idx * q_nb3 + (head_kv_idx * FAD_G + h) * q_nb2 + (ulong) q_idx * q_nb1);
        qf[i] = qr[d];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const global char * k_base = (const global char *) k_void + k_offset + batch_idx * k_nb3 + head_kv_idx * k_nb2;
    const global char * v_base = (const global char *) v_void + v_offset + batch_idx * v_nb3 + head_kv_idx * v_nb2;

    const global half * mask_row[FAD_G];
    {
        const global char * mb = mask_void == NULL ? NULL :
            (const global char *) mask_void + mask_offset + (batch_idx % mask_ne3) * mask_nb3 + (ulong) q_idx * mask_nb1;
        for (int h = 0; h < FAD_G; ++h) {
            mask_row[h] = mb == NULL ? NULL :
                (const global half *) (mb + ((head_kv_idx * FAD_G + h) % mask_ne2) * mask_nb2);
        }
    }

    float  m[FAD_G], l[FAD_G];
    float4 o[FAD_G];
    for (int h = 0; h < FAD_G; ++h) { m[h] = FAD_MINIT; l[h] = 0.0f; o[h] = (float4)(0.0f); }

    for (int t0 = kv_start; t0 < kv_end; t0 += 64) {
        const int  r     = t0 + lid;
        const bool valid = r < kv_end;

        float s[FAD_G];
        for (int h = 0; h < FAD_G; ++h) s[h] = 0.0f;
        if (valid) {
            const global half * kr = (const global half *) (k_base + (ulong) r * k_nb1);
            for (int c = 0; c < FAD_DK / 8; ++c) {
                const float8 kv8 = vload_half8(c, kr);
                for (int h = 0; h < FAD_G; ++h) {
                    const float8 q8 = vload8(c, qf + h * FAD_DK);
                    const float8 pr = kv8 * q8;
                    s[h] += ((pr.s0 + pr.s1) + (pr.s2 + pr.s3)) + ((pr.s4 + pr.s5) + (pr.s6 + pr.s7));
                }
            }
            for (int h = 0; h < FAD_G; ++h) {
                s[h] *= scale;
                if (mask_row[h] != NULL) s[h] += (float) mask_row[h][r];
            }
        } else {
            for (int h = 0; h < FAD_G; ++h) s[h] = FAD_MINIT;
        }

        for (int h = 0; h < FAD_G; ++h) {
            const float tmax  = sub_group_reduce_max(s[h]);
            const float m_new = fmax(m[h], tmax);
            const float alpha = exp(m[h] - m_new);
            const float p     = valid ? exp(s[h] - m_new) : 0.0f;
            l[h] = l[h] * alpha + sub_group_reduce_add(p);
            m[h] = m_new;
            o[h] *= alpha;
            P[lid * FAD_G + h] = p;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        const int nr = min(64, kv_end - t0);
        for (int i = 0; i < nr; ++i) {
            const float4 vv = vload_half4(lid, (const global half *) (v_base + (ulong) (t0 + i) * v_nb1));
            const float4 p0 = vload4(0, P + i * FAD_G);
            const float4 p1 = vload4(1, P + i * FAD_G);
            o[0] = mad((float4)(p0.s0), vv, o[0]);
            o[1] = mad((float4)(p0.s1), vv, o[1]);
            o[2] = mad((float4)(p0.s2), vv, o[2]);
            o[3] = mad((float4)(p0.s3), vv, o[3]);
            o[4] = mad((float4)(p1.s0), vv, o[4]);
            o[5] = mad((float4)(p1.s1), vv, o[5]);
            o[6] = mad((float4)(p1.s2), vv, o[6]);
            o[7] = mad((float4)(p1.s3), vv, o[7]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int h = 0; h < FAD_G; ++h) {
        const int head_idx = head_kv_idx * FAD_G + h;
        global float * rec = partial_void +
            ((((ulong) batch_idx * n_head + head_idx) * n_q + q_idx) * n_splits + split_idx) * rec_stride;
        if (lid == 0) { rec[0] = m[h]; rec[1] = l[h]; }
        vstore4(o[h], lid, rec + 2);
    }
}
#endif // cl_khr_integer_dot_product
