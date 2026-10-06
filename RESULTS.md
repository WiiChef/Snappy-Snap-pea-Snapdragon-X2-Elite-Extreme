# llama.cpp MTP speed work (Qualcomm OpenCL fork, Adreno X2-90) — 2026-10-04

Branch `mtp-smallbatch` in `D:\llm\src\llama-qc-mtp` (worktree of `D:\llm\src\llama-qc`), build `D:\llm\build\qc-mtp`.
Model `D:\llm\scan\scan-all-q6k.gguf` (Qwen3.6-35B-A3B, Q6_K experts / Q8_0 dense, MTP head blk.40).
Benchmark: `D:\ninfer\analysis\mtp_test_me.ps1` (warmed; refrigerator prose + fibonacci code, 200 tokens, greedy).

## Result

| Config | Prose tok/s | Code tok/s | Mean |
|---|---:|---:|---:|
| Plain decode (any build) | 32.4 | 32.7 | 32.6 |
| MTP, stock fork (N=3) | 25–28 | 25–28 | ~26 (slower than plain) |
| **MTP, this branch, `--spec-draft-n-max 7 --spec-draft-p-min 0.6`** | **39.8** | **50.6** | **45.2 (+39%)** |

Plain pp512 560 / tg128 32.9 — unchanged. Output coherent; matches plain greedy up to near-ties.

## Commits (on top of fb62b8a)
- `133d62e` multi-column Q8_0 flat GEMV (had an activation-index bug)
- `15ff684` temporary guard; `ef2de56` fix the `ix` -> `ib` activation index, drop guard
- `29de3d1` reduced-vocabulary MTP draft head (`LLAMA_MTP_DRAFT_VOCAB`, default 65536; 0 = full) +
  view-aware Q8_0 dispatch (row views judged by `view_src`)
- `9980092` int8 dp4a multi-column Q8_0 verify GEMV (`GGML_OPENCL_Q8_MC_DP4A=0` to disable)

## Why it stops here (measured, profiling build)
Verify cost by width: 1 col 29.4 ms, 2 cols 26.4, 4 cols 32.2, 8 cols 39.6. A 2–4 token verify
costs ~1.0x a plain step; MoE grows only with distinct experts (6.8 -> 9.7 -> 13.5 -> 17.1 ms),
already at its floor. Draft step ~2 ms. Rounds average ~2.2 tokens (prose less, code more), limited
by the MTP head's acceptance and the p-min cutoff, plus ~6 ms/round host overhead in llama-server.
Tried and rejected: wider row tiles for the multi-column head (spills), broadcast-free / texture
multi-column dense GEMV (slower than the existing COK GEMM), GPU target sampling (no change),
draft vocab 16k/32k (acceptance loss), N=8 (9-column verify exceeds the 8-column kernels).

## Deploy (not applied)
Run the live server from `D:\llm\build\qc-mtp\bin\llama-server.exe` with
`--spec-type draft-mtp --spec-draft-n-max 7 --spec-draft-p-min 0.6` (OpenMP runtime dir on PATH,
see mtp_test_me.ps1).

## DFlash test (2026-10-04)
Draft `pythoneer/qwen36-35b-a3b-dflash-llamacpp` (named Q8_0 but bf16 weights; requantized: D:\llm\dflash\dflash-q8real.gguf).
Agent-style bench (tune\results.csv), q8_0 KV: bf16 draft n4 24.9; Q8_0 draft n4 33.5, n6 33.6, n8 19.1 (9-col verify falls off the
8-col kernels) vs MTP 7/0.6 38.9. Acceptance is much better (~3.5 tok/round on code vs MTP ~2.2) but each round costs ~100 ms
(verify ~33 ms; rest is drafter + target hidden-state extraction + the draft's full 248k-vocab head via the target's lm_head, 8.6 ms).
Possible future work: reduced-vocab head for DFlash, profile the extraction/host path, 16-col verify kernel.

## Long-context decode: new q8_0 FA decode kernel (2026-10-04, commit on mtp-smallbatch, from branch fa-decode-dk256)
Profile at d32768: flash_attn_f32_q8_0_q1_vec_mq_split = 17.4 of 48.9 ms/token (~20 GB/s; 8 serial subgroup reductions per KV row).
Env knobs (split size, C8, head-sub, WG mult, decompose MIN_NQ) did nothing; two Codex rounds failed (best 6.2 tok/s).
New kernel_fa_q8_d256_g8_dec (kernels/mul_mv_q8_0_f32_flat.cl): lane-per-KV-row dp4a QK, per-64-row-tile softmax, lane-owns-dims PV.
Plain decode tg32 q8_0 KV: d0 32.7 -> 32.6, d8192 27.4 -> 31.0, d32768 20.3 -> 27.7 (+36%). FA tests 2985/2985; greedy output
matches up to near-ties; 85k-token MTP prompt OK. Opt out: GGML_OPENCL_FA_Q8_D256_LPR=0. Attention at 32k ~5.4 ms/token (floor ~3).

## Round 3 (2026-10-05): verify-path kernels, all on mtp-smallbatch
Profile of an MTP round (code prompt, ~3.7 tok/round): verify ~68 ms (MoE GEMM ~27 ms = at its distinct-expert
bandwidth floor; dense ~13; lm_head 5-col ~8), draft side ~24 ms (315 draft decodes x 2.9 ms incl ~1.2 ms host each,
graph reused 238/239; process 2.8 ms/round; sampling 0.27 ms/step).
Kept (commits a6f184c, f9b4ef5):
- small-M (<=2048) q8_0 2..8-column split-K int8 GEMV (512x2048 n5 45.4 -> 15.8 us) + lm_head mc local-memory activations
  (8.7 -> 7.9 ms); agent bench 39.6 -> 40.9 tok/s.
- verify attention (2..8 queries, n_kv >= 2048) routed to the row-per-lane FA kernel: pp5 d16384 55.4 -> 65.6,
  d32768 42.3 -> 55.5 t/s; MTP decode after 85k prompt 15.2 -> 20.5 tok/s.
Tried and rejected: dp4a single-column dense GEMV (slower: GEMVs are ~80% of measured streaming bandwidth, not ALU bound);
grouped small-batch MoE GEMV (slower; GEMM already at the distinct-expert floor); dense 2..8-col dp4a for large M (= COK);
mc lm_head 2/8 rows per subgroup (worse than 4); recordable queue (no gain, output identical); draft vocab 48k/40k (no gain);
draft backend sampling (no change); 2-query-per-WG FA kernel (wrong results on Adreno compiler, shelved:
D:\ninfer\analysis\fa_dec2_wip.cl.txt); Qualcomm binary kernel lib needs a Software Center login and covers Q4/MXFP4 MoE only.
Prefill at depth (pp2048: 550 / 473 @8k / 332 @32k) is near compute peak for attention (~4.5 TFLOPS).
- c42a99a skip padded slots in the q6_K MoE reorder+quant pre-pass (ragged GEMM): pp5 74.4 -> 77.5; agent bench 41.9 -> 43.0.
- e445d32 Q4_0 copy of the MTP draft head, 96k rows (drafter only): agent bench 42.6 -> 43.9 tok/s, acceptance 77.8 -> 80.5%.
  (64k Q4 43.1; 80k 43.1; 112k 43.2; 128k 42.8.) Deployed via env in start-qwen-server.ps1.
- Rejected: deferring the MTP catch-up decode into the first draft step (acceptance 73 -> 62%: a draft row decoded inside a
  multi-row batch drafts worse; decoded separately it matches but gains nothing); parallel FA merge (no gain); KV split size (256 best).
- f563561 f16-KV twin of the row-per-lane decode kernel for the MTP draft context (its KV stays f16): MTP decode after an
  85k prompt 22.9 -> 26.5 tok/s. (q8_0 draft KV -ctkd/-ctvd: 27.3 at 85k but 42.4 vs 43.75 tok/s at short context, lower
  draft acceptance; f16 multi-query routing: no gain, not kept.)
- 4844dcc clFlush every 8 nodes on small graphs (MTP draft step: enqueue 0.7 ms ran back to back with ~1.5 ms GPU):
  agent bench 43.6 -> 44.35 tok/s.
- Measured and dismissed: OpenCL object creation per GEMV (6 us), process priority (none), power mode (already Best performance),
  K-tile staging through local memory in the FA kernel (much slower), parallel FA merge (none).
Agent workload split (agentbench): generation ~85% of wall time, prefill <10%.
- 54f880f MTP draft top-10 without the sampler (scan only real draft-head rows; output byte-identical): agent bench
  44.35 -> 45.4 tok/s.
- Dismissed at the end: reduced logits readback for the MTP context (no change); chained in-graph drafting (PR #27173 style)
  estimated at only ~2-3% here (a fixed-length chain cannot stop at the p-min cutoff; current drafting is ~4.7 steps x 2.4 ms)
  for a large graph rewrite -- not done.

## Where it ends (2026-10-05)
Agent-prompt bench (4 prompts x 2 seeds, temp 0.6): plain 31.9 -> MTP start of day 39.6 -> now 45.4 tok/s (+42% vs plain).
MTP decode after an 85k-token prompt: 15.2 -> ~26.5 tok/s. Accuracy unchanged (verified output; draft-only changes).
Remaining time per MTP round (~90 ms): verify ~68% (MoE at the distinct-expert bandwidth floor, dense GEMVs ~80% of
measured streaming bandwidth), drafting ~16% (GPU ~1.45 ms/step + ~0.9 ms llama_decode host overhead), catch-up ~3%.
- c47251b/d56c802 chained MTP drafting (LLAMA_MTP_CHAIN=k, opt-in, default off): drafts identical to the per-step path, but
  not faster (k=7 41.3, k=4 43.0 vs 43.7 tok/s): the draft context alternates catch-up and chain graphs, so the ~600-node chain
  graph is rebuilt every round (0 reuse), and a fixed chain cannot stop at p_min. Re-allocating a cached non-active graph
  instead of rebuilding hit GGML_ASSERT(src_backend_id != -1) in the scheduler; reverted. Per-step GPU in the chain ~1.5 ms,
  same as standalone, so the remaining draft cost is GPU work, not host round trips.

## 2026-10-05 outside-method sweep
- Qualcomm binary kernel library (adreno-opencl-kernels.dll, 2026-09-24, adds q6_K/q8_0 ILA GEMMs): 34.1 vs 45.3 tok/s - replaces our tuned kernels; rejected.
- Qualcomm fork merge to 8085b4e (add+rms_norm fusion etc., mostly Adreno 850/dk128): 45.4 vs 45.3 - neutral (dev branch only).
- Upstream #27694 probabilistic MTP drafting + rejection verify, ported (dev 8f59bcd, opt-in): p-min 0.6 43.8, 0.8 45.1, 0.9 42.1 vs greedy 45.4 - no gain; draft already ~81% accepted.
- Context: GPU shared allocs fail at ~31.0 GiB; 128k/160k crash mid-prefill; 98k is the ceiling. Only ~6 GB RAM free with server up, so --cache-ram stays 4096.

- 2026-10-05 late: fixed-width (-DMC_NCOL) verify GEMVs +2.2% (45.4 -> 46.4, identical outputs); FA dk256 PV unroll 2 +1.5-2% at 64k. n-max 10/12: accepted tokens flat (2198/2214 vs 2227), drafts past 7 mostly rejected -> wider verify kernels and adaptive n-max not worth it. Live (mtp-smallbatch 8867485): 46.3 tok/s, agentbench 9/10 (09-wait-health timing check, flaky: fails ~half of all q6k runs).
- Live switched to d289633 (no fork merge; outputs identical to merged build, 2227 acc). Task 9 reruns: new 3/5 vs morning binary 6/6 (chance ~18%).
- 2026-10-06: seeded bench shows morning binary == a89c4bf == d289633 == fork-merged build token for token (2227). Live = a89c4bf, 46.4 tok/s.

## 2026-10-06 NPU + profile
- Latest upstream Hexagon (43fe9c6, skel re-signed with the existing GGML.HTP.v1 test cert): Q6_K MUL_MAT_ID 128x768x2048: n1 77us (GPU 77), n4 278us (GPU 403), n8 2618us (GPU 632). A crash in the NPU perf run then wedged the NPU (session open 0x80000406) and the GPU could no longer allocate the dev build's buffers (even at 64k) -> reboot needed. GPU+NPU concurrency was already known to corrupt output.
- Profile (prof_live.csv, code prompt, 51.5 tok/s): verify MoE dp4a GEMM 33% + reorder 5%, target lm_head verify (mc_lds) 10% at ~68 GB/s vs 128 GB/s single-col, draft head Q4_0 6%.
- GGML_OPENCL_MOE_GEMV_MAX_TOK=8 (fork hook, needs the 8085b4e merge): verify MoE via the f32 GEMV per token: 48.0 vs 46.5 tok/s (+3.2%); cutoff 4 = 48.0, 6 = 48.4 (noise). Changes outputs (fp32 activations instead of q8_1) -> needs agentbench before deploy.
- lm_head row-group loop (GGML_OPENCL_Q8_MC_LDS_RG) built, untested (blocked by the post-NPU memory state).

## 2026-10-06 Qualcomm audit + NPU
- Branch audit (git, 80+ qualcomm OpenCL branches): features either already in x2-unified-everything (rebased/reworded) or for other chips/quants. Option audit (264 relevant GGML_OPENCL_* knobs): opt-in/gated ones measured on dev+GEMV8 (base 47.6 tg / 337.5 pp): FUSE_RMS_ROPE_SET_ROWS 47.9/337.8 (noise, changes output), ADRENO_USE_LARGE_BUFFER 47.1, XMEM_SDPA 46.6/334.6, Q8_DENSE_DP4A_WIMG 46.4/343, Q8_0_DP4A_ALDS4 47.0/353.1 (+4.6% pp, identical output) KEEP, MOE_RAGGED_STATICIDX 46.2/334.5. GLU fusions are q4_K-only (no q6_K gate+up fused MoE GEMV exists).
- lm_head row-group loop: raised register use past the 512-item WG -> enqueue failure; reverted.
- NPU after reboot (upstream 43fe9c6, our shapes): MUL_MAT_ID n1 53us n2 100 n4 198 (= GPU per-token GEMV), n5-8 1.3-2.2 ms. Full model on HTP0: pp512 314, tg32 18.2 (GPU 550 / 32). No NPU route adds speed.
- Deployed 2026-10-06: 46b7cc1 + MOE_GEMV_MAX_TOK=8 + Q8_0_DP4A_ALDS4=1; agentbench 10/10 (q6k-oct6).
