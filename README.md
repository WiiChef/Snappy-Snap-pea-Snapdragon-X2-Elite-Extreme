# Snappy Snap-pea - Snapdragon X2 Elite Extreme

### The fastest and most accurate Snapdragon Qwen3.6-35B-A3B configuration?

Getting **Qwen3.6-35B-A3B** with all-**Q6_K** experts to decode as fast as possible on a
**Snapdragon X2 Elite Extreme** laptop (Adreno X2-90 GPU, OpenCL, 48 GB unified memory) with
**MTP speculative decoding**.

This repo isn't llama.cpp. It records what we changed on top of Qualcomm's llama.cpp fork
([qualcomm/llama.cpp](https://github.com/qualcomm/llama.cpp), branch `opencl/x2-unified-everything` @ `fb62b8a`):

- the kernels
- the patch series
- the server config
- the benchmarks
- everything we tried that didn't help

## Headline numbers

Agent-prompt bench: 4 coding and agent prompts × 2 seeds, temp 0.6, 400 generated tokens each.

| Setup | Decode tok/s |
|---|---|
| Plain decode (no speculation) | 31.9 |
| MTP on the stock fork | 39.6 |
| **MTP + snap-pea** | **45.4** (+43% vs plain, +15% vs stock MTP) |
| MTP decode after an 85k-token prompt | 15.2 → **26.5** |

- **Prefill:** about 550 tok/s from an empty context, 473 tok/s at 8k and 332 tok/s at 32k, close to the GPU's compute peak.
- **Accuracy:** agentbench 10/10 on two runs. The draft-side changes never change the verified output; drafts only decide how many tokens get checked per round.

Where the time goes per MTP round:
- **About 68% verifying the drafted tokens.**
  - The MoE expert reads run at the bandwidth floor for the distinct experts each batch touches.
  - The dense GEMVs reach about 65–80% of measured streaming bandwidth.
- **About 16% drafting.** That's GPU work of about 1.5 ms per step, not host overhead.

## Model and config

The model is `scan-all-q6k.gguf`, 27.8 GB: routed and shared experts in Q6_K, everything else in Q8_0, including the MTP layer `blk.40`.

```
llama-server -m scan-all-q6k.gguf -ngl 99 -fa on -ub 512 -b 2048 -t 8 -np 1 \
  --spec-type draft-mtp --spec-draft-n-max 7 --spec-draft-p-min 0.6 \
  -ctk q8_0 -ctv q8_0 -c 98304 --cache-ram 4096 \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 0
env: LLAMA_MTP_DRAFT_HEAD_Q4=1  LLAMA_MTP_DRAFT_VOCAB=98304
```

Full launcher: [scripts/start-qwen-server.ps1](scripts/start-qwen-server.ps1).

Memory limits:
- **98k is the context ceiling.** The GPU's shared allocations fail at about 31 GiB. At 128k and 160k, a scratch buffer that grows with context crashes the prefill.
- **The KV cache has to be q8_0.** With f16 KV, the MTP draft context runs out of memory at around 30k tokens. q8_0 survives 85k with KLD within noise.
- **The 4 GB prompt cache is already the limit.** Only about 6 GB of RAM stays free with the server up.

## Kernels and changes

The code each kernel adds is extracted in [kernels/](kernels/). The full, applicable series is in
[patches/live](patches/live) (`git am` onto `fb62b8a`). Everything is on by default unless marked opt-in, and each has an env opt-out.

| Change | What it does | Opt-out |
|---|---|---|
| `kernel_fa_q8_d256_g8_dec` | Flash-decoding for head dim 256, GQA 8, q8_0 KV. Each lane owns one KV row and computes the QK dot with dp4a against Q requantized in local memory. Softmax runs per 64-row tile, and each lane accumulates 4 dims of PV. Partial records and merge are the same as the fork's split FA. | `GGML_OPENCL_FA_Q8_D256_LPR=0` |
| Verify attention routing | Uses the same kernel for the 2–8 query verify batch when n_kv ≥ 2048, instead of decomposing. | `GGML_OPENCL_FA_Q8_D256_LPR_NQ=0`, `..._MIN_KV` |
| `kernel_fa_f16_d256_g8_dec` | f16 twin of the kernel above, for the MTP draft context's f16 KV. | `GGML_OPENCL_FA_F16_D256_LPR=0` |
| `kernel_gemv_noshuffle_q8_0_q8a_dp4a_mc_splitk` | Split-K multi-column Q8_0 GEMV for small-M projections (M ≤ 2048, 2–8 columns) in the verify batch. It quantizes the activations once and reduces in a second pass. | `GGML_OPENCL_Q8_NS_MC_SPLITK=0` |
| `kernel_mul_mv_q8_0_q8a_dp4a_mc_lds` | Multi-column dp4a GEMV with the activations held in local memory, 8 subgroups. | `GGML_OPENCL_Q8_MC_LDS_NSG` |
| MoE reorder pad skip | `kernel_moe_reorder_quant_a_q8_1_ragged` returns early on padded slots in the ragged q6_K dp4a MoE path. | `GGML_OPENCL_MOE_REORDER_SKIP_PAD=0` |
| Q4_0 draft head | At load, requantizes the first 96k rows of the Q8_0 output head to Q4_0 (108 MiB), used by the drafter only. It raised acceptance from 77.8% to 80.5% and decode from 42.6 to 43.9 tok/s. | opt-in: `LLAMA_MTP_DRAFT_HEAD_Q4=1`, `LLAMA_MTP_DRAFT_VOCAB` |
| Fast draft top-k | Scans the real vocab rows for the top 10 and admits zeros from the padded rows exactly where they rank. This replaces the sampler's full 248k sort. | `GGML_MTP_FAST_TOPK=0` |
| Small-graph flush | `clFlush` every 8 nodes on graphs of 256 nodes or fewer (the draft steps). | `GGML_OPENCL_FLUSH_EVERY=N` |
| Timing | Prints the host/GPU split of draft time at exit. | opt-in: `GGML_MTP_TIMING=1` |
| Chained drafting | k draft steps in one graph. Identical drafts, but not faster (41.3 vs 43.7 tok/s). | opt-in: `LLAMA_MTP_CHAIN=k` |

[patches/experimental](patches/experimental) holds our port of upstream #27694, probabilistic MTP drafting with rejection-sampling verify (`--spec-draft-sampling probabilistic`). It measured no gain: 45.1 tok/s at p-min 0.8 vs 45.4 greedy.

## What didn't help

All of these are measured, with details in [RESULTS.md](RESULTS.md).

**Draft-side alternatives:**
- DFlash draft: 33.6 tok/s.
- n-gram speculation: +0%.
- Other draft-head vocab sizes (40k–128k): 96k Q4_0 was best.
- p-min 0.5 / 0.7, n-max 6.
- q8 draft KV: worse at short context.
- Chained drafting.
- Probabilistic drafting.
- Deferring the catch-up decode: acceptance dropped from 73% to 62%.

**Kernel experiments:**
- Grouped MoE GEMV.
- 2/8-row multi-column variants.
- dp4a 1-column GEMV.
- K-tile local-memory staging.
- Parallel merge.
- Split-size sweeps.

**Runtime and outside options:**
- Recordable queues: they never engage with flash attention.
- Qualcomm's precompiled kernel library (`adreno-opencl-kernels.dll`): **34.1 vs 45.3**. Don't ship it next to these kernels.
- Merging the fork's commits up to Oct 5: neutral.
- Fewer active experts (6 or 4): KLD 7–25× worse, rejected for accuracy.
- Smaller expert quants: rejected for accuracy, and they gave no real speed gain here.
- Hexagon NPU: llama.cpp can't run GGUF on it. Running it alongside the GPU server also corrupted the GPU model's output.

## Scripts

- [scripts/bench_cfg.ps1](scripts/bench_cfg.ps1) + `prompts.json`: the agent-prompt bench.
- `long_check.ps1`: decode after an 85k-token prompt.
- `ctx_fill.ps1`: context ceiling and GPU memory peak.
- `build-qc-mtp.cmd`: incremental build (MSVC arm64, OpenCL).
- `official.jinja`: the Qwen chat template used.

Paths in the scripts point to the original machine, so adjust them for yours.

## License

The patches and kernels modify llama.cpp and are MIT-licensed like it.
