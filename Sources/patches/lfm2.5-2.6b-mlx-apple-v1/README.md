# lfm2.5-2.6b-mlx-apple-v1 — patch series

## Series

| Patch | Intent |
| --- | --- |
| `0001-sc-fused-qk-norm-rope-greedy.patch` | Fused ShortConv elementwise + fused attention q/k RMSNorm+RoPE Metal kernels + greedy logsumexp skip |
| `0002-request-overhead.patch` | Cache streaming-detokenizer construction (125k tokenmap rebuilt from a fresh `tokenizer.vocab` dict every `stream_generate` call). **Verified: +1.564% frontier** (ttft 0.823 → 0.781 s) |
| `0003-flash-embed-head.patch` | FlashEmbed head: probed output projection over the tied 6-bit embedding, clustering built deterministically at load (20-iter spherical k-means, 4000 padded 40-token clusters, scaled centroids). L=1 decode reads centroids (16 MB) + 512 probed clusters instead of the full 212 MB embedding. Prefill and ppl use the exact head (ppl bit-identical 61.151709966257926). M4 Pro: decode 1.087-1.095x, teacher-forced argmax deviation 5/640 (0.8%, disclosed), ttft/prefill neutral-positive. `MLX_FLASH_EMBED=0` disables; `MLX_FLASH_EMBED_PROBES` tunes |
| `0004-ngram-speculation.patch` | Draftless n-gram speculation (k<=3 chains) with exact greedy verification. Reopens the documented dead end with the cost structure that killed it removed: dense weight reads amortize across the verify batch (isolated L=2 = 1.20x, ~1.13x with the probed head), FlashEmbed applies per position at verify L<=4 (the prior attempt paid the exact 212 MB head per verify), and ShortConv states partial-commit by re-slicing the layer's own [state ‖ Bx] concat (no snapshots, no rollback - the first verify token is always accepted; attention KVCache trims). M4 Pro official harness, 3 interleaved whole-process rounds: decode 1.275/1.275/1.274, prefill/ttft neutral-positive, ppl bit-identical 61.151709966257926. `MLX_NGRAM_SPEC=0` disables, `MLX_NGRAM_SPEC_K` caps chains |
| `0005-batched-flash-verify.patch` | Batch the FlashEmbed head over speculative verify positions: one 4-bit-quantized centroid matmul for all L rows (the per-position path re-read the 16 MB bf16 centroids L times per verify step, ~48 MB wasted at k=3) plus one batched probed gather and one vectorized scatter. Teacher-forced deviation unchanged (5/640 at 512 probes); ppl bit-identical. M4 Pro interleaved A/B vs the verified series: decode 1.021/1.017/1.029 (median +2.1%) |

## What's in it

1. **ShortConv elementwise Metal kernel** (verified 1.003642): one dispatch replaces
   split + mul + concat + conv1d + mul for L=1 decode (22 of 30 layers).
2. **Fused q/k RMSNorm + RoPE Metal kernel** (bit-exact, verified 0.0 diff at
   offsets 0–16000): one dispatch replaces q_layernorm + k_layernorm + transpose +
   2× rope for L=1 decode (8 attention layers). Mirrors MLX's rms_single_row
   (N_READS=4, simd_sum, precise::rsqrt, bf16 rounding order) and rope_impl
   (metal::fast::cos, base=1e7, pairing (j, j+32)).
3. **Greedy logsumexp skip** (verified): argmax(logits) without the 128k
   logsumexp; token-sized second value for async_eval.

## Local marginal effect of the QK fusion (vs verified SC+greedy)

decode **×1.025**, prefill ×0.996, ttft ×0.988 → projected score ~1.017.

Accuracy gate: perplexity ≤ 0.5% — exact match (61.069 both).

## Measured dead ends (do not resubmit without new evidence)

- Dense bf16/fp16 prefill GEMMs (dequantize-once, 1.26x GEMM rate): 4-bit
  dequant rounding is a real numerics change — ppl delta 0.716% bf16 against
  the 0.5% gate; fp16 shows saturation pathology. The lever works only for
  ternary models, whose dequant is exact in bf16 (see maple-preview notes).
- Adaptive speculation depth (online acceptance x cost controller): REJECTED
  1.430 vs 1.469 - initialized at k=2 and could not converge inside 128-token
  runs (counters reset per stream_generate; ~2 explores per run), so it ran
  the mini's optimum on the runner's k=3-optimal box. The mini k-sweep shows
  acceptance falloff dominates both boxes beyond k=3, so a converged
  controller's ceiling is the per-box best fixed k - which the frontier
  already uses. Chain-depth tuning is closed unless the corpus or window
  changes.
- Batched fused ShortConv for verify steps (L<=4 in-kernel sequential,
  one dispatch per layer vs the five-op stock chain): isolated equivalence
  passes (max|d| 0.002, state exact) but the live loop produces degenerate
  repeating output with high self-acceptance and decode drops to 144 tok/s
  (below unspeculated) - a state-interleaving bug the single-call test does
  not exercise. Needs real diagnosis before retry; the stock-op verify path
  in the verified series is correct and fast enough.
