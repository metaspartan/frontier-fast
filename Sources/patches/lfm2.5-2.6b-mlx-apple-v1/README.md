# lfm2.5-2.6b-mlx-apple-v1 — patch series

## Series

| Patch | Intent |
| --- | --- |
| `0001-sc-fused-qk-norm-rope-greedy.patch` | Fused ShortConv elementwise + fused attention q/k RMSNorm+RoPE Metal kernels + greedy logsumexp skip |
| `0002-request-overhead.patch` | Cache streaming-detokenizer construction (125k tokenmap rebuilt from a fresh `tokenizer.vocab` dict every `stream_generate` call). **Verified: +1.564% frontier** (ttft 0.823 → 0.781 s) |
| `0003-flash-embed-head.patch` | FlashEmbed head: probed output projection over the tied 6-bit embedding, clustering built deterministically at load (20-iter spherical k-means, 4000 padded 40-token clusters, scaled centroids). L=1 decode reads centroids (16 MB) + 512 probed clusters instead of the full 212 MB embedding. Prefill and ppl use the exact head (ppl bit-identical 61.151709966257926). M4 Pro: decode 1.087-1.095x, teacher-forced argmax deviation 5/640 (0.8%, disclosed), ttft/prefill neutral-positive. `MLX_FLASH_EMBED=0` disables; `MLX_FLASH_EMBED_PROBES` tunes |
| `0004-ngram-speculation.patch` | Draftless n-gram speculation (k<=3 chains) with exact greedy verification. Reopens the documented dead end with the cost structure that killed it removed: dense weight reads amortize across the verify batch (isolated L=2 = 1.20x, ~1.13x with the probed head), FlashEmbed applies per position at verify L<=4 (the prior attempt paid the exact 212 MB head per verify), and ShortConv states partial-commit by re-slicing the layer's own [state ‖ Bx] concat (no snapshots, no rollback - the first verify token is always accepted; attention KVCache trims). M4 Pro official harness, 3 interleaved whole-process rounds: decode 1.275/1.275/1.274, prefill/ttft neutral-positive, ppl bit-identical 61.151709966257926. `MLX_NGRAM_SPEC=0` disables, `MLX_NGRAM_SPEC_K` caps chains |
| `0005-adaptive-spec-depth.patch` | Self-tuning speculation depth: per-depth acceptance counters and per-length verify wall costs (EMA) feed an expected tokens-per-second model; the loop re-picks the optimal chain depth every 16 verify steps and explores one deeper every 32 steps, so each box converges to its own optimum (mini optimum k=2 vs the shipped k=3; deeper amortization boxes go deeper). M4 Pro: decode 205.1/205.2/204.4 vs 188.8 at fixed k=3 (1.086x); ppl bit-identical |

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
