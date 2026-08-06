# lfm2.5-2.6b-mlx-apple-v1 — patch series

## Series

| Patch | Intent |
| --- | --- |
| `0001-sc-fused-qk-norm-rope-greedy.patch` | Fused ShortConv elementwise + fused attention q/k RMSNorm+RoPE Metal kernels + greedy logsumexp skip |

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
