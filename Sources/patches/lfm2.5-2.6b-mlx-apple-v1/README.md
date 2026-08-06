# lfm2.5-2.6b-mlx-apple-v1 — patch series

## Series (verified frontier — score 1.006225)

| Patch | Intent |
| --- | --- |
| `0001-sc-fused-qk-norm-rope-greedy.patch` | SC elementwise + q/k RMSNorm+RoPE Metal kernels + greedy logsumexp skip |

## Kernels (all bit-identical to stock ops, verified 0.0 diff)

1. **ShortConv elementwise** (verified 1.003642): one dispatch replaces
   split+mul+concat+conv1d+mul for L=1 decode (22 conv layers).
2. **q/k RMSNorm+RoPE** (verified 1.006225): one dispatch replaces
   q_layernorm + k_layernorm + transpose + 2× rope for L=1 decode
   (8 attention layers). Mirrors MLX rms_single_row (N_READS=4, simd_sum,
   precise::rsqrt, bf16 cast-then-multiply) and rope_impl (metal::fast::cos,
   base=1e7, (j, j+32) pairing).
3. **Greedy logsumexp skip** (verified 1.003831): argmax(logits) without the
   128k logsumexp; token-sized second value for async_eval.

## Tried after the frontier (all rejected / noise — do not resubmit)

- Fused residual add + RMSNorm (30 layers): 1.0033 on runner (below frontier)
- Fused operator_norm for conv input: +0.11% local (noise)
- KVCache step 256→512/1024: neutral (update cost is slice-write, not realloc)
- mx.compile variants: slower on M4
- Engine qmv tuning, n-gram spec, requant: see earlier notes

## Leaderboard

MLX baseline 1.0 → SC 1.003642 → SC+greedy 1.003831 → **+QK 1.006225** (current)
