# lfm2.5-2.6b-mlx-apple-v1 — patch series

## Series

| Patch | Intent |
| --- | --- |
| `0001-sc-qk-rn-fused-greedy.patch` | Four Metal kernels + greedy skip: SC elementwise, q/k RMSNorm+RoPE, residual+RMSNorm, greedy logsumexp skip |

## Kernels

1. **ShortConv elementwise** (verified 1.003642): one dispatch for L=1 decode (22 conv layers).
2. **q/k RMSNorm+RoPE** (verified 1.006225): one dispatch replaces q_layernorm +
   k_layernorm + transpose + 2× rope (8 attention layers). Bit-identical
   (mirrors rms_single_row N_READS=4 + rope_impl rotation).
3. **Residual add + RMSNorm** (NEW): one dispatch replaces h=x+r + ffn_norm(h)
   for L=1 decode (30 layers). Mirrors MLX rms reduction order exactly:
   N_READS=4 per thread, 16 simdgroups with sequential group-partial sum,
   precise::rsqrt, bf16 cast-then-multiply.
4. **Greedy logsumexp skip** (verified 1.003831).

## Accuracy

Perplexity exact match (61.069 both arms). All kernels bit-identical to the
stock ops they replace (verified 0.0 diff on random inputs across offsets).
