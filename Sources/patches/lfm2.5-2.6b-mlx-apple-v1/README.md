# lfm2.5-2.6b-mlx-apple-v1 — patch series

## Series

| Patch | Intent |
| --- | --- |
| `0001-sc-qk-rn-fused-greedy.patch` | SC elementwise + q/k RMSNorm+RoPE + residual+RMSNorm Metal kernels + greedy skip |

## Kernels (all bit-identical to stock ops)

1. **ShortConv elementwise** (verified 1.003642): L=1 decode, 22 conv layers.
2. **q/k RMSNorm+RoPE** (verified 1.006225): L=1 decode, 8 attention layers.
   Mirrors rms_single_row N_READS=4 + rope_impl rotation exactly.
3. **Residual add + RMSNorm** (NEW): L=1 decode, 30 layers. Replaces h=x+r +
   ffn_norm(h). Mirrors MLX reduction order: N_READS=4, 16 simdgroups with
   sequential group-partial sum, precise::rsqrt, bf16 cast-then-multiply.
4. **Greedy logsumexp skip** (verified 1.003831).

## Marginal effect of the residual+norm fusion (vs 1.006225 frontier)

decode **×1.012** locally (tight range 1.008–1.015, 6 pairs, same harness).
ppl exact match (61.069 both).
