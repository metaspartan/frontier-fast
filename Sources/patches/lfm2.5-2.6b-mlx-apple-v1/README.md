# lfm2.5-2.6b-mlx-apple-v1 — patch series (verified frontier: 1.006225)

| Patch | Intent |
| --- | --- |
| `0001-sc-fused-qk-norm-rope-greedy.patch` | SC elementwise + q/k RMSNorm+RoPE Metal kernels + greedy logsumexp skip |

Kernels (all bit-identical to stock, ppl exact 61.069):
1. ShortConv elementwise (verified 1.003642): one dispatch for L=1 decode (22 layers)
2. q/k RMSNorm+RoPE (verified 1.006225): one dispatch for L=1 decode (8 attention layers)
3. Greedy logsumexp skip (verified 1.003831)

## Measured dead ends (do not resubmit without new evidence)
- N-gram speculative decoding (K=1/K=3, pipelined, reference-snapshots): 0.85x harness —
  batch M=2 costs 1.47x at growing cache, online accept 78%, overlap incomplete
- Fused residual add + RMSNorm (30 layers): 1.0033 runner (below frontier)
- Fused operator_norm: +0.11% local (noise)
- mx.compile variants, engine qmv tuning, requant, KVCache step: all dead
