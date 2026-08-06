# lfm2.5-2.6b-mlx-apple-v1 — patch series

| Patch | Intent |
| --- | --- |
| `0001-rn256-qk-sc-greedy.patch` | SC elementwise + q/k RMSNorm+RoPE + residual+RMSNorm(256-thread) + greedy skip |
| `0002-request-overhead.patch` | Port of the maple-verified request-path fix: cache the BPE detokenizer tokenmap (125k-entry vocab rebuilt from a fresh `tokenizer.vocab` dict on every `stream_generate` call — ~34 ms M1 Ultra, ~50 ms M4 Pro, inside measured TTFT); stop flushing the MLX buffer pool after the final prefill chunk and at decode step 0. Numerics-free: ppl bit-identical (61.151709966257926 both arms). M4 Pro interleaved A/B over 7 whole-process rounds: ttft 1.08–1.12x, prefill +0.8% order-corrected, decode wash (within the ±0.6% floor). Same lever verified on maple-preview at +2.1pp score |

Kernels (ppl exact 61.069):
1. ShortConv elementwise (verified 1.003642): one dispatch, L=1 decode, 22 layers
2. q/k RMSNorm+RoPE (verified 1.006225): one dispatch, L=1 decode, 8 attention layers
3. residual add + RMSNorm (256-thread / 8-simdgroup geometry): one dispatch replaces
   h=x+r + ffn_norm(h) for L=1 decode, all 30 layers. 256-thread threadgroups pack
   onto M4's 10 GPU cores; the previous 512-thread (16-simdgroup) version scored
   below the frontier on M4 despite winning locally on M1 Ultra (64 cores).
4. Greedy logsumexp skip (verified 1.003831)
