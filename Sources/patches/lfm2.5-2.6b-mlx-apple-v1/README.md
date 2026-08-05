# lfm2.5-2.6b-mlx-apple-v1 — patch series

## Series

| Patch | Intent |
| --- | --- |
| `0001-greedy-decode-host-overhead-kvcache-step.patch` | Greedy `generate_step`: skip 128k logsumexp + avoid async_eval of full-vocab logprobs; defer per-token `get_peak_memory` to final yield; KVCache step 256→512 |

## Engine series (rebuild)

See `Sources/mlx-engine-patches/lfm2.5-2.6b-mlx-apple-v1/` for the M4-targeted
`affine_qmv_wide` M=1 decode path (`nv_1` instantiation + dispatch).

## What a patch targets here

MLX has no engine binary to rebuild for the Python series. **Your patch applies
to the installed `mlx_lm` Python tree**. The runner copies pristine `mlx_lm`,
applies the series with `git apply`, and selects the arm with `PYTHONPATH`.

## Measured stock on the ranking runner (Apple M4, 16 GB)

- decode **~61.6 tok/s**, prefill **~654 tok/s** (latest baseline phase)

## Local notes (Apple M1 Ultra, directional only)

- ppl stock = cand exact on `fixtures/gainz-corpus.txt`
- greedy text match on interleaved process A/B
- cooler paired ratios for this python series (noisy): score ≈ **1.02–1.03**
- Custom Metal SwiGLU / residual kernels: ppl-ok but **slower** + text drift
- Fused gate+up `quantized_matmul`: bit-exact, **slower** on Metal here
- ZMLX stock patches on dense LFM2.5: token-identical but much slower locally

## Gates

Accuracy is **perplexity equivalence within 0.5%**. Runner scores median of
per-round ratios on whole-process A/B.
