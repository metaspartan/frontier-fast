# lfm2.5-2.6b-mlx-apple-v1 — patch series

## Series

| Patch | Intent |
| --- | --- |
| `0001-kvcache-step-512.patch` | KVCache step 256→512 for the ranked 512-token prefill |

## Engine series

See `Sources/mlx-engine-patches/lfm2.5-2.6b-mlx-apple-v1/` for the M4-targeted
`qmv_wide` M=1 decode optimization (the real lever).

## Gates

Accuracy: perplexity ≤ 0.5% relative to stock. The engine change only affects
dispatch path selection (same `qmv_wide_impl` kernel math), not computation.
