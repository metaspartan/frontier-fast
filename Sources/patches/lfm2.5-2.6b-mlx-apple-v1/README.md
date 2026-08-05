# lfm2.5-2.6b-mlx-apple-v1 — patch series

## Series

| Patch | Intent |
| --- | --- |
| `0001-greedy-decode-host-overhead-kvcache-step.patch` | Greedy `generate_step`: skip 128k logsumexp + avoid async_eval of full-vocab logprobs; defer per-token `get_peak_memory` to final yield; KVCache step 256→512 |

## Why no engine patches

`Sources/mlx-engine-patches/...` is empty on purpose. A non-empty engine series
forces a full MLX rebuild; the trusted runner currently fails that path with
`No such file or directory: 'cmake'`. This series is a pure `mlx_lm` Python
overlay (no rebuild).

## What a patch targets here

The runner copies pristine `mlx_lm`, applies this series with `git apply`, and
selects the arm with `PYTHONPATH`.

## Measured stock on the ranking runner (Apple M4, 16 GB)

- decode **~61.6 tok/s**, prefill **~654 tok/s**

## Local notes (Apple M1 Ultra, directional)

- ppl stock = cand exact
- greedy text match on interleaved process A/B
- cooler paired ratios (3 pairs): decode **×1.015**, prefill **×1.027**,
  ttft **×1.032** → score ≈ **1.020**

## Gates

Accuracy is **perplexity equivalence within 0.5%**. Runner scores median of
per-round ratios on whole-process A/B.
