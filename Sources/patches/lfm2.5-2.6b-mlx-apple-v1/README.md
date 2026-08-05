# lfm2.5-2.6b-mlx-apple-v1 — patch series

## Series

| Patch | Intent |
| --- | --- |
| `0001-lfm2-shortconv-decode-fma.patch` | L=1 ShortConv decode: depthwise FIR as 3 float32 FMAs (skip concat + grouped conv1d) |

## Why this lever

LFM2.5-2.6B is a hybrid: **22 of 30 layers are ShortConv**, not attention.
On the ranked decode path every token hits those 22 layers with `L=1` and
`L_cache=3`. Stock builds a `(B,3,H)` concat and runs general grouped `conv1d`
for what is already three multiplies and an add per channel.

This patch keeps prefill and lengths-based cache on the stock path, and only
rewrites the common single-token decode path.

## Engine patches

`Sources/mlx-engine-patches/...` is empty. Non-empty series force a full MLX
rebuild; the trusted runner currently lacks `cmake` for that path.

## Correctness

- Ranked gate: **perplexity ≤ 0.5%** relative to stock
- Local: **exact ppl match** (61.069) on `fixtures/gainz-corpus.txt`
- Greedy text match on same-process A/B (FMA uses float32 accumulate)

## Local signal (Apple M1 Ultra, directional)

Cooler interleaved process A/B (4 pairs, official `tools/mlx_bench.py`):

| | decode | prefill | ttft | score |
| --- | ---: | ---: | ---: | ---: |
| median paired ratio | **×1.071** | ×1.006 | ×1.006 | **~1.048** |

Thermal noise on this box is large; official score is the M4 runner.

## Dead ends (do not resubmit without new evidence)

- Greedy logsumexp skip / peak_memory defer: noise floor on M4 (~0.9996)
- Engine `qmv_wide` M=1: blocked by missing runner cmake
- Custom Metal SwiGLU / residual_norm: ppl drift + slower
- Fused gate+up `quantized_matmul`: bit-exact, slower
