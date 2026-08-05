# MLX engine patches — lfm2.5-2.6b-mlx-apple-v1

## Series

| Patch | Intent |
| --- | --- |
| `0001-affine-qmv-wide-m1-decode-nv1.patch` | Instantiate `affine_qmv_wide` for `vecs_per_tg=1` and route **M=1** decode QMV through it on gen-15+ (Apple M4) |

## Why this is the lever

LFM2.5-2.6B decode is almost entirely **M=1 affine quantized matvecs**
(gate/up/down, QKV, ShortConv projections). Stock MLX v0.32.0:

1. Instantiates `affine_qmv_wide` only for `nv∈{2,3,4,5}` (no `nv_1`)
2. Gates `dispatch_qmv` with `M >= 2 && use_qmv_wide(...)`

So even on M4 (`use_qmv_wide` true for affine because gen ≥ 15), **single-token
decode never uses the wide kernel**. This patch adds the missing `nv_1`
specialization and opens the dispatch for M=1.

`use_qmv_wide` still returns false for affine on gen-13 (M1/M2), so older Macs
keep the stock qmv path — no regression there.

## Surfaces

| Where | What | Rebuild |
| --- | --- | --- |
| `Sources/patches/<track>/` | `mlx_lm` Python overlay | none |
| `Sources/mlx-engine-patches/<track>/` | MLX C++/Metal (this dir) | full engine build |

Pinned engine: **MLX v0.32.0** (`7a1d4f5c`).

## Local validation

- Patch applies cleanly to v0.32.0
- Built locally with `nv_1` + forced wide on gen-13: kernels load, matmul runs
- Gen-13 absolute speed is not the target (wide is gen-15+ for affine); ranked
  box is **Apple M4** where stock already enables wide for M≥2

## Accuracy

Kernel math is the existing `qmv_wide_impl`; only dispatch + instantiation
change. Gate remains perplexity ≤ 0.5% vs stock.
