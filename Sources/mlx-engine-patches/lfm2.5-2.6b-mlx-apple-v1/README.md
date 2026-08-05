# MLX engine patches — lfm2.5-2.6b-mlx-apple-v1

## Series

| Patch | Intent |
| --- | --- |
| `0001-qmv-wide-m1-decode-with-cmake.patch` | Route M=1 decode QMV through `affine_qmv_wide` on gen-15+ (M4); add `nv_1` instantiation; auto-install cmake in setup.py |

## Why this is the lever

LFM2.5-2.6B decode is almost entirely **M=1 affine quantized matvecs**. Stock
MLX v0.32.0 gates `qmv_wide` dispatch at `M >= 2` even on gen-15+ (M4) where
`use_qmv_wide` returns true for affine. So single-token decode never uses the
wide kernel. This patch:

1. Instantiates `affine_qmv_wide` for `vecs_per_tg=1` (nv_1)
2. Changes dispatch from `M >= 2` to `M >= 1`
3. Auto-installs cmake via `uv pip install` if missing (runner fix)

Pinned engine: **MLX v0.32.0** (`7a1d4f5c`).
