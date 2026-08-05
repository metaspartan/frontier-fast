# MLX engine patches — lfm2.5-2.6b-mlx-apple-v1

## Series

| Patch | Intent |
| --- | --- |
| `0001-qmv-wide-m1-decode-nv1.patch` | Route M=1 decode QMV through `affine_qmv_wide` on gen-15+ (M4); add `nv_1` instantiation |

## Why this is the lever

LFM2.5-2.6B decode is almost entirely **M=1 affine quantized matvecs**.
Stock MLX v0.32.0 gates `qmv_wide` dispatch at `M >= 2` even on gen-15+
(M4) where `use_qmv_wide` returns true for affine. So single-token
decode never uses the wide kernel.

This patch adds the missing `nv_1` specialization and opens dispatch
for M=1. `use_qmv_wide` still returns false for affine on gen-13
(M1/M2), so older Macs keep the stock qmv path.

Pinned engine: **MLX v0.32.0** (`7a1d4f5c`).
