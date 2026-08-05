# MLX engine patches — lfm2.5-2.6b-mlx-apple-v1

## Series

| Patch | Intent |
| --- | --- |
| `0001-qmv-fast-widen-bn16.patch` | Widen qmv_fast threadgroup: 4 simdgroups (bn=16, was bn=8) |

## Why

Stock qmv_fast uses 2 simdgroups (64 threads, bn=8 outputs per threadgroup).
For LFM2.5's shapes (N=10752, K=2048), this creates 1344 threadgroups.
Doubling to 4 simdgroups (128 threads, bn=16) halves to 672 threadgroups,
reducing scheduling overhead while maintaining the same per-thread work.

All LFM2.5 N values (10752, 6144, 2048, 512) are divisible by 16.
