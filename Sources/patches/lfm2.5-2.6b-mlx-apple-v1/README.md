# lfm2.5-2.6b-mlx-apple-v1 — patch series

## Series

| Patch | Intent |
| --- | --- |
| `0001-fused-shortconv-decode-elementwise.patch` | L=1 ShortConv: fused elementwise (split+mul+concat+conv1d+mul → one Metal kernel) |
| `0002-fused-sc-elem-decode-gate-up.patch` | Same + fused gate+up MLP for decode (M=1) |

## Verified

| Patch | Score | Decode | Prefill | TTFT |
| --- | --- | --- | --- | --- |
| 0001 | **1.00364** | +0.39% | +0.30% | +0.34% |

## Engine patches

Empty — runner lacks cmake for MLX rebuild.
