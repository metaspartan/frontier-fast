# maple-preview-mlx-apple-v1 — patch series

| Patch | Intent |
| --- | --- |
| `0001-registry-load-flashhead.patch` | Registry-preferred model load (no trust_remote_code needed) + auto-enable FlashHead approximate lm_head for L=1 decode |

## Why registry-preferred load

The pinned deepgrove fork (`mlx-lm @ eba96c1`) ships `mlx_lm/models/maple.py`
— a superset of the checkpoint's `maple.py` with self-checked fused kernels
(`_matches` verifies each fast path against its portable equivalent on live
weights once, then commits to it). The checkpoint config carries
`model_file: maple.py`, which stock `load_model` refuses without
`trust_remote_code`. The harness calls `load(args.model)` with no such flag,
so **no candidate can load at all without this fix**.

The patch prefers `get_model_classes(config)` (registry) when the model type
is registered, falling back to the checkpoint module otherwise. Loading the
fork's registered maple.py also gives the newer fused QK norm+rope kernel.

## Why FlashHead

The checkpoint ships `model-flashhead.safetensors` and `flash_head` metadata,
but `load()` defaults `use_flash_head=False` — the runner measures the full
151936×2048 4-bit lm_head. On M4 (~120 GB/s) that head reads ~155 MB/step,
~1.3 ms of the 5.9 ms decode step. FlashHead scores 512×32-token cluster
centroids, then gather_qmm's the probed rows: ~22 MB/step.

- L=1 decode only; prefill (and the ppl gate) use the exact lm_head → **ppl
  identical (115.817 both)**
- Greedy argmax is exact whenever the true argmax is in probed clusters +
  force tokens; verified 0/256 mismatches on the benchmark corpus, and the
  harness greedy text is byte-identical

## Local (M1 Ultra, directional)

FlashHead decode 179.4 vs exact 180.5 tok/s (−0.6%: M1's bandwidth hides the
head cost). M4 is the target: 155→22 MB/step should give ~1.2x decode there.
