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
151936×2048 4-bit lm_head. FlashHead scores 512×32-token cluster centroids
then gather_qmm's the probed rows, cutting the lm_head read ~7x.

- L=1 decode only; prefill (and the ppl gate) use the exact lm_head → **ppl
  identical**
- Greedy argmax is exact whenever the true argmax is in probed clusters +
  force tokens; verified 0 mismatches on the benchmark corpus

## Verified on trusted runner

- **score 1.1829** (frontier), decode 218.68 tok/s (+29.4%), prefill +0.3%,
  ttft +0.0% — the track's first verified submission.

## Measured-and-closed levers (do not resubmit without new evidence)

Decode profile on the M4 Pro (frontier, per-24-layer): attention 0.052 ms
(qkv 0.0105 + fused-QK 0.0126 + rotating-KV update 0.0053 + SDPA 0.0068 +
o_proj 0.0072), fused router gate 0.012, switch (up_gate gather + swiglu +
down gather) 0.054, fused add+norm ×2 ~0.01, aggregate ~0.004 → 0.14/layer.

- **Custom ternary up+swiglu+down kernel (decode)**: scalar 2-bit decode is
  branch-bound; 0.096 ms vs 0.054 stock (1.8x slower). gather_qmm is already
  ~110 GB/s effective on the 6 MB expert read. CLOSED.
- **Custom ternary kernel (prefill M=4096)**: at fp32 compute floor, equal to
  gather_qmm (Apple's kernel is already optimal). CLOSED.
- **mx.compile of the decode switch**: 1.005x (noise); compile doesn't merge
  matmul-heavy graphs. CLOSED.
- **N-gram M=2 speculation**: M=2 batch costs 1.5x at the growing rotating
  cache while accept ≈ 78% — net loss. CLOSED.
- **bf16 RMSNorm**: 1.002x (noise). CLOSED.
- **Packed QK norm+rope (4 heads/TG, 1.6x isolated)**: 0.9985 end-to-end —
  GPU already saturated; launch count irrelevant in-context. CLOSED.
- **Router reference vs fused**: fused is 2.4x faster; keep fused. CLOSED.
- **KV slice-write**: already in-place rotating buffer (0.0053 ms/layer).
  CLOSED.
- **In-place KV write from QK kernel**: mlx metal_kernel wraps inputs as
  `const device` — writes are compile errors. CLOSED.

## Local numbers (M4 Pro, directional)

Frontier official harness: decode 310.7 tok/s, prefill 1613 tok/s, ttft 0.36s,
ppl 113.97. Trusted runner (same patch) measured 218.7/982/0.572 — the runner
box is ~30% slower; always A/B against the same box.
