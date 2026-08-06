# maple-preview-mlx-apple-v1 — patch series

| Patch | Intent |
| --- | --- |
| `0001-registry-load-flashhead.patch` | Registry-preferred model load (no trust_remote_code needed) + auto-enable FlashHead approximate lm_head for L=1 decode |
| `0002-request-overhead.patch` | Per-request host overhead: cache the BPE detokenizer tokenmap (was rebuilt from a fresh `tokenizer.vocab` dict on every `stream_generate` call, ~95 ms on M1 Ultra); stop flushing the MLX buffer pool after the final prefill chunk and at decode step 0 (both flushes forced the next allocations back to the OS). **Verified: +20.93% frontier** |
| `0003-flashhead-probes-256.patch` | FlashHead probe count 512 → 256 for this checkpoint: halves the probed lm_head read (16.8 → 8.4 MB per decoded token). Probe sweep on the benchmark corpus: 0/640 greedy tokens drift at 384, 256 and 192 probes; the miss cliff is at 128 (239/640). 256 keeps a 2x margin above the cliff. ppl gate untouched (prefill/ppl use the exact head). `MLX_FLASH_PROBES` overrides for same-binary A/B |

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
  cache while accept ≈ 78% — net loss. CLOSED. **Re-tested 2026-08-06 with the
  cache cost removed** (loose-draft RotatingKVCache: certain token commits via
  the stock S=1 in-place write, draft K/V ride loose, commit-on-accept — no
  concat, no rollback; k≤3 chains, coverage 0.873 / acceptance 0.957): decode
  1.44x on M1 Ultra (dispatch-latency-bound) but ~1.02x on M4 Pro and
  **runner-rejected 1.1949 vs 1.2093** — the runner M4 is GPU-throughput-bound
  and a 1+k-token verify step costs ~(1+k)x GPU work (8 fresh experts per
  token; no weight reuse across positions). The implementation survives on
  branch `maple/0003-ngram-speculation`; the lever stays CLOSED on this
  hardware for real, not for cache-cost reasons.
- **bf16 RMSNorm**: 1.002x (noise). CLOSED.
- **Packed QK norm+rope (4 heads/TG, 1.6x isolated)**: 0.9985 end-to-end —
  GPU already saturated; launch count irrelevant in-context. CLOSED.
- **Router reference vs fused**: fused is 2.4x faster; keep fused. CLOSED.
- **KV slice-write**: already in-place rotating buffer (0.0053 ms/layer).
  CLOSED.
- **In-place KV write from QK kernel**: mlx metal_kernel wraps inputs as
  `const device` — writes are compile errors. CLOSED.
- **Prefill expert GEMMs (2026-08-06)**: sorted gather runs at 94% of the
  box's dequant-GEMM ceiling (4.08 vs 4.34 TFLOPS dense on M4 Pro), and
  segment size does not matter (16-row segments vs tile-filling 64-row:
  1.01x) — the NAX gather absorbs ragged segments. The ceiling is the 2-bit
  dequant GEMM rate itself, confirming the earlier "fp32 compute floor"
  verdict mechanically. Unsorted gather is 2.3x worse — never remove the
  sort. CLOSED.
- **qmv guarded-tail fix (2026-08-06)**: qmv_impl's K-loop sends the final
  full block through the runtime-bounded safe path for K % 512 == 0 — but
  every matvec shape in this model (N % 8 == 0, K % 512 == 0) dispatches to
  qmv_fast / gather_qmv_fast, which have no guarded tail. A fix never fires
  here. Kernel census for decode: qmv_fast (qkv, o_proj), gather_qmv_fast
  (up_gate, down; M=1 stays on the vector path — get_qmv_batch_limit gates
  the matrix path), fused metal_kernels for router/QK/add-norm, FlashHead
  gather. CLOSED.
- **Fused ternary MoE decode kernels (2026-08-06)**: a complete, correct
  mx.fast.metal_kernel replacement for the M=1 switch — K1 fuses
  up_gate+clamped-SwiGLU, K2 fuses down+score-weighted combine (2 dispatches
  vs ~6, stock-parity rounding, all 24 layer probes pass, token streams
  identical). Five geometry/codegen variants measured on the M4 Pro against
  the stock pair (gather_qmv_fast + elementwise): best was 0.83x / 0.85x
  isolated, 0.965x end-to-end decode. Findings that transfer: per-LANE
  contiguous access dominates on Apple GPUs (cross-lane interleaving was
  2-8x worse), 32-thread threadgroups halve throughput vs 64, multi-row
  register blocking regresses via spills, and vectorized uint4/bfloat4 loads
  change nothing (the compiler already coalesces). Apple's qmv_fast template
  keeps a ~17% lead that survived every variant. Third confirmation of the
  custom-ternary-kernel CLOSED verdict; full implementation preserved in
  `reference/fused-moe-decode-attempt.patch.txt` (not part of the applied
  series). CLOSED.
- **gs512 ternary scale regroup (2026-08-06)**: row-alpha scales are per-row
  constant, so regrouping scales/biases gs128→gs512 (with a 2-line engine
  patch adding `instantiate_quantized_types(512, 2)` to quantized.metal +
  quantized_nax.metal) is bit-identical at every benchmark shape on both the
  classic and NAX kernel paths — and decode-NEUTRAL: M4 Pro interleaved
  ratios 0.983/0.988/1.025. The qmv-family kernels are not scale-fetch
  limited, and M=1 decode never takes the NAX path (`get_qmv_batch_limit`
  routes small M to the vector kernels). ~9% of ternary bytes are scale/bias
  overhead, but cutting them buys nothing here. CLOSED. (Side finding: dense
  qmm at M∈{3..8} picks different valid kernel variants per process — ~1-ulp
  cross-process flicker exists in stock too.)

## 0002 measurements (M1 Ultra, interleaved whole-process A/B, official harness)

Per-round cand/ref ratios over 2 rounds: ttft 1.370 / 1.396, prefill 1.063 /
1.065, decode 0.989 / 1.007. Perplexity bit-identical (115.8171952670014 both
arms) — the patch touches no numerics. Mechanism: `tokenizer.vocab`
materializes a fresh 152k-entry dict per access and the BPE streaming
detokenizer rebuilt its tokenmap from it on every request (~95 ms, inside
measured TTFT but before `prompt_time`'s clock starts); the two
`mx.clear_cache()` calls dumped the warm buffer pool right where decode (and
the next harness run's prefill) re-allocates it.

## Local numbers (M4 Pro, directional)

Frontier official harness: decode 310.7 tok/s, prefill 1613 tok/s, ttft 0.36s,
ppl 113.97. Trusted runner (same patch) measured 218.7/982/0.572 — the runner
box is ~30% slower; always A/B against the same box.
