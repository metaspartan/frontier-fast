# lfm2.5-2.6b-gguf-gb10cuda-v1 — patch series

Applied in order against pinned llama.cpp **b10237**.

| # | Patch | Measured on this track |
| --- | --- | --- |
| 0001–0011 | laguna-xs gb10cuda engine family, unchanged | **+0.9% decode, +0.8% prefill** (see "the 11-patch family" below) |
| 0012 | `cuda-sm121-wide-q4k-mmvq-vecdot` | **+7.76% decode** (5/5 interleaved rounds disjoint; prefill neutral) |
| 0013 | `cuda-rms-norm-register-cached-row` | **+0.65% prefill** (5/5 disjoint), decode +0.19% (overlapping); **bit-exact** |

## 0012: the sm_121 wide Q4_K mmvq vec_dot (+7.76% decode)

The largest lever found on this track, and the first one that is not a
launch-count play. Read the profile section below before proposing anything
else here: it is what made this patch obvious and it closes several families.

Stock `vec_dot_q4_K_q8_1` runs at `vdr = 2`: every lane fetches its weights
as **two separate 4-byte loads**. The patch adds an `MMVQ_PARAMETERS_SM121`
table whose only entries are `Q4_K vdr = 4` and `Q4_K nwarps = 2`, plus the
wide `vec_dot_q4_K_q8_1` that `vdr = 4` needs. One call now covers the column
pair `(iqs, iqs+2)`: the pair shares `bq8_offset`, the unpacked scale/min pair
and `d8`, so that work happens once instead of twice, and each weight fetch
becomes an 8-byte `int2` load.

**The alignment argument, which is where a blind port of the R9700 0028 patch
goes wrong.** That patch reads its 8 bytes with `__builtin_memcpy` from a
1-byte-aligned `uint8_t *`, which gfx12 handles in hardware; nvcc has to
honour the declared alignment and would lower it to byte loads. Cast to
`const int2 *` instead — valid here because `block_q4_K` is 144 bytes with
`qs` at offset 16 and a row is a whole number of blocks, so `qs` is 16-byte
aligned and `q4 + i0` (`i0` in `{0, 2}`) is 8-byte aligned. It is **not**
valid for the q8_1 activation (`block_q8_1` is 36 bytes, so `qs` is only
4-byte aligned) and **not** valid for Q6_K (210-byte blocks, 2-byte aligned
`ql`/`qh`). Those two keep their narrow accessors.

**nwarps 4 → 2 is not a tuning choice, it is coverage.** `blocks_per_iter`
is `vdr*nwarps*warp_size/qi`, so `vdr` 2→4 with `nwarps` 4→2 leaves it at 8:
the same single k-loop pass over the 8 blocks of a 2048-column row that stock
makes. The two must ship together — **`nwarps = 2` on its own is a 4.4%
regression** (measured, see the dead-lever table). Every other type and every
other `ncols_dst` mirrors GENERIC exactly.

### Measured (runner box, parked vllm container, series-only control build)

- **cross-tree no-op floor first**: the control (`series`) and a second tree
  built from byte-identical sources measured 112.96 vs 112.92 tg128 (ratio
  **0.9996**), so a build-to-build A/B on this box is trustworthy to ~0.05%.
- toggle-free A/B, 5 interleaved whole-process rounds, `llama-bench -p 512
  -n 128 -r 3`:

| round | control tg128 | 0012 tg128 | control pp512 | 0012 pp512 |
| --- | --- | --- | --- | --- |
| 1 | 112.39 | **121.11** | 8056.09 | 8096.83 |
| 2 | 112.46 | **120.95** | 8002.48 | 8104.12 |
| 3 | 112.29 | **121.14** | 8057.62 | 8108.67 |
| 4 | 112.46 | **120.97** | 8015.28 | 7969.82 |
| 5 | 112.31 | **121.14** | 8039.43 | 7967.10 |

  decode median ratio **1.0776**, 5/5 arms disjoint; prefill overlapping in
  both directions, i.e. neutral — expected, since prefill takes MMQ and never
  enters this kernel.
- **kernel census diff** (nsys `cuda_gpu_trace`, same command, control vs
  0012) — the firing proof, since both arms run identically named kernels:

| kernel | grid | control | 0012 |
| --- | --- | --- | --- |
| `mul_mat_vec_q<Q4_K,1,fusion,->` | 10752 | 270 × **32x4** | 270 × **32x2** |
| `mul_mat_vec_q<Q4_K,1,fusion,->` | 2048 | 216 × 32x4 | 216 × 32x2 |
| `mul_mat_vec_q<Q4_K,1,-,->` | 6144 | 198 × 32x4 | 198 × 32x2 |
| `mul_mat_vec_q<Q4_K,1,-,->` | 2048 | 270 × 32x4 | 270 × 32x2 |
| `mul_mat_vec_q<Q4_K,1,-,->` | 512 | 108 × 32x4 | 108 × 32x2 |

  Every Q4_K launch moves to the 2-warp block, counts and grids identical;
  every Q6_K row is unchanged. Nothing else in the census moves.
- **perplexity**: official gate `-c 512 --chunks 8` on the runner corpus is
  **bit-identical, 22.7466 both arms** — and that is not a pass, it is a
  blind spot: at the gate's shapes the consumer takes MMQ and this kernel
  never runs. Decode-path perplexity (`-b 512 -ub 1 --chunks 8`, so mmvq runs
  at every position) is **22.5380 → 22.6048, +0.296%**, inside the ±0.5% band.
  Report both; do not quote the gate number alone.
- **server greedy** (`llama-server`, 6 prompts, `temperature 0`, non-emptiness
  asserted — 6/6 completions, 5318 and 5633 bytes): **1/6 byte-exact, 5/6
  diverge mid-completion**. Reassociation class, the same as the R9700
  0024/0026/0028 wide/multi-row rewrites: the wide call moves an addition out
  of the warp reduction tree and into a lane, which cannot be undone by
  mirroring the accumulator's declared shape (the 0038 trick) because the two
  columns genuinely live in different lanes at stock `vdr`. Perplexity is the
  arbiter and it is in band.
- the full 12-patch series applies clean to pristine `b10237` and the
  resulting tree is `diff -r` identical to the measured tree.

Projection: decode `1.0091 × 1.0776 = 1.0874`, prefill `1.0083`, ttft
unchanged → score **~1.056–1.064** against the **1.0135** bank.

## 0013: the folded rms_norm keeps its row in registers (+0.65% prefill, bit-exact)

`rms_norm_pre_add_f32` walks its row twice — once to build the sum of squares
after the folded residual add, once to scale/multiply/quantize — and the second
walk re-reads exactly the floats the first one computed. When `ncols` is an
exact multiple of `block_size` and the per-thread column count is <= 8, both
loops become `#pragma unroll` loops over that count and the values stay in
registers. The exact-multiple requirement is what keeps every lane of every
warp live on every step, so the warp reductions in the quantize epilogue see
the participation they see today; any other shape takes the stock loops.

Bit-exact by construction, and measured so: gate **22.7466** and decode-path
**22.6048**, matching the 12-patch control digit for digit.

- **pp512: control max 8046.86 vs patch min 8054.78 — 5/5 arms disjoint,
  median ratio 1.0065.** At prefill the kernel is 4.4% of the pass and the
  re-read is of a genuinely large tensor, so this is where it pays.
- tg128 120.76 vs 120.99 median, **+0.19% with arms overlapping by 0.02** —
  neutral-to-slightly-positive, not a decode win. The decode-shape grid is one
  block for one row, so the saved read was already an L2 hit; what is left is
  one dependent round trip out of a ~3.6 us kernel, 61 times a token.

## The decode profile that produced 0012 — read this first

`nsys --cuda-graph-trace=node`, `llama-bench -p 0 -n 34 -r 1`, on the
11-patch build with the vllm container parked. Per token (30 layers: 22
shortconv + 8 attention, `n_embd` 2048, `n_ff` 10752, `n_vocab` 128000):

| what | n/token | share of token | achieved on weights |
| --- | --- | --- | --- |
| ffn gate+up, fused Q4_K 10752×2048 ×2 | 30 | **42%** | ~199 GB/s |
| ffn down, Q6_K ×14 + Q4_K ×16 | 30 | 29% | ~175 GB/s |
| output head, Q6_K 128000×2048 (215 MB) | 1 | 12% | ~199 GB/s |
| conv in_proj, Q4_K 6144×2048 | 22 | 10% | ~181 GB/s |
| conv out_proj + attn o, Q4_K 2048×2048 | 30 | 5% | ~155 GB/s |
| attn k/v, 512×2048 | 16 | 1% | small, latency-bound |
| **all mul_mat_vec_q** | **137** | **91%** | |
| norms, quantize, ssm_conv, concat, rope, flash-attn, copies | ~250 | 9% | |

**Decode on this track is 91% quantized matvec.** That single number closes
the entire launch-removal family here (LFM's graph has ~250 non-matvec
dispatches worth 9% in total, none of them individually above 2.5%), and it
is why every launch-class lever measured on this box has come back at or
below +1%.

### The ceiling is measured, not assumed

A standalone CUDA probe (`~/bwprobe.cu` on the box) reads a 2 GiB buffer in
several geometries:

| pattern | GB/s |
| --- | --- |
| one block per 1152-byte row, 128 threads, `float4` (**the mmvq geometry**) | **251.6** |
| one block per 2304-byte row, 128 threads, `float4` | 247.6 |
| grid-stride `float4`, large grid | 242.2 |
| grid-stride `float` (4-byte loads) | 227.6 |

So GB10's achievable DRAM read rate for exactly the shape mmvq uses is
**~251 GB/s**, the block-per-row geometry is the *best* pattern available
(not a handicap), and 4-byte loads cost ~6% against 16-byte loads even in the
easy grid-stride case. Against that, the stock Q4_K matvec was running its
weights at ~205 GB/s. **The gap was issue-side load width, and 0012 collects
most of it.** Anyone profiling here again: `ncu` is installed at
`/usr/local/cuda-13.0/bin/ncu` but returns `ERR_NVGPUCTRPERM` — the counters
are admin-restricted, so use the standalone probe plus census diffs instead.

## Dead levers — measured on this track, do not re-buy

| lever | result |
| --- | --- |
| sm_121 mmvq table ported from the qwen3.6 gb10 series | **-4.4% decode** |
| `nwarps = 2` for K-quants at `ncols_dst == 1` (alone, stock vdr) | **-4.4% decode** |
| `nwarps = 8` for K-quants at `ncols_dst == 1` | +0.22%, and it moves decode-path ppl +0.17% |
| `rows_per_block = 2` at `ncols_dst == 1` | **-3.1% decode** |
| `rows_per_block = 4` at `ncols_dst == 1` | **-5.2% decode** |
| `GGML_CUDA_GRAPH_OPT=1` (multi-stream graph optimizer) | neutral |
| `GGML_CUDA_ENABLE_MMVQ_GROUP=1` (grouped mmvq on) | **-2.5%**, keep 0010's default |
| `GGML_CUDA_PDL=0` | -3%, i.e. PDL is already on and already paying |
| Q6_K wide vec_dot (`vdr` 1 → 2, same sharing as 0012) | +0.24%, arms overlapping = neutral |
| Q4_K `vdr = 8` (16-byte int4 loads, single-warp block) | 0.9987 |
| ssm-conv + c-gate mul fold (R9700 port) | 0.996 |
| ngram self-speculation | neutral, and off-board anyway |

Notes that cost real runner time to learn:

- **The two negative wide-vec_dot results pin the mechanism, and they matter
  more than the win does.** The Q6_K pair form does the *same* index/scale/q8/d8
  sharing as 0012 and halves the call count, and it buys nothing — because
  `block_q6_K` is 210 bytes, so `ql`/`qh` are 2-byte aligned and must go through
  `get_int_b2`, i.e. pairs of 16-bit loads, and the load cannot widen. And
  `vdr = 8` widens Q4_K further to 16-byte `int4` loads *and* collapses the block
  to a single warp with no LDS reduction at all — and gives back 0.13%. So on
  GB10 the payoff is **load width and nothing else**, and it **saturates at 8
  bytes per lane**. Target only quant types whose block size permits an `int2`
  cast; do not chase call-count sharing, redundant-work removal, or the
  reduction tail.
- **The qwen3.6 gb10 sm_121 mmvq table does not transfer to a dense model.**
  There it pays through `rows_per_block` under `small_k`; LFM2.5 is dense with
  `n_embd` 2048, so Q4_K has `blocks_per_row_x = 8`, `small_k` never fires,
  `rows_per_block` stays 1, and the table's only live effect is `nwarps`
  4 → 2 — which is the 4.4% regression above.
- **`rows_per_block` never moves perplexity at all** (gate 22.7466 and
  decode-path 22.5380 identical at rpb 1, 2 and 4). It is the one geometry
  knob that keeps bit-identity — it just has no upside on this hardware.
- **CUDA graphs are engaged** on this track (33 `cudaGraphLaunch` for 35
  decode tokens); there is no lever there.
- **Park the vllm container before measuring.** LFM decode reads **112 tok/s**
  parked against the 74–75 recorded in this README's original 11-patch
  section, so every pre-2026-08-08 number on this track was taken under
  memory contention and is not comparable to anything above.
- **Never run `llama-bench` while a compile is running on this box** — it
  costs every arm a uniform ~8% and produced one discarded round tonight.

## The 11-patch family (0001–0011)

The laguna-xs gb10cuda engine family, unchanged: q8_1 requant dedupe, grouped
launches with grouped-mmvq off by default, norm/rope/set-rows groups,
quantize folds, mmvf batched k-loads. Originally measured on the runner box
at decode 1.0091 / prefill 1.0083 with ppl bit-identical, under the
memory-contended conditions noted above.

## Open, in the order I would try them

1. **Cross-port 0012 to the other three gb10cuda tracks.** laguna-xs, laguna-s
   and qwen3.6 are all `Q4_K_M` on this box, so the wide vec_dot applies to
   their (ids-path) expert matvecs too, and those tracks are far more
   matvec-dominated than this one. Note the coverage arithmetic changes with
   `n_embd`: `blocks_per_iter = vdr*nwarps*warp_size/qi`, so a 4096-column row
   has 16 k-blocks and takes `vdr = 4` with `nwarps` **unchanged** at 4, while a
   2048-column row needs `nwarps = 2` as here. Those tracks already carry an
   `MMVQ_PARAMETERS_SM121` table, so only the `get_vdr_mmvq(type, table_id)`
   overload and the wide `vec_dot_q4_K_q8_1` need adding — and pin their
   `should_use_small_k` predicate to the *stock* vdr so the `rows_per_block`
   decision does not move at the same time.
2. **Prefill**: `unary_gated_op_kernel` (silu) is **12.9% of pp512** and
   `quantize_mmq_q8_1` another **6.9%** — the glu writes f32 and a separate
   pass reads it back to build the MMQ-layout q8_1. 0008 already folds the
   *mmvq*-layout quantize into the glu; extending
   `ggml_cuda_norm_quant_register` to the MMQ layout would delete ~44 MB of
   round-trip per layer. Worth ~6–8% prefill.
3. The conv-block bookkeeping (`k_get_rows_float` 86 µs, `k_bin_bcast` 82 µs,
   `concat_cont` 44 µs, `cpy_scalar` 37 µs per token) — the R9700 0030
   recurrent-state identity view is the model, but the whole pool is ~2.5%.
