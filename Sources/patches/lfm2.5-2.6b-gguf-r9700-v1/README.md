# lfm2.5-2.6b-gguf-r9700-v1 — patch series

## Landed / pending

All three landed wins are verified on the trusted R9700 runner. The scores
below are the runner's, not local ones.

| Patch | Status | Notes |
| --- | --- | --- |
| `0001-mmvq-narrow-starved-blocks.patch` | **won** — score 1.0999 (+9.99%), decode 1.162x / 209.7 tok/s | Drop idle warps on starved K=2048 q4_K shapes |
| `0002-mmvq-dedupe-q8-1-requant.patch` | **won** — score 1.1067 (+10.67%), decode 1.170x / 211.3 tok/s | Per-graph q8_1 activation cache |
| `0003-mmvq-grouped-launch.patch` | **won** — score 1.1103 (+11.03%), decode 1.176x / 212.2 tok/s | Group same-activation Q/K/V matvecs into one launch |
| `0004-rms-norm-grouped-launch.patch` | **won** (score 1.1117, decode 1.180x / 212.8 tok/s) | Group independent rms_norm+mul launches |
| `0005-rms-norm-fold-residual-and-q8-1-quant.patch` | **won** (score 1.1519, decode 1.228x / 222.1 tok/s) | Fold residual add + q8_1 quant into rms_norm |
| `0006-glu-fold-q8-1-quantize.patch` | candidate | Fold q8_1 quantize of GLU into GLU kernel |
| `0007-mmvf-single-warp-decode-block.patch` | candidate | mmvf single-warp decode + batched k-loads |
| `0008-ssm-conv-fuse-mul.patch` | candidate | Fuse the shortconv c-gate MUL into ssm_conv |
| `0009-mmvq-narrow-q6k-output-head.patch` | candidate | Narrow the q6_K output-head matvec |
| `0010-shortconv-decode-chain-fold.patch` + `0011-rs-state-identity-view.patch` | **won** — score **1.2124 (+21.24%)**, decode 1.3307x / **238.3 tok/s**, prefill 1.0214, ttft 1.0179 | The whole shortconv decode chain in ONE kernel, plus the conv-window gather as a cache view. Took the track from Cybara's 1.1659. |

`0001` measured **+17.177%** decode locally (180.325 -> 211.439 tok/s) and
landed at **+9.99%** overall on the trusted runner. A clean local figure
routinely lands a point or two lower once the runner's paired protocol and
the prefill term are applied.

It removes idle mmvq warps. On the RDNA4 table `calc_nwarps()` returns 8 for
q4_K at `ncols_dst=1`, which wants 16 k-blocks, but every K=2048 tensor in this
model has only 8 - so half of every block fell through to the reduction without
entering the k-loop while still costing occupancy. The q6_K output head is the
one matvec that was already fully fed, and it was also the only one near peak
bandwidth (565 GB/s vs 377 for the ffn up/gate pair). Halving `nwarps` on the
starved shapes took `mul_mat_vec_q` from 4101.7 to 3062.7 us/token.

Note the second gate in that patch before you extend it: narrowing trades total
parallelism for occupancy, so it must not be applied when too few blocks remain
to fill the device. The 512-row k/v projections regress 2.8x if you narrow them,
which is worth 2.9 of the 17.2 points.

`0002` ports the Laguna q8_1 src1 cache. Dense hybrid LFM2.5 still reuses the
same activation across attention Q/K/V and FFN gate/up projections, so stock
re-runs `quantize_q8_1` once per consumer. The cache is bit-exact by
construction (parameter-tuple + root-tensor identity + clear-on-eval).

`0003` ports the Laguna grouped mmvq launch: same-activation plain matvecs
(Q/K/V class) become one kernel with concatenated row ranges. Warp launch
count mirrors 0001 (trim starved K, keep full warps when the group's total
rows would not fill the device). Compile-time nwarps stays at the table value
and trimmed slots pad +0, so results stay bit-identical.

`0004` ports the Laguna grouped rms_norm launch: independent rms_norm+mul pairs
(Q/K-norm class and similar) share one kernel with concatenated row ranges.
Per-row work matches stock so results are bit-identical. Do **not** blindly
port Laguna MoE-only patches after this.

## Measured-neutral: the remaining laguna family ports (do not spend a slot)

The four laguna-series patches this series does not carry — rope-grouped
launch, set-rows-grouped launch, mul-fold-q8-1-quantize, and the pool-teardown
fix — were ported (with seam repairs) onto 0001-0009 and measured on the
runner box, 2026-08-06: 5-round interleaved whole-process A/B vs the 0001-0009
build read decode median ratio ~1.001 and neutral prefill; ppl bit-identical
(22.5182 = stock). LFM2.5's graph is mostly conv layers: there are too few
rope/set_rows/foldable-mul sites for the launch savings to clear noise. The
ports are correct but not worth a submission on this model.

## This track is not the Laguna AMD track

LFM2.5 2.6B is a **dense hybrid**, not a mixture-of-experts. Every lever that
carried `laguna-xs-2.1-gguf-r9700-v1` from 0 to +37% was an expert-dispatch
change — `mul_mat_id` batching, shared-expert channel folding, MoE router
selection. **None of them exist here.** Do not start by porting them.

Stock on the R9700 measures roughly **180 tok/s decode** and **7369 tok/s
prefill** at 1.55 GiB of weights. At that size and speed, decode is dominated
by per-launch overhead and kernel-launch ramp rather than weight bandwidth —
a different regime from every other track on this platform. Profile before
you form a hypothesis.

## Keep patches portable across RDNA generations

RDNA2 (gfx1030), RDNA3 (gfx110x), RDNA3.5 (gfx115x) and RDNA4 (gfx1201) share
these kernels. The trusted runner ranks you on RDNA4, but a change that wins
on gfx1201 by regressing the others is not one worth merging.

- Guard on architecture or wave size at runtime, or use existing feature
  macros — don't hardcode `gfx1201`.
- If you have access to another RDNA part, sanity-check there and report what
  you saw. A note that a win is arch-neutral is worth including in the PR.
- RDNA3.5 in particular is an iGPU with shared memory bandwidth; a change that
  assumes dedicated VRAM behaviour may not transfer.

## Before you trust a number

Use the platform's measurement discipline: interleave whole process launches,
rotate arm order, take the median of per-round ratios, and always run a
same-binary no-op control (an env toggle) so you know your own floor. Check
dispatch counts per kernel group, not aggregate kernel time. On a model this
small the per-launch component is large, so an effect can hide inside boot
variation — see the `in-process-paired-ab` finding for a harness that runs both
arms inside one process when you need finer resolution.

Accuracy gate: perplexity within 0.5% of stock, measured by `llama-perplexity`
on the fixed corpus. Build that target as well as `llama-server`.


## Measured-neutral: ngram self-speculation (do not spend a slot)

The maple 0022 lever ported and A/B'd 2026-08-07: decode 223.8->208.2 and
222.5->236.6 across two seeds (-7%/+6%, net neutral with high prompt
variance; ppl untouched at 22.5182). LFM's continuations accept
inconsistently; the variance makes it unshippable against a +16.59%
frontier. Maple-specific economics (its degenerate 2-bit loops accept
~0.8+) do not transfer.

## Round 1 (2026-08-09): the shortconv decode chain, 420 -> 347 dispatches/token

**RANKED: verified at 1.2124 (+21.24%), decode 238.3 tok/s** against the previous
frontier of 1.1659 / 225.0 tok/s. Local bench predicted +5.38% decode; the runner
read +6.6% (1.2488 -> 1.3307), i.e. the ranked path did slightly BETTER than the
bench, which matches the llama-server census note below.

### The census this round is built on

`rocprofv3 --kernel-trace`, `llama-bench -p 0 -n 34 -r 1`, sorted by
`Dispatch_Id`. **420.5 dispatches/token, 3718 us of kernel in a ~4400 us
token** (227 tok/s), i.e. kernel is 84% of wall and the inter-dispatch gap is
about 1.6 us. `mul_mat_vec_q` alone is 117 launches and 2997 us - **1.66 GB of
weights in 3.0 ms is 557 GB/s against a ~640 GB/s part, so the matvecs are 87%
of the practical ceiling and are NOT the lever.** The README's older claim that
"decode is dominated by per-launch overhead rather than weight bandwidth" is
half right: the *matvecs* are bandwidth-bound and closed; the attackable pool is
the ~700 us of small kernels plus ~700 us of gap, about a third of the token.

| n/tok | us/tok | each | kernel |
|---|---|---|---|
| 117 | 2997 | 25.6 | `mul_mat_vec_q` (2048/10752/6144/512-block shapes) |
| 39 | 114 | 2.9 | `rms_norm_pre_add_f32<1024,1>` |
| 60 | 95 | 1.6 | `quantize_q8_1` |
| 12 | 79 | 6.6 | `mul_mat_vec_q_grouped` (attention q/k/v) |
| 22 | 73 | 3.3 | `rms_norm_pre_add_f32<1024,2>` |
| **22** | **44** | 2.0 | **`k_bin_bcast` (the shortconv b*x MUL)** |
| **22** | **42** | 1.9 | **`cpy_scalar` (conv-state writeback)** |
| **22** | **41** | 1.8 | **`ssm_conv_f32`** |
| **24** | **39** | 1.6 | **`k_get_rows_float` (conv-window gather)** |
| **22** | **32** | 1.4 | **`concat_cont`** |
| 8 each | ~120 | | flash_attn tile/combine, rope x2, rms_norm_grouped, set_rows |

The five bolded rows are one chain, repeated in each of the 22 shortconv layers:
**112 launches, 198 us of kernel, 27% of every dispatch in the token**, and all
of it is staging - copy the window out of the cache, copy it into a concat
buffer, copy the tail back to the cache, then convolve the copy.

### The cost constant for this model

The first (14-layer) version of 0010 removed exactly 42 dispatches/token and
measured +2.15%: **about 2.2 us per removed small dispatch**, kernel time plus
gap. Price candidates on this track with that number; the full round removed 77
and landed at +5.4%, which tracks.

### 0010 + 0011: the chain in one kernel (+5.38% decode, BIT-IDENTICAL)

0011 turns the gather into a view of the cache; 0010 turns the remaining four
nodes into a single per-channel kernel. Details and the measurement protocol are
in the patch headers. Census after both: **347.5 dispatches/token**,
`concat_cont`/`cpy_scalar`/`ssm_conv_f32`/`k_bin_bcast` all ~0, `k_get_rows_float`
24 -> 2, `shortconv_fold_f32<3>` 21/token.

Perplexity is **bit-identical on both shapes** (22.4212 decode-path, 22.5182
gate-shape, 3 loads per arm against a parent-commit control) and server greedy
is **6/6 byte-identical** including a ~1000-token prompt.

### The trap that cost this round a debugging cycle - read this before folding anything else here

A fold that issues its stores EARLIER than the nodes it replaces needs a
hoisting-legality check, and this graph is built to punish you for skipping it.
The shortconv fold has to be *detected and launched at the b*x MUL*, because the
conv window for layer L+1 is gathered into the SAME allocator block ~9 nodes
later - launch at the CONCAT or the SSM_CONV and you read a window that has
already been overwritten. But launching there hoists the layer output and the
conv-state writeback ~20 nodes early, onto blocks that nodes in between still
read and write.

The failure mode is what makes it expensive: **llama-bench reported a clean
+2.15% with 5/5 rounds separated and never noticed**, because llama-bench does
not look at what it decoded. Perplexity read 1.7e5 against 22.4. The detector now
requires both destinations to be disjoint from every input and output of every
node remaining in the window; 21 of 22 layers still fold.

Two portable rules for this track:

- **run the decode-path perplexity before you believe any A/B**, not after;
- when a fold looks numerically wrong, bisect by *which stores the fold owns*
  (have it materialize the intermediate and let stock recompute downstream)
  before you audit the arithmetic. That bisect proved the accumulation was
  right in one build and pointed straight at the hoist.

### Measured this round, not worth a slot on its own

`0011` alone reads only +1.11% on llama-bench even though it removes 22
launches, because deleting the gather nodes shifts the allocator layout enough
to block the grouped-mmvq and grouped-rms_norm detectors in the 8 attention
layers (`mul_mat_vec_q_grouped` 12 -> 4, `rms_norm_f32_grouped` 8 -> 0).
**Under `llama-server` both groupings survive** - a rocprofv3 census of the
serving path shows `mul_mat_vec_q_grouped` 11.8/token and
`rms_norm_f32_grouped` 7.9/token alongside the fold - so this is a llama-bench
artifact, not a real loss. Recovering it on the bench path (~12 dispatches,
~0.6%) is open but low value.

### What is left

Post-round census is 347.5 dispatches/token, ~4160 us. The remaining non-matvec
pool: `quantize_q8_1` 60, `rms_norm_pre_add` 61, and the 8 attention layers'
rope/set_rows/flash-attn glue. The matvec side is closed on bandwidth grounds
(above). The next largest coherent cut is the 60 `quantize_q8_1` launches.
