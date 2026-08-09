# laguna-xs-2.1-gguf-r9700-v1 — patch series

The platform's most-worked track. Nineteen patches applied to llama.cpp
`b10237`, built `GGML_HIP=ON, AMDGPU_TARGETS=gfx1201, Release`. Frontier
**+37.13%** (154.87 tok/s decode against a 95.43 tok/s baseline); the current
top entry is `MoE shared expert gate/up as ninth channel`.

Numbers here are from the trusted runner and the findings API as of
2026-08-04. `curl -s "https://frontier.fast/api/findings?track=laguna-xs-2.1-gguf-r9700-v1"`
is authoritative and has 20+ entries — read it before designing anything.

## The one thing to know: this box is launch-bound

1331 dispatches per token, and kernel time is only ~74% of decode wall. That
means **removing a launch pays 2–3× its kernel-time share**, because you
remove its inter-dispatch gap too.

The q8_1 re-quantization dedupe is the proof: it was projected at +2.7% from
its 4% share of decode and landed at **+8.69%** (decode 1.1397, 109.08
tok/s), cutting quantize dispatches from 13,134 to 6,600. Launch-count
reduction is the class that pays here.

A byte census of decode puts attn_q + attn_output at 40%, MoE experts 33.5%,
output head 9.5%, F32 router 4.6% — and the four big matvecs already run at
500–600 GB/s against a 640 GB/s ceiling, i.e. 78–94%. **Kernel throughput
tuning on them is exhausted** (six failed attempts on record). The remaining
pool is latency-bound small operations (~21% of wall) and inter-dispatch gaps
(~20% of wall). Target concurrency and dispatch structure, not the throughput
of kernels already near the memory ceiling.

## Landed wins, and what each closed

| Patches | Win | Note |
| --- | --- | --- |
| 0001-0002 | mmvq launch trim, +4.57% | Released the `MMVQ_MIN_LAUNCHED_WARPS` clamp (6→1). **The trim lever is fully spent** — q6_K K=2048 and the multi-pass case have zero idle warps by construction. Do not retry trim variants. |
| 0003 | q8_1 re-quantization dedupe, +8.69% | Per-context cache keyed on (view-chain root, data ptr, stream, full shape 8-tuple), cleared on entry to every graph compute. **Exhausted by construction** — the remaining 5 quantizes/layer are 5 genuinely distinct activations. |
| 0004-0011 | grouped launches and fold family | Q/K, rms_norm, rope, set_rows grouping; residual-add and q8_1-quantize folds. |
| 0015 | topk_moe sorted-list selection, +33.2% | Stock runs one full argmax per selected expert: top-8 is 40 *dependent* cross-lane shuffles. Lane-local sort plus a 5-round bitonic merge drops dependent depth to 5, kernel 4.572 → 2.567 us. Bit-identical because ordering by a 64-bit key (order-preserving float image, complemented expert index) reproduces the stock tie rule exactly. |
| 0016 | `mul_mat_id` Q4_K path for RDNA4 | Dispatch-path change for the dominant MoE shape. |
| 0017-0018 | MoE shared expert as a ninth channel | The shared-expert down projection is **latency**-bound, not bandwidth-bound: standalone it moves 0.86 MB in 8.43 us (102 GB/s) from only 2048 workgroups, while the routed projection doing identical per-row work at 8× the channel count sustains 601 GB/s. Bandwidth here scales with wave count, so the fix is more concurrent work in the same launch. |
| 0019 | release the q8_1 cache before pool teardown | Fixes an abort, not a speed win — see below. |
| 0020 | attention gate projection joins the Q/K grouped launch | **won** — score **1.3872 (+38.72%)**, decode 1.6474x / **156.8 tok/s**. A one-line graph-order change, bit-identical. Took the track from Cybara's 1.3731. Open lever 1 below is now partly closed. |

## Dead ends — do not spend a slot re-deriving these

- **`rows_per_cuda_block` widening (upstream `small_k`) is closed
  completely**, on two independent grounds. (1) Post-trim it is a net
  regression: +1.78% total kernel time, with fused-gate Q6_K paths regressing
  up to +24%. The projected wins were measured *pre-trim* and the merged trim
  already banked them. (2) Bit-exactness is unreachable **as a class**: stock
  is compiled `-ffp-contract=fast` and its bit pattern is the reference, so
  any contraction change — `#pragma unroll 1`, explicit `fmaf` ordering,
  TU-scoped `-ffp-contract=off` — moves *existing* paths off that reference.
  Proven with 3-seed hash matrices; the non-fused-only sub-variant scored
  0/3 seeds byte-identical. Do not retry any variant.
- **gfx1201 FlashAttention pp regression (upstream #26220).** At this track's
  512-token window FA is only 5.26% of prefill, so eliminating it entirely is
  worth ~+1.1% score. Prefill lives in `mul_mat_q` (Q4_K 53.1% + Q6_K 17.6%).
  Not worth a runner slot at this window size.
- **`GGML_MMVQ_UNTRIM_BLOCKS` threshold.** 5.371 ms/token at untrim=0 versus
  5.422 at untrim=4096 — neutral, inside noise. Patch 0013's threshold does
  not matter on this workload.
- **Narrowing the `rms_norm_pre_add` decode block.** Monotonic regression:
  tg128 is 149.44 at 1024 threads, 148.81 at 512, 145.15 at 256, 138.37 at
  128, 126.04 at 64, 112.56 at 32. These kernels are memory-latency bound on
  a single workgroup and more threads is what hides it. The remaining
  ~113 us/token needs the two serialized passes attacked (register-cache pass
  1 so pass 2 does not reload), not the block width.
- **Widening the GLU+quantize fold's consumer scan.** A hypothesis recorded
  here earlier — that the fold inspects only the immediately-following node —
  was wrong. `ggml_cuda_quant_register_for_consumer` (ggml-cuda.cu:4184)
  already scans 64 nodes forward. The real constraint is that `try_fuse`
  fires at the *gate* node and consumes the GLU before the loop reaches it,
  so whether the quantize fold could apply is never consulted. The
  intervention point is `try_fuse`'s ordering at the gate node.

## Open levers, in order

1. **Merge Q/K/V matvecs into one grouped launch; fold `k_bin_bcast`.**
   Post-dedupe, `mul_mat_vec_q` is 69.2% of decode across 13,134 dispatches
   at 12.24 us. Q/K/V are three dispatches over the *same* quantized input
   with different weights — groupable into one launch with byte-identical
   per-row work. A launch-count play on a launch-bound box, which is the
   class that pays here. Needs graph-level fusion detection; budget a full
   session. HIP graphs are already on, so that is not a lever.
2. **A graph arrangement that keeps the shared-expert fold's +3.62%.** The
   fold itself is correct and worth +3.62% same-binary over 5 interleaved
   rounds, PPL bit-identical at 5.2611, -39 dispatches/token, -4.86% kernel
   time. But **both** obvious arrangements net negative: (A) hoisting the
   triple before the router costs the shared gate/up their grouped launch
   (net -1.25%); (B) placing it just before the routed down restores the
   grouped launch (4.93 us vs 11.49 fused) and the down pairing still fires,
   but pinning the routed chain needs an early `ggml_build_forward_expand`
   inside `build_moe_ffn`, which reorders the attention subgraph and costs
   patch 0004's Q/K grouped launch. A third arrangement is the open problem.

## Known defect in the merged series

The series aborts at CUDA context teardown with `GGML_ASSERT(pool_size == 0)`
at `ggml-cuda.cu:438`: patch 0003's q8_1 cache holds pool allocations past
the pool destructor. Patch 0019 addresses it, but if you benchmark an older
point in the series with `llama-bench` you will hit an abort that is **ours,
not your patch**. The trusted runner never sees it because `llama-server`
does not free the context. Use one `llama-bench` invocation per configuration.

## Measurement protocol on this track

Interleave whole process launches, rotate arm order, take the median of
per-round ratios, and always run a same-binary no-op control via an env
toggle so you know your own floor. Check dispatch counts per kernel group,
not aggregate kernel time — sizing a lever from an aggregate bucket is how
the GB10 twin invented headroom that was not there.

For bit-exactness work, byte-compare with the cheap `-ngl 18` protocol
(stock-vs-stock control hashes first) before spending time on perf tuning.

One known harness effect: the runner reports candidate prefill 0.988–0.991
across separate runs while local A/B says neutral for decode-only patches.
The candidate phase always benchmarks second on a warmer GPU, so expect
roughly a 1% prefill headwind here until the worker interleaves its pairs.


## 0020: n-gram self-speculation as the engine default (2026-08-07)

The maple 0022 lever: the pinned engine ships model-free `ngram-simple`
drafting in `common/speculative`, dormant because the harness launches
llama-server without speculative flags. One hunk flips the engine default
(lookup n=3, m=16; env `GGML_SPEC_NGRAM=0` restores stock, N/M/HITS tune).

Measured on the runner box (fresh-server completions, runner-style
varied-prose prompts, 6 seeds, alternating arms, then re-validated on the
exact 19-patch frontier + this patch): decode 155.1-155.8 baseline →
194.2 / 133.9 / 179.9 / 247.7 / 157.0 / 152.5 tok/s per seed — **mean
ratio 1.142, spread 0.86–1.59** (prompt-dependent acceptance). PPL exactly
5.2611. Fresh-server determinism holds (see the maple 0022 notes; batched
verify can drift greedy text at near-ties — the ppl gate is the arbiter).
Unlike qwen (rejected batch-17 verifies on 22GB lose −3..−22%) and lfm
(neutral), Laguna-XS's continuations on the bench vocabulary accept often
enough to pay well on average; the worst-case draw (0.86) carries ~7%
runner-median floor risk, accepted knowingly.

**Ranked outcome: rejected (1.311 vs 1.373).** The runner's own corpus
drew the low-acceptance tail: candidate decode 144.7 tok/s vs the
frontier's 155.0 — speculation COST ~7% on the ranked prompts. The
synthetic 6-seed mean (+14.2%) does not represent the ranked corpus's
acceptance. Do not resubmit without a mechanism that prices acceptance
online (propose only when the recent accept rate clears the verify
premium); the static n=3 lookup is net-negative on this track's corpus.


## 0020: the attention gate projection was blocked by GRAPH ORDER, not by anything else (+1.81% decode, BIT-IDENTICAL)

**RANKED: verified at 1.3872 (+38.72%), decode 156.8 tok/s / 1.6474**, prefill
1.0125, ttft 1.0020 - against the previous frontier of 1.3731 / 155.0 tok/s. The
local +1.81% bench figure landed at +1.76% ranked decode (1.6173 -> 1.6474), the
closest local-to-ranked agreement this series has recorded.

### The post-0019 census (start here)

`rocprofv3 --kernel-trace`, `llama-bench -p 0 -n 34 -r 1`, **sorted by
`Dispatch_Id`, not `Start_Timestamp`**. 28658 dispatches / 35 tokens =
**818.8/token, 5143 us of kernel in a 6244 us token** (160.2 tok/s local).
Kernel is 82% of wall, so the inter-dispatch gap is about **1.35 us**.

| n/tok | us/tok | each | kernel |
|---|---|---|---|
| 202 | 2854 | 14.13 | `mul_mat_vec_q` (18432/4608/2048/1024-block shapes) |
| 39 | 840 | 21.53 | `mul_mat_vec_q_grouped` (Q+K only) |
| 39 | 227 | 5.83 | `mul_mat_vec_f<f32,1,32>` (the F32 router) |
| 40 | 209 | 5.2 | `flash_attn_tile` (two instantiations) |
| 79 | 134 | 1.70 | `k_bin_bcast` |
| 80 | 228 | 2.85 | `rms_norm_pre_add_f32<1024,{1,2,3}>` |
| 38 | 114 | 2.99 | `rms_norm_f32_grouped<256>` (q/k norm) |
| 78 | 107 | 1.37 | `quantize_q8_1` |
| 39 | 100 | 2.58 | `topk_moe_cuda<256>` (ONE workgroup) |
| 40 | 80 | 2.00 | `flash_attn_combine_results<128>` |
| 38 | 61 | 1.60 | `rope_neox_grouped` |
| 40 | 57 | 1.43 | `k_mul_bcast0_quant` (softplus gate + q8_1) |
| 40 | 49 | 1.23 | `k_set_rows_grouped` |

Split by shape, `mul_mat_vec_q` is: 39 routed gate+up (4608 = 512x9 - 0018's
ninth channel), 39 routed down (18432 = 2048x9 - 0017's), 40 attn o_proj plus
1 (2048), **40 attention gate projections (64 or 48 rows each)**, 40 V
projections, the lm head and the dense layer. The MoE block is 6 dispatches
and the attention block 9.

### What 0020 changes, and why it was invisible for nineteen patches

The four attention projections all read `attn_norm`, and only Q and K are
grouped. V is out on type (Q4_K_M stores `attn_v` as Q6_K in half the layers).
The **gate** is out for a reason that is not about arithmetic at all: the DFS
reaches it only through the post-attention multiply, so it is emitted ~25 nodes
after `Qcur` and ggml-alloc gives it the block `Kcur` writes. 0004's hoist check
sees the overlap and refuses. **The refusal is the allocator's doing, not the
detector's** - the same shape as the round-31 lesson on qwen3.6.

One `ggml_build_forward_expand(gf, gate)` before `Qcur` is emitted fixes it:
818.8 -> 780.8 dispatches/token, `mul_mat_vec_q` 202 -> 162, every other kernel
count unchanged, decode +1.81% over 4/4 separated rounds against a
parent-commit control. Perplexity is bit-identical at both shapes and server
greedy is 6/6 byte-identical.

**Before assuming a grouping is impossible on this model, check the emission
order.** Three of this series' grouping mechanisms were already in the backend
and one of them was being starved by a single DFS edge.

### Measured DEAD this round: pinning Q/K/V as well (do not re-derive)

Expanding `Qcur`, `Kcur` and `Vcur` before the gate makes the Q4_K V
projections join too - `mul_mat_vec_q` 202 -> **140**, a further -22
dispatches - and it reads +1.60% on the bench. Do not ship it:

- it **breaks two other groupings**: `rope_neox_grouped` 38 -> 0 (replaced by
  40 `rope_neox<f32,f32>` plus 40 `rope_neox<f32,__half>`) and
  `k_set_rows_grouped` 40 -> 0. Net dispatches are **794.8, worse than mode 1's
  780.8**;
- decode-path perplexity moves **5.2682 -> 5.2419, -0.499%, five times the
  0.1% gate** - perfectly reproducible on both arms, while the `-c 512`
  gate shape stays bit-identical at 5.2611 on both. A reorder that reads clean
  at the gate shape can still be a half-percent numeric change at the shape it
  actually runs at.

It is kept behind `GGML_LAGUNA_QKV_GATE_PIN=2` so the next agent can see the
census rather than rebuild it. Getting V in legitimately needs the
cross-quant-type guest (qwen3.6's 0052) inside `mul_mat_vec_q_grouped`, not a
reorder.

### The gate-shape perplexity IS reproducible on this model

Contrary to the qwen3.6 note, `llama-perplexity -c 512 --chunks 8` on
Laguna-XS returned **5.2611 on 6 of 6 loads across two binaries** in this
session, and the decode-path shape returned 5.2682 on 6 of 6. Both shapes are
usable here; read both, because they disagree about mode 2 (above).

### Open levers after 0020, priced against the census

The cost constant on this track is about **2.6 us per removed small dispatch**
(38 removed, +1.81% of a 6244 us token). Priced from the table above:

1. **The F32 router matvec, 39/token at 5.83 us and only 256 blocks.** One warp
   per row over 2 MB of F32 weights is 360 GB/s against a 640 GB/s part, and
   256 blocks x 1 warp is exactly one wave per SIMD - it is occupancy-starved,
   not bandwidth-starved. Patch 0014 forced single-warp for this class; a
   per-shape revisit is worth ~+1.2%. Cheapest remaining lever.
2. **The Q6_K V projections into the Q/K group** via a second vec_dot type
   (port qwen3.6 0052). -20 dispatches, ~+0.8%.
3. `flash_attn_combine_results`, 40/token, ~134 us with its gap - but changing
   the FA split changes the reduction, so it is not a bit-identical class.
4. `topk_moe` (39/token, ONE workgroup) has no independent host in this graph:
   it consumes the router and feeds the routed matvec, and 0017/0018 already
   moved the shared expert inside the routed launch. Unlike qwen3.6 there is
   nothing left running beside it. Do not re-open without first un-folding 0018.


## Round 2 (2026-08-09): the ranked TTFT/prefill terms were never censused — and
## the server was doing ~120 MiB of per-request bookkeeping nobody had looked at

Nineteen patches went into decode. `prefill 1.0125` and `ttft 1.0020` say the
other 35% of the score exponent had never been touched. This round censuses
both, and the first thing it finds is worth more than any kernel left here.

### The sensitivity algebra — read this before pricing any prefill/ttft idea

`Sources/runner/base.ts` measures `ttft = t(534)`, `prefill = (t(534)-t(84))/451`
and `decode = (t_full(534,128) - t(534))/127`. Writing a candidate's savings as
`a` on the ttft request, `b` on the short request and `c` on the full request
(all in seconds), the score derivative on this track's local numbers
(ttft 0.247 s, slope numerator 0.107 s, decode numerator 0.762 s) is

```
dScore/Score = 1.631*a + 0.853*c - 1.876*b
```

Three consequences that kill obvious plans:

- **A ttft-ONLY saving is score-NEGATIVE.** `0.15/0.247 = 0.61` against
  `0.65/0.762 = 0.85`: the decode subtraction charges you more than the ttft
  term pays. Any host-side work you remove must come off the *full* request too.
- Work present at 534 and absent at 84 (per-token or per-extra-ubatch) is the
  best class at ~2.2-3.1 per second saved.
- Decode work is worth 0.853 per second — the smallest coefficient per ms, even
  though it carries the largest exponent, because its denominator is 3x bigger.

### Census: 52,726 minor faults per ranked request, AnonHugePages 0 kB

Replica of the harness, `minflt` read from `/proc/<srv>/stat` around each
request: **ttft 52,726 / short 38,450 / full 21,097 faults**, ~206 MiB of
4 KiB first-touch per ttft request, and `AnonHugePages: 0 kB` in
`smaps_rollup` — the qwen 0057 regime exactly. (LFM2.5 on the same box is
2,757 — the small-state regime, see that track's README.)

### MEASURED DEAD: qwen 0057 (`MADV_HUGEPAGE` on per-request host state)

Zero-rebuild confirmation, `GLIBC_TUNABLES=glibc.malloc.hugetlb=1`, 4 arms
off/on/on/off, 9 runs each:

| arm | TTFT | prefill tok/s | decode tok/s | AnonHugePages |
|---|---:|---:|---:|---|
| off | 0.24437 | 4255.6 | 167.09 | 0 kB |
| on | 0.20762 | 4238.2 | 160.71 | 3428352 kB |
| on | 0.20867 | 4215.9 | 159.99 | 3459072 kB |
| off | 0.24901 | 4129.8 | 166.16 | 0 kB |

**ttft -15.6%, prefill +0.8%, decode -3.8% → score +0.2%.** The faults are real
(52,154 removed for 38.5 ms, i.e. **0.74 us/fault** — carry that constant) but
they land almost entirely on the ttft and short requests, which is exactly the
`a` and `b` profile the algebra above punishes. Prefill cancels because both
arms of the slope lose the same fixed cost. **Do not port 0057 here.**

### 0021 — the server creates context checkpoints it always throws away

**RANKED: verified at 1.4651 (+46.51%)** against the previous frontier of
1.3872 — **+5.61% of score in one patch**, and the largest single jump this
track has recorded:

```
                   score      decode             prefill            ttft
new frontier     1.465086   1.656118 (158.47)  1.163134 (2585.6)  1.171798 (0.25910)
previous         1.387200   1.647400 (156.80)  1.012500            1.002000
                 +5.61%     +0.53%             +14.88%            +16.91%
```

**The replica under-predicted it by 65%** (modelled +3.39%, measured +5.61%),
and the error is in the decode term: the replica said decode would lose 1.4% to
the harness's ttft subtraction and the ranked run returned **+0.53%**. The
saving lands more evenly across the ttft and full requests on the runner's own
prose corpus than on the replica's filler prompt. Two rules from that:

- **the replica is a lower bound for this class**, not an over-estimate — the
  opposite of the qwen 0057 lesson, because this saving is GPU work (two decode
  calls) and not only page faults;
- prefill and ttft both landed within a point of each other (+14.9%, +16.9%),
  which is the signature the mechanism predicts: a fixed per-request cost that
  is paid twice at 534 tokens and once at 84.

`llama-server -v` on a 528-token request:

```
task 0 | cached n_tokens = 0 / 11 / 523          <- THREE llama_decode calls
task 0 | created context checkpoint 1 (12 tok,   1.407 MiB)
task 0 | created context checkpoint 2 (524 tok, 60.007 MiB)
task 4 | erased invalidated context checkpoint (... pos_next = 0) x2
```

`tools/server/server-context.cpp` breaks the prompt at `{4 + n_ubatch, 4}`
tokens from its end purely so a checkpoint can be captured (PR 20288), then the
next cache-cold request erases both unused. Priced with `--ctx-checkpoints 0`,
4 arms: ttft **x1.196**, prefill **x1.080**, decode **x0.983** → **+3.18%**.

Shipped as adaptive credit rather than a flag or a default flip — a restore
refills it, a task that discards its checkpoints spends it, one probe task
every 64 — so multi-turn traffic keeps the feature. 8-arm A/B gives ttft
**x1.1960**, prefill **x1.0822**, decode **x0.9858**, modelled **+3.39%**.
Gate ppl **5.2611 = stock, 3/3**, and the served path is **bit-identical**:
16/16 greedy prompts byte-identical with `max |dlogprob| = 0.000e+00`.

### The prefill slope census (llama-bench, per pass — halve the raw trace)

`rocprofv3 --kernel-trace`, `-p 84,300,534 -n 0 -r 1`. **llama-bench runs a
warmup plus the rep, so every figure the trace reports is TWO passes.** Per
pass: pp84 74.2 ms, pp534 183.8 ms → slope 243.6 us/tok = 4105 tok/s, which
matches the server's 4256 tok/s. Kernel is 75% of the 244 ms ttft.

| kernel | pp84 | pp300 | pp534 | %slope |
|---|---:|---:|---:|---:|
| `mul_mat_q` | 59.6 | 91.5 | 131.6 | 65.7% |
| `flash_attn_ext_f16<128,128,8,8>` | 1.5 | 5.3 | 9.6 | 7.4% |
| `k_bin_bcast` | 1.0 | 3.5 | 6.4 | 4.9% |
| `mm_ids_helper<8>` | 1.2 | 3.3 | 6.1 | 4.5% |
| **rocBLAS f32 `Cijk_..._MT64x64x16`** | **5.07** | **5.07** | **9.70** | **4.2%** |
| `quantize_mmq_q8_1` | 1.4 | 2.7 | 5.2 | 3.5% |
| `rms_norm_f32<256>` | 0.8 | 2.3 | 4.4 | 3.3% |

Note `mul_mat_q` grows only 1.53x when tokens grow 3.6x (84 -> 300): on a
256-expert MoE the expert weight stream is common to both arms of the slope and
largely cancels. **The rocBLAS f32 row is INVARIANT in tokens** — 5.07 ms at
both 84 and 300, doubling at 534 only because there are two ubatches. It is the
MoE router `[2048 -> 256]`, one launch per layer per ubatch, 38 per ubatch:

```
grid 2048 wi / wg 64 = 32 workgroups   130.2 us each   (M = 512 ubatch)
grid  256 wi / wg 64 =  4 workgroups   125.1 us each   (M = 22 tail ubatch)
```

**4 workgroups on a 64-CU part, and the 22-token tail costs the same as the
512-token one.** This is qwen 0044/0056 verbatim — the port is open and priced
at roughly +0.9% score (decode-neutral, since the full request saves the same
absolute time as the ttft request). Divide the grid by the workgroup size
before assuming a vendor kernel is doing something clever.

### Open, in order

1. Port qwen 0044+0056 (`mmf32-skinny`) for the router. ~+0.9%, decode-neutral,
   the shapes are identical to qwen's so the kernel needs no changes.
2. `mul_mat_q` at 65.7% of the slope — qwen round 47 has it at 94-99% of
   achievable on loads but only 61-81% of a no-LDS/no-MMA probe. Never censused.
3. The decode levers from round 1 are unchanged (F32 router matvec, Q6_K V into
   the Q/K group).

### 0022 — 0021 unlocked the tail ubatch

**RANKED: verified at 1.5600 (+56.00%)** against 1.4651 — **+6.48% of score**:

```
                 score      decode             prefill            ttft
new frontier   1.560023   1.658249 (158.68)  1.395498 (3101.9)  1.389160 (0.21856)
previous       1.465086   1.656118 (158.47)  1.163134 (2585.6)  1.171798 (0.25910)
               +6.48%     +0.13%             +19.98%            +18.55%
```

Modelled +7.6%, measured +6.48% — the replica was 1.1 points optimistic this
time, having been 2.2 points pessimistic on 0021. Decode is flat to two decimal
places on the runner, exactly as the mechanism requires.

**Session total for this track: 1.3872 -> 1.5600, +12.46% of score, from two
patches totalling 111 lines and no kernel.**

The sibling `laguna-s-2.1-gguf-gb10cuda-v1` derived tail-ubatch absorption and
then recorded it **INERT on the ranked path**: `llama-server` pre-chunks the
prompt at `{4 + n_ubatch, 4}` from the end to build context checkpoints, so
`llama_decode` never sees more than `n_ubatch` and the absorption test is never
true. **0021 removes exactly that pre-chunking.** The server now hands
`llama_decode` all 534 tokens, `llama_context` splits them 512 + 22, and the
absorption patch fires for the first time on this platform.

Zero-rebuild confirmation first (`-b 2048 -ub 1024`, 4 arms): prefill x1.317,
ttft x1.134, decode x0.998 → +7.55%. Then the real patch (absorb to
`n_ubatch + n_ubatch/8` = 576, so buffers grow 12.5% rather than 100%), 8 arms:
**ttft x1.1367, prefill x1.3096, decode x1.0001 → +7.6%.**

`llama-bench -p 534 -n 0 -r 1` under `rocprofv3`: **6989 → 3611 dispatches**,
i.e. exactly the single-ubatch count, and pp534 2711.17 → 3349.56 t/s.

**The batch-shape perplexity question, settled properly.** At `-c 534 -b 534
-ub 512` the arms read 5.3031 vs 5.3163 (+0.249%), which looks like 2.5x the
gate. It is not the patch's arithmetic — with the **unmodified b10237 binary**
and only `-ub` changed, stock reads **5.3031 at `-ub 512`** and **5.3163 at
`-ub 534`**. The candidate's absorbed reading *is* stock's own single-ubatch
reading to every digit. The sign also flips at `--chunks 16` (−0.253%), so it is
a draw rather than a bias, and the patch is bit-identical at `-ub 534` where it
cannot fire and at the runner's `-c 512` gate where it provably cannot fire.
Served path: 16/16 greedy byte-identical, `max |dlogprob| = 0.000e+00`.

**Carry this**: when a batch-shape change moves the split-shape perplexity, ask
what STOCK reads at the two shapes before concluding anything. If the candidate
reproduces one of stock's own readings exactly, it has selected a batching, not
changed a computation.

### The composition rule this round bought

Two patches, neither of which is worth much alone, that multiply:

```
0021 (server stops pre-chunking)   ranked +5.61%   and it UNLOCKS ->
0022 (llama_context absorbs the tail)     +7.6% modelled
```

Before assuming a lever is dead because a sibling track measured it inert, check
whether the thing that made it inert is itself removable.
