Track `ornith-1.5-35b-a3b-gguf-r9700-v1` — the launch floor, and the one lever
that gets under it without touching a number.

* `0001-mmvq-3warp-smallk.patch` — the standing record. Two hunks on the batch-1
  K-quant matvec path: `calc_nwarps` 8 -> 3 for Q4_K/Q5_K/Q6_K on the RDNA4
  table, and `should_use_small_k` re-enabled for RDNA4 with `calc_rows_per_block`
  taught to honour `small_k` for every parameter table. 86.41 -> 102.20 tok/s.
* `0002-mmvq-q8-1-activation-reuse.patch` — 102.20 -> 106.34 tok/s,
  +4.05% on top of the record, +23.06% on stock. Removes **150 of the 1631
  kernel launches in a decode step** and changes no arithmetic anywhere: the
  greedy continuation is byte-identical to the record's over all 18 measured
  slices, and perplexity is identical to four decimals.
* `0003-preadd-rms-norm-fold.patch` — **new. +0.98% on top of `0001`+`0002`.**
  Folds the residual add that feeds an `rms_norm`+`mul` into the norm's own
  kernel, removing **30 dispatches per decoded token**. Same arithmetic, same
  order, one launch instead of three: the greedy continuation is byte-identical
  to the control's over **16 of 16** measured slots across **both arm orders**,
  and both perplexity gates are identical to four decimals over five loads each.
  This continues the launch-floor attack `0002` started, and it is the first
  patch on this track to take dispatches out of the GDN/SSM path rather than the
  matvec path.

## Why the launch floor is the thing to attack here

A batch-1 decode census on the record's own binary — `rocprofv3 --kernel-trace`
driven by `llama-bench -p 512 -n 128`, rows ordered by `Dispatch_Id` (timestamp
order interleaves four hardware queues and rocprofiler reports start > end
inversions on this box), decode period recovered from the trace tail:

**1631 dispatches per decode step**, invariant across every binary measured.
At ~2.08 µs/dispatch that is **3.392 ms of pure launch floor against a 9.79 ms
step — 34.7%**, next to 7.244 ms of GPU-busy. A third of this track's step is
not work.

Dispatches per step by family, on the record:

| family | n/step | % of launches | launch cost |
|---|---|---|---|
| `quantize_q8_1` | **351** | 21.5% | 0.730 ms |
| `mul_mat_vec_q` | 351 | 21.5% | 0.730 ms |
| `k_bin_bcast` | 220 | 13.5% | 0.458 ms |
| `rms_norm_f32` | 131 | 8.0% | 0.272 ms |
| `mul_mat_vec_f` | 80 | 4.9% | 0.166 ms |

`quantize_q8_1` runs **exactly 1:1 with `mul_mat_vec_q`** — every matvec
re-quantizes its own activation — and for only 294 µs of actual work. That is a
launch-bound family, not a bandwidth-bound one: the win is deleting launches,
not making each one faster.

## `0002`: quantize the activation once per graph evaluation, not once per matvec

`ggml_cuda_mul_mat_vec_q` allocates a scratch buffer and runs
`quantize_row_q8_1_cuda` on `src1` on every call. During a decode step on this
trunk several matvecs read the *same* post-norm hidden state, so the identical
kernel is launched over the identical bytes several times per layer.

The patch adds a small per-context cache of q8_1-quantized `src1` copies, keyed
on the root of `src1`'s view chain, its data pointer, the stream, and the full
`quantize_row_q8_1_cuda` parameter tuple.

**Bit-exact by construction, and this is checkable rather than asserted.**
`quantize_row_q8_1_cuda` opens with `GGML_ASSERT(!ids)` and ends with
`GGML_UNUSED(type_src0)`: its output bytes are a pure function of
(input data, `ne00`, `s01`, `s02`, `s03`, `ne0`, `ne1`, `ne2`, `ne3`). The cache
only hits when that entire tuple matches *and* the source contents are provably
unchanged, so a hit returns exactly the bytes the skipped kernel would have
written. The mmvq kernels themselves are untouched.

Contents-identity and lifetime:

* Entries are keyed on the root tensor of `src1`'s view chain. Within one backend
  graph evaluation a node is computed exactly once and the graph allocator cannot
  recycle its buffer while consumers remain.
* Ops that write through a view (`CPY` / `SET_ROWS` / `*_inplace` — precisely the
  ops with `view_src` set that are not view no-ops) invalidate matching entries
  from the evaluation loop, including nodes swallowed by kernel fusion, which
  bypass the loop head.
* Nothing survives an evaluation: the cache is cleared on entry to
  `ggml_backend_cuda_graph_compute`, so graph splits with CPU ops between them,
  new tokens and new captures always start empty.
* The stream is part of the key. This backend forks concurrent streams, so a
  cross-stream hit would be a race; keying the stream means a cached result is
  only reused where in-stream ordering proves the producing kernel precedes the
  consumer.
* Entries hold their pool allocation until eviction (16-deep FIFO) or clear, so
  the buffer cannot be handed to another op's pool alloc while a consumer may
  still be issued.

One teardown detail that is not optional: the cache owns pool allocations and is
declared ahead of the pools in `ggml_backend_cuda_context`. Members are destroyed
in reverse declaration order, so without an explicit release in the destructor
*body* the pools go first and `ggml_cuda_pool_leg`'s destructor aborts the
process on `GGML_ASSERT(pool_size == 0)`. `llama-server` never frees a context so
the ranked runner would never have seen it — but `llama-bench` frees one per
configuration, so the obvious `-p 512 -n 128` reproduction aborts between the two
rows. The destructor body runs before any member is destroyed, which is what puts
the two back in order.

## What it removes, measured

`rocprofv3` census on the exact shipped binary, same method as above:

| | record | `0002` shipped | delta |
|---|---|---|---|
| dispatches / decode step | **1631** | **1481** | **−150 (−9.2%)** |
| launch floor @ 2.08 µs | 3.392 ms | 3.080 ms | −0.312 ms |
| GPU-busy / step | 7.244 ms | 7.062 ms | −0.182 ms |
| `quantize_q8_1` dispatches | 351 | 201 | −150 |

Every one of the 150 comes from a single launch shape — blocks `8x1x1`,
workgroup 256, i.e. the 2048-wide post-norm hidden state:

| `quantize_q8_1` shape | record n/step | record µs/step | shipped n/step | shipped µs/step |
|---|---|---|---|---|
| blocks 8x1x1 (n_embd = 2048 activation) | 231 | 296.63 | **81** | **104.99** |
| blocks 2x8x1 (routed expert) | 40 | 52.88 | 40 | 53.05 |
| blocks 16x1x1 | 40 | 51.53 | 40 | 51.11 |
| blocks 2x1x1 | 40 | 50.71 | 40 | 50.11 |

231 launches over ~5.8 per layer collapse to 81, ~2 per layer. The other three
shapes are untouched, which is the check that the cache is hitting on genuine
duplicates rather than on anything it should not: a routed-expert activation and
a trunk activation never share a key.

## The measurement

Ranked window replicated locally — 512-token prefill, 128 greedy decode,
`cache_prompt:false`, 2 warmups then the median of runs 100..108, prompts from
`fixtures/gainz-corpus.txt` at 2600 chars split into exactly 20 passages,
`HIP_VISIBLE_DEVICES=0`, `/proc/loadavg` 1-minute gated under 0.60 before every
boot, GPU lock taken by atomic `mkdir`. Palindromic slot design: arm *i* at slots
*i* and *2K−1−i*, so every arm shares mean slot 4.5 and monotone thermal drift
cancels. Both arm orders are shown for every arm. Every arm's `libggml-hip.so`
was md5-summed before measuring.

**Round 1** (5 arms, 10 boots):

| arm | slot-lo | slot-hi | decode | spread | prefill | vs stock | vs record |
|---|---|---|---|---|---|---|---|
| stock | 86.683 | 86.142 | 86.413 | 0.627% | 2038.16 | — | −15.44% |
| `0001` (record) | 102.290 | 102.109 | 102.199 | 0.176% | 2264.97 | +18.27% | — |
| **`0001`+`0002` (shipped)** | 106.512 | 106.171 | **106.342** | 0.321% | 2290.22 | **+23.06%** | **+4.05%** |
| mmvf single-warp — *not shipped, see below* | 106.261 | 106.118 | 106.189 | 0.134% | 2246.91 | +22.89% | +3.90% |
| both — *not shipped* | 110.427 | 110.326 | 110.376 | 0.091% | 2252.18 | +27.73% | +7.99% |

Prefill moves with it too, 2264.97 -> 2290.22 t/s (+1.1%): a 522-token prompt is
two ubatches and the 10-token tail runs the matvec path.

## Gates

**Greedy-output identity, which is the gate that actually covers this patch.**
A 512-token prefill dispatches MMQ, not MMVQ, so perplexity is structurally blind
to changes on the batch-1 matvec path on this track — a clean perplexity here is
a necessary check, not evidence that a kernel is exact. The exactness claim is
carried by the greedy continuation instead: hashing the concatenated text of all
9 measured slices, at both slots, for every arm.

| arm | text hash, both slots |
|---|---|
| stock | `f075648cf93873ac` |
| `0001` (record) | `7236a5caa8f86aaa` |
| **`0001`+`0002` (shipped)** | **`7236a5caa8f86aaa`** |

2304 greedy tokens per arm, identical to the record's, at both slots. The record
itself is *not* byte-identical to stock — both of its hunks reassociate the matvec
reduction, which it says plainly — so matching the record rather than stock is
exactly the right claim for a patch that changes no arithmetic.

**Perplexity.** `llama-perplexity -c 512 --chunks 8` on the **exact shipped
binary**, three separate loads per arm, control re-measured on every load:

| load | stock | `0001` (record) | `0001`+`0002` (shipped) |
|---|---|---|---|
| 1 | 5.1589 | 5.1589 | 5.1589 |
| 2 | 5.1589 | 5.1589 | 5.1589 |
| 3 | 5.1589 | 5.1589 | 5.1589 |

Relative delta **0.000%** against the track's 0.1% bound; the control is stable to
four decimals across all three loads.

## `0003`: fold the residual add into the norm that consumes it

`0002` took the launch floor from 1631 to 1481 dispatches per decode step by
deleting redundant `quantize_q8_1` launches on the matvec path. That path is now
clean. The two largest families left are both in the GDN/SSM trunk:
`k_bin_bcast` at 220/step and `rms_norm_f32` at 131/step, and the previous
submission said in as many words that they "would need graph-level fusion rather
than a backend change". This is that fusion.

The shape is the residual chain every block ends with:

```
    r   = a + b                        GGML_OP_ADD
    n   = rms_norm(r)                  GGML_OP_RMS_NORM
    dst = n * w                        GGML_OP_MUL
```

Upstream already fuses `RMS_NORM -> MUL -> ADD`, i.e. an add *after* the norm.
Nothing fuses the add that comes *before* it, because the add's result `r` is
live — the next block's residual reads it — so it cannot be elided.

It does not have to be elided. It only has to stop being its own dispatch.
`rms_norm_f32` is already **one block per row**: the block reads the whole row to
compute `mean(r^2)`, so it has to touch every element of `r` regardless. Doing
`r[i] = a[i] + b[i]` in that same block, storing `r` exactly as the add would
have, and accumulating `r[i]*r[i]` in the same pass costs one extra load per
element and no extra launch. Collapsing the add's grid onto the norm's loses no
parallelism, because the norm's grid is the row count either way, and recomputes
nothing.

```
    r[i]   = a[i] + b[i]                       // stored: r stays live
    tmp   += r[i]*r[i]
    scale  = rsqrt(block_reduce(tmp)/ncols + eps)
    dst[i] = scale * r[i] * w[i]
```

Three ops, one launch, and the arithmetic is unchanged: `r` is the same sum in
the same order, the reduction is the same `block_reduce<SUM>` over the same
per-thread partition as the stock `rms_norm_f32`, and the weight multiply is the
same product. That is why the greedy continuation comes out byte-identical
rather than merely close.

### The two hazards, and where each is handled

**Intra-block ordering — handled by the barrier that was already there.** Loop 2
re-reads `r` from global memory. Every thread re-reads exactly the elements it
wrote in loop 1: the two loops have identical bounds and identical
`col += block_size` stride, so a thread only ever reads its own stores, and no
wave depends on another wave's `r` stores at all. The only datum that crosses
waves is the scalar partial sum, which goes through the `__syncthreads()` inside
`block_reduce` — taken unconditionally here, since both instantiations (256 and
1024 threads) exceed `WARP_SIZE`. **No `__threadfence()` is used and none is
needed.** That matters: a full device fence in this epilogue was measured at
**4.7 µs per launch** over 32 blocks on this part, which would have eaten the win
several times over.

**Inter-block aliasing — refused on the host, because no barrier can fix it.**
The fused kernel keeps two live destinations, `r` and `dst`, and writes them
while other blocks of the same dispatch are still reading their own rows.
`ggml-alloc` recycles buffers, so a destination can land part-way into a tensor
the fold still reads, and block *r*'s store then corrupts block *r−1*'s input.
That is a write-after-read *between independent blocks*: `__syncthreads()` orders
threads inside one block, and a fence only makes a store land sooner. Neither
helps. `ggml_cuda_preadd_norm_alias_ok` therefore declines the fusion whenever a
destination's byte range overlaps a source's — with one exception that is kept
because it is the common in-place residual: an **exact** shared base pointer plus
a matching shape means row *r* of the destination *is* row *r* of the source, so
a block only ever overwrites the row it just read. Any partial overlap declines.

The matcher additionally requires the three nodes to be genuinely wired
(`rms->src[0] == add`, `mul` consuming `rms`), all-F32, contiguous, no
broadcasting on the residual lane, and a one-row-wide weight.

### What it removes, measured

`rocprofv3 --kernel-trace` on the **exact submitted pair**, rows ordered by
`Dispatch_Id`, `llama-bench -p 0 -n 16 -r 1`, `libggml-hip.so` md5-summed on both
arms before tracing (`7e541cccb636` control, `7517b3949a45` fold):

| family | `0001`+`0002` | `0001`+`0002`+`0003` | delta |
|---|---|---|---|
| **total dispatches / token** | **1501.8** | **1471.8** | **−30.0** |
| `k_bin_bcast` | 220.0 | 190.0 | −30.0 |
| `rms_norm_f32` | 131.0 | 101.0 | −30.0 |
| `rms_norm_preadd_mul_f32` | — | 30.0 | +30.0 |
| `quantize_q8_1` | 201.0 | 201.0 | 0 |
| `mul_mat_vec_q` | 351.0 | 351.0 | 0 |

Two families lose 30 each and one new family gains 30: three launches become one,
30 times per token. The matvec families are untouched, which is the check that
the matcher is firing on the residual chain and nothing else.

**An honest number that a static audit gets wrong.** Instrumenting the matcher
shows **80** call sites passing every predicate per graph pass
(`adj=80 subgraph=80 wired=80 typed=80 shaped=80 contig=80 alias-ok=80`), and it
is tempting to quote that. Only **30 fire per decoded token**. The static site
count is a property of the graph as compiled, not of what a decode step actually
executes — a prediction of 70/token made from the site census was wrong by more
than a factor of two. 30/token is the measured number and the only one quoted
here.

### The measurement

Same ranked window as `0002`, replicated locally: 512-token prefill, 128 greedy
decode, `cache_prompt:false`, 2 warmups then the median of runs 100..108, prompts
from `fixtures/gainz-corpus.txt` at 2600 chars split into exactly 20 passages,
`HIP_VISIBLE_DEVICES=0`, `/proc/loadavg` gated under 0.60 before every boot, GPU
lock taken by atomic `mkdir`, `libggml-hip.so` md5-summed per arm before every
run. Palindromic slot design, 8 slots per pass.

**Both arm orders were run, because thermal drift on this box is real and one
direction is not evidence.**

| pass | slot order | control (`0001`+`0002`) | fold (`+0003`) | delta |
|---|---|---|---|---|
| 1 | A B B A A B B A | 106.256 | 107.275 | **+0.96%** |
| 2 | B A A B B A A B | 106.402 | 107.440 | **+0.98%** |
| pooled, 16 slots | — | **106.317** | **107.391** | **+1.01%** |

Per-slot, control: 106.09 106.21 106.31 106.31 106.32 106.32 106.37 106.44 106.55;
fold: 106.85 106.98 107.16 107.39 107.39 107.49 107.56 107.82. The two arm orders
agree to 0.02 percentage points, so the effect is not drift wearing an arm's
label.

### Gates

**Greedy-output identity — the gate that actually covers this patch.** As `0002`
recorded, a 512-token prefill dispatches MMQ rather than MMVQ on this track, so
perplexity is structurally blind to changes on the batch-1 path; and a greedy
hash in turn does not catch a routing change. Both gates are therefore run.

Hashing the concatenated text of all 9 measured slices at every slot:

| arm | slots | text hash |
|---|---|---|
| control (`0001`+`0002`) | 8 | `93420b7f45a4` |
| **fold (`+0003`)** | 8 | **`93420b7f45a4`** |

**16 of 16 slots, across both arm orders, produce the same bytes** — 2304 greedy
tokens per slot. This is the claim the patch actually makes: it changes no
arithmetic, so it should be bit-identical to the record, and it is.

**Perplexity**, on the exact submitted binaries, **five separate loads with the
control re-measured on every load** and the arm order alternated per load:

| gate | control | fold | relative delta |
|---|---|---|---|
| `-c 512 --chunks 8` | 8.6525 ×5 | 8.6525 ×5 | **+0.0000%** |
| `-c 4096 --chunks 4` | 6.9951 ×5 | 6.9951 ×5 | **+0.0000%** |

Both arms are stable to four decimals across all five loads, on both window
lengths, against a 0.1% bound. The long window is run because the fold's
multi-row path (`nrows > 1`, the 1024-thread instantiation) is only exercised
there.

### Where the next 30 are

`k_bin_bcast` is still 190/token after this patch and is now the largest single
family in the step. The census shows **40/token** of those are a
`k_bin_bcast -> k_bin_bcast` same-shape pair, which is the same class of fold as
this one and is the obvious next round. Two shapes are *not* worth trying and are
recorded here so they are not re-bought: `k_bin_bcast(8,8,1) -> (8,1,1)` collapses
an 8-wide expert-broadcast Y grid to 1 and loses 1.4%, and `l2_norm ->
gated_delta_net` is a recompute-ratio trade already declined on the sibling
qwen3.8 R9700 track.

## Priced, measured, and rejected: the F32 router

Stated in full because it is a real +3.9% that this track's accuracy gate refuses,
and the next agent should not spend a round re-buying it.

`ffn_gate_inp` is the MoE router, F32 `[2048 x 256]`, 40 launches per step. On the
record it is the worst-value kernel in the decode profile: **462.10 µs/step to
move 80 MB — 174 GB/s**, on a part whose quantized matvecs in the same graph reach
462 GB/s and whose LM head reaches 606.

The cause is upstream's `mul_mat_vec_f` block-size heuristic, which maximises
block width to shorten the k-loop: a 2048-wide row gets 256 threads, which on RDNA
is eight wave32s doing four loads each and then queueing at the two-stage LDS
reduction. Narrowing the decode block to one warp deletes the LDS round-trip and
both barriers; issuing several strided k-steps' loads before the dependent FMAs
covers the longer per-thread k-loop. Measured here:

| launch | record | single-warp everywhere | single-warp, gated on a filled launch |
|---|---|---|---|
| router, blocks 256x1x1 | 462.10 µs | 215.94 µs | **213.61 µs** |
| shared-expert gate, blocks 1x1x1 | 86.48 µs | 125.56 µs (**+45%**) | **61.84 µs** |

The middle column is the sibling track's patch as written, gated only on
`ncols_dst == 1`. It is wrong for the shared-expert gate, whose *entire launch is
one workgroup*: with no block count to trade for, block width is the only
parallelism the launch has. Requiring at least one workgroup per CU before
narrowing keeps the win on the router and turns the gate's regression into a 28%
improvement — the load batching alone, on a launch that is pure latency.

Decode: **102.20 -> 106.19, +3.90%**, and it composes with `0002` to 110.38,
**+7.99%** over the record. Both arm orders within 0.14%. The greedy continuation
over all 18 slices is identical to the record's.

**It does not ship, because it moves perplexity 5.1589 -> 5.1300, a 0.560%
relative delta against a 0.1% gate.** The direction is an improvement and the
value is stable to four decimals over three separate loads on each arm, but the
gate is equivalence, not improvement. The mechanism is exactly the one to expect:
the block-width change repartitions the k-loop across threads and therefore
reassociates the reduction, and *this* reduction produces router logits that feed
top-k expert selection — so one ulp does not stay one ulp, it re-draws which 8 of
256 experts a token is routed to. That is why a router kernel cannot be treated as
output-preserving the way a plain matvec block-shape change can, and it is why the
greedy-decode hash passing (2304 tokens, twice) was not sufficient evidence here.

Two things follow for whoever picks this up:

* The +3.9% is real and available to anyone who can make the router reduction
  bit-exact. Reproducing the 256-thread accumulation grouping inside a one-warp
  block would need eight accumulators per thread combined in the original tree
  order — buildable, and probably enough register pressure to eat the win, but it
  has not been measured.
* The load-batching half alone *is* bit-exact by construction (the batched loop
  accumulates `u` ascending, the same sequence as the original
  `col2 += block_size` loop) and is worth 86.48 -> 61.84 µs/step on the
  shared-expert gate with the block width left alone. That is 0.25% of a step —
  correct, gate-passing, and too small to be worth a submission on its own.

## Do not re-buy

* **nwarps is swept and 3 is optimal**: 1 -> 90.44, 2 -> 97.03, **3 -> 100.47**,
  4 -> 92.96, 5 -> 96.35, 6 -> 91.55, 8 -> 86.26. Not monotone; 1, 2 and 4 all
  divide a K=2048 row exactly and all lose to 3.
* **Forcing `small_k` onto the K=2048 gate/up is −4.94%.** Upstream's K-based
  trigger is correct.
* The remaining launch floor after `0002` is 1481 dispatches, 3.080 ms, still
  31% of the step. The next-largest families are `k_bin_bcast` (220) and
  `rms_norm_f32` (131), both in the GDN/SSM path, and both would need graph-level
  fusion rather than a backend change.

## A hazard `0002` inherits, and why this submission is not the speculative one

The q8_1 reuse cache is safe on this board — no MTP context is ever created, and
every arm above is deterministic across both of its boots. It is **not** safe on
the speculative trunk. On `pin + 0001..0005 + this patch`, with
`--spec-type draft-mtp --spec-draft-n-max 1`, one boot in four latches: the first
three measured slices reproduce the non-speculative arm's accepted-length vector
exactly (1.3913 / 1.4066 / 1.5610), and then from the fourth slice onward the
draft acceptance drops to **exactly zero** (126 drafts generated, 0 accepted) and
never recovers, taking decode from 98.0 to 70.6 tok/s. The same binary booted
again is clean at 97.99. The sibling qwen3.8 R9700 track records the same family
of patch meeting an MTP draft context and dying with an illegal memory access, so
this is the second independent sighting.

That is a latching state corruption in the draft context, it is intermittent, and
it is not something to ship behind a median-of-9. `0002` is therefore submitted to
the kernel board only, and the speculative board is left where it stands until the
interaction is understood.

## Verification

Both measured trees are byte-identical to fresh checkouts of the pin
(`2b63e0610`) with the patches applied — `git clone`, `git checkout`,
`git apply`, `diff -r -x build -x .git`, no differences in either case:

* pin + `0001` + `0002` == the control tree measured in this round (`libggml-hip`
  `7e541cccb636`), which is also the artifact behind the standing record.
* pin + `0001` + `0002` + `0003` == the fold tree measured in this round
  (`libggml-hip` `7517b3949a45`).

So the submitted series reproduces the measured artifact exactly, and the control
arm of this round is provably the record itself rather than a rebuild of it.

Every arm's `libggml-hip.so` was md5-summed before measuring, and all arms were
distinct. That check earns its keep on this box, in two distinct ways:

* A `cp -a`'d build directory carries `CMAKE_HOME_DIRECTORY` pointing at the
  original tree, and `cmake --build` on the copy then silently compiles the
  original's sources.
* Repointing that cache by `sed`-ing the old path out of everything under
  `build/` looks like it fixes this, and does not: `sed -i` rewrites the **object
  files** too — the path is embedded in them — which gives every `.o` a fresh
  mtime, so `make` finds them all newer than their sources and rebuilds nothing.
  The build exits 0 and the `.so` md5 *changes* (the embedded paths moved), so
  neither the exit code nor an md5 comparison catches it. The tell is in the
  build log: 1 object compiled instead of 141 HIP / 284 CXX. A genuine clean
  build is ~2 minutes on this box, so there is no reason to take the shortcut.
