Track `ornith-1.5-35b-a3b-gguf-r9700-v1` — MMVQ block shape on a sparse trunk.

* `0001-mmvq-3warp-smallk.patch` — two hunks on the batch-1 K-quant matvec path.
  **86.53 -> 102.39 tok/s, +18.33%**, measured against the shipped 5-warp record
  at 96.22 in the same round: **+6.41% on top of the current record.**

  1. `calc_nwarps` 8 -> 3 for Q4_K/Q5_K/Q6_K on the RDNA4 table. The record
     ships 5, which was inherited from the dense qwen3.8 track's
     `0002-mmvq-verify-width` and had never been swept against this model's
     shapes.
  2. `should_use_small_k` re-enabled for RDNA4 on those three types, and
     `calc_rows_per_block` made to honour `small_k` for every parameter table
     instead of only GENERIC/GCN/TURING. Upstream disables the small-K row
     blocking for **all** of RDNA *and* the RDNA4 table cannot express it — two
     independent switches, both of which have to move before anything happens.

## Why this trunk is nothing but small-K reductions

A batch-1 decode census (`rocprofv3 --kernel-trace`, rows ordered by
`Dispatch_Id` — timestamp order interleaves four queues and rocprofiler reports
start > end inversions on this box) finds **1631 dispatches in one decode step**,
of which 351 are `mul_mat_vec_q` and account for 68% of GPU-busy time.

The two routed-expert shapes are `[2048 x 512]` gate/up and `[512 x 2048]` down
— 8 and **2** Q4_K super-blocks per row. An 8-warp block consumes 16 super-blocks
per K-loop iteration, so on the down projection 32 of a block's 256 threads have
a K block to reduce and the other 224 are launched to do nothing.

Priced on the current record's binary:

| launch | n/step | µs/step | bytes/step | GB/s |
|---|---|---|---|---|
| routed gate+up, K=2048, 8 experts | 40 | 818 | 378 MB | **462** |
| routed down, K=512, 8 experts, Q6_K | 20 | 668 | 138 MB | **206** |
| routed down, K=512, 8 experts, Q4_K | 20 | 511 | 94 MB | **184** |
| dense LM head, K=2048, Q6_K | 1 | 689 | 417 MB | **606** |

Same experts, same quantisation, same part, adjacent in the same graph: the only
thing separating the 462 GB/s gate/up from the 184 GB/s down is K. That is
exactly the case `small_k` exists for upstream, and exactly the case RDNA is
excluded from.

## The sweep, because this curve cannot be derived

Ranked window replicated locally — 512-token prefill, 128 greedy decode,
`cache_prompt:false`, 2 warmups then the median of runs 100..108, prompts from
`fixtures/gainz-corpus.txt` at 2600 chars split into exactly 20 passages,
`HIP_VISIBLE_DEVICES=0`, `/proc/loadavg` gated under 0.60 before every boot, GPU
lock taken by atomic `mkdir`. Palindromic slot design: arm *i* at slots *i* and
*2K−1−i*, so every arm shares mean slot and monotone thermal drift cancels. Both
arm orders are shown for every arm.

**Round 1** (7 arms, 14 boots, base = stock):

| arm | slot-lo | slot-hi | decode | spread | vs stock |
|---|---|---|---|---|---|
| stock (nwarps 8) | 86.184 | 86.344 | 86.264 | 0.185% | — |
| nwarps 6 | 91.569 | 91.527 | 91.548 | 0.046% | +6.12% |
| small_k + nwarps 4 | 92.190 | 92.323 | 92.257 | 0.145% | +6.95% |
| nwarps 4 | 92.939 | 92.986 | 92.963 | 0.051% | +7.76% |
| nwarps 5 (shipped record) | 96.437 | 96.264 | 96.350 | 0.180% | +11.69% |
| small_k + nwarps 5 | 97.025 | 96.681 | 96.853 | 0.355% | +12.27% |
| **nwarps 3** | 100.447 | 100.498 | **100.473** | 0.051% | **+16.47%** |

**Round 2** (5 arms, 10 boots, base = the 5-warp record):

| arm | slot-lo | slot-hi | decode | spread | vs record |
|---|---|---|---|---|---|
| nwarps 1 | 90.522 | 90.364 | 90.443 | 0.175% | −6.12% |
| nwarps 5 (record) | 96.482 | 96.204 | 96.343 | 0.289% | — |
| nwarps 2 | 97.102 | 96.966 | 97.034 | 0.141% | +0.72% |
| nwarps 3 | 100.380 | 100.322 | 100.351 | 0.058% | +4.16% |
| **small_k + nwarps 3 (shipped)** | 102.405 | 102.262 | **102.333** | 0.139% | **+6.22%** |

The record reproduced across rounds at 96.350 / 96.343 / 96.216, so the two
bases are the same box.

The curve is **not monotone and not a thread-utilisation curve**: nwarps 1, 2 and
4 all divide the 8 super-blocks of a K=2048 Q4_K row exactly, and all three lose
to 3, which leaves a third of its last iteration idle. What moves is occupancy
and the LDS cross-warp reduction, which is why the value has to be swept on the
part rather than reasoned out — and why 8, upstream's RDNA4 default, is the worst
of the seven.

Note the `small_k` trigger is K-dependent, so at nwarps=3 it reshapes only the
K=512 down projection and leaves the K=2048 gate/up on `rows_per_block = 1`. It
is worth +2.0% there and +0.5% at nwarps=5; at nwarps=4 it is −0.8%.

**Confirmation, on the exact shipped binary** (3 arms, 6 boots, and again in a
5-arm round):

| arm | slot-lo | slot-hi | decode | spread | prefill | vs stock |
|---|---|---|---|---|---|---|
| stock | 86.808 | 86.250 | 86.529 | 0.644% | 2033.40 | — |
| 5-warp record | 96.349 | 96.084 | 96.216 | 0.275% | 2043.97 | +11.20% |
| **shipped** | 102.422 | 102.357 | **102.389** | 0.063% | 2262.96 | **+18.33%** |
| stock | 86.628 | 85.804 | 86.216 | 0.956% | 2055.36 | — |
| **shipped** | 102.336 | 102.534 | **102.435** | 0.193% | 2286.94 | **+18.81%** |

Prefill moves with it, 2043.97 -> 2262.96 t/s (+10.7%): a 522-token prompt is two
ubatches and the 10-token tail runs the matvec path.

## Accuracy gate

`llama-perplexity` on the **exact shipped binary**, three separate loads per arm,
with the control re-measured on every load — the gate is prefill-shaped, a
greedy-decode bit-exactness check does not gate it, and re-measuring the control
is what separates a flaky gate from a wrong patch.

| load | stock (control) | shipped |
|---|---|---|
| 1 | 5.1589 | 5.1589 |
| 2 | 5.1589 | 5.1589 |
| 3 | 5.1589 | 5.1589 |

Long corpus `-c 32768 --chunks 1`: **7.5415** both. Relative delta **0.000%**
against a 0.1% bound, control stable to four decimals across all three loads.

Both hunks reassociate the matvec reduction, so the greedy continuation moves —
as it did for the 8->5 record this supersedes. Every arm is deterministic across
its two boots: identical text hash at both slots, for all twelve arms measured.

Worth stating plainly for whoever tunes this next: a 512-token prefill dispatches
MMQ, not MMVQ, so **perplexity is structurally blind to a matvec block-shape
change on this track**. A clean perplexity here is a necessary check, not
evidence that the kernel is exact.

## Verification

The measured tree is byte-identical to a fresh checkout of the pin
(`2b63e0610`) with this patch applied: `git clone`, `git checkout 2b63e0610`,
`git apply`, `diff -r -x build -x .git` → no differences.

Every arm's `libggml-hip.so` was md5-summed before measuring. That check earned
its keep: a first attempt produced six "variants" that were all byte-identical to
the record, because a `cp -a`'d build directory carries
`CMAKE_HOME_DIRECTORY` pointing at the original tree and `cmake --build` on the
copy silently compiles the original's sources.

## Speculation on this track

See the speculative board. The GGUF carries its own NextN head at `blk.40`, but
each extra verified position costs ~10% of a full pass because every token routes
to its own 8 of 256 experts, so depth stays at 1 and speculation is a net loss
against leaving it off. This patch widens that gap rather than narrowing it: it
makes the non-speculative step cheaper without making the node price any lower.
