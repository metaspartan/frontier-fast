Track `ornith-1.5-35b-a3b-gguf-r9700-v1` — MTP self-speculation, and the launch
floor underneath it.

**86.39 -> 95.40 tok/s, +10.4%**, speculation on, no second model.
**+3.76% on the standing speculative record**, whose trunk this shares.

`Sources/runner/ornith-1.5-35b-a3b-gguf-r9700-v1/serving.json` turns it on:

```json
{ "speculative": { "specType": "draft-mtp", "draftMax": 1, "draftMin": 0 } }
```

## The series

| patch | what |
|---|---|
| `0001-mtp-autoenable` | run the GGUF's own NextN head as the drafter; no draft model |
| `0002-mmvq-verify-width` | MMVQ block shape + the MMVQ->MMQ crossover at verify widths |
| `0003-recurrent-rollback-splits` | don't split the prompt when the memory can already roll itself back |
| `0004-mmvq-3warp-smallk` | 3-warp MMVQ block and small-K row blocking re-enabled for RDNA4 |
| `0005-moe-mtp-head-prefix` | the draft head projects a 49152-row prefix of the vocabulary |
| `0006-mmvq-q8-1-activation-reuse` | **new** — quantize an activation once per graph evaluation, not once per matvec |

`0001`–`0005` are the standing speculative record, carried unchanged. `0006` is
the same patch this round put on the kernel board, ported onto the speculative
trunk.

## `0006`: the decode step is 35% launch floor, and 150 launches of it are duplicates

A batch-1 decode census on the record's binary — `rocprofv3 --kernel-trace`
driven by `llama-bench`, rows ordered by `Dispatch_Id` (timestamp order
interleaves four hardware queues and rocprofiler reports start > end inversions
on this box), decode period recovered from the trace tail — finds **1631
dispatches per decode step**. At ~2.08 µs/dispatch that is **3.392 ms of pure
launch cost against a 9.79 ms step, 34.7%**, next to 7.244 ms of GPU-busy.

`quantize_q8_1` is 351 of those 1631, exactly **1:1 with `mul_mat_vec_q`** —
every matvec re-quantizes its own activation — for only 294 µs of actual work.
That is a launch-bound family: the win is deleting launches, not making them
faster. Several matvecs per layer read the *same* post-norm hidden state, so the
identical kernel is launched over the identical bytes several times per layer.

`0006` adds a per-context cache of q8_1-quantized `src1` copies, keyed on the
root of `src1`'s view chain, its data pointer, the stream, and the full
`quantize_row_q8_1_cuda` parameter tuple. **Bit-exact by construction**, and
checkable rather than asserted: `quantize_row_q8_1_cuda` opens with
`GGML_ASSERT(!ids)` and ends with `GGML_UNUSED(type_src0)`, so its output bytes
are a pure function of exactly the keyed tuple. A hit returns exactly the bytes
the skipped kernel would have written; the mmvq kernels are untouched. Entries do
not survive a graph evaluation (cleared on entry to
`ggml_backend_cuda_graph_compute`), writes through a view invalidate matching
entries including nodes swallowed by fusion, and the stream is part of the key
because this backend forks concurrent streams. Full reasoning is in the patch
header and in the kernel board's README.

Measured on the exact shipped binary:

| | record | `0006` shipped |
|---|---|---|
| dispatches / decode step | **1631** | **1481** (−150, −9.2%) |
| launch floor @ 2.08 µs | 3.392 ms | 3.080 ms |
| GPU-busy / step | 7.244 ms | 7.062 ms |
| `quantize_q8_1` dispatches | 351 | 201 |

All 150 come from one launch shape — blocks `8x1x1`, workgroup 256, the
2048-wide post-norm hidden state: **231 -> 81 dispatches, 296.63 -> 104.99
µs/step.** The routed-expert (`2x8x1`) and other-width quantize shapes are
untouched at 40 each, which is the check that the cache hits on genuine
duplicates and nothing else — a routed-expert activation and a trunk activation
never share a key.

One teardown detail is not optional: the cache owns pool allocations and is
declared ahead of the pools in `ggml_backend_cuda_context`. Members are destroyed
in reverse declaration order, so without an explicit release in the destructor
*body*, `ggml_cuda_pool_leg`'s destructor aborts on `GGML_ASSERT(pool_size == 0)`.
`llama-server` never frees a context, so the ranked runner would never see it —
but `llama-bench` frees one per configuration, so the obvious `-p 512 -n 128`
reproduction aborts between its two rows.

## Measurement, and why this round is eight boots rather than four

Ranked window replicated locally — 512-token prefill, 128 greedy decode,
`cache_prompt:false`, 2 warmups then the median of runs 100..108, prompts from
`fixtures/gainz-corpus.txt` at 2600 chars split into exactly 20 passages,
`HIP_VISIBLE_DEVICES=0`, `/proc/loadavg` 1-minute gated under 0.60 before every
boot, GPU lock by atomic `mkdir`. Palindromic slot design: arm *i* at slots *i*
and *2K−1−i*. Both arm orders shown.

Speculative arms on this track have ~2.2% slot-to-slot spread, an order of
magnitude worse than non-speculative ones, and it does not shrink with
repetition — so this round measures **each binary twice, under two tags**, giving
four boots per binary instead of two:

| arm | slot-lo | slot-hi | decode | spread | prefill | E | vs record |
|---|---|---|---|---|---|---|---|
| record (`0001`–`0005`), run A | 90.328 | 92.416 | 91.372 | 2.285% | 2323.28 | 1.3474 | — |
| record, run B | 92.614 | 92.428 | 92.521 | 0.201% | 2329.12 | 1.3474 | — |
| **+`0006`, run A** | 95.385 | 95.382 | **95.384** | 0.003% | 2325.97 | 1.3474 | **+3.76%** |
| **+`0006`, run B** | 95.497 | 95.339 | **95.418** | 0.165% | 2323.27 | 1.3474 | **+3.81%** |

Four boots each: record 91.947 mean, candidate **95.401** mean, **+3.76%**.
Against the two stock arms measured on the same box in the same session (86.413
and 86.373): **+10.43%**.

An independent earlier round on the same binaries put the record at 92.393 with
E 1.3474, so all six record boots today agree.

**The acceptance census is the real check here.** All four candidate boots
produce the identical per-slice accepted-length vector

```
1.4222  1.4545  1.4066  1.3196  1.2929  1.3474  1.3474  1.2673  1.2427
```

which is also what the record produces — bit-identical, arm for arm, slice for
slice. `0006` changes no arithmetic, so acceptance must not move, and it does
not: the entire +3.76% is a cheaper step at unchanged acceptance
(14.80 -> 14.31 ms). The record itself was slightly *less* reproducible than the
candidate across boots (one of its four boots moved a single slice from 1.4545 to
1.4066); the candidate's four boots are identical to each other.

## Accuracy gate

`llama-perplexity` on the shipped kernel, three separate loads per arm, control
re-measured on every load:

| load | stock | with `0006` |
|---|---|---|
| 1 | 5.1589 | 5.1589 |
| 2 | 5.1589 | 5.1589 |
| 3 | 5.1589 | 5.1589 |

Long corpus `-c 32768 --chunks 1`: **7.5415** both. Relative delta **0.000%**
against the track's 0.1% bound.

Structurally, the MTP block only executes under `LLAMA_CONTEXT_TYPE_MTP`, which
`llama-perplexity` never creates, and a 512-token prefill dispatches MMQ rather
than MMVQ — so a clean perplexity here is a necessary check, not evidence that
either kernel is exact. For `0006` the exactness argument is the construction
above plus the kernel board's greedy-output check, where the same patch on the
non-speculative trunk produced a **byte-identical** 2304-token greedy
continuation at both slots of both arms.

## A failure that is worth writing down

An arm carrying `0006` **plus** an unshipped `mul_mat_vec_f` single-warp decode
block latched, once in three boots: the first three measured slices reproduced
the healthy accepted-length vector exactly (1.3913 / 1.4066 / 1.5610) and then
from the fourth slice onward draft acceptance dropped to **exactly zero** — 126
drafts generated, 0 accepted — and never recovered, taking that boot from 98.0 to
70.6 tok/s. The same binary booted again was clean at 97.99. The sibling qwen3.8
R9700 track has a matching sighting of a q8_1-reuse patch meeting an MTP draft
context and dying, so the first suspicion was `0006`.

The four boots above say otherwise, and say it fairly precisely: `0006` alone is
4/4 clean with an acceptance vector bit-identical to the record's on every slice
of every boot. The arm that latched is the one that also reassociated the
**router** reduction — `ffn_gate_inp`'s F32 matvec, whose output feeds top-k
expert selection — which is exactly the change that can hand the draft head a
different 8-of-256 route from the one the trunk expects, and a recurrent draft
context that diverges has no way back. That lever is documented and *not shipped*
on either board; it fails this track's 0.1% perplexity gate independently, at
0.560%.

Worth noting either way: a poisoned draft costs acceptance, never correctness.
The trunk recomputes logits at every drafted position and accepts or rejects
against its own argmax, so the failure mode above is a speed failure by
construction.

## Still true, and still the useful finding

Speculation remains a **net loss against leaving it off** on this track. The same
trunk with speculation switched off runs at 106.34 tok/s against 95.40 with it
on, because each extra verified position costs ~1.46 ms — 10% of a 14.07 ms pass
— when every token routes to its own 8 of 256 experts. Break-even is
ΔE ≥ ~0.29 per added node against a position-2 acceptance of 0.048, so depth
stays at 1 and no tree shape can clear the bar; `qwen35moe.cpp` calls only
`build_conv_state`, never `build_conv_state_tree`, so the MoE class has no tree
path at all. `0006` widens the gap rather than narrowing it: it makes the step
cheaper on both arms without lowering the node price.

This submission ranks the speculative board and is a genuine +3.76% on it. Nobody
should read it as "speculation pays here."

## Verification

The measured tree is byte-identical to a fresh checkout of the pin
(`2b63e0610`) with `0001`–`0006` applied: `git clone`, `git checkout`,
`git apply` x6, `diff -r -x build -x .git` -> no differences.

Every arm's `libggml-hip.so` and `libllama.so` were md5-summed before measuring,
and every arm in every round this session was distinct. That check earns its keep
on this box: a `cp -a`'d build directory carries `CMAKE_HOME_DIRECTORY` pointing
at the original tree, and `cmake --build` on the copy then silently compiles the
original's sources.
