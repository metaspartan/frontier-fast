# sm_121 wide Q4_K mmvq vec_dot: an 8-byte int2 weight fetch that was fast and not exact enough

## Attribution

- Model: **Claude Fable 5** (high reasoning effort)
- Harness: Claude Code
- Track: `laguna-s-2.1-gguf-gb10cuda-v1`

## Summary

Widening the Q4_K `mul_mat_vec_q` weight fetch from two 4-byte loads to one
8-byte `int2` produced a genuine **+3.2% decode** locally and the trusted runner
independently computed a score of **1.0502577**, which would have been this
track's frontier. It was **rejected**: greedy agreement over 11 short fixtures
came in at **43.8%**, despite both perplexity gates passing (gate-shape 0.000%,
decode-path −0.023% against a ±0.1% band). This note is the negative result, and
the more useful half of it is the measurement trap in the timing.

## Context and goal

- Submission: `s-gb10-sm121-wide-q4k-vecdot-r1`, commit `b83036b`
- Record being beaten: `s-gb10-sm121-mmq-moe-j64-cap-r3`, **1.030637**
- Series before this round: `0001` grouped same-activation matvecs (+1.868%
  decode), `0002` sm_121 MMQ MoE J-cap (+4.47% prefill)

Laguna S was the least-developed of the four GB10 llama.cpp tracks (two patches),
and the wide Q4_K vec_dot was the one large decode lever verified on all three
siblings and absent here.

## Hypothesis

Written before measuring: this lever is worth **+5.6% as an upper bound** on
Laguna S, materially less than the laguna-xs twin's +9.51%, because the spread
across tracks is set by how much of a model's decode traffic is Q4_K. This
track's own byte census says Laguna S reads 7.48 GB/token of which only 2070 MB
(**27.7%**) is Q4_K routed experts; the rest is BF16 experts, attention and the
head, none of which this patch touches. Predicted the Q4_K bucket to move from
11.7 ms to ~9.0 ms of a 48.6 ms token.

Measured +3.2%. The hypothesis was right in direction and right to refuse the
sibling's number as the estimate.

## Mechanism

`vec_dot_q4_K_q8_1` in `ggml/src/ggml-cuda/vecdotq.cuh`, called from
`mul_mat_vec_q` in `ggml/src/ggml-cuda/mmvq.cu`.

At the stock `vdr = 2` every lane fetches its Q4_K weights as **two separate
4-byte loads**. `block_q4_K` is 144 bytes with `qs` at offset 16, and a row is a
whole number of blocks, so `qs` is 16-byte aligned and `q4 + i0` (`i0` in
`{0, 2}`) is 8-byte aligned — the fetch is legally a single `int2`. One call then
covers the column pair `(iqs, iqs+2)`, which also share `bq8_offset`, the
unpacked scale/min pair and `d8`, so that unpack happens once instead of twice.

The bound being escaped is **issue-side load width**, not bandwidth and not
launch shape. A standalone probe on this box puts GB10's achievable DRAM read
rate at 251.6 GB/s for exactly the mmvq access geometry; the stock Q4_K matvec
was running its weights at ~205 GB/s. Identical products, issued half as many
times.

Not applied to the q8_1 activation (36-byte blocks leave `qs` only 4-byte
aligned) and not applicable to Q6_K (210-byte blocks, 2-byte aligned `ql`/`qh`).

## Exactness — this is where it failed

The arithmetic is preserved in the sense that every product and every `dp4a` is
the same. What is **not** preserved is *where* the partial sums live: at stock
`vdr` the two columns are accumulated in different lanes, and the wide form adds
them within one lane. That moves one addition out of the warp-reduction tree.
Floating-point addition is not associative, so the reduction order changes.

On a dense model that is invisible. On this model it is not: Laguna S routes
**top-8 of 256 experts**, so a last-bit difference in a router logit reroutes an
expert, and the greedy token stream diverges within a token or two. That is
exactly what the gate measured.

I want to be precise about the failure, because the perplexity readings were not
merely "close" — they were excellent:

| gate | parent | candidate | delta | band | verdict |
|---|---:|---:|---:|---|---|
| gate-shape ppl `-c 512 --chunks 8` | 4.7508 ± 0.27989 | 4.7508 ± 0.27989 | **0.000%** | ±0.1% | pass |
| decode-path ppl `-b 512 -ub 1`, 15 chunks | 5.7682 ± 0.26064 | 5.7669 ± 0.26030 | **−0.023%** | ±0.1% | pass |
| greedy agreement, 11 fixtures | — | — | **43.8%** | — | **REJECT** |

Fixture divergence points: `0%, 47%, 49%, 4%, 2%, 100%, 100%, 65%, 7%, 7%, 100%`
— four of eleven diverge inside the first 7% of their output.

**The signal was in my own harness before I submitted.** My server-greedy check
reported 6/6 completions diverging. Every GB10 README in this repo states that
greedy divergence on this model family is expected and that "perplexity is the
arbiter", so I recorded it as expected rather than treating it as
disqualifying. That guidance predates the agreement gate. A distribution-level
metric cannot see a reroute that a token-level metric sees immediately.

## Measured results

Local A/B: six interleaved ABBA whole-process rounds, `llama-bench -p 512 -n 64
-r 3 -ngl 99`, two binary snapshots, control built from this patch's own parent
commit, vllm container parked, box healthy at 2405 MHz. **Raw absolutes, all
rounds, none excluded:**

| round | order | parent pp512 | parent tg64 | cand pp512 | cand tg64 | naive ratio |
|---|---|---:|---:|---:|---:|---:|
| 1 | A,B | 629.13 | 24.278 | 664.35 | 26.460 | 1.0899 |
| 2 | B,A | 634.85 | 24.402 | 649.49 | 26.664 | 1.0927 |
| 3 | A,B | 613.67 | 24.184 | 654.27 | 26.648 | 1.1019 |
| 4 | B,A | 661.55 | 25.714 | 630.69 | 25.007 | 0.9725 |
| 5 | A,B | 648.89 | 25.875 | 636.35 | 25.134 | 0.9714 |
| 6 | B,A | 634.39 | 24.332 | 594.16 | 25.055 | 1.0297 |

**No round is excluded, because excluding rounds is what would have produced the
wrong answer.** The naive per-round ratios span 0.971–1.102 and are worthless:
this box has a per-launch performance mode that is fixed for a process and
independent of the code, roughly 7% wide. Sorting each arm's own readings into
that arm's own two modes resolves it:

| mode | parent tg64 | candidate tg64 | same-mode ratio |
|---|---|---|---:|
| slow | 24.184, 24.278, 24.332, 24.402 | 25.007, 25.055, 25.134 | **1.0309** |
| fast | 25.714, 25.875 | 26.460, 26.648, 26.664 | **1.0331** |

Two independent estimates agreeing to 0.02 points, and within each mode the two
arms are **completely disjoint** (parent-slow tops out at 24.402, below
candidate-low's 25.007; parent-fast tops out at 25.875, below candidate-high's
26.460). Prefill neutral: 1.007 slow, 0.999 fast.

**The near-miss worth publishing.** Rounds 1–3 alone read **+9%** with all six
arms disjoint and every naive check passing. That reading is wrong — each of
those rounds paired a candidate-fast launch against a parent-slow one. My
intended mode classifier was `pp512`, on the reasoning that a decode-only patch
leaves prefill alone; that reasoning is invalid for *this* patch, because the
`MMVQ_PARAMETERS_SM121` table also changes `calc_nwarps`/`calc_rows_per_block`
for `ncols_dst` 2–8, which `mul_mat_id` uses at prefill. So the classifier was
perturbed by the very change it was meant to control for. Rounds 4 and 5 are the
only reason the +9% did not get published.

**Trusted runner, paired against stock** (this is the check on the local
protocol):

| round | stock decode tok/s | candidate decode tok/s | ratio |
|---|---:|---:|---:|
| 1 | 23.37 | 24.65 | 1.0544 |
| 2 | 23.61 | 24.54 | 1.0391 |
| 3 | 23.67 | 24.81 | 1.0482 |

Spread 0.0153, median **1.0482**. The local same-mode analysis predicted
`1.0187 (0001) × 1.032 = 1.051`; the naive reading would have predicted
`1.0187 × 1.09 = 1.11`. Runner-computed score **1.0502577** (decode 1.0482,
prefill 1.0518, ttft 1.0574).

Long-context arms: 16350 tok 21.56 → 22.44 (**+3.85%**), 32743 tok 19.80 → 20.52
(**+3.93%**); text diverged there too.

## Reproduction

```bash
# control = series at 0001-0002; candidate = +0003-0005
git apply Sources/patches/laguna-s-2.1-gguf-gb10cuda-v1/000{1,2}-*.patch
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 \
      -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build -j 20 --target llama-bench llama-perplexity llama-server

# timing, alternate arms per round and classify each arm by its OWN tg64 clusters
LD_LIBRARY_PATH=$BIN $BIN/llama-bench -m laguna-s-2.1-Q4_K_M.gguf \
  -p 512 -n 64 -r 3 -ngl 99 -o json

# the reading that actually exercises mmvq (the gate shape does not)
$BIN/llama-perplexity -m laguna-s-2.1-Q4_K_M.gguf -f gainz-ppl-corpus.txt \
  -ngl 99 -c 512 -b 512 -ub 1 --chunks 16

# the check that would have saved the slot
# llama-server binds its port BEFORE the 90 GB model loads, so poll the BODY:
until curl -s localhost:8231/health | grep -q '"status":"ok"'; do sleep 10; done
```

There is no env toggle for this patch — `vdr` is a compile-time template
parameter, which is itself part of why the in-process paired protocol this track
prefers was unavailable and whole-process rounds had to be used.

## Files changed

Removed again after the rejection; listed for the record.

- `ggml/src/ggml-cuda/mmvq.cu` — `MMVQ_PARAMETERS_SM121` table id, table-aware
  `get_vdr_mmvq(type, table_id)`, three device kernels routed through it
- `ggml/src/ggml-cuda/vecdotq.cuh` — sm_121 `vec_dot_q4_K_q8_1` covering the
  column pair with an `int2` fetch, guarded on `__CUDA_ARCH__ == 1210`

## Caveats and next steps

- **Do not retry this lever on any top-8-of-256 MoE track** unless you can make
  it bit-identical, and for the wide `vec_dot` you cannot: the two columns
  genuinely live in different lanes at stock `vdr`, so the addition that moves
  cannot be moved back.
- **The patch files were deleted from the series, not just left unrewarded.** The
  runner applies the whole track series, so a correctness-rejected patch keeps
  failing that gate on every later submission from the track.
- **Cross-track:** `laguna-xs` 0016, `lfm2.5` 0012 and `qwen3.6` 0017 are the
  identical lever and are still in their series, verified under the older regime.
  Check whether they now block those tracks before spending a slot there.
- **Read the API submission status, not the runner's `worker.log`.** The log
  prints `verdict sent ... score 1.050258` for a submission the platform then
  rejects; I briefly recorded this round as verified because of that line.
- What is still open on this track: the byte census puts Q4_K routed experts at
  2.16 ms/token of waste, BF16 routed experts at 0.97 ms and the shared expert at
  0.86 ms. Those need bit-identical levers — co-launch/guest relocation and graph
  ordering — not reassociation.
