# laguna-s-2.1-gguf-gb10cuda-v1 — patch series

Applied in order against pinned llama.cpp **b10237**
(`2b63e0610bbc2be990ae1360d5256efcdc3f9efb`).

| # | Patch | Measured on this track |
| --- | --- | --- |
| 0001 | `cuda-mmvq-group-same-activation-matvecs` | **+1.868% decode** (in-process paired A/B, 12 rotated cycles, no-op floor 0.99994) |
| 0002 | `sm121-mmq-moe-j64-cap` | **+4.47% prefill** (median of same-mode interleaved toggle rounds; decode neutral — the cap is MMQ/prefill-only) |
| 0003 | `llama-absorb-trailing-ubatch` | **INERT on this harness** — verified 1.0313 but the +0.07% was noise; llama-server pre-chunks the prompt so it never fires. See the correction below. |

The verified frontier on this track is **1.030637** (0001+0002). Stock baseline
is 23.627 tok/s decode, 343.1 tok/s prefill, 0.66 s TTFT.

## Measured-neutral and removed: the attention-gate graph-order pin

The R9700 laguna-xs twin's `0020` is worth **+1.81% decode, bit-identical**
there. Ported here it is **not resolvable at six rounds** and was removed. It is
documented rather than deleted because the *reason* is a fact about this model,
not about the lever.

**It does fire, and it is exactly as exact as advertised.** Same-binary
`GGML_MMVQ_GROUP_STATS`: grouped-mmvq goes `groups=93 members=233` →
`groups=95 members=238`, with `n_nodes` unchanged at **3806** (a pure reorder),
and `GGML_LAGUNA_QKV_GATE_PIN=0` on the candidate binary returns exactly
`groups=93 members=233`. Greedy output is **byte-exact** across three arms
(parent, candidate pin-on, candidate pin-off — 3325 bytes each).

**But it only recruits 5 of 233 members.** On the R9700 twin the gate joins in
all 40 attention layers; here it joins in a couple. One
`ggml_build_forward_expand` is evidently *necessary but not sufficient* on this
model — most layers' gates are still refused by the hoist-legality check for a
reason the reorder does not address. **That diagnosis, not the timing, is what
the next agent should pick up**: find why the other ~40 gates still fail, and
the lever may be worth its R9700 value.

Six interleaved ABBA rounds, control = 0001+0002, candidate = +pin:

| round | order | parent pp512 | parent tg64 | cand pp512 | cand tg64 | naive ratio |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | A,B | 639.98 | 24.608 | 638.11 | 24.437 | 0.9931 |
| 2 | B,A | 600.33 | 24.138 | 630.56 | 24.350 | 1.0088 |
| 3 | A,B | 657.98 | 25.877 | 633.78 | 24.477 | 0.9459 |
| 4 | B,A | 633.38 | 23.512 | 635.06 | 24.197 | 1.0291 |
| 5 | A,B | 639.29 | 24.143 | 648.23 | 25.859 | 1.0711 |
| 6 | B,A | 641.11 | 24.563 | 644.24 | 24.387 | 0.9928 |

Same-mode: slow (n=5 each) parent 24.143 vs cand 24.387 = **1.0101**; fast
(n=1 each) 25.877 vs 25.859 = **0.9993**. Unlike 0003–0005, **the arms overlap
within the mode** (parent slow spans 23.512–24.608, candidate 24.197–24.477),
and the +1.01% rests entirely on the single parent outlier at 23.512 — drop it
and the ratio is **1.0014**. That is not a result. Bounded somewhere in
−0.1%…+1%, below what six whole-process rounds resolve on this track.

Not submitted: a submission that does not beat the current best is auto-rejected
("score did not improve current best"), and there is no measured gain here to
put against a runner slot.

# CORRECTION (2026-08-09): 0003 IS INERT HERE — llama-server pre-chunks the prompt

`0003` was submitted and **verified at 1.0313** (from 1.030637), and that
+0.07% is **noise, not the patch**. Read this before building on any prefill
number in the section below it.

**The patch cannot fire on the ranked path.** `llama_context::decode` chooses the
effective ubatch size from the batch it is handed — but `llama-server` splits the
prompt *before* calling `llama_decode`. Instrumented on the server path, a
537-token ranked prompt arrives as **three separate decode calls of 21 + 512 + 4**,
so `llama_decode` never sees more than `n_ubatch` and the absorption test
(`n_tokens > n_ubatch`) is never true. It fires only for callers that hand
`llama_decode` a whole batch — `llama-bench` does, which is exactly why the local
`pp534` measurement showed +21% and the ranked run showed nothing.

**The scored prefill is also not a slope on this runner.** `rocm-worker` takes
`body.timings.prompt_per_second` from a **single cache-cold `/completion`
request**, median of 9 runs at indices 100–108, with the prompt sized in
**characters** (`GAINZ_PROMPT_CHARS = 2600`). So `prompt_n` varies run to run
(measured: 524, 544, 562, 551, 541, 547, 536, **511**, 536) and there is no
`(t(534) − t(84))/451` anywhere in this harness. The `+38.4% slope` figure was
computed with a formula this runner does not use, on a code path that does not
run.

**The control that settles it was already in the data.** Run index 107 yields a
**511-token** prompt — under `n_ubatch`, so absorption is *structurally
impossible* on it. Exact ranked-shape replica, both arms:

| prompt_n | absorb=0 | absorb=1 | ratio | can absorb? |
| --- | --- | --- | --- | --- |
| 524 | 484.6 | 497.0 | 1.0256 | yes |
| 544 | 466.1 | 478.7 | 1.0270 | yes |
| 562 | 462.3 | 468.8 | 1.0141 | yes |
| 551 | 465.5 | 476.3 | 1.0232 | yes |
| 541 | 472.1 | 481.7 | 1.0203 | yes |
| 547 | 475.5 | 485.4 | 1.0208 | yes |
| 536 | 480.3 | 487.5 | 1.0150 | yes |
| **511** | **545.8** | **555.8** | **1.0183** | **NO** |
| 536 | 473.4 | 482.4 | 1.0190 | yes |

The impossible run moved **+1.83%**, against a +2.06% mean on the runs where
absorption "could" fire. The two server boots drew different launch modes
(decode 23.98 vs 25.36), and that drift *is* the entire difference.

**But the prize is real and it is bigger than the tail estimate.** That same
table shows the 511-token prompt — the only one processed in a single decode
call — running at **545.8 tok/s against ~472.5 for every prompt that gets
split: +15.5%**, sitting on the ranked path today. The lever is right; the layer
was wrong. **It has to be taken in `tools/server/server-context.cpp`, whose
prompt loop bounds each chunk by `n_ubatch`**, not in `llama_context`.

### And the split is NOT `n_ubatch` chunking — it is context checkpointing

Chasing the server-side fix found the real cause, and it closes the follow-up
rather than opening it. `tools/server/server-context.cpp` deliberately breaks the
prompt at `checkpoint_offsets = {4 + n_ubatch, 4}` counted **from the end** of the
prompt — *"process the last few tokens of the prompt separately in order to allow
for a checkpoint to be created"* (ref llama.cpp PR 20288). At `n_ubatch = 512`
those offsets are 516 and 4, which reproduces every observed split exactly:

| prompt_n | decode calls | |
| --- | --- | --- |
| 511 | **2** | 507 + 4 (under the 516 offset) |
| 524 | 3 | 8 + 512 + 4 |
| 537 | 3 | 21 + 512 + 4 |
| 562 | 3 | 46 + 512 + 4 |

The accumulation loop is bounded by `n_batch` (2048), **not** `n_ubatch` — the
earlier reading in this README was wrong. And `do_checkpoint` keys off
`params_base.n_ctx_checkpoints > 0`, default **32**, a *server startup* parameter
(`--ctx-checkpoints`) that the runner does not pass and that is **not**
conditioned on the request's `cache_prompt`. So the ranked path builds context
checkpoints on every cache-cold request that can never be reused.

**Size of the prize, controlling for prompt length.** Within the eight ranked
prompts that get three decode calls, rate falls linearly with length at
0.59 tok/s per token (`rate = 793.2 − 0.591·n`), which extrapolates to
**491.2 tok/s** at n=511. The one prompt that gets two calls measures **545.8**.
So one fewer decode call is worth **+11.1%** prefill — and the earlier headline
"+15.5%" was partly a length effect, now corrected.

**Not built.** Suppressing checkpoint creation *is* the prompt-cache-checkpoint
removal already declined as harness-fitting, and there is no per-request gate to
hook. The general form — skip checkpoint creation when the request sets
`cache_prompt: false`, since those checkpoints can never be reused — is arguably
a real fix, but it is the same category and needs a ruling before anyone spends a
slot on it.

**The mechanism is worth carrying to every llama.cpp server track:** on a
sparse-MoE model every extra `llama_decode` call costs a full routed-expert
sweep, and the server splits prompts by default at `4 + n_ubatch` and `4` tokens
from the end.

Two rules this cost a slot to learn:

1. **`llama-bench pp<N>` is not the ranked measurement.** The ranked path is a
   `llama-server` request measuring `prompt_per_second`; llama-bench hands
   `llama_decode` the whole batch while the server pre-chunks it. Price from a
   server request or not at all.
2. **Always include a run whose prompt is under `n_ubatch`.** Absorption cannot
   fire on it, so anything it moves is drift — a free built-in control.

# THE PREFILL SCORE IS 18% TAIL UBATCH — read this before profiling prefill again

Nobody had censused GB10 prefill as the **slope** the score actually uses. Doing
so finds the largest single structural fact on this track, and it is not in any
kernel.

Scored prefill is `(t(534) − t(84)) / 451`. The runner boots
`llama-server -m … -ngl 99 -c 8192 --parallel 1` with **no `-b`/`-ub` flag**
(`SERVER_FLAGS` in the worker), so llama.cpp's defaults apply and a 534-token
prompt is split by `n_ubatch = 512` into **512 + 22**.

`llama-bench -p 84,300,534 -n 0 -r 1` on the 0001+0002 control, nsys trace
segmented per ubatch (segment spans reproduce the reported t/s to 0.1 ms):

| | launches | time | per token |
| --- | --- | --- | --- |
| ubatch A (512 tok) | 4105 | 801.1 ms | **1.56 ms** |
| ubatch B (22 tok, the tail) | 2453 | **175.1 ms** | **7.96 ms** |

**The 22-token tail is 17.9% of t(534) for 4.1% of the tokens — a 5.09x
per-token penalty.** It issues 2453 launches, 60% as many as the 512-token
ubatch, because on a 256-expert MoE the launch structure is set by experts and
layers, not by tokens. Its cost is `mul_mat_q` 109.5 ms + `mul_mat_f_ids`
(BF16 experts) 51.0 ms = 160 of its 175 ms, and that is a **full sweep of the
expert weights** — ~30 GB at this box's ~200 GB/s — performed for 22 tokens.

**No kernel change can recover it.** It is bandwidth, not tiling: the tail must
read the experts it routes to whatever the tile geometry is. The only way to
not pay a second sweep is to not have a second ubatch.

### Measured: the whole thing is reachable from `n_ubatch`

`llama-bench -p 84,534 -n 32 -r 2 -ub 512,1024`, one load, same binary:

| n_ubatch | pp84 | pp534 | tg32 |
| --- | --- | --- | --- |
| 512 (default) | 251.92 | 561.42 | 25.98 |
| 1024 | 248.85 | **691.49** | 25.67 |

Converted to what is scored:

| | 512 | 1024 | |
| --- | --- | --- | --- |
| t(84) | 333.4 ms | 337.6 ms | |
| t(534) | 951.2 ms | 772.2 ms | |
| **scored prefill slope** | 730.1 tok/s | **1037.5 tok/s** | **+42.1%** |
| decode tg32 | 25.98 | 25.67 | −1.2% |
| ttft ≈ t(534) | | | +23.2% |

`0.9881^0.65 × 1.4211^0.20 × 1.2317^0.15` = **1.0983**, i.e. this track's
1.0306 would become **≈1.132**. pp84 is one ubatch either way and is unchanged,
which is the control that says the effect is the tail and nothing else.

### SHIPPED as 0003 — adaptive absorption, not the flag

Both preconditions below cleared and the patch is `0003`. It does **not** raise
`n_ubatch`: `llama_context::decode` picks the ubatch size for the batch before
handing it to `init_batch`, and runs the batch as one ubatch when it overhangs
`n_ubatch` by at most `min(n_batch, n_ubatch + n_ubatch/8)`. Both
`graph_reserve` sites are sized at the same bound.

**The first version never fired, and the reason is worth carrying forward.** It
put the absorption inside `split_simple`; these models reach the allocator
through `split_equal`. The symptom was a "measurement" that was pure launch-mode
lottery — the candidate happened to draw the slow mode and read 4% *worse*, with
pp84 (one ubatch under both arms) moving by the same 4%. **Choosing the ubatch
size before `init_batch` covers `split_simple`, `split_equal` and `split_seq`
at once.** Verify firing from the ubatch sizes the allocator emits
(`n_tokens = 534` on, `512` off) — a llama-bench timing cannot tell you.
`llama-bench` also swallows `LLAMA_LOG_INFO` unless you pass `-v`.

Measured, same binary toggled, three interleaved ABBA rounds, each arm
classified against its own tg64 clusters (stock drew slow 3/3; absorbed drew
slow 1, fast 2):

| same-mode (slow) | stock | absorbed | ratio |
| --- | --- | --- | --- |
| pp84 | 239.81 | 238.34 | 0.9939 |
| pp534 | 547.19 | 663.88 | 1.2133 |
| **scored prefill slope** | **720.9 tok/s** | **997.9 tok/s** | **1.3843** |
| tg64 | 24.414 | 24.300 | 0.9953 |

Score `0.9953^0.65 × 1.3843^0.20 × 1.2133^0.15` = **1.0953** → ≈**1.129**.

Preconditions, both cleared: decode is **−0.47%**, with every absorbed reading
inside the stock arm's own clusters; and `llama-server -ngl 99 -c 8192
--parallel 1` boots and listens with absorption on. Gate ppl is
**4.7508 ± 0.27989** on absorption-on, absorption-off and a separately built
control.

### The two preconditions as they stood before shipping

It is a change to a llama.cpp **default**, not a kernel, and it is shaped to the
fact that the scored prompt is 534. Three things must be settled first, and none
of them were in this session:

1. **Is it in the spirit of the track?** Raising `n_ubatch` removes a fixed
   second sweep; it does not improve marginal per-token throughput, which is
   what a slope is meant to isolate. The defensible form is *adaptive tail
   absorption* — when the trailing ubatch is a small fraction of `n_ubatch` and
   the batch still fits `n_batch`, grow the previous ubatch instead of issuing a
   sweep for it. That is a general MoE engine fix (any prompt slightly over
   `n_ubatch` pays this), not a 534-shaped constant. **Balanced splitting is
   NOT the answer and was checked: two 267-token ubatches pay two full sweeps,
   ~1140 ms against 976 ms — worse than 512+22.**
2. **The −1.2% decode is unresolved.** This track has a documented ~7%
   per-launch bimodality; `r=2` cannot see 1.2%. It must be re-measured with the
   in-process rotated protocol in this README before any of the +9.8% is
   believed. Note the score is still strongly positive even if the −1.2% is real.
3. **Memory on the ranked boot.** llama-bench used default context; the runner
   uses `-c 8192` on an 89.43 GiB model in 124.5 GiB. Doubling ubatch activation
   buffers must be shown not to OOM *there* before spending a slot.

Cross-track: laguna-xs, laguna-S and qwen3.6 gb10cuda are all MoE and all get
the same 512+22 split from the same flagless boot, so the same tail exists on
all three. lfm2.5 is dense — its tail costs almost nothing, which is consistent
with this being an expert-sweep effect.

# READ THIS FIRST: this track enforces GREEDY-OUTPUT AGREEMENT, not just perplexity

**A patch that passes both perplexity gates can still be rejected**, and one of
mine was. This is the single most expensive thing to learn here, so it is at the
top.

`s-gb10-sm121-wide-q4k-vecdot-r1` (the sm_121 wide Q4_K mmvq vec_dot, formerly
0003–0005 of this series) was **REJECTED**:

```
status        = rejected
statusReason  = correctness gate failed - agreement over 11 short fixtures: 43.8%
                (fixture divergence points: 0%, 47%, 49%, 4%, 2%, 100%, 100%,
                 65%, 7%, 7%, 100%)
measured      = score 1.0502577, decode 1.0482, prefill 1.0518, ttft 1.0574
```

The timing was real and excellent — the runner computed a **1.0502577** score,
which would have been this track's frontier. It was thrown away on correctness.

**The patches were removed from this series** (commit that follows this README
edit). They are not merely unrewarded: the runner applies the *whole* track
series, so leaving them in would have failed the agreement gate on every future
submission from this track regardless of what the new patch did.

### The stale guidance that caused it

Every GB10 README in this repo — including this one — carried some form of
"perplexity is the arbiter" and treated greedy divergence on a top-8-of-256 MoE
as *expected and acceptable*. The laguna-xs twin's 0016 section says so
explicitly ("1/6 byte-exact … Perplexity is the arbiter"), and that patch was
**verified** under the older regime. That is no longer true. Both readings can
be immaculate and the submission still dies:

| gate | this patch | verdict |
| --- | --- | --- |
| gate-shape ppl `-c 512 --chunks 8` | 4.7508 both arms, **0.000%** | pass |
| decode-path ppl `-b 512 -ub 1`, 15 chunks | 5.7682 → 5.7669, **−0.023%** | pass |
| greedy agreement, 11 fixtures | **43.8%** | **REJECT** |

The signal was in my own harness before I submitted: the server-greedy check
reported **6/6 completions diverging**. I read that against the READMEs' advice
instead of treating it as the gate it now is.

### What to do instead

**Treat bit-identity as the requirement on this track, not as a nice-to-have.**
Before spending a runner slot, ask whether the change moves any addition between
lanes or reorders any reduction. If it does, on a 256-expert top-8 router it will
reroute an expert within a token or two and agreement will collapse — 43.8% here,
with 4 of 11 fixtures diverging in the first 7% of their output.

A float-reassociation lever is now worth a slot only if you can make it exact.
For the wide `vec_dot` specifically you cannot: the two columns genuinely live in
different lanes at stock `vdr`, so the addition that moves cannot be moved back
(the lfm2.5 README's 0038-trick note explains why).

**Cross-track warning.** `laguna-xs` 0016, `lfm2.5` 0012 and `qwen3.6` 0017 are
the same lever and are all still in their series, verified under the older
regime. If the agreement gate applies to them as it did here, those series are
blocked for future submissions until the patch is dropped. Verify before
spending a slot on any of those tracks.

## The rejected 0003–0005: the sm_121 wide Q4_K mmvq vec_dot (+3.2% decode)

Kept as a record because the *timing* result and the measurement protocol are
sound and reusable; only the exactness was unacceptable. The patch files
themselves have been removed from this directory.

Port of the lever that is `0016` on laguna-xs gb10cuda (+9.51%), `0012` on
lfm2.5 gb10cuda (+7.76%) and `0017` on qwen3.6 gb10cuda (+1.77%). Stock
`vec_dot_q4_K_q8_1` runs at `vdr = 2`, so every lane fetches its weights as two
separate 4-byte loads. `block_q4_K` is 144 bytes with `qs` at offset 16 and a
row is a whole number of blocks, so `qs` is 16-byte aligned and the fetch can
legally become an 8-byte `int2`. The three patches are one unit: 0005 keys off
the `MMVQ_PARAMETERS_SM121` table id that 0003/0004 introduce.

**Why S gets +3.2% where XS gets +9.5%, and why that was predictable.** The
cross-track spread is set by how much of a model's decode is Q4_K. This
README's own byte census says Laguna S reads 7.48 GB/token of which only
2070 MB (27.7%) is Q4_K routed experts — the rest is BF16 experts, attention
and the head, none of which this patch touches. Sizing the lever from that
census predicts the Q4_K bucket moving 11.7 ms → ~9.0 ms of a 48.6 ms token,
i.e. +5.6% as an upper bound; +3.2% measured is the same story with the
shared-expert Q4_K share and the untouched dense path diluting it. **Do not
size this family from the XS number.**

### The measurement, and a correction to this track's protocol

Six interleaved ABBA whole-process rounds, `llama-bench -p 512 -n 64 -r 3`,
two binary snapshots (control = this series at 0001–0002, i.e. the patch's own
parent commit; candidate = +0003–0005):

| round | order | parent pp512 | parent tg64 | cand pp512 | cand tg64 | naive ratio |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | A,B | 629.13 | 24.278 | 664.35 | 26.460 | 1.0899 |
| 2 | B,A | 634.85 | 24.402 | 649.49 | 26.664 | 1.0927 |
| 3 | A,B | 613.67 | 24.184 | 654.27 | 26.648 | 1.1019 |
| 4 | B,A | 661.55 | 25.714 | 630.69 | 25.007 | 0.9725 |
| 5 | A,B | 648.89 | 25.875 | 636.35 | 25.134 | 0.9714 |
| 6 | B,A | 634.39 | 24.332 | 594.16 | 25.055 | 1.0297 |

The naive per-round ratios span **0.971 to 1.102** and are worthless. Sorting
each arm's own `tg64` readings into that arm's two launch modes resolves it
completely:

| | parent | candidate | same-mode ratio |
| --- | --- | --- | --- |
| slow mode | 24.184, 24.278, 24.332, 24.402 | 25.007, 25.055, 25.134 | **1.0309** |
| fast mode | 25.714, 25.875 | 26.460, 26.648, 26.664 | **1.0331** |

Two independent estimates agreeing to 0.02 points, and **within each mode the
two arms' readings are completely disjoint** (parent-slow tops out at 24.402
below candidate-low's 25.007; parent-fast tops out at 25.875 below
candidate-high's 26.460). Prefill is neutral: 633.5/629.1 = 1.007 in the slow
mode, 654.3/655.2 = 0.999 in the fast mode.

**The correction.** 0002's section proposed classifying launch mode by `pp512`.
That works for a decode-only patch but **not for this one, and the trap is
live**: `MMVQ_PARAMETERS_SM121` changes `calc_nwarps`/`calc_rows_per_block` for
`ncols_dst` 2–8, which `mul_mat_id` uses during prefill, so the candidate has
its own prefill behaviour and `pp512` is no longer an arm-independent mode
indicator. Reading rounds 1–3 that way would have published **+9%** — three
rounds, all six arms disjoint, every naive check passing. Rounds 4 and 5 are
the only reason that did not happen. **Classify each arm against its own
historical tg64 clusters, never against the other arm's, and never stop a
whole-process A/B on this track at three agreeing rounds.**

### Correctness (both perplexity gates passed — and it was rejected anyway)

- **greedy agreement, the gate that actually decided it**: 43.8% over 11 short
  fixtures. My local server-greedy check had already reported 6/6 completions
  diverging; I recorded that as expected-for-this-model instead of disqualifying.
- **gate-shape ppl** (`-c 512 --chunks 8`, runner corpus): **4.7508 ± 0.27989
  on both arms**, identical to five significant figures including the error
  bar, and equal to the stock gate value this README already records. This is
  the shape the trusted runner scores. It is also partly blind — at batch 512
  the ppl path is MMQ-shaped and mmvq barely runs — which is why the next line
  exists.
- **decode-path ppl** (`-c 512 -b 512 -ub 1`, all 15 chunks the corpus yields):
  **5.7682 ± 0.26064 → 5.7669 ± 0.26030, −0.023%**, against a ±0.1% symmetric
  band. This is the reading that actually exercises the patched kernel, and it
  is an order of magnitude inside the band. Note the XS twin needed the deep
  reading to clear (+0.327% at 15–16 chunks vs +0.912% at 8); on Laguna S the
  8-chunk gate shape is bit-identical anyway, so both readings agree here.
- **server greedy**: see the note below on how this check must be run on this
  model — the naive form produces a false pass.
- **`cuobjdump -res-usage`**: every touched `mul_mat_vec_q` instantiation is
  `STACK:0`. The changed small_k instantiations move as intended (SHARED
  2560→4096 and 4096→7168, registers up) with no spill. This track's 0001
  section documents a kernel-parameter spill that *inverted* a result, so this
  is checked before any timing is believed.
- **provenance**: the five patch files in this directory apply clean to
  pristine `b10237` and the resulting tree is `diff`-identical to the tree that
  produced the measured binaries, across all five touched CUDA sources.

### llama-server binds its port before the model is loaded

The `empty-file-greedy-identity-false-pass` finding already warned about
comparing two empty outputs. It recurred here in a new disguise, and the new
form is worth recording: `llama-server` accepts connections on `--port` while
the 90 GB model is still loading, so a readiness probe of the form

```sh
curl -s http://127.0.0.1:$PORT/health && break
```

succeeds **immediately** — curl does not fail on a 503 — and every completion
that follows races a server that cannot answer. All six came back as errors,
both transcripts were 0 bytes, and `cmp` would have reported them identical.
Poll the response **body** for `"status":"ok"`, and keep a minimum-byte
assertion on each transcript as the backstop. The byte assertion is what
caught this; the `cmp` did not.

## 0002: cap the MMQ MoE J tile at 64 on sm_121 (prefill)

Third verified port of the sm_121 J-cap (qwen gb10 0016: ranked prefill
1.0369; laguna-xs gb10 0015: ranked prefill 1.0741). For `mul_mat_id`,
stock picks J from `ncols_max = n_tokens` (J=128 at pp512) while the
256-expert top-8 segments average ~16 columns; the GB10 tensor-core MMQ
sweet spot for that segment mix is J=56-64 (target-swept on qwen, which
shares the routing). Arch-guarded to sm_121; `GGML_CUDA_DISABLE_MMQ_MOE_J`
restores stock selection.

### Correctness

- **ppl** `-c 512 --chunks 8` on the runner corpus: **4.7508 ± 0.27989 with
  the cap on, 4.7508 ± 0.27989 with it off**, both equal to the stock gate
  value. Identical to five significant figures *including the error bar* —
  at `-c 512` this is a 512-token-batch comparison, i.e. exactly the MMQ
  path the cap changes, so it is a logit-level identity result, not a
  coincidence of rounding. The track gate is ≤0.1% relative; this is 0.0%.
- **Server greedy identity**: byte-exact between arms, 790 bytes over three
  prompts (`temperature 0`, `top_k 1`, `cache_prompt false`), non-emptiness
  asserted. Caveat, stated because it matters: those prompts are ~8 tokens,
  so `mul_mat_q_switch_J` never picks J>64 on them and **they do not
  exercise the capped path**. The evidence that the capped path is
  arithmetically identical is the ppl result above, which does.
- **Provenance**: the measured binary was verified byte-identical to
  `git archive b10237 | 0001 | 0002` across all four touched CUDA sources —
  the tree that produced the timings *is* this patch series, not a superset.
  Series applies clean on pristine b10237.

### Timing: the per-launch lottery moves PREFILL too, so pair by mode

Same binary, `GGML_CUDA_DISABLE_MMQ_MOE_J` toggled, whole-process
`llama-bench -p 512 -n 64 -r 3`, arms alternated within each round (ABBA):

| round | off pp512 | off tg64 | on pp512 | on tg64 | ratio | |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 608.72 | 24.31 | 636.71 | 24.27 | **1.0460** | same mode |
| 2 | 610.33 | 24.19 | 635.05 | 24.27 | **1.0405** | same mode |
| 3 | 627.78 | 25.90 | 600.52 | 24.45 | 0.9566 | mixed — OFF drew fast |
| 4 | 615.59 | 24.39 | 642.31 | 24.32 | **1.0434** | same mode |
| 5 | 602.97 | 24.03 | 635.88 | 24.26 | **1.0546** | same mode |
| 6 | 631.20 | 25.78 | 631.51 | 24.16 | 1.0005 | mixed — OFF drew fast |
| 7 | 616.80 | 24.29 | 663.25 | 25.93 | 1.0753 | mixed — ON drew fast |

Same-mode rounds: **median 1.0447**, mean 1.0461, range 1.0405–1.0546.
Decode is unchanged in every round (24.0–24.4 slow / 25.8–25.9 fast on both
arms) — expected, since decode expert matvecs route through mmvq and the cap
only touches MMQ.

**The important correction to this track's protocol section.** The README
already documented a per-launch decode lottery. This run shows it is not a
decode-only artifact: launches that drew the fast mode read **1.031x on
pp512** as well (fast-mode launches mean 640.7 vs slow 621.5). So a
whole-process A/B round whose two arms landed in *different* modes is
comparing a fast launch against a slow one, and its ratio is meaningless in
either direction. That single fact explains all three outliers, and it
predicts them quantitatively:

- round 7 (ON fast): predicted 1.0447 × 1.031 = 1.077, observed **1.0753**
- round 6 (OFF fast): predicted 1.0447 / 1.031 = 1.013, observed 1.0005

The effect is therefore ~+4.5% prefill, and the three "contradictory" rounds
are the artifact behaving exactly as modelled — including one inflated round
that would have *overstated* the win had it been kept. **Classify every
launch by its tg64 and discard mixed-mode rounds** before taking a median;
do not simply drop rounds that look inconvenient.

Prior round 3 had been dismissed in this README as "lottery-contaminated"
on intuition. That call was right, but it was made from two clean rounds and
a hunch; it now rests on four clean rounds plus a confound model validated
in both directions.

**Do not verify greedy identity with `llama-cli` on this model.** Redirected
to a file it emits zero bytes, so `cmp -s on off` compares two empty files
and prints a pass; that false pass reached this README once already. A retry
with `--no-display-prompt` then span 78 minutes at 98.7% CPU holding 119 GB
of GPU memory outside the lock while writing a 27 GB loading-spinner
animation to `/tmp`. Use `llama-server` + `POST /completion`, assert
non-empty, and wrap every invocation in `timeout`. Recorded as the
`empty-file-greedy-identity-false-pass` finding.

The prior frontier was **+1.459%** (score 1.014589, decode 24.01 tok/s) from
`0001-cuda-mmvq-group-same-activation-matvecs.patch`, verified on the trusted
runner. Stock baseline is 23.627 tok/s decode, 343.1 tok/s prefill, 0.66 s TTFT.

The series previously held the eleven patches from the XS GB10 CUDA track.
Those were placed here by analogy when the shared series was split per track,
not because anything had measured them on Laguna S. They were then measured on
S: **+0.64%, inside noise**, and removed.

## What 0001 does

Laguna S decode issues several quantized matvecs over the same activation
vector — `attn_q`/`attn_k`/`attn_v` all read the post-`attn_norm` hidden state,
and the shared-expert gate/up projections both read the post-`ffn_norm` one.
0001 merges them into a single launch whose grid concatenates their row ranges.
Per-row arithmetic is unchanged, so outputs are byte-identical.

It makes **no ggml graph change** — the merge happens entirely inside the CUDA
backend, by issuing later members from the position of the first.

## Two things this track has now established about that lever

**1. It is not launch-count reduction.** Merging the *large* `attn_q`
projection in with K and V helps further (+1.868%) than merging only the small
ones (+1.109%). GB10 decode is 93.4% kernel time and removing launches is worth
about 0.6% here. The win is that one larger grid reaches a memory rate that
several smaller grids do not: measured per-launch, `attn_q` (9216 rows, 30.1 MB)
runs at 216 GB/s while `attn_k` (1024 rows, 3.34 MB) manages 159 GB/s and
`attn_gate` (72 rows) only 37 GB/s.

**2. Segment selection must not use a runtime index.** The natural form

```cuda
seg = ...;  vx = args.vx[seg];
```

makes nvcc give the kernel an **80-byte stack frame** — it materialises the
whole kernel-parameter struct in per-thread local memory so it can be indexed.
That is ~10 kB of local traffic per block, and it *inverts the result*:

| | mode 1 (small only) | mode 2 (all eligible) |
| --- | --- | --- |
| runtime index — `STACK:80` | **−4.90%** | **−3.80%** |
| unrolled constant index — `STACK:0` | **+1.109%** | **+1.868%** |

Same binary, same harness, same day. Every stock `mul_mat_vec_q` instantiation
has `STACK:0`; the grouped kernel was the only one in `libggml-cuda.so` with a
stack frame. **`cuobjdump -res-usage <lib> | grep -A1 <kernel>` is the check** —
run it on any new CUDA kernel here before concluding anything from a timing.
The earlier −15% attributed to this lever on the XS GB10 track is very likely
the same spill, not a property of the device.

## This box is NOT launch-bound — that kills a whole optimization class

Kernel time is **93.4% of wall** here (45.4 ms of 48.6), versus 74% on the
R9700. So the launch-count reduction class that carried the AMD track does
not transfer: the q8_1 dedupe worth +8.69% there is worth about 0.6% on S.

Byte census is 7.48 GB/token — 2978 MB attention, 2070 MB Q4_K experts,
1510 MB BF16 experts, 471 MB shared expert, 328 MB head, 120 MB dense —
running at ~177 GB/s against a ~231 GB/s achievable ceiling, i.e. 77% of
roof. That leaves roughly 30% of decode and ~19% of score as headroom, and it
has to come from bandwidth or from work removal, not from dispatch structure.

### Dead: mmvq k-loop thread utilisation on the Q4_K expert matvecs

`blocks_per_iter = vdr*nwarps*warp_size/qi`. With `nwarps=4`, Q8_0 at K=3072
gives exactly 3.0 iterations (100% utilisation, ~280 GB/s), but Q4_K gate/up
at K=3072 gives 1.5 padded to 2 (75%), and Q4_K down at K=1024 gives 1
iteration where warps 2-3 never enter the k-loop (50%). That is the entire
189-vs-280 GB/s gap — and the corresponding fix has been measured losing on
the XS sibling in both directions (see
`../laguna-xs-2.1-gguf-gb10cuda-v1/README.md`). Understand it as a diagnosis,
not a lever.

## Before you trust a number on this track

Laguna S has a decode-rate state that is **fixed for a process launch and
independent of your code**: bimodal, roughly 7% wide (~23.4-24.1 vs
~25.5-25.6 tok/s), while the reading *within* a launch is rock steady. Low
variance inside one launch is the artifact's signature, not evidence that your
effect is real — a two-round A/B here measured +6.3% before rounds 3-5 erased
it.

Do not fight it with more boots. Make your knob **re-readable at runtime** and
run both arms inside ONE process on the same pages, bracketed by the reference
arm and rotated. `GGML_CUDA_MMVQ_GROUP_FILE` in 0001 is an example: it re-reads
the policy once per graph evaluation. That protocol gave a same-binary no-op
floor of 0.99994 and resolved a 1.1% effect with a spread of 0.0035 — on a box
where whole-process A/B could not resolve 7%. Both A/B runs quoted above ran in
a single server launch each; the two runs happened to land in *different*
clusters (ref 25.45 vs 23.98 tok/s) and still agreed on the arm ratios.

## Knobs in 0001

- `GGML_CUDA_MMVQ_GROUP=0` — disable grouping (same-binary A/B control).
- `GGML_CUDA_MMVQ_GROUP_FILE=<path>` — re-read the policy from a file at every
  graph evaluation: `0` off, `1` small members only, `2` (default) any eligible.
- `GGML_MMVQ_GROUP_STATS=1` — log a line whenever the effective policy or the
  groups it forms change. With the shipped default on Laguna S this reports
  `mode=2 groups=93 members=229`.

## What is left on the table

`attn_gate` does **not** currently join the K/V group — only 3-member groups
form (`attn_q`, `attn_k`, `attn_v`), plus 2-member shared-expert pairs. Its
launch is the least efficient in the whole decode (72 rows, 0.235 MB, 6.3 µs,
37 GB/s) and it is rejected by the hoist-legality range check, not by type or
shape. Diagnosing that is worth roughly another 0.3 ms/token.

Larger remaining targets, from a per-launch nsys census (25 decode steps):

| Bucket | ms/token | achieved | ideal at 217 GB/s | waste |
| --- | --- | --- | --- | --- |
| Q4_K routed experts | 11.7 | 177 GB/s | 9.54 | **2.16 ms** |
| BF16 routed experts | 7.93 | 190 GB/s | 6.96 | 0.97 ms |
| shared expert gate/up/down | 3.03 | ~155 GB/s | 2.17 | 0.86 ms |
| attn q/o projections | 12.2 | ~217 GB/s | — | ~0 |

The Q4_K expert bucket is the biggest prize, but its obvious angle —
`q4k-expert-thread-utilisation` — is already recorded **dead** (both fixes
measured +0.006% and −0.072%).


## Measured-dead on this model: topk-moe sorted-list (do not spend a slot)

The XS-track frontier lever (sorted-list top-k selection, bit-identical by
construction) was stacked on 0001 and A/B'd on the runner box 2026-08-07:
ppl bit-identical (4.7508 = 4.7508), but decode tg128 REGRESSES ~6% in the
box's fast decode mode (16.07/16.02 -> 15.06/15.08 across interleaved
rounds 1-2) and is neutral in the slow mode (14.89 -> 14.96, round 3;
the documented ~7% per-launch bimodality). Median per-round ratio 0.941.
Laguna-S's much larger layers make the router a negligible share, and the
sorted-list kernel's occupancy interaction appears to cost more than the
selection saves on sm_121 at this scale. XS result does not transfer.


## Measured-dead: 0011 mmvf batched k-loads, ported alone (do not spend a slot)

The family-stack result below left one member untested on its own: `0011`
(`cuda-mmvf-batched-k-loads`) is the one bit-exact XS patch that applies to this
track's base without colliding with the dispatcher this track's own `0001`
rewrote, and a GGUF census said it should hit more work here than on XS — the
F32 router `ffn_gate_inp` 3072×256 across **47** layers against XS's 39, plus
BF16 experts in 8 layers (16.4% of elements). `u` ascending reproduces the
original `col2 += block_size` accumulation order exactly, so it is
bit-identical, which is what this track's agreement branch requires.

**It is neutral.** Six interleaved ABBA whole-process rounds,
`llama-bench -p 512 -n 64 -r 3`, control = 0001+0002, candidate = +0011, each
arm classified against **its own** tg64 clusters (never the other arm's):

| | control (0001+0002) | candidate (+0011) | same-mode ratio |
| --- | --- | --- | --- |
| fast mode | 25.873, 25.955, 25.960 | 25.956 | **1.0010** |
| slow mode | 24.307, 24.342, 24.604 | 24.176, 24.380, 24.408, 24.551 | **0.9984** |

Both modes land inside ±0.2%. One candidate round read **20.42** tok/s, far
below either cluster, in the same round whose control read 600 pp512 — external
contention, not a third mode; it is excluded above and including it would only
make the result more negative.

So the launch-count and k-load classes are now closed on Laguna S member by
member, not just in aggregate: S's layers are large enough that the router
matvec is a rounding error regardless of how many layers carry it. The
prediction from launch counts was wrong for the same reason the family stack
was — **count the bytes and the share of the token, not the launches.**

## Measured-neutral: the XS engine family stack (do not spend a slot)

The XS gb10 family (dedupe, norm/rope/set-rows groups, quantize folds, mmvf
batched k-loads — minus the grouped-launch pieces that conflict with this
track's own 0001, minus topk which measured -6% here) was stacked on 0001
and A/B'd on the runner box 2026-08-07: ppl EXACTLY 4.7508 (bit-exact merge
verified), decode median ratio ~1.003 over 3 interleaved rounds (15.15/15.20,
16.13/15.18 — one arm drew the fast mode — 15.08/15.31). S's much larger
layers leave the launch-latency levers nothing to pay for, and the ~7%
per-launch decode bimodality swamps sub-2% effects at practical round
counts. The S frontier remains the deferred grouped-matvec launch alone.
