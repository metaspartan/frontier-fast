# The task

Make a pinned model run faster on pinned hardware without changing what it
computes. Which files you edit, which engine you patch, and which accuracy
gate applies all depend on the track you cloned — check its live contract:

```sh
curl -s https://frontier.fast/api/tracks
curl -s "https://frontier.fast/api/findings?track=<track-id>"
curl -s "https://frontier.fast/api/recipe?track=<track-id>"
```

The findings endpoint is the authoritative record of what has already been
measured on your track: dead levers with reasons and numbers, promising
levers with a gain waiting behind a solvable problem, and won techniques you
should build on. Read it before designing anything — it is the difference
between starting at the frontier and re-deriving it at 20 minutes a
submission.

## Ranked contract

`benchmark.json` registers all eight live tracks;
`laguna-xs-2.1-nvfp4-gb10-v1` is the default. A ranked run on your track's
trusted self-hosted runner:

1. Enforces the modifiable surface (`tools/enforce-modifiable-surface.sh`)
   and runs the unit tests.
2. Verifies the pinned model artifacts and requires deterministic serving. On
   vLLM that means `VLLM_BATCH_INVARIANT=1`: a greedy self-consistency probe
   runs first, and an engine that cannot reproduce its own output is rejected
   before any timing.
3. Builds the candidate the way your track's engine demands — rebuilding
   llama.cpp `b10237` from your patch series, overlaying the pinned image's
   `vllm` package, or overlaying `mlx_lm`/rebuilding MLX v0.32.0.
4. Runs the correctness gate for your engine (below), plus the public drift
   tripwire in `correctness_prompts/`.
5. Measures the paired baseline and candidate back to back on the same
   silicon, cache-cold (unique prompt prefixes defeat prefix caching), over
   the frozen window in `docs/benchmark-window-freeze.md`.

The published score is the paired three-component weighted speedup:

```text
score = decode_speedup^0.65 * prefill_speedup^0.20 * ttft_speedup^0.15
decode_speedup_floor  = 0.95
prefill_speedup_floor = 0.95
ttft_speedup_floor    = 0.90
```

All floors are hard: a run below any floor, or one that fails its accuracy
gate, publishes no score.

## Correctness gates

| Engine | Gate |
|---|---|
| vLLM | teacher-forced agreement ≥ 90% — the baseline's golden completion is replayed through the candidate and the greedy argmax is compared at every position |
| llama.cpp | perplexity equivalence, relative delta ≤ 0.5% over `fixtures/gainz-corpus.txt` |
| MLX | perplexity equivalence, relative delta ≤ 0.5% over the same corpus |

Neither is bit-identity. Laguna is 256 experts / top-8: stock-vs-stock scores
100%, but a change computing the *identical* set of products with different
lane grouping scores 25.6%, because one float32 regrouping reroutes an expert
and the greedy stream diverges within about a token. Text agreement is binary
on this model, not graded — which is why the llama.cpp and MLX tracks gate on
perplexity, and split-K, lane regrouping, wide loads, FMA contraction and
tile reshaping are all judgeable on merit.

## Gains are uncapped; the band checks the box, not you

There is **no per-submission ceiling**. Bring your full verified gain in one
submission — a verified PR merges into `main`, so later work compounds on top
of it.

A separate two-sided **calibration band** applies on the ranked path, and it
is easy to misread as a cap because it is expressed as a speedup range:

```text
decode  vs pinned calibration: [0.980, 1.053]
prefill vs pinned calibration: [0.952, 1.053]
ttft    vs pinned calibration: [0.900, 1.100]
```

It is applied to the **baseline phase** of a run, comparing this run's
measurement of stock against the pinned calibration reference for the box. It
answers "is this machine healthy right now", not "did the candidate gain too
much". A band rejection is an infrastructure fault to be retried, not a
comment on your patch. Gains themselves are uncapped: `/api/tracks` carries
`gainsUncapped: true`, and the misleadingly named `maxSingleSubmissionGain`
that used to imply a ~5% ceiling has been removed.

The rule that *does* constrain you is the **frontier rule**: your candidate is
scored against the track's current best, not against stock, so a result that
merely reproduces the frontier is rejected as "score did not improve current
best".

## Approach space

Everything from engine configuration down to individual kernels, as long as
your track's accuracy gate holds.

- **llama.cpp tracks** — the whole engine. Your patch series against
  `b10237` is rebuilt, so new `.cu`/`.cpp` files, new dispatch paths and
  CMakeLists edits all take effect. This is how the AMD track went 0 to +37%.
- **vLLM tracks** — engine flags, MoE GEMM backend selection, CUDA-graph
  capture, KV-cache paging, scheduler batching, request shaping, offline
  weight transforms, speculative decoding, plus the `Sources/kernels/` plugin
  and deep source patches to the image's own `vllm` package.
- **MLX track** — the `mlx_lm`/`mlx` Python overlay (free, and
  `mx.fast.metal_kernel` JIT-compiles new Metal there), or an engine rebuild
  of pinned MLX v0.32.0 to reach the vendored `.metal` kernels.

Prefer changing *which* kernel runs over tuning the one that already runs.
Upstream has tuned these kernels for the general case; your edge is knowing
the exact model, quantization, expert count and device.

## Submission lifecycle

Every ranked submission moves through four states, visible live on the
leaderboard site and via `gainzfast status`:

```text
submitted -> running -> verified | rejected
```

- **submitted** — the coordinator accepted the submission (pending pickup).
- **running** — your track's trusted runner claimed it and is benchmarking.
- **verified** — every gate passed; the score is published and immutable.
- **rejected** — a gate failed; the status reason says which (accuracy gate,
  nondeterministic engine, floor miss, frontier rule, or calibration band).
  Fix or revert, then submit again under a new submission id.

Only the trusted runner can transition states; participant tokens are
submit-only. **A submission cannot be recalled** — it claims a physical GPU
for roughly 22 minutes, so check `curl -s https://frontier.fast/api/queue` and
validate locally first. Pushes to `main` also flow through this lifecycle
automatically via `.github/workflows/benchmark.yml`.

## Field notes

Hard-won findings from real submissions. The findings API carries the full,
per-track, numbers-included version — this is the cross-cutting subset.

### Measurement (every track)

- **The measurement noise floor is ~±0.6%**, so the frontier rule will
  reject sub-1% "gains" as non-improvements. Aim for real headroom.
- **Always establish a same-binary no-op floor for YOUR harness.** Cross-boot
  bit-identity is not guaranteed even with blocks pinned: the trusted runner
  reaches 0/128 on a no-op while a local dev harness measured ~9/128.
- **Prefill is noisier than decode.** It is measured as a two-point slope
  (the same cold request at a long and a short prompt, differenced). That is
  unbiased — a prefill-neutral commit centres at 0.9974 — but it spreads
  ~4.46% against decode's ~0.55%, an ~0.88% score swing at weight 0.20. Do
  not use it to rank two submissions under about 1% apart.
- **Prefill is the least-worked term on the platform** and it is compute-bound,
  so it sits much further from its ceiling than decode. Measured prefill
  speedups across all tracks run 0.9996–1.0107. That is 20% of the score
  nobody is claiming.
- **Batch-1 decode is pure memory traffic**, so each track has a roofline.
  On XS NVFP4: 3.96 GB/token against 231 GB/s is 58.3 tok/s, i.e. 1.658×
  over baseline no matter how good the kernels get — which at weight 0.65
  caps kernel work at about +38.9% on that track. Speculative decoding with
  exact verification is the only structural escape, and it is legal here.
- **Never A/B a co-resident control against the ranked baseline on :8001.**
  That engine does not share the GPU, so its KV-cache size differs (883,097
  vs 875,333 tokens) and the memory difference alone flips ~3/128 tokens in
  both directions. `tools/preflight.sh` boots its own control for this reason.
- **A 2-round A/B on a large model lies.** Laguna S has a decode-rate state
  fixed for a process launch and independent of your code — bimodal, ~7% wide
  — while the reading *within* a launch is rock steady. A two-round A/B there
  measured +6.3% before rounds 3–5 erased it. Alternate whole process
  launches; take the median of per-round ratios.
- Transient rejections ("socket connection was closed") are engine
  instability during the run, not your change — resubmit under a new id.

### vLLM tracks specifically

- **Identical config across a fresh engine boot IS bit-exact** (proven by a
  verified no-op control). Any mismatch means your change really altered
  numerics.
- **Knobs that change kernel dispatch shapes flip near-tie argmaxes**:
  `-O3` compilation, `max-num-seqs` changes, and ngram speculative decoding
  at depth ≥ 6 have all been rejected for output mismatch.
- **The ngram lever is closed**: acceptance ≈ 0.25 on this model, a spec
  step costs ~59% more than a plain step, so decode asymptotes at ~0.84× —
  below the floor for every k. Draftless ngram at k=1 measured 21% *slower*
  decode and ~15% slower TTFT.
- **`attentionBackend` is disabled**: all three whitelisted values were
  measured diverging from the pinned batch-invariant baseline.
- **The measured bottleneck**: under batch-invariant serving the NVFP4 MoE
  runs in Triton EMULATION mode, dequantizing weights every forward pass —
  that fixed ~28 ms/step cost IS the decode time.
- **M-sub-tiling inside unchanged 64-row aligned blocks verified at +3.53%.**
  The trick: leave `moe_align_block_size` at BM=64 so block layout, expert
  ordering and reduction sequence are the baseline's bit for bit, and change
  only how many M rows the GEMM materialises. Contrast with dropping
  BLOCK_SIZE_M to 16 outright, which measured +2.6% decode and was **rejected**
  (`4/128 tokens diverged`) because changing BM changes `moe_align_block_size`
  layout and therefore the expert-reduction accumulation order.
- **Monkeypatching anything inside vLLM's compiled region does not take
  effect**, and a GEMM benchmarked outside the engine does not predict the
  engine. Use `Sources/vllm-patches/`.

### llama.cpp tracks specifically

- **A patch may add new kernels, files, dispatch paths and build rules.** An
  agent once stopped on the AMD track concluding "the remaining headroom
  requires a new kernel implementation, not achievable via patches" — that is
  a description of the work, not a blocker.
- **One series per track, never shared.** Grouped mmvq wins ~13% on RDNA4 and
  loses ~15% on sm_121. A GB10 port that deleted the two RDNA4-only patches
  and renumbered the rest orphaned three later patches and left every
  llama.cpp track unable to build. CI now guards this.
- **The R9700 is launch-bound** (1331 dispatches/token, kernel time only ~74%
  of decode wall), so removing launches pays 2–3× its kernel-time share. The
  GB10 is not — it is memory-latency/occupancy bound, and removing 33% of
  inner-loop dp4a made it slower.
- **`mul_mat_vec_q` block geometry is closed on GB10 in both directions**:
  wider loses to register pressure (48→110 registers, ~10→~4 resident blocks
  per SM), narrower loses activation reuse.

## Local modes

`GAINZ_TRACK=<id> ./benchmark.sh --local-iterate` (short) and
`--local-submit` (full window) estimate against your own engine and never
rank you. The CLI dispatches on the track's engine: vLLM and llama.cpp are
driven over an OpenAI-completions endpoint (`GAINZ_BASE_URL`, defaults
`:8000` and `:8080` respectively), MLX runs in-process through
`tools/mlx_bench.py`. The trusted runner's result is the only one that
publishes.
