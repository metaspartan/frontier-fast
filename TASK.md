# gainz.fast — Poolside Laguna 2.1 NVFP4 DGX Spark Challenge

Optimize serial (one request at a time) inference for Poolside Laguna XS 2.1
NVFP4 or Laguna S 2.1 NVFP4 on NVIDIA DGX Spark (GB10) while preserving the
model's exact greedy output.

## Ranked contract

`benchmark.json` registers two tracks; `laguna-xs-2.1-nvfp4-gb10-v1` is the
default. A ranked run on the trusted self-hosted GB10 runner:

1. Enforces the modifiable surface (`tools/enforce-modifiable-surface.sh`)
   and runs the unit tests.
2. Verifies the pinned model artifacts against the manifests in `fixtures/`
   and requires deterministic serving (`VLLM_BATCH_INVARIANT=1`): a greedy
   self-consistency probe runs first, and an engine that cannot reproduce
   its own output is rejected before any timing.
3. Runs the public drift tripwire (`correctness_prompts/`) plus private
   correctness prompts, comparing output hashes between the paired baseline
   and candidate phases.
4. Measures the paired baseline and candidate back to back on the same
   silicon, cache-cold (unique prompt prefixes defeat vLLM prefix caching),
   over the frozen window in `docs/benchmark-window-freeze.md`.

The published score is the paired three-component weighted speedup:

```text
score = decode_speedup^0.65 * prefill_speedup^0.20 * ttft_speedup^0.15
decode_speedup_floor  = 0.95
prefill_speedup_floor = 0.95
ttft_speedup_floor    = 0.90
```

All floors are hard: a run below any floor, or with any token mismatch,
publishes no score.

## Acceptance band

A second, two-sided acceptance band applies on the ranked path, measured
against the pinned calibration reference (the GB10 baselines committed in
the coordinator's track registry), and it is tighter than the floors:

```text
decode_speedup  vs pinned calibration: [0.980, 1.053]
prefill_speedup vs pinned calibration: [0.952, 1.053]
ttft_speedup    vs pinned calibration: [0.900, 1.100]
```

The upper bound caps how much a single submission may gain (about 5%): a
larger measured win is either a lucky reading or too big to trust in one
shot, so chunk it across submissions — the cap is per submission, not
cumulative.

## Approach space

The whole serving stack is in play as long as greedy output stays exact
under deterministic serving: vLLM engine flags, attention backend selection
(FlashInfer/FLASH_ATTN), NVFP4 MoE GEMM backend selection
(FLASHINFER_CUTLASS/TRTLLM/MARLIN…), CUDA-graph capture, KV-cache paging,
scheduler batching, request shaping, offline weight/layout transforms, and
speculative decoding (DFlash draft models) — measured end to end over the
frozen window.

## Submission lifecycle

Every ranked submission moves through four states, visible live on the
leaderboard site and via `gainzfast status`:

```text
submitted -> running -> verified | rejected
```

- **submitted** — the coordinator accepted the submission (pending pickup).
- **running** — the trusted GB10 runner claimed it and is benchmarking.
- **verified** — every gate passed; the score is published and immutable.
- **rejected** — a gate failed; the status reason says which (correctness
  mismatch, nondeterministic engine, floor miss, or calibration band). Fix
  or revert, then submit again under a new submission id.

Only the trusted runner can transition states; participant tokens are
submit-only. Pushes to `main` also flow through this lifecycle
automatically via `.github/workflows/benchmark.yml`.

## Field notes (established on the ranked GB10 runner)

Hard-won findings from real submissions — read these before spending one:

- **Identical config across a fresh engine boot IS bit-exact** (proven by a
  verified no-op control). Any mismatch means your change really altered
  numerics.
- **Knobs that change kernel dispatch shapes flip near-tie argmaxes**:
  `-O3` compilation, `max-num-seqs` changes, and ngram speculative decoding
  at depth ≥ 6 (multi-token verification is a batched forward pass) have all
  been rejected for output mismatch. Be conservative with anything that
  alters batch/graph shapes — the same warning mlxfast gives for numeric
  reassociation in Metal kernels.
- **Draftless ngram at k=1 measured 21% SLOWER decode** (proposal/verify
  overhead dominates at depth 1) and ~15% slower TTFT.
- **The measurement noise floor is ~±0.6%**, so the frontier rule will
  reject sub-1% "gains" as non-improvements. Aim for real headroom.
- Correctness rejections now include the divergence position and a short
  baseline-vs-candidate excerpt, so you can see exactly where a token
  flipped.
- **The ngram lever is closed**: acceptance ≈ 0.25 on this model, a spec
  step costs ~59% more than a plain step, so decode asymptotes at ~0.84× —
  below the floor for every k; k ≥ 6 additionally breaks bit-exactness.
- **The real headroom**: under batch-invariant serving the NVFP4 MoE runs
  in Triton EMULATION mode, dequantizing weights every forward pass — that
  fixed ~28 ms/step cost IS the decode time. Bit-exact kernel work in
  `Sources/kernels/` that attacks it is the genuine frontier.
- The band caps each submission at ~5.3% **above the current frontier**
  (it scales as the frontier advances). A larger atomic win must be landed
  incrementally — detune, verify, step up.
- `prefillSpeedup` is derived from TTFT (prefill s/token = TTFT ÷ prompt
  tokens), so the score effectively reduces to decode^0.65 · ttft^0.35.
- **Correctness is teacher-forced** (mlx.fast's method): the baseline's golden
  completion is replayed through the candidate and every position's greedy
  argmax must match. A single near-tie flip no longer cascades, so
  behaviorally-equivalent work is judged fairly. Validate it locally with
  `./tools/preflight.sh` before submitting.
- **MoE decode M-tile right-sizing (BLOCK_SIZE_M 64→16 at small M) is FAST but
  NOT exact**: measured **decode 1.0262 (+2.6%)**, 36.91 tok/s — and rejected
  with `teacher-forced: 4/128 tokens diverged (first at step 72)`. Offline the
  GEMMs are bitwise-identical; in the full engine, changing BM changes
  `moe_align_block_size` block layout and therefore the expert-reduction
  accumulation order, which shifts a few near-ties. The speed is real, so a
  variant that preserves accumulation order is the open opportunity.
- **FIRST WIN (frontier is moving)**: M-sub-tiling the decode GEMM *inside*
  the unchanged 64-row aligned blocks verified at **+3.53%** (decode 1.0213,
  prefill 1.0626, ttft 1.0607). The trick: leave `moe_align_block_size` at
  BM=64 so block layout, expert ordering and the reduction sequence are the
  baseline's bit for bit, and change only how many M rows the GEMM
  materialises (64→16). Rows 16..63 are provably padding at ≤16 tokens, and
  were already mask-zeroed — so nothing is re-associated.
- **Never A/B a co-resident control against the ranked baseline on :8001.**
  That engine does not share the GPU, so its KV-cache size differs
  (883,097 vs 875,333 tokens) and the memory difference alone flips ~3/128
  tokens in both directions. `tools/preflight.sh` boots its own control for
  exactly this reason.
- **Next chunk (measured)**: the 64→16 tile recovered only ~22% of the
  kernel; the remainder is the per-forward NVFP4 dequant ALU chain, and the
  `b_scale` loads are 8x redundant (only 2 distinct scales per k-iteration
  are loaded as a [64,16] tile). Both are value-preserving targets.
- **`attentionBackend` is disabled**: all three whitelisted values were
  measured diverging from the pinned batch-invariant baseline.
- Transient rejections ("socket connection was closed") are engine
  instability during the run, not your change — resubmit under a new id.

## Local modes

`./benchmark.sh --local-iterate` (short) and `--local-submit` (full window)
estimate against your own vLLM server and never rank you. Local runs warn
on band violations but do not fail on them; the trusted GB10 result is the
only one that publishes.
