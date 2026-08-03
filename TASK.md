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

## Local modes

`./benchmark.sh --local-iterate` (short) and `--local-submit` (full window)
estimate against your own vLLM server and never rank you. Local runs warn
on band violations but do not fail on them; the trusted GB10 result is the
only one that publishes.
