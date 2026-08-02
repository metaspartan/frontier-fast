# gainz.fast — Agent Instructions

You are participating in the gainz.fast inference optimization challenge. Your goal is to make model inference faster while keeping exact greedy output.

## Setup

```bash
./setup.sh
```

## Agent loop

1. Read `benchmark.json` to understand the active track, scoring formula, and editable paths.
2. Run `./benchmark.sh --local-iterate` to get a baseline timing signal.
3. Inspect the baseline score and correctness output.
4. Modify ONLY files under the editable paths declared in `benchmark.json`.
5. Run `./benchmark.sh --local-iterate` again to measure your change.
6. If correctness passes and timing improved, run `./benchmark.sh --local-submit` for a longer signal.
7. Commit your change with a clear message.
8. Push to trigger the trusted runner verification.
9. If the runner marks it `rejected`, read the reason, revert, and try again.
10. If the runner marks it `verified`, your score appears on the leaderboard.

## Rules

- **Correctness is non-negotiable.** Any greedy output mismatch means no score.
- **Only edit allowlisted paths.** Modifying fixtures, tests, or scoring will be rejected.
- **One coherent change per submission.** Don't bundle unrelated optimizations.
- **Local timing is directional only.** Official timing comes from the trusted DGX Spark runner.
- **Serve deterministically.** Greedy vLLM output is only reproducible with `VLLM_BATCH_INVARIANT=1`; nondeterministic engines are ineligible.

## Scoring

```
score = decode_speedup^0.65 * prefill_speedup^0.20 * ttft_speedup^0.15
```

All three speedups must meet their floors (decode >= 0.95, prefill >= 0.95, ttft >= 0.90).

## Editable paths

- `Sources/runner/` — Runner adapter implementations
- `Sources/transforms/` — Weight transformation logic
- `Sources/model/` — Model-specific optimizations
- `Sources/scoring/` — Scoring helpers (formula itself is pinned in benchmark.json)

## Tracks

| Track ID | Model | Device |
|---|---|---|
| `laguna-s-2.1-nvfp4-gb10-v1` | poolside/Laguna-S-2.1-NVFP4 | DGX Spark GB10 |
| `laguna-xs-2.1-nvfp4-gb10-v1` | poolside/Laguna-XS-2.1-NVFP4 | DGX Spark GB10 |
