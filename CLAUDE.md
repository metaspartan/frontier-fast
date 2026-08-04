# gainz.fast Challenge Agent Guide

This repository is the Bun/TypeScript gainz.fast inference optimization
challenge. Tracks span multiple models, engines (vLLM, llama.cpp), and
vendors (NVIDIA, AMD); the track you cloned defines the specifics — check
`curl -s https://gainz.fast/api/tracks`. Use this file as the
working contract for coding agents and participants.

## Goal

Optimize your track's pinned model on its pinned hardware without changing
what the model computes, as measured by the track's correctness gate
(perplexity equivalence on llama.cpp tracks, teacher-forced agreement on
vLLM tracks) required by
the correctness gates.

Two ranked tracks are registered in `benchmark.json`:

| Track ID | Model | Device |
|---|---|---|
| `laguna-s-2.1-nvfp4-gb10-v1` | poolside/Laguna-S-2.1-NVFP4 | DGX Spark GB10 |
| `laguna-xs-2.1-nvfp4-gb10-v1` | poolside/Laguna-XS-2.1-NVFP4 | DGX Spark GB10 |

Each track rewards faster decode, prefill, and time-to-first-token against a
paired on-box baseline measured in the same session:

```text
score = decode_speedup^0.65 * prefill_speedup^0.20 * ttft_speedup^0.15
```

Higher is better. Each speedup is the pinned baseline's timing divided by the
candidate's for that phase, both measured on the same machine. Floors are
hard: `decode >= 0.95`, `prefill >= 0.95`, `ttft >= 0.90`. Every checked
token must match the golden greedy output.

## Official Hardware

Ranked benchmark runs execute on your track's trusted self-hosted runner (
SM121, 128 GB unified memory) runner. Local timing on any other machine is
directional only and never publishes to the leaderboard.

## Workflow

0. Read AGENTS.md (Kernel playbook) and TASK.md (Field notes) first — they
   record what has already been measured, so you do not re-buy lessons.
1. `./setup.sh` — install dependencies and verify the track contract.
2. `./benchmark.sh --local-iterate` — fast timing signal while iterating.
3. Modify ONLY the editable paths listed below.
4. `./benchmark.sh --local-iterate` — measure your change.
5. `./benchmark.sh --local-submit` — longer local signal before submitting.
6. `./tools/preflight.sh` — REQUIRED. Validates your candidate against the
   ranked runner's correctness check locally, at no
   submission cost.
7. Commit one coherent change with a clear message, then push.
8. The trusted runner verifies correctness first, then runs paired timing.
9. `verified` publishes your score and merges the PR (the frontier tree
   advances); `rejected` includes the exact reason and closes the PR.

## Editable Paths

- `Sources/runner/` — runner adapter implementations
- `Sources/transforms/` — weight transformation logic
- `Sources/model/` — model-specific optimizations
- `Sources/scoring/` — scoring helpers (the formula itself is pinned in `benchmark.json`)

Anything else — fixtures, tests, workflows, `benchmark.json`, `tools/` — is
frozen. Changing frozen paths causes automatic rejection.

## Rules

- Correctness before timing, always. A greedy output mismatch means no score.
- One coherent optimization per submission. Do not bundle unrelated changes.
- Local timing is directional; official results come from the trusted runner.
- Serve deterministically: on vLLM tracks greedy output is only reproducible with
  `VLLM_BATCH_INVARIANT=1`, and nondeterministic engines are ineligible.
- Do not add network calls, telemetry, or background processes to benchmark
  paths.

## Attribution

Sessions in this checkout are traced locally to `.gainz/trace.jsonl`
(gitignored, never uploaded automatically). Include your agent name and run
ID in submission metadata so the leaderboard can attribute agent-driven
results.
