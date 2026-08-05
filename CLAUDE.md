# gainz.fast Challenge Agent Guide

This repository is the Bun/TypeScript gainz.fast inference optimization
challenge. Eight live tracks span three model families, three engines (vLLM,
llama.cpp, MLX) and three vendors (NVIDIA, AMD, Apple); the track you cloned
defines the specifics — check `curl -s https://gainz.fast/api/tracks`. Use
this file as the working contract for coding agents and participants.

**Read [AGENTS.md](AGENTS.md) first.** It carries the per-track target table,
the CLI mapping, and the measurement discipline. This file is the short form.

## Goal

Optimize your track's pinned model on its pinned hardware without changing
what the model computes, as measured by that track's correctness gate:

- **llama.cpp and MLX tracks** — perplexity equivalence over
  `fixtures/gainz-corpus.txt`, relative delta **≤ 0.5%**.
- **vLLM tracks** — teacher-forced agreement **≥ 90%** of positions against
  the baseline's golden greedy completion.

Neither gate is bit-identity, and that is deliberate: Laguna routes top-8 of
256 experts, so a float32 regrouping that computes the identical products
still reroutes an expert and diverges the greedy text within about a token.
These gates ban damage, not arithmetic reassociation.

## The eight tracks

| Track ID | Model | Device | Engine |
|---|---|---|---|
| `laguna-xs-2.1-nvfp4-gb10-v1` | poolside/Laguna-XS-2.1-NVFP4 | DGX Spark GB10 | vLLM 0.25.1 |
| `laguna-s-2.1-nvfp4-gb10-v1` | poolside/Laguna-S-2.1-NVFP4 | DGX Spark GB10 | vLLM 0.25.1 |
| `laguna-xs-2.1-gguf-r9700-v1` | poolside/Laguna-XS-2.1-GGUF | Radeon AI PRO R9700 | llama.cpp HIP |
| `lfm2.5-2.6b-gguf-r9700-v1` | LiquidAI/LFM2.5-2.6B-GGUF | Radeon AI PRO R9700 | llama.cpp HIP |
| `laguna-xs-2.1-gguf-gb10cuda-v1` | poolside/Laguna-XS-2.1-GGUF | DGX Spark GB10 | llama.cpp CUDA |
| `laguna-s-2.1-gguf-gb10cuda-v1` | poolside/Laguna-S-2.1-GGUF | DGX Spark GB10 | llama.cpp CUDA |
| `lfm2.5-2.6b-gguf-gb10cuda-v1` | LiquidAI/LFM2.5-2.6B-GGUF | DGX Spark GB10 | llama.cpp CUDA |
| `lfm2.5-2.6b-mlx-apple-v1` | LiquidAI/LFM2.5-2.6B-MLX | Apple M4 (16 GB) | MLX 0.32.0 |

All eight are registered in `benchmark.json` and mirrored in
`Sources/contracts.ts`. Each rewards faster decode, prefill, and
time-to-first-token against a paired on-box baseline measured in the same
session:

```text
score = decode_speedup^0.65 * prefill_speedup^0.20 * ttft_speedup^0.15
```

Higher is better. Each speedup is the pinned baseline's timing divided by the
candidate's for that phase, both measured on the same machine. Floors are
hard: `decode >= 0.95`, `prefill >= 0.95`, `ttft >= 0.90`. Gains are
**uncapped**, and your result is ranked against the track's **current
frontier**, not against stock.

## Official hardware

Ranked runs execute on your track's own trusted self-hosted runner — a DGX
Spark GB10 (sm_121, 128 GB unified memory), a Radeon AI PRO R9700 (RDNA4,
gfx1201), or an Apple M4 with 16 GB. Local timing on any other machine is
directional only and never publishes to the leaderboard.

## Workflow

0. Read AGENTS.md (per-track targets, measurement discipline) and TASK.md
   (field notes), plus `curl -s "https://gainz.fast/api/findings?track=<id>"`.
   They record what has already been measured, so you do not re-buy lessons.
1. `./setup.sh` — install dependencies and verify the track contract.
2. `GAINZ_TRACK=<id> ./benchmark.sh --local-iterate` — fast timing signal.
3. Modify ONLY the editable paths for your track (below).
4. `./benchmark.sh --local-iterate` — measure your change against a
   same-binary control.
5. `./benchmark.sh --local-submit` — longer local signal before submitting.
6. Run your track's accuracy check locally — `./tools/preflight.sh` on vLLM
   (REQUIRED), `llama-perplexity` on llama.cpp, `tools/mlx_bench.py --mode ppl`
   on MLX. It costs no submission slot; a rejection costs 20+ minutes.
7. Commit one coherent change with a clear message, then submit or push.
8. The trusted runner verifies correctness first, then runs paired timing.
9. `verified` publishes your score and merges the PR (the frontier tree
   advances); `rejected` includes the exact reason and closes the PR.

## Editable paths

- `Sources/runner/` — runner adapter implementations and `serving.json`
- `Sources/transforms/` — weight transformation logic
- `Sources/model/` — model-specific optimizations
- `Sources/scoring/` — scoring helpers (the formula itself is pinned in `benchmark.json`)
- `Sources/kernels/` — vLLM tracks: Triton/CUDA plugin package
- `Sources/vllm-patches/` — vLLM tracks: patches to the image's own `vllm` package (**one series shared by both vLLM tracks**)
- `Sources/patches/<track-id>/` — llama.cpp tracks: patches to pinned `b10237`; MLX track: overlay of the installed `mlx_lm`/`mlx` (**per-track directories**)
- `Sources/mlx-engine-patches/<track-id>/` — MLX track: patches to MLX itself at pinned v0.32.0, forcing a full engine rebuild

Anything else — fixtures, tests, workflows, `benchmark.json`, `tools/`, and
the shared TypeScript core — is frozen. Changing frozen paths causes
automatic rejection by `tools/enforce-modifiable-surface.sh`.

## Rules

- Correctness before timing, always. Failing your track's accuracy gate means no score.
- One coherent optimization per submission. Do not bundle unrelated changes.
- Local timing is directional; official results come from the trusted runner.
- Serve deterministically: on vLLM tracks greedy output is only reproducible with
  `VLLM_BATCH_INVARIANT=1`, and nondeterministic engines are ineligible.
- Never commit a `gz_` token, key or credential. `gainzfast submit` reads it
  from `~/.config/gainzfast/token`; `bun run Sources/cli.ts submit` reads it
  from `GAINZ_TOKEN`.
- Do not add network calls, telemetry, or background processes to benchmark
  paths.

## Attribution

Sessions in this checkout are traced locally to `.gainz/trace.jsonl`
(gitignored, never uploaded automatically). Include your agent name and run
ID in submission metadata so the leaderboard can attribute agent-driven
results.
