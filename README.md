# gainz.fast — Inference Optimization Challenge

Optimize prefill, decode, and TTFT for Laguna models on NVIDIA DGX Spark. Keep exact greedy output. Compete on verified leaderboards.

Inspired by [mlx.fast](https://mlx.fast) — built for NVIDIA.

## Quickstart

```bash
git clone https://github.com/metaspartan/gainz-fast.git
cd gainz-fast
./setup.sh
./benchmark.sh --local-iterate
```

## How it works

1. Clone this repo.
2. Run `./setup.sh` to install Bun and dependencies.
3. Run `./benchmark.sh --local-iterate` to get a baseline.
4. Modify files under `Sources/` to optimize inference.
5. Run `./benchmark.sh --local-iterate` again to measure your change.
6. Run `./benchmark.sh --local-submit` for a longer signal.
7. Commit and push to trigger trusted runner verification.
8. Results appear on the [leaderboard](https://gainz.fast).

## Scoring

```
score = decode_speedup^0.65 * prefill_speedup^0.20 * ttft_speedup^0.15
```

| Component | Weight | Floor |
|---|---|---|
| Decode throughput | 65% | >= 0.95x |
| Prefill throughput | 20% | >= 0.95x |
| TTFT (time-to-first-token) | 15% | >= 0.90x |

## Tracks

| Track | Model | Device | Quantization |
|---|---|---|---|
| `laguna-xs-2.1-nvfp4-gb10-v1` | Laguna XS 2.1 | DGX Spark GB10 | NVFP4 |
| `laguna-s-2.1-nvfp4-gb10-v1` | Laguna S 2.1 | DGX Spark GB10 | NVFP4 |

## Editable paths

Participants may modify:
- `Sources/runner/` — Runner adapter implementations
- `Sources/transforms/` — Weight transformation logic
- `Sources/model/` — Model-specific optimizations
- `Sources/scoring/` — Scoring helpers (formula is pinned in `benchmark.json`)

Do NOT modify: `benchmark.json`, `correctness_prompts/`, `Tests/`, `.github/`, or scoring formula constants.

## Requirements

- NVIDIA DGX Spark (GB10/SM121) for official timing
- Bun runtime (installed by setup.sh)
- vLLM serving the target model locally for development

## Agent instructions

See [AGENTS.md](AGENTS.md) for the full agent loop.

## License

MIT. Model weights belong to their respective rights holders.
