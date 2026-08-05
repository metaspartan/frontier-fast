# Frozen benchmark window

The ranked timing window is frozen per track version. All eight live tracks
share the same window except for the number of measured runs:

| Knob | Value |
|---|---|
| Prompt (prefill) tokens | 512 |
| Decode steps | 128 |
| Warmup runs | 2 |
| Measured runs | 9 (median) — except `lfm2.5-2.6b-mlx-apple-v1`, which is 5 |
| Temperature | 0 (greedy) |
| Corpus | `fixtures/gainz-corpus.txt` (varied prose) |
| TTFT measurement | wall time of a max_tokens=1 completion over a fresh prompt |
| Decode rate | full-window wall time minus TTFT, divided by decode steps − 1 |
| Prefill rate | two-point slope: the same cold request at a long and a short prompt, differenced |
| Prompt caching | defeated: every timing prompt carries a unique prefix |
| Serving determinism | `VLLM_BATCH_INVARIANT=1` on the vLLM tracks; llama.cpp and MLX are deterministic as built |

`measuredRuns` per track is authoritative in `/api/tracks` and mirrored in
`Sources/contracts.ts`.

## Why prefill is a slope

Prefill throughput used to be computed as `ttftSeconds / promptTokens`, which
made `prefillSpeedup` exactly equal to `ttftSpeedup` and rested 0.35 of the
score on a single measurement — the wall time of one cold prompt plus one
token. Every vLLM leaderboard row from that era shows the two identical to
four decimal places.

It is now a two-point slope: the same request at a long and a short prompt,
differenced, so the fixed per-request cost cancels and what remains is the
marginal prompt-processing rate. The trusted runners, `tools/mlx_bench.py`
and the local `Sources/runner` harness all use this method.

The slope is unbiased — a commit that is prefill-neutral by construction
centres at 0.9974 — but noisier than decode, because differencing two
measurements amplifies their noise: ~4.46% spread versus ~0.55% for decode.
At weight 0.20 that alone is a ~0.88% score swing, so it should not be used
to rank two submissions under about 1% apart.

Changing any of these knobs requires a new track version with a fresh
leaderboard frontier; existing scores are never silently re-based.
