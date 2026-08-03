# Frozen benchmark window

The ranked timing window is frozen per track version. For
`laguna-xs-2.1-nvfp4-gb10-v1` and `laguna-s-2.1-nvfp4-gb10-v1`:

| Knob | Value |
|---|---|
| Prompt (prefill) tokens | 512 |
| Decode steps | 128 |
| Warmup runs | 2 |
| Measured runs | 5 (median) |
| Temperature | 0 (greedy) |
| TTFT measurement | wall time of a max_tokens=1 completion over a fresh prompt |
| Decode rate | full-window wall time minus TTFT, divided by decode steps − 1 |
| Prompt caching | defeated: every timing prompt carries a unique prefix |
| Serving determinism | VLLM_BATCH_INVARIANT=1 required |

Changing any of these knobs requires a new track version with a fresh
leaderboard frontier; existing scores are never silently re-based.
