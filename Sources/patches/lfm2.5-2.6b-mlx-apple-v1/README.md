# lfm2.5-2.6b-mlx-apple-v1 — patch series

**Intentionally empty.** No submission has beaten the baseline, so this track's
frontier is the pristine `mlx_lm`. Add yours as `0001-`.

## What a patch targets here

MLX has no engine binary to rebuild. **Your patch applies to the installed
`mlx_lm` Python tree** — the same shape as the vLLM tracks' source overlay, not
the llama.cpp tracks' C++ rebuild. The runner copies the pristine `mlx_lm` out
of its venv, applies the series with `patch -p1`, and selects the arm with
`PYTHONPATH`, so stock and candidate cannot contaminate each other.

That means everything in `mlx_lm` is fair game: the model implementation, the
cache, the sampler, the quantized matmul paths, custom Metal kernels via
`mx.fast.metal_kernel`, or a different algorithm entirely.

## Measured stock on the ranking runner (Apple M4, 16 GB)

- decode **58.47 tok/s**, prefill **597 tok/s**, peak memory **2.2 GB**

The model is 4-bit and peaks near 2.2 GB, so a change here should stay valid on
8 GB Apple Silicon. If you have another Mac, say what you saw on it in the PR —
an arch-neutrality note across M-series generations is valuable.

## Gates and method

Accuracy is **perplexity equivalence within 0.5%** over the same fixed
varied-prose corpus the llama.cpp and vLLM tracks use, so scores mean the same
thing across engines. Greedy text is recorded but never ranks: a numeric change
can preserve text and still shift the distribution.

The runner alternates whole process launches and scores the **median of
per-round ratios**. Do the same locally, always with a same-binary control, and
check that your change actually fires before trusting a delta — this platform
has repeatedly been fooled by measurements that never executed the new path.

Note the box also runs a GitHub Actions runner. The worker refuses to measure
while other accelerator work is live, so a contended run is requeued rather
than scored.
