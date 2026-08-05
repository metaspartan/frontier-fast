# MLX engine patches — lfm2.5-2.6b-mlx-apple-v1

**Intentionally empty.** Patches here modify **MLX itself** — the C++ backend and
the Metal kernels — and force a full engine rebuild on the runner. Add yours as
`0001-` when you have one.

This is the deeper of the two surfaces on this track:

| Where | What it changes | Rebuild |
|---|---|---|
| `Sources/patches/<track>/` | `mlx_lm` and `mlx` Python: attention, MoE path, KV cache, sampler, and Metal kernels written with `mx.fast.metal_kernel` | none |
| `Sources/mlx-engine-patches/<track>/` | MLX's own C++ backend and vendored `.metal` kernels | full engine build |

Use the Python surface unless you actually need to change a kernel that already
exists. It has no build cost, and `mx.fast.metal_kernel` already lets you author
and JIT-compile new Metal without touching the engine.

## The engine

Pinned to **MLX v0.32.0** (`7a1d4f5c`), built from source on the runner as the
stock arm. The editable kernel surface is `mlx/backend/metal/kernels/` — 26
`.metal` files plus the steel GEMM sources under
`mlx/backend/metal/kernels/steel/`. Patches must apply to that pin with
`git apply`; the runner tells you the pin if yours does not.

Your patch gets its own engine tree, its own venv and a clean build, so it
cannot contaminate the stock arm. Build failures return the actual compiler
diagnostics.

## Toolchain on the runner

Xcode 17F113, `metal` 32023.883, `metallib` linker, cmake and ninja. The
worker scopes `DEVELOPER_DIR` to itself, so the box's other CI keeps the
system toolchain — do not assume `xcode-select -p` points at Xcode.

## Cost, and what it buys

An engine rebuild is minutes, not seconds, and it happens inside your
submission's slot on a shared machine. That is the price of reaching kernels
that are already compiled into the engine. Measure locally first — the runner
alternates whole process launches and scores the median of per-round ratios,
and a change that does not clear a same-binary control there will not clear it
here either.

Accuracy gate is unchanged: perplexity within 0.5% of stock over the shared
varied-prose corpus.
