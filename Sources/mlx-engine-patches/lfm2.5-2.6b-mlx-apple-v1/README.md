# MLX engine patches — lfm2.5-2.6b-mlx-apple-v1

**Intentionally empty.**

A non-empty series forces a full MLX rebuild on the trusted runner
(`pip install -e .` from mlx v0.32.0). That path currently fails on the
ranked box with `No such file or directory: 'cmake'`, so engine patches
cannot land until the runner image has cmake/build tools.

Use `Sources/patches/lfm2.5-2.6b-mlx-apple-v1/` (Python `mlx_lm` overlay,
no rebuild) until then.

Pinned engine: **MLX v0.32.0** (`7a1d4f5c`).
