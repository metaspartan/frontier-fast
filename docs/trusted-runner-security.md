# Trusted runner security

Ranked results are produced only by the trusted self-hosted runner that owns
your track — a DGX Spark GB10, a Radeon AI PRO R9700, or an Apple M4.
Nothing a participant submits is believed about performance.

- The benchmark workflow runs on push to `main` and manual dispatch only —
  never on pull requests — so fork code cannot reach the trusted hardware
  without maintainer review. Workflow approval is required for all outside
  contributors.
- Patch series are applied inside the engine's own tree — llama.cpp `b10237`,
  the pinned image's `vllm` package, or `mlx_lm`/MLX v0.32.0 — and cannot
  reach the rest of the container or the machine. Non-source artifacts
  (`.so`, `.sh`, archives) are rejected at deploy time: submissions ship
  source, not binaries.
- The coordinator's submission lifecycle is runner-token gated: only the
  trusted worker can transition `submitted → running → verified/rejected`,
  and verified results are immutable. Participant tokens carry a
  submit-only scope.
- The paired baseline and candidate are measured in the same session on the
  same silicon; the published ratio cancels host drift. Telemetry
  (unified-memory headroom via /proc/meminfo on GB10) gates each run. The
  Apple runner additionally refuses to measure while other accelerator work
  is live on the box, requeueing rather than scoring a contended run.
- Hidden correctness prompts are never committed to this repository; the
  public prompts in `correctness_prompts/` are drift tripwires only.
- Verified result artifacts are HMAC-signed by the runner before
  publication.
