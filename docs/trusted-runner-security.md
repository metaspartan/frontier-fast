# Trusted runner security

Ranked results are produced only by the trusted self-hosted DGX Spark
runner; nothing a participant submits is believed about performance.

- The benchmark workflow runs on push to `main` and manual dispatch only —
  never on pull requests — so fork code cannot reach the trusted hardware
  without maintainer review. Workflow approval is required for all outside
  contributors.
- The coordinator's submission lifecycle is runner-token gated: only the
  trusted worker can transition `submitted → running → verified/rejected`,
  and verified results are immutable. Participant tokens carry a
  submit-only scope.
- The paired baseline and candidate are measured in the same session on the
  same silicon; the published ratio cancels host drift. Telemetry
  (unified-memory headroom via /proc/meminfo on GB10) gates each run.
- Hidden correctness prompts are never committed to this repository; the
  public prompts in `correctness_prompts/` are drift tripwires only.
- Verified result artifacts are HMAC-signed by the runner before
  publication.
