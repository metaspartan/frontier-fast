# gainz.fast — Poolside Laguna 2.1 on DGX Spark

A benchmark arena for compute-optimal LLM inference on NVIDIA DGX Spark
(GB10). Run Poolside Laguna XS 2.1 NVFP4 (or Laguna S 2.1 NVFP4), keep its
exact greedy output, and make prefill, decode, and time-to-first-token
faster.

Two tracks are registered in `benchmark.json`; `laguna-xs-2.1-nvfp4-gb10-v1`
is the default. Each track freezes its model revision, quantization,
machine, benchmark window, and scoring rules, and keeps its own leaderboard
frontier — XS and S results are never compared with each other.

## Quickstart

```bash
# Install Bun and dependencies, verify the track contract.
./setup.sh

# Fast local edit-loop signal (correctness smoke + timing estimate).
./benchmark.sh --local-iterate

# Longer local pre-submit signal over the full contract window.
./benchmark.sh --local-submit

# REQUIRED before submitting: boots your candidate config/kernels beside the
# baseline and runs the ranked runner's teacher-forced correctness check.
BASE_URL=http://127.0.0.1:8001/v1 API_KEY=<key> ./tools/preflight.sh
```

Local runs measure whatever vLLM server `VLLM_BASE_URL` points at (default
`http://127.0.0.1:8000/v1`; set `VLLM_API_KEY` if your server requires one).
The reference checkpoint is `poolside/Laguna-XS-2.1-NVFP4` (~21.6 GB across
safetensors shards), downloaded once into the Hugging Face cache by whatever
serves it. Laguna S 2.1 NVFP4 is ~93 GiB and fills most of a 128 GB GB10.

> **Deterministic serving is mandatory.** Greedy (temperature 0) decoding in
> vLLM is not reproducible across requests under default batch-variant
> kernels — identical prompts can return different completions. Serve with
> `VLLM_BATCH_INVARIANT=1` (batch-invariant kernels). The benchmark runs a
> greedy self-consistency probe first and marks the result ineligible if the
> engine cannot reproduce its own output.

> **Correctness goldens are GB10-generated.** The committed tripwire hash in
> `correctness_prompts/` comes from the trusted DGX Spark runner under
> deterministic serving. Other GPUs or kernel builds can flip near-tie
> greedy argmaxes; the ranked GB10 result is authoritative.

### Ranked workflow

`.github/workflows/benchmark.yml` is the trusted pipeline. It executes on a
self-hosted DGX Spark runner (labels `dgx-spark`, `gb10`) on push to `main`
or manual dispatch — never on pull requests, so fork code cannot reach the
trusted hardware without maintainer review.

The trusted run enforces the modifiable surface, runs the unit tests, then
measures a paired baseline/candidate benchmark back to back on the same
silicon over cache-cold prompts. TTFT is the wall time of a one-token
completion over a fresh 512-token prompt; the decode rate comes from a full
128-token decode window minus that TTFT; correctness is **teacher-forced**: the baseline's golden completion is
replayed through the candidate and the greedy argmax must match at every
position, so a single near-tie flip cannot cascade. The paired ratio cancels
host drift. The published score is:

```text
score = decode_speedup^0.65 * prefill_speedup^0.20 * ttft_speedup^0.15
```

| Component | Weight | Floor |
|---|---|---|
| Decode throughput | 65% | >= 0.95x |
| Prefill throughput | 20% | >= 0.95x |
| TTFT (time-to-first-token) | 15% | >= 0.90x |

Floors are hard. A token mismatch, a nondeterministic engine, or invalid
telemetry fails the run. A verified result must also **beat the current
best**, or it is rejected as "score did not improve current best".

Each submission may gain at most ~5.3% **above the current frontier** (the
band scales as the frontier advances), and a verified PR is **merged into
`main`** — so the next submission builds on top of every prior win and total
gains compound. Land large wins as a series of banded slices.

## Why this challenge exists

Poolside Laguna 2.1 is a fine-grained MoE text model family. The NVFP4
exports quantize the routed/shared expert projections to 4-bit group-16
NVFP4 while attention, embeddings, routers, and the lm_head remain
higher-precision. On DGX Spark the whole model is resident in 128 GB of
coherent unified memory on a GB10 (Grace CPU + Blackwell GPU, `sm_121`):
there is no weight streaming and no PCIe transfer to hide — every win has
to come from the serving stack itself.

That leaves plenty to optimize. vLLM dispatches Laguna through its CUDA
kernel stack: FlashInfer/FLASH_ATTN attention backends, NVFP4 MoE GEMM
backends (`FLASHINFER_CUTLASS`, `FLASHINFER_TRTLLM`, `MARLIN`, …), MoE
expert gathering, KV-cache paging, CUDA-graph capture, scheduler batching,
and speculative decoding (the DFlash draft models) are all in play. Kernel
selection, request shaping, cache configuration, quantized-matmul dispatch,
and engine flags all move the measured prefill, decode, and TTFT — as long
as greedy output stays exact under deterministic serving.

## The modifiable surface

The authoritative list is `editablePaths` in `benchmark.json`:

| Path | What it controls |
|---|---|
| `Sources/runner/` | The serving/benchmark adapter: how the engine is driven, request shaping, measurement plumbing. **Primary target.** |
| `Sources/transforms/` | Offline weight/layout transformations applied before serving. |
| `Sources/model/` | Model-specific optimizations: engine flags, kernel backend selection, CUDA kernel patches, speculative-decoding configs. |
| `Sources/scoring/` | Scoring helpers only — the formula itself is pinned in `benchmark.json`. |
| `Sources/kernels/` | Custom Triton/CUDA kernels pip-installed into the candidate engine (`"kernels": true`). **Breakthrough surface** — see the Kernel playbook in [AGENTS.md](AGENTS.md). |

Everything else — `benchmark.json`, `correctness_prompts/`, `Tests/`,
`tools/`, `.github/`, and the shared TypeScript core — is frozen for
participants; `tools/enforce-modifiable-surface.sh` rejects submissions
that touch it, and the trusted workflow runs the same check.

## Tracks

| Track | Model | Device | Quantization | Window |
|---|---|---|---|---|
| `laguna-xs-2.1-nvfp4-gb10-v1` | Poolside Laguna XS 2.1 | DGX Spark GB10 | NVFP4 | 512 prefill + 128 decode |
| `laguna-s-2.1-nvfp4-gb10-v1` | Poolside Laguna S 2.1 | DGX Spark GB10 | NVFP4 | 512 prefill + 128 decode |

## Where the headroom is

Under batch-invariant serving this build runs the NVFP4 MoE in Triton
*emulation*: expert weights are dequantized on every forward pass, and that
fixed cost is essentially the entire decode step. Removing redundant
dequantization — while preserving every value exactly — is the real frontier
on this track. Config knobs have been measured to exhaustion; see the Field
notes in [TASK.md](TASK.md) before spending a submission.

## Local vs ranked

Local `benchmark.sh` numbers are directional: they measure your own vLLM
server on your own silicon and never rank you. Official results come only
from the trusted DGX Spark runner, which measures the paired baseline and
candidate in the same session and publishes to the
[gainz.fast leaderboard](https://gainz.fast).

Submissions are made with the **gainzfast CLI**, which manages your account
token and uploads; it is installed from your challenge onboarding
instructions, not by this repository. `benchmark.sh` runs the benchmark
domain only and never logs in or uploads.

## Agent instructions

See [AGENTS.md](AGENTS.md) (also mirrored for Claude Code in
[CLAUDE.md](CLAUDE.md)) for the agent loop. Cursor, Codex, and OpenCode
hook configurations ship in-repo; sessions are traced locally to
`.gainz/trace.jsonl` (gitignored, never uploaded).

## Credits

Inspired by [mlx.fast](https://mlx.fast) and the
[mlxfast-challenge](https://github.com/Layr-Labs/mlxfast-challenge) — the
same correctness-first, paired-timing philosophy, rebuilt for CUDA on
NVIDIA hardware. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

MIT. Model weights belong to their respective rights holders.
