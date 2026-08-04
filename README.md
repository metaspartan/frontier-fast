# gainz.fast — the inference optimization arena

A benchmark arena for compute-optimal LLM inference across GPUs, engines,
and model families. Take a pinned model on pinned hardware, keep what it
computes intact, and make prefill, decode, and time-to-first-token faster.

Every track freezes its own model revision, quantization, engine build,
machine, benchmark window, correctness gate, and scoring rules, and keeps
its own leaderboard frontier — tracks are never compared with each other.
Today that spans **NVIDIA GB10 and AMD RDNA4** hardware running **vLLM** and
**llama.cpp**; new models, engines, quantizations, and vendors are added as
trusted runners come online ([vote or host one](RUNNERS.md)).

```sh
# The live registry — always authoritative
curl -s https://gainz.fast/api/tracks
```

## Quickstart

```bash
# Install Bun and dependencies, verify the track contract.
./setup.sh

# Fast local edit-loop signal (correctness smoke + timing estimate).
./benchmark.sh --local-iterate

# Longer local pre-submit signal over the full contract window.
./benchmark.sh --local-submit

# vLLM tracks — REQUIRED before submitting: boots your candidate beside the
# baseline and runs the ranked runner's correctness check.
BASE_URL=http://127.0.0.1:8001/v1 API_KEY=<key> ./tools/preflight.sh

# llama.cpp tracks — check accuracy the way the runner will:
llama-perplexity -m <model> -f <corpus> -ngl 99 -c 512 --chunks 8
```

What a local run measures depends on your track's engine. **vLLM tracks**
drive whatever server `VLLM_BASE_URL` points at (default
`http://127.0.0.1:8000/v1`; set `VLLM_API_KEY` if required). **llama.cpp
tracks** build the pinned tree with your patch series and drive
`llama-server`. Both download their pinned checkpoint once — sizes range
from ~20 GB (Laguna XS, 4-bit) to ~96 GB (Laguna S), so check your track's
`recommendedVramGiB` before starting.

> **Deterministic serving is mandatory.** A track's engine must reproduce
> its own greedy output, or nothing can be measured against it. On vLLM this
> requires `VLLM_BATCH_INVARIANT=1` — default batch-variant kernels return
> different completions for identical prompts. llama.cpp is deterministic as
> built (verified per track at commissioning). Every ranked run probes this
> first and marks the result ineligible if the engine fails its own probe.

> **Goldens and calibration are runner-generated.** Correctness references
> and calibration numbers come from each track's own trusted runner. Other
> GPUs or kernel builds legitimately flip near-tie argmaxes; the ranked
> result on the track's pinned machine is authoritative.

### Ranked workflow

`.github/workflows/benchmark.yml` is the trusted pipeline. It executes on a
self-hosted runner for your track (e.g. labels `dgx-spark`/`gb10`, or
`rocm`/`rdna4`) on push to `main`
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

Every track pairs a **pinned model** with a **pinned engine** on **pinned
hardware**, then asks a simple question: can you make it faster without
changing what it computes? Nothing is streamed or hidden — the weights are
resident, the window is fixed, and both the baseline and your candidate are
measured back-to-back on the same silicon in the same session. Every win has
to come from the serving stack itself.

The families currently in the arena are Poolside Laguna 2.1 (fine-grained
MoE text models) served two ways — **vLLM** with NVFP4 quantization, and
**llama.cpp** with GGUF quantization — across **NVIDIA** and **AMD** GPUs.
More models, engines, quantizations, and vendors get added as trusted
runners come online; see the [roadmap](https://gainz.fast/#roadmap) to vote
for one or [host a runner](RUNNERS.md).

## Tracks

Live tracks, their surfaces, and their pinned baselines:

| Track | Model | Device | Engine · Quant | Baseline decode |
|---|---|---|---|---|
| `laguna-xs-2.1-gguf-r9700-v1` | Laguna XS 2.1 | Radeon AI PRO R9700 | llama.cpp HIP · Q4_K_M | 95.43 tok/s |
| `laguna-xs-2.1-gguf-gb10cuda-v1` | Laguna XS 2.1 | DGX Spark GB10 | llama.cpp CUDA · Q4_K_M | 90.62 tok/s |
| `laguna-s-2.1-gguf-gb10cuda-v1` | Laguna S 2.1 | DGX Spark GB10 | llama.cpp CUDA · Q4_K_M | 23.63 tok/s |
| `laguna-xs-2.1-nvfp4-gb10-v1` | Laguna XS 2.1 | DGX Spark GB10 | vLLM · NVFP4 | 35.18 tok/s |
| `laguna-s-2.1-nvfp4-gb10-v1` | Laguna S 2.1 | DGX Spark GB10 | vLLM · NVFP4 | 14.19 tok/s |

All tracks share the same window (512-token prefill + 128 greedy decode
steps, median of 9 cache-cold runs), the same score
(`decode^0.65 · prefill^0.20 · ttft^0.15`), the same floors, and the same
frontier rule. Query any track's live contract with
`curl -s https://gainz.fast/api/tracks`.

## Two kinds of surface

Which files you edit depends on the engine your track pins.

**llama.cpp tracks — full source surgery.** Your submission is a git patch
series in `Sources/patches/0001-*.patch` applied to the pinned llama.cpp
tree; the runner rebuilds the entire engine with it and races your binary
against stock. Every kernel is yours to rewrite. See
[Sources/patches/README.md](Sources/patches/README.md).

**vLLM tracks — plugin and deep source.** `Sources/kernels/` is a Triton/CUDA
plugin package loaded into the candidate engine (`"kernels": true`), and
`Sources/vllm-patches/0001-*.patch` patches the pinned image's own vLLM
Python/Triton tree (`"vllmSource": true`) — including the quantized MoE
modules themselves. The rest of the participant surface:

| Path | What it controls |
|---|---|
| `Sources/runner/` | Serving/benchmark adapter and `serving.json` engine overrides. |
| `Sources/transforms/` | Offline weight/layout transformations applied before serving. |
| `Sources/model/` | Model-specific optimizations: engine flags, backend selection, speculative configs. |
| `Sources/scoring/` | Scoring helpers only — the formula is pinned in `benchmark.json`. |

Everything else — `benchmark.json`, `correctness_prompts/`, `Tests/`,
`tools/`, `.github/`, and the shared TypeScript core — is frozen;
`tools/enforce-modifiable-surface.sh` rejects submissions that touch it.

## Correctness: what "without changing what it computes" means

Two gates, matched to what each engine can measure:

- **llama.cpp tracks — perplexity equivalence.** The runner measures PPL
  over a fixed corpus on stock and on your build in the same paired run and
  accepts a relative delta of **≤ 0.5%** (identical output is a fast path).
- **vLLM tracks — teacher-forced agreement.** The baseline's greedy
  completion is replayed through your candidate and the argmax must match at
  **≥ 90%** of positions.

Both are *accuracy-preserving* rather than bit-preserving, and that is
deliberate. Laguna routes top-8 of 256 experts per layer, so any float32
regrouping — even one computing the identical set of products — reroutes an
expert and the greedy text diverges within about one token (measured:
stock-vs-stock scores 100%, identical-product lane regrouping scores 26%).
A bit-identity gate would therefore ban split-K, wide loads, tile reshaping
and FMA contraction while proving nothing about model quality. These gates
ban *damage* instead.

## Where the headroom is

It differs per track, and the platform publishes what has already been
measured. Before designing anything:

```sh
curl -s "https://gainz.fast/api/findings?track=<track-id>"
```

Each finding carries a verdict (`dead` / `promising` / `won`), the measured
numbers, why it failed, and concrete advice — dead levers will waste a
runner slot, and `won` entries describe techniques already merged that you
build on top of. A few examples of what that ledger currently holds: the
R9700 is launch-bound (removing dispatches pays 2–3× its kernel-time share),
the GB10 is not (its decode matvec is memory-latency bound), and the vLLM
NVFP4 MoE runs in Triton emulation under batch-invariant serving.

## Local vs ranked

Local `benchmark.sh` numbers are directional: they measure your own engine
on your own silicon and never rank you. Official results come only
from your track's trusted runner, which measures the paired baseline and
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
