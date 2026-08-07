# frontier.fast — the inference optimization arena

A benchmark arena for compute-optimal LLM inference across GPUs, engines,
and model families. Take a pinned model on pinned hardware, keep what it
computes intact, and make prefill, decode, and time-to-first-token faster.

Every track freezes its own model revision, quantization, engine build,
machine, benchmark window, correctness gate, and scoring rules, and keeps
its own leaderboard frontier — tracks are never compared with each other.
Today that spans **NVIDIA GB10**, **AMD RDNA4** and **Apple M4** hardware
running **vLLM**, **llama.cpp** and **MLX**; new models, engines,
quantizations, and vendors are added as trusted runners come online.

```sh
# The live registry — always authoritative
curl -s https://frontier.fast/api/tracks
```

## Quickstart

```bash
# Install Bun and dependencies, verify the track contract.
./setup.sh

# Fast local edit-loop signal (correctness smoke + timing estimate).
GAINZ_TRACK=<track-id> ./benchmark.sh --local-iterate

# Longer local pre-submit signal over the full contract window.
GAINZ_TRACK=<track-id> ./benchmark.sh --local-submit
```

Then check accuracy the way *your track's* runner will:

```bash
# vLLM tracks — REQUIRED before submitting: boots your candidate beside an
# identically-configured control and runs the ranked teacher-forced check.
BASE_URL=http://127.0.0.1:8001/v1 API_KEY=<key> ./tools/preflight.sh

# llama.cpp tracks — perplexity equivalence vs your stock build.
llama-perplexity -m <model> -f fixtures/gainz-corpus.txt -ngl 99 -c 512 --chunks 8

# MLX track — the same harness the runner uses, on stock and on your overlay.
python3 tools/mlx_bench.py --model LiquidAI/LFM2.5-2.6B-MLX \
  --corpus fixtures/gainz-corpus.txt --mode ppl
```

What a local run measures depends on your track's engine, and the CLI now
dispatches on it:

| Engine | What `./benchmark.sh` drives | Default endpoint |
|---|---|---|
| vLLM | your running vLLM server | `http://127.0.0.1:8000/v1` |
| llama.cpp | your running `llama-server` | `http://127.0.0.1:8080/v1` |
| MLX | `tools/mlx_bench.py`, in-process | no server |

Override the first two with `GAINZ_BASE_URL` (`VLLM_BASE_URL` still works),
and set `VLLM_API_KEY` if your server requires one. Every track downloads its
pinned checkpoint once — sizes range from ~1.6 GB (LFM2.5 2.6B) to ~96 GB
(Laguna S) — so check your track's `recommendedVramGiB` before starting.

> **Deterministic serving is mandatory.** A track's engine must reproduce
> its own greedy output, or nothing can be measured against it. On vLLM this
> requires `VLLM_BATCH_INVARIANT=1` — default batch-variant kernels return
> different completions for identical prompts. llama.cpp and MLX are
> deterministic as built (verified per track at commissioning). Every ranked
> run probes this first and marks the result ineligible if the engine fails
> its own probe.

> **Goldens and calibration are runner-generated.** Correctness references
> and calibration numbers come from each track's own trusted runner. Other
> GPUs or kernel builds legitimately flip near-tie argmaxes; the ranked
> result on the track's pinned machine is authoritative.

### Ranked workflow

`.github/workflows/benchmark.yml` is the trusted pipeline. It executes on a
self-hosted runner for your track (e.g. labels `dgx-spark`/`gb10`,
`rocm`/`rdna4`, or `apple`/`m4`) on push to `main` or manual dispatch —
never on pull requests, so fork code cannot reach the trusted hardware
without maintainer review.

The trusted run enforces the modifiable surface, runs the unit tests, then
measures a paired baseline/candidate benchmark back to back on the same
silicon over cache-cold prompts. TTFT is the wall time of a one-token
completion over a fresh 512-token prompt; the decode rate comes from a full
128-token decode window minus that TTFT. The paired ratio cancels host
drift. The published score is:

```text
score = decode_speedup^0.65 * prefill_speedup^0.20 * ttft_speedup^0.15
```

| Component | Weight | Floor |
|---|---|---|
| Decode throughput | 65% | >= 0.95x |
| Prefill throughput | 20% | >= 0.95x |
| TTFT (time-to-first-token) | 15% | >= 0.90x |

Floors are hard. A correctness-gate failure, a nondeterministic engine, or
invalid telemetry fails the run. A verified result must also **beat the
current best on its track**, or it is rejected as "score did not improve
current best" — you are measured against the frontier, not against stock.

Gains are **not capped**. Bring your full verified win in one submission —
a verified PR merges into `main`, so later submissions build on top of it and
totals compound. The calibration band only sanity-checks the *baseline* phase
of each run (box health); a band rejection is an infrastructure fault, not a
comment on your patch.

## Why this challenge exists

Every track pairs a **pinned model** with a **pinned engine** on **pinned
hardware**, then asks a simple question: can you make it faster without
changing what it computes? Nothing is streamed or hidden — the weights are
resident, the window is fixed, and both the baseline and your candidate are
measured back-to-back on the same silicon in the same session. Every win has
to come from the serving stack itself.

The families currently in the arena are Poolside Laguna 2.1 (fine-grained
MoE text models) and Liquid AI LFM2.5 2.6B (a dense hybrid), served three
ways — **vLLM** with NVFP4, **llama.cpp** with GGUF Q4_K_M, and **MLX** with
4-bit weights — across **NVIDIA**, **AMD** and **Apple** silicon. More
models, engines, quantizations, and vendors get added as trusted runners
come online.

## Tracks

Eight live tracks. Baselines and frontiers below were read from
`/api/leaderboard` and `/api/recipe` on 2026-08-04; the API is authoritative
and moves.

| Track | Model | Device | Engine · Quant | Baseline decode | Frontier |
|---|---|---|---|---|---|
| `laguna-xs-2.1-gguf-r9700-v1` | Laguna XS 2.1 | Radeon AI PRO R9700 | llama.cpp HIP · Q4_K_M | 95.43 tok/s | **+37.13%** |
| `laguna-xs-2.1-gguf-gb10cuda-v1` | Laguna XS 2.1 | DGX Spark GB10 | llama.cpp CUDA · Q4_K_M | 90.62 tok/s | +2.01% |
| `laguna-s-2.1-gguf-gb10cuda-v1` | Laguna S 2.1 | DGX Spark GB10 | llama.cpp CUDA · Q4_K_M | 23.63 tok/s | none yet |
| `lfm2.5-2.6b-gguf-r9700-v1` | LFM2.5 2.6B | Radeon AI PRO R9700 | llama.cpp HIP · Q4_K_M | 179.87 tok/s | +9.99% |
| `lfm2.5-2.6b-gguf-gb10cuda-v1` | LFM2.5 2.6B | DGX Spark GB10 | llama.cpp CUDA · Q4_K_M | 110.63 tok/s | none yet |
| `laguna-xs-2.1-nvfp4-gb10-v1` | Laguna XS 2.1 | DGX Spark GB10 | vLLM 0.25.1 · NVFP4 | 35.18 tok/s | +12.42% |
| `laguna-s-2.1-nvfp4-gb10-v1` | Laguna S 2.1 | DGX Spark GB10 | vLLM 0.25.1 · NVFP4 | 14.19 tok/s | +8.70% |
| `lfm2.5-2.6b-mlx-apple-v1` | LFM2.5 2.6B | Apple M4 (16 GB) | MLX 0.32.0 · 4-bit | 58.46 tok/s | none yet |

All tracks share the same window (512-token prefill + 128 greedy decode
steps), the same score (`decode^0.65 · prefill^0.20 · ttft^0.15`), the same
floors, and the same frontier rule. Ranked runs take the median of 9
cache-cold runs, except `lfm2.5-2.6b-mlx-apple-v1`, which takes 5. Query any
track's live contract with `curl -s https://frontier.fast/api/tracks`, and the
exact build pins with `curl -s "https://frontier.fast/api/recipe?track=<id>"`.

## Three kinds of surface

Which files you edit depends on the engine your track pins. Every engine on
this platform now takes real kernel work; only the mechanism differs.

**llama.cpp tracks — full source surgery.** Your submission is a git patch
series in `Sources/patches/<track-id>/0001-*.patch` applied to llama.cpp
`b10237` (`2b63e061`); the runner rebuilds the entire engine with it and
races your binary against that track's accumulated series. New `.cu`/`.cpp`
files, new dispatch paths and CMakeLists edits all work. **The directories
are per track** — the tracks pin the same commit but want different patches.
See [Sources/patches/README.md](Sources/patches/README.md).

**MLX track — two depths.** `Sources/patches/<track-id>/` overlays the
installed `mlx_lm` and `mlx` Python packages with no rebuild (including new
Metal authored through `mx.fast.metal_kernel`);
`Sources/mlx-engine-patches/<track-id>/` patches MLX itself at pinned
**v0.32.0** and forces a full engine build, which is the path to the
vendored `.metal` kernels. See
[Sources/mlx-engine-patches/lfm2.5-2.6b-mlx-apple-v1/README.md](Sources/mlx-engine-patches/lfm2.5-2.6b-mlx-apple-v1/README.md).

**vLLM tracks — plugin and deep source.** `Sources/kernels/` is a
Triton/CUDA plugin package loaded into the candidate engine
(`"kernels": true`), and `Sources/vllm-patches/0001-*.patch` patches a
pristine copy of the pinned image's own `vllm` package
(`"vllmSource": true`) — including the quantized MoE modules themselves.
Unlike the llama.cpp tracks this series is **shared by both vLLM tracks**.
See [Sources/vllm-patches/README.md](Sources/vllm-patches/README.md).

The rest of the participant surface, shared by every track:

| Path | What it controls |
|---|---|
| `Sources/runner/` | Serving/benchmark adapter and `serving.json` engine overrides. |
| `Sources/transforms/` | Offline weight/layout transformations applied before serving. |
| `Sources/model/` | Model-specific optimizations: engine flags, backend selection, speculative configs. |
| `Sources/scoring/` | Scoring helpers only — the formula is pinned in `benchmark.json`. |

Everything else — `benchmark.json`, `correctness_prompts/`, `fixtures/`,
`Tests/`, `tools/`, `.github/`, and the shared TypeScript core — is frozen;
`tools/enforce-modifiable-surface.sh` rejects submissions that touch it.
`benchmark.json`'s `editablePaths` is the machine-readable version of this
section.

## Correctness: what "without changing what it computes" means

**Perplexity equivalence applies on every track**, so the accuracy question
is the same whichever engine you are optimizing:

- **All tracks — perplexity equivalence.** The runner measures PPL over a
  fixed text on stock and on your build within the same paired run and
  accepts a relative delta of **≤ 0.5%**. On llama.cpp this is
  `llama-perplexity` over `fixtures/gainz-corpus.txt`; on MLX it is
  `tools/mlx_bench.py --mode ppl`; on vLLM it is computed from
  `prompt_logprobs`.
- **vLLM tracks additionally — teacher-forced agreement ≥ 90%.** The
  baseline's greedy completion is replayed through your candidate and the
  argmax must match at ≥ 90% of positions. These engines are deterministic
  under `VLLM_BATCH_INVARIANT=1`, so the check is available there and is kept
  as a second, coarser signal.

Why both, and why perplexity is the one that generalizes: teacher-forcing
asks whether the greedy *argmax sequence* still matches, while perplexity
asks whether the model still computes the same *distribution*. Once you are
writing real kernels, those come apart — an accumulation reorder flips
knife-edge argmaxes without changing what the model believes, and a genuine
numeric error can move the distribution while the greedy text happens to
survive. Note the gate is **equivalence, not quality**: a candidate whose
perplexity *improves* by more than 0.5% is also rejected, because changing
what the model computes is out of scope even when the new numbers look
better.

Both are *accuracy-preserving* rather than bit-preserving, and that is
deliberate. Laguna routes top-8 of 256 experts per layer, so any float32
regrouping — even one computing the identical set of products — reroutes an
expert and the greedy text diverges within about one token (measured:
stock-vs-stock scores 100%, identical-product lane regrouping scores 25.6%).
A bit-identity gate would therefore ban split-K, wide loads, tile reshaping
and FMA contraction while proving nothing about model quality. These gates
ban *damage* instead.

## Where the headroom is

It differs per track, and the platform publishes what has already been
measured. Before designing anything:

```sh
curl -s "https://frontier.fast/api/findings?track=<track-id>"
```

Each finding carries a verdict (`dead` / `promising` / `won`), the measured
numbers, why it failed, and concrete advice — dead levers will waste a
runner slot, and `won` entries describe techniques already merged that you
build on top of. A few examples of what that ledger currently holds: the
R9700 is launch-bound (removing dispatches pays 2–3× its kernel-time share),
the GB10 is not (its decode matvec is memory-latency bound), the vLLM NVFP4
MoE runs in Triton emulation under batch-invariant serving, and **prefill is
essentially untouched on every track** at 20% of the score.

## Local vs ranked

Local `benchmark.sh` numbers are directional: they measure your own engine
on your own silicon and never rank you. Official results come only
from your track's trusted runner, which measures the paired baseline and
candidate in the same session and publishes to the
[frontier.fast leaderboard](https://frontier.fast).

Submissions are made with the **gainzfast CLI**
(`curl -fsSL https://frontier.fast/install.sh | sh`), which manages your account
token and uploads; `gainzfast setup` and `gainzfast run` are thin wrappers
around `./setup.sh` and `./benchmark.sh` in this repository. If you cannot
install it, `bun run Sources/cli.ts submit --name "..." --track <id>` does
the same POST with a token from `GAINZ_TOKEN`. `benchmark.sh` runs the
benchmark domain only and never logs in or uploads.

## Agent instructions

See [AGENTS.md](AGENTS.md) (also mirrored for Claude Code in
[CLAUDE.md](CLAUDE.md)) for the agent loop. Cursor, Codex, and OpenCode
hook configurations ship in-repo; sessions are traced locally to
`.gainz/trace.jsonl` (gitignored, never uploaded).

## Credits

Inspired by [mlx.fast](https://mlx.fast) and the
[mlxfast-challenge](https://github.com/Layr-Labs/mlxfast-challenge) — the
same correctness-first, paired-timing philosophy, extended to CUDA, HIP and
Metal.

## License

MIT. Model weights belong to their respective rights holders.
