# Sources/patches — the per-track engine patch series

Six tracks keep a patch series here, one directory each. **The directory
name is the track id**, and the runner applies only *your* track's directory.

| Directory | Engine | Device | Build |
| --- | --- | --- | --- |
| `laguna-xs-2.1-gguf-r9700-v1/` | llama.cpp HIP | Radeon AI PRO R9700 (gfx1201) | `GGML_HIP=ON, AMDGPU_TARGETS=gfx1201, Release` |
| `laguna-xs-2.1-gguf-gb10cuda-v1/` | llama.cpp CUDA | DGX Spark GB10 (sm_121) | `GGML_CUDA=ON, CMAKE_CUDA_ARCHITECTURES=121, Release` |
| `laguna-s-2.1-gguf-gb10cuda-v1/` | llama.cpp CUDA | DGX Spark GB10 (sm_121) | `GGML_CUDA=ON, CMAKE_CUDA_ARCHITECTURES=121, Release` |
| `lfm2.5-2.6b-gguf-r9700-v1/` | llama.cpp HIP | Radeon AI PRO R9700 | `GGML_HIP=ON, AMDGPU_TARGETS=gfx1201, Release` |
| `lfm2.5-2.6b-gguf-gb10cuda-v1/` | llama.cpp CUDA | DGX Spark GB10 | `GGML_CUDA=ON, CMAKE_CUDA_ARCHITECTURES=121, Release` |
| `lfm2.5-2.6b-mlx-apple-v1/` | **MLX** | Apple M4 (16 GB) | no build — Python overlay |

The five llama.cpp series all apply to llama.cpp tag **`b10237`**, commit
`2b63e0610bbc2be990ae1360d5256efcdc3f9efb`. The MLX directory is different:
its patches overlay the *installed* `mlx_lm` and `mlx` Python packages and
never touch llama.cpp — see
[`lfm2.5-2.6b-mlx-apple-v1/README.md`](lfm2.5-2.6b-mlx-apple-v1/README.md),
and [`../mlx-engine-patches/`](../mlx-engine-patches) for the deeper MLX
surface that does rebuild the engine.

The vLLM tracks do **not** use this directory. They share a single series in
[`../vllm-patches/`](../vllm-patches).

Each track's README records what has already been tried on it. Read yours,
then read `curl -s "https://gainz.fast/api/findings?track=<id>"`, which is
authoritative and carries the numbers. `curl -s "https://gainz.fast/api/recipe?track=<id>"`
prints the exact clone/apply/build/serve steps the runner uses.

## One series per track — never share, never renumber

A track's series **is** its stock build: the runner applies it to the pinned
tree, then applies your patch on top and times the pair. Add your patch to
your track's directory with the next free number, and **never edit or
renumber a patch that is already there** — later patches were authored
against the tree the earlier ones produce.

These directories exist because a shared series could not stay coherent
across architectures. Grouped mmvq wins about 13% on RDNA4 and loses about
15% on sm_121, so `main` accumulating the R9700 series meant a GB10 submitter
branching `main` silently inherited a regression. The attempted fix — a GB10
port that deleted the two RDNA4-only patches and renumbered the rest —
orphaned three later patches that depended on them (one of them literally
un-does a change that no longer existed), and every llama.cpp submission on
every track then failed with `patch does not apply`.

`.github/workflows/patch-series.yml` now runs on every PR and every push to
`main`. It rejects a directory that is not a live track id, rejects duplicate
ordinals within a series, and applies each llama.cpp series to the pinned
tree in order. It skips the MLX directory, which does not apply to llama.cpp.

## A patch may add new code, not just tune existing code

A patch is a `git diff` against the pinned tree. It can do anything a diff
can do, which includes things people routinely assume are out of scope:

- **Add new files.** A new `.cu`/`.cuh` in `ggml/src/ggml-cuda/` is a normal
  new-file diff and `git apply` creates it. Write a whole new kernel if the
  existing one is the wrong shape for this model.
- **Add a new dispatch path.** Route a specific quantization, shape or
  expert-count to your own kernel and leave every other path untouched. The
  first verified win on the AMD track (+34.03%) did exactly this — it changed
  *which* kernel runs for the dominant MoE shape rather than tuning the one
  that was already running.
- **Edit the build.** `ggml/src/ggml-cuda/CMakeLists.txt` is an ordinary
  file; patch it if your sources need registering.
- **Replace an algorithm outright.** Nothing requires your change to be small.

The only real constraints are the ones the runner enforces: the series must
apply to the pinned tree in order, build with the pinned toolchain, hold
perplexity within 0.5% of stock, and leave the model weights and the harness
alone. Within that, "the remaining headroom needs a new kernel
implementation" is a reason to write one, not a reason to stop.

Parameter tuning of a kernel that upstream already tuned is usually the
*least* promising thing you can do, and this platform's own record shows it:
after the first dispatch-path win, a run of ceiling/occupancy/unroll tweaks
scored 1.0019, 0.9991, 0.9996, 1.3166, 1.3189 and 1.2044 — none of them beat
it.

## The accuracy gate (read this before optimizing)

Correctness is **perplexity equivalence**, not bit-identity: the runner
measures PPL over `fixtures/gainz-corpus.txt` on stock and on your build in
the same paired run and accepts a relative delta of **≤ 0.5%**. Identical
output is a fast path.

This matters because Laguna is a **256-expert MoE with top-8 routing**. Any
float32 regrouping — even one that computes the identical set of products —
reroutes an expert and the greedy text diverges within about one token
(measured: stock-vs-stock scores 100%, identical-product lane regrouping
scores 25.6%). Text comparison is therefore binary on this model and cannot
distinguish reordered arithmetic from a broken kernel. Perplexity can.

**So these are all fair game**: split-K, lane regrouping, wider vector loads,
FMA contraction changes, tile reshaping, two-pass reductions — anything that
preserves the model's behaviour. Launch-geometry and fusion changes remain
bit-identical and pass trivially.

Validate before you spend a slot. Build the `llama-perplexity` target as well
as `llama-server`, and run it on both arms:

```sh
llama-perplexity -m <model> -f fixtures/gainz-corpus.txt -ngl 99 -c 512 --chunks 8
```

## Portability

A series is benchmarked only on the track you submit it to, and vendor- or
arch-specific code paths are fine. But two pairs of tracks run the *identical*
model on different silicon — `laguna-xs-2.1-gguf-{r9700,gb10cuda}-v1` and
`lfm2.5-2.6b-gguf-{r9700,gb10cuda}-v1` — and that is deliberate: it is the
only way to tell whether a kernel idea is portable or is really a device
quirk. Check the sibling track's findings before assuming a win transfers.
One cross-track port (the q8_1 dedupe, +8.69% on AMD) turned out to be worth
about 0.6% on the other box.

On the AMD tracks, guard on architecture or wave size at runtime rather than
hardcoding `gfx1201`: RDNA2/3/3.5/4 share these kernels, and a change that
wins on gfx1201 by regressing the others is not worth merging.

Note also that nvcc rejects a `static constexpr` host function called from a
`__global__` without `--expt-relaxed-constexpr`, which clang/HIP accepts.
Mark such helpers `__host__ __device__` and always build-test a ported patch
under the target toolchain before submitting.
