# Sources/patches — the llama.cpp full-source tracks

TWO tracks share this surface, both pinned to llama.cpp `b10237`
(`2b63e0610bbc2be990ae1360d5256efcdc3f9efb`) and the same Q4_K_M GGUF:

| Track | Device | Build | Baseline |
| --- | --- | --- | --- |
| `laguna-xs-2.1-gguf-r9700-v1` | Radeon AI PRO R9700 (HIP, gfx1201) | `GGML_HIP=ON, gfx1201` | 95.43 decode tok/s |
| `laguna-xs-2.1-gguf-gb10cuda-v1` | DGX Spark GB10 (CUDA, sm_121) | `GGML_CUDA=ON, arch 121` | 90.62 decode tok/s |

Your patch series is benchmarked on whichever track you submit to; a series
can target both (submit twice). Vendor-specific code paths are fine as long
as each track's build stays byte-exact vs its stock binary.

**This branch carries the `laguna-xs-2.1-gguf-gb10cuda-v1` (GB10 CUDA) series.**
The runner applies every `Sources/patches/*.patch` of the submitted commit, so a
branch carries exactly one track's set. These four patches are the CUDA port of
the R9700 series 0003-0006 (dedupe, grouped matvecs, grouped rms_norm, grouped
rope), with the per-arch differences documented in each patch header. The R9700
series 0001/0002 (idle-warp launch trim) is deliberately absent: the geometry it
targets does not occur on NVIDIA, where the upstream `small_k` path is enabled
and on RDNA it is not.

Track `laguna-xs-2.1-gguf-r9700-v1` is **full-source kernel surgery**,
mlx.fast style: your submission is a patch series against the pinned
llama.cpp tree, and the trusted runner rebuilds the whole engine with it.

## Contract

- Base: llama.cpp tag `b10237`, commit `2b63e0610bbc2be990ae1360d5256efcdc3f9efb`.
- Put ordered patches here: `0001-<name>.patch`, `0002-<name>.patch`, …
  (`git format-patch` output or `git diff` applied with `git apply --index`).
- The runner applies them in lexical order onto a clean worktree, rebuilds
  (`GGML_HIP=ON, gfx1201, Release`), and paired-benchmarks your build against
  stock on the same R9700: 9 cache-cold runs each, median decode/prefill/TTFT.
- Model: official `Laguna-XS-2.1-Q4_K_M.gguf`
  (sha256 `1ac7079101fca5a6df8c5a7523a3c30ea7d1c0e4b1258090e7d6d4039287f6cb`), `-ngl 99 -c 8192 --parallel 1`.
- Correctness: byte-identical greedy output vs the stock build (determinism
  verified 3/3 on this stack). Baseline: 96.18 decode tok/s, 2,968 prefill
  tok/s, 222 ms TTFT.

## Where the headroom likely is

Any HIP/ggml kernel is in scope: the Q4_K dequant path, MoE expert routing,
flash-attention (a known pp regression exists on RDNA4 at depth), graph
scheduling. Change values → rejected; remove provably-dead work → verified.
