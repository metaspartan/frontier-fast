# Sources/patches — the llama.cpp full-source tracks

TWO tracks share this surface, both pinned to llama.cpp `b10237`
(`2b63e0610bbc2be990ae1360d5256efcdc3f9efb`) and the same Q4_K_M GGUF:

| Track | Device | Build | Baseline decode |
| --- | --- | --- | --- |
| `laguna-xs-2.1-gguf-r9700-v1` | Radeon AI PRO R9700 (HIP, gfx1201) | `GGML_HIP=ON, gfx1201` | 95.43 tok/s |
| `laguna-xs-2.1-gguf-gb10cuda-v1` | DGX Spark GB10 (CUDA, sm_121) | `GGML_CUDA=ON, arch 121` | 90.62 tok/s |
| `laguna-s-2.1-gguf-gb10cuda-v1` | DGX Spark GB10 (CUDA, sm_121) | `GGML_CUDA=ON, arch 121` | 23.63 tok/s |

## The accuracy gate (read this before optimizing)

Correctness is **perplexity equivalence**, not bit-identity: the runner measures
PPL over a fixed corpus on stock and on your build in the same paired run and
accepts a relative delta of **<= 0.5%**. Identical output is a fast path.

This matters because Laguna is a **256-expert MoE with top-8 routing**. Any
float32 regrouping — even one that computes the identical set of products —
reroutes an expert and the greedy text diverges within about one token
(measured: stock-vs-stock scores 100%, identical-product lane regrouping
scores 26%). Text comparison is therefore binary on this model and cannot
distinguish reordered arithmetic from a broken kernel. Perplexity can.

**So these are all fair game**: split-K, lane regrouping, wider vector loads,
FMA contraction changes, tile reshaping, two-pass reductions — anything that
preserves the model's behaviour. Launch-geometry and fusion changes remain
bit-identical and pass trivially.

Your patch series is benchmarked on whichever track you submit to; a series
can target both (submit twice). Vendor-specific code paths are fine as long
as each track's build stays byte-exact vs its stock binary.

**This branch carries the `laguna-xs-2.1-gguf-gb10cuda-v1` (GB10 CUDA) series.**
The runner applies every `Sources/patches/*.patch` of the submitted commit, so a
branch carries exactly one track's set. These four patches are the CUDA port of
the R9700 series 0003-0006 (dedupe, grouped matvecs, grouped rms_norm, grouped
rope), with the per-arch differences documented in each patch header. The R9700
series 0001/0002 (idle-warp launch trim) is absent for a measured reason, not
the one originally written here. **Correction:** an earlier version of this
file claimed the idle-warp geometry "does not occur on NVIDIA". That is false
— `small_k` only widens `rows_per_cuda_block` to `nwarps`, giving warp 0 more
rows; it never hands k-blocks to the other warps. On Laguna XS the idle warps
are real (`ffn_down_exps` Q4_K K=512 runs at 25% warp utilisation, Q6_K at
50%). The trim is absent because a row-per-warp rewrite was *measured* at
-0.154% on GB10 (slower in 7 of 8 interleaved rounds): warp parallelism is not
what that kernel lacks — spreading rows across warps loses the q8_1 activation
reuse and turns one coalesced 4-row store into four scattered ones.

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
