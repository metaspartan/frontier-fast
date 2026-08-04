## One series per track

Each track keeps its accumulated wins in its own directory:

- `laguna-xs-2.1-gguf-r9700-v1/` — RDNA4 (Radeon AI PRO R9700)
- `laguna-xs-2.1-gguf-gb10cuda-v1/` — GB10 sm_121
- `laguna-s-2.1-gguf-gb10cuda-v1/` — GB10 sm_121

A track's series **is** its stock build: the runner applies it to the pinned
llama.cpp tree, then applies your patch on top and times the pair. Add your
patch to your track's directory with the next free number, and never edit or
renumber a patch that is already there — later patches were authored against
the tree those earlier ones produce.

These directories exist because a shared series could not stay coherent across
architectures. A launch-geometry change that wins on RDNA4 measurably regresses
on sm_121, so a GB10 port deleted two RDNA4 patches and renumbered the rest —
which silently orphaned three later patches that depended on them (one of them
literally un-does a change that no longer existed). Every llama.cpp submission
on every track then failed with `patch does not apply`. CI now applies each
track's series to the pinned tree on every PR, so this cannot recur silently.

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
series 0001/0002 (idle-warp launch trim) is deliberately absent, because the trim
changes the launch geometry and the small grids on this device want more memory
requests in flight, not fewer wave slots.

### Correction: the idle warps DO occur on NVIDIA

The reason previously given here and in the header of
`0002-cuda-mmvq-grouped-launch.patch` -- that the geometry the R9700 trim targets
"does not occur on NVIDIA, where the upstream `small_k` path is enabled" -- is
**wrong**, and it is worth correcting because it reads as "nothing to see here".

`should_use_small_k` fires when `blocks_per_row_x < nwarps*blocks_per_iter_1warp`,
and all it then does is widen `rows_per_cuda_block` from 1 to `nwarps`. It gives
warp 0 more rows. It does **not** give the other warps any k-blocks: the k-loop in
`mul_mat_vec_q` still starts at `tid/(qi/vdr)` and still strides by the full
`blocks_per_iter`, so a warp runs an iteration only if its `tid` maps to a `kbx`
below `blocks_per_row_x`. On Laguna-XS-2.1 (`ffn_down_exps` / `ffn_down_shexp`,
K=512, i.e. two k-blocks per row) that means:

| tensor | K | k-blocks | `blocks_per_iter` | warps that enter the k-loop |
| --- | --- | --- | --- | --- |
| `ffn_gate_exps`, `ffn_up_exps` (Q4_K) | 2048 | 8 | 8 | 0,1,2,3 â€” one exact iteration |
| `ffn_down_exps`, `ffn_down_shexp` (Q4_K) | 512 | 2 | 8 | **0 only** |
| `ffn_down_exps`, `ffn_down_shexp` (Q6_K) | 512 | 2 | 4 | **0,1 only** |

So a 128-thread `ffn_down` block issues its loads from 32 threads (Q4_K) or 64
(Q6_K); the rest enter the kernel, run zero iterations and contribute only the
`+0.0f` they park in their shared-memory reduction slot.

### ...but waking them up does not help (measured)

Giving each warp its own output row -- so every warp issues loads and the
cross-warp reduction, its staging buffers and the `__syncthreads` all disappear --
is **bit-identical** whenever `small_k` holds (each warp then runs at most one
k-iteration, so a lane's operand sequence and the final `warp_reduce_sum` tree are
unchanged; only the idle warps' `+0.0f` are dropped). Confirmed: perplexity over
the fixed corpus is identical to every digit, per chunk and final
(1.1086/1.1247/1.1313, final 1.1313 +/- 0.01416), for stock, candidate, and
candidate with the escape hatch off.

It is nonetheless **slower**. Same binary, arms interleaved within each round,
`llama-bench -p 512 -n 128 -r 3`, Laguna-XS-2.1-Q4_K_M, `-ngl 99`, sm_121,
CUDA 13.0, on top of this track's 0001-0010 series (decode tok/s):

| round | stock `small_k` | row-per-warp |
| --- | --- | --- |
| 1 | 95.834 | 95.699 |
| 2 | 95.835 | 95.547 |
| 3 | 95.779 | 95.532 |
| 4 | 95.813 | 95.498 |
| 5 | 95.606 | 95.618 |
| 6 | 95.605 | 95.568 |
| 7 | 95.560 | 95.472 |
| 8 | 95.548 | 95.467 |

Mean 95.6975 vs 95.5501 (**-0.154%**), median -0.160%, candidate slower in 7 of 8
rounds. Prefill neutral (~2435-2461 tok/s in both arms).

The conclusion is not "the idle warps are fine", it is that **warp-level
parallelism is not what this kernel is short of**. Warp 0's four rows already give
it four independent loads in flight, and spreading those across four warps costs
the activation (`y`) reuse that the stock `small_k` block gets by holding one
`q8_1` block in registers across `rows_per_cuda_block` rows, plus it turns one
coalesced 4-row store into four scattered single-row stores. That trade is
slightly net-negative, which also means the remaining `mul_mat_vec_q` headroom on
this device is not in the block's warp geometry.

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
