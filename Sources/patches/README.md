# Sources/patches — the AMD R9700 llama.cpp track surface

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
