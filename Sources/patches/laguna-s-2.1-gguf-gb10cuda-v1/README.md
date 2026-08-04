# `laguna-s-2.1-gguf-gb10cuda-v1` — DGX Spark GB10, llama.cpp CUDA, Laguna S 2.1

Track-scoped directory. The runner prefers this over the shared
`Sources/patches/`, which carries the **XS** GB10 series — measured below, that
series is worth nothing here, so S should not inherit it.

Base: llama.cpp `b10237` (`2b63e0610bbc2be990ae1360d5256efcdc3f9efb`).
Build: `cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 -DCMAKE_BUILD_TYPE=Release`.
Model: `laguna-s-2.1-Q4_K_M.gguf` (89.4 GiB), `-ngl 99 -c 8192 --parallel 1`.

| # | patch | default | status |
| --- | --- | --- | --- |
| 0001 | `cuda-mmvq-wide-decode-block` | `GGML_CUDA_MMVQ_NW=6` | correctness-proven; **speed unproven, see the artifact below** |

## READ THIS FIRST: the local harness cannot resolve <10% on this track

The 89.4 GiB model on a 121 GiB box produces a **bimodal, per-process-launch
performance state**. Same binary, same env, same flags, `llama-bench -p 512
-n 128 -r 3`: every arm lands in either a ~23.4-24.1 tok/s cluster or a
~25.5-25.6 tok/s cluster, and which one is decided at model-load time — not by
the code under test. Within a launch the reading is rock steady (stddev
0.002-0.06 tok/s), which makes a single round look deceptively authoritative.

Fifteen end-to-end observations, three separate lock windows:

| arm | observations (tok/s) |
| --- | --- |
| pristine b10237 | 24.106, 23.859, 25.591, 23.851, 25.516 |
| patched, wide block ON (`NW=6`) | 25.468, 25.515, 25.641, 23.347, 23.411 |
| patched, wide block OFF (`NW=0`) | 23.843, 23.713, 23.831, 23.902, 25.535 |

Every arm, including pristine stock and the provably-inert `NW=0` build, visits
both clusters. Cluster membership is independent of the arm. Arm means are
24.585 / 24.676 / 24.165 — i.e. **the wide block is +0.4% vs stock, which is
nothing**, and the first two rounds that suggested +6.3% were the lottery.

`kswapd0` runs at ~17% CPU throughout, so memory reclaim / page placement for
an 89 GiB allocation is the leading suspect.

**Normalising against untouched kernels does not rescue it.** An `nsys`
decode capture comparing `mul_mat_vec_q` against kernels the patch cannot
touch (`mul_mat_f` bf16 experts, `rms_norm`, `flash_attn`, `k_bin_bcast`)
gave `mmvq/ref` of 3.4659 vs 3.4042 in one pair (+1.81%) and 3.3464 vs 3.4672
in the next (−3.48%) — the sign flips. The same arm's two captures of the
Q4_K fused kernel differ by 16.6% (286.2 vs 245.3 ms), and the reference
kernels swing ~6% but not proportionally, so the ratio does not cancel.

Consequence for anyone working this track: **budget many paired launches per
arm, or find a placement-stable protocol first.** A 2-3 round A/B here will
produce a confident-looking number that is pure noise. This very likely also
explains the earlier `s-gguf-gb10cuda-mmid-cuda-graphs-v3` verdict
("stock phase measured 23.35 tok/s") and the 23.35 / 23.627 spread already in
`per-track-runner-constants`.

## The model, measured

`laguna` arch, 48 layers, 256 experts, **top-10**, `d_model` 3072, expert
`d_ff` 1024. 39 MoE layers hold Q4_K experts, **8 hold BF16 experts**, all
attention is Q8_0. Per-token decode from an `nsys` capture:

| kernel | ms/token | share |
| --- | --- | --- |
| `mul_mat_vec_q<Q8_0,1,no-fusion>` — attn k/v/o, shexp down, output head | 11.9 | 27% |
| `mul_mat_f` bf16 MoE experts (8 layers) | 7.6 | 17% |
| `mul_mat_vec_q<Q8_0,1,fusion>` — attn q+gate, shexp gate+up, dense FFN | 7.4 | 17% |
| `mul_mat_vec_q<Q4_K,1,fusion>` — MoE gate+up, 39 layers | 7.3 | 17% |
| `mul_mat_vec_q<Q4_K,1,small_k>` — MoE down, 39 layers | 3.8 | 9% |
| rms_norm / rope / k_bin_bcast / topk_moe / set_rows / silu / FA | 3.5 | 8% |
| `quantize_q8_1` | 1.4 | 3% |

`mul_mat_vec_q` is **70% of decode**. Summed kernel time ≈ 43.7 ms/token
against a ~42.3 ms/token decode wall: **this track is NOT launch-gap-bound**
(contrast R9700, where kernel time is only ~74% of wall). Launch-count
reduction — the class that pays 2-3x on RDNA4 — is therefore structurally dead
here, and that independently explains why the earlier "CUDA graphs for
MUL_MAT_ID" attempt regressed decode to 0.9455: there are no gaps to recover,
only overhead to add.

## What 0001 does

`mul_mat_vec_q` strides the k loop by `blocks_per_iter = vdr*nwarps*warp_size/qi`
and a thread runs an iteration only when `tid/(qi/vdr)` lands below
`blocks_per_row_x = K/qk`. With the GENERIC table's `nwarps = 4`:

| tensor | type | K | k-blocks | `blocks_per_iter` | trips | slots used |
| --- | --- | --- | --- | --- | --- | --- |
| `ffn_gate/up_exps` | Q4_K | 3072 | 12 | 8 | 2 | 12/16 = **75%** |
| `attn_q`, `attn_output`, `output` | Q8_0 | 3072 | 96 | 32 | 3 | 100% |
| `attn_output` (72-head layers) | Q8_0 | 9216 | 288 | 32 | 9 | 100% |

`nwarps = 6` makes `blocks_per_iter` 12 for Q4_K (12/12 → one balanced trip,
zero idle slots) and 48 for Q8_0 (96/48 → 2 trips instead of 3; 288/48 → 6
instead of 9), against a 6-way rather than 4-way cross-warp reduction. Taken
only where `blocks_per_row_x` is an exact multiple of the new
`blocks_per_iter`, so `ffn_down_exps` (4 k-blocks), `ffn_down_shexp` (32) and
every non-Laguna model keep stock geometry.

**Correctness is settled** even though speed is not:
`llama-perplexity` over the fixed corpus is **identical to every digit**,
per chunk and final: 5.7341 ± 0.25780 for both arms (gate is ≤0.5%).
`test-backend-ops test -b CUDA0`: 865/865 MUL_MAT_ID and 1186/1186 MUL_MAT
pass for every geometry tried.

Escape hatch, same binary, both arms: `GGML_CUDA_MMVQ_NW=0|6`,
`GGML_CUDA_MMVQ_NW_Q4K=0` to restrict the wide block to Q8_0.

## Dead levers measured on this track

**1. `rows_per_cuda_block` widening is dead — and it is a cache-residency
trap.** Every decode matvec in this model except `ffn_down_exps` runs
`rows_per_cuda_block == 1`: a 128-thread block re-reads the whole q8_1
activation, walks ONE weight row, then pays a `__syncthreads` + shared-memory
cross-warp reduction + a 5-step warp shuffle to produce a single float. For the
output head that is 100,352 blocks each re-reading the same 3.2 KB activation.
Widening to 2/4/8 rows amortises the epilogue, reuses the activation out of L2
and adds independent weight streams — the variant the XS ledger lists as
untested. In an isolated `test-backend-ops` harness it looked like a huge win:
**+44% on Q4_K gate/up, +15-71% on the 3.3 MB shapes.** It is an illusion.
`eval_perf` builds the graph once with **fixed expert ids** and replays the
same op, so the 10 selected experts (~18 MB) become L2-resident. On the one
genuinely DRAM-bound shape — the 327 MB output head at 228 GB/s — widening
measured **−5.1%**, and end to end it was indistinguishable from its own
control. Real decode streams ~7.5 GB/token cold. This is
`isolated-gemm-must-match-strides` in its **cache-residency** form: reproduce
the working-set residency, not only the strides.

**2. The shared XS GB10 series does not transfer to S.** Applying
`Sources/patches/0001-0011,0015,0016` (13 of 16 apply; 0012-0014 conflict) to
Laguna-S measured **+0.64% decode** — inside the noise floor, and inside the
artifact above. Those patches are dispatch-count and small-kernel folds, and S
is not launch-gap-bound. Concrete case for `per-track-patch-series`.

**3. Isolated op benchmarks mis-rank geometry on this model.** The isolated
harness ranked `nw6` at **−21.4%** on Q4_K gate/up and **+2.5%** on the output
head; end to end neither reproduced. Do not tune GB10 decode geometry on
`test-backend-ops` alone.
