# qwen3.8-27b-gguf-r9700-v1 — patch series

The series applies to llama.cpp `2b63e0610bbc2be990ae1360d5256efcdc3f9efb`, in
filename order.

- `0001` RDNA4 5-warp block for the K-quant batch-1 matvec
- `0002` cache the q8_1 activation quantization across matvecs sharing one src1
- `0003` `rms_norm` writes the q8_1 form of the activation it produces
- `0004` the fused unary+mul gating the attention output does the same
- `0005` `gated_delta_net` computes its own gate and beta
- `0006` `gated_delta_net` reads its recurrent state in place, eliding the `build_rs` gather
- `0007` `gated_delta_net` normalises q and k in register, eliding both `l2_norm` launches
- `0008` one placement kernel replaces the conv-state gather, concat and write-back cpy

## The model

Qwen3.8-27B Q4_K_M (`unsloth/Qwen3.8-27B-GGUF`), a **dense** 27B with a hybrid
attention stack: 64 trunk layers, of which **48 are linear-attention and 16 are
full-attention** — one full layer every fourth.

Two things follow from that, and they are what make this track different from the
MoE tracks next to it:

1. **Dense means every weight is read every token.** There is no routing slack.
   Decode is weight-bandwidth bound from the first token, and a patch that wins
   here has to move fewer bytes or spend fewer launches — it cannot win by
   touching less of the model.
2. **The linear-attention layers carry recurrent state**, the way a Mamba stack
   does. On the sibling Nemotron tracks that state turned out to be a large,
   *uncounted* traffic term, and correcting for it moved a published ceiling by
   8.7%. Censused here by `0006`: the state is 128x128 floats per head across 48
   heads, so **3.1 MB per linear layer and 151 MB per token** read and written -
   about 1.8% of the token's bytes on top of the 16.74 GB of weights, and it is
   not in the published roofline.

## Engine

Stock pinned engine — no custom pin. The GGUF declares
`general.architecture = qwen35` and llama.cpp `2b63e0610` already implements
`LLM_ARCH_QWEN35`, so the standard build works unmodified.

## The MTP block

The file declares `block_count = 65` against the config's 64 hidden layers,
because `qwen35.nextn_predict_layers = 1` puts a multi-token-prediction block at
`blk.64` (15 tensors, including `nextn.eh_proj`). llama.cpp loads it but executes
it only when the context type is `LLAMA_CONTEXT_TYPE_MTP`, so during a ranked
decode its **424,699,392 params / 289,527,808 bytes never move**.

The track's published parameter and byte figures are the trunk only, for that
reason. If you are computing bytes-per-token, exclude `blk.64.*` — counting it
inflates your denominator and will make your kernel look better than it is.

## Where to look first

**Correct the launch-cost arithmetic before you price anything.** A dispatch costs
~2.08 µs on this box, but "1% of decode" is `0.01 × token_duration / dispatch_cost`,
so it scales with the token. This model's token is ~32.9 ms, which makes 1% of decode
about **170 launches** counting launch structure alone, or ~118 once you also count
the kernel time a removed dispatch takes with it. An earlier revision of this file
said "about 20", which is the figure for the ~6 ms tokens on the Laguna and Nemotron
boards; carrying it here understates every launch-structure idea by a factor of eight.

Census at the `0001-0005` frontier, `rocprofv3 --kernel-trace` ordered by
`Dispatch_Id` with full template names: **1316.0 dispatches and ~30.0 ms of kernel
per decoded token**, i.e. kernel time is ~91% of the token and the remaining ~2.5 ms
of inter-kernel gap is still the largest single pool.

The matvecs are ~82% of the token and are already at ~94% of achievable streaming
bandwidth, so the pool that is left is launch structure, and it is concentrated in the
**48 linear-attention layers**, which issue about 11 small kernels each:

| kernel | per token | µs each |
|---|---|---|
| `l2_norm_f32<32>` (q and k) | 96.1 | 1.99 |
| `k_bin_bcast` | 48.0 | 1.6 |
| `cpy_scalar` | 64.0 | 2.29 |
| `k_get_rows_float` / `_vec` | 49.1 / 48.0 | 1.70 / 6.39 |
| `concat_cont` | 48.0 | 1.65 |
| `ssm_conv_f32` | 48.0 | 1.52 |
| `unary_gated<silu>` | 48.1 | 1.42 |

`0006` took `k_get_rows_float_vec`, which is not a launch-structure win at all: it is
`build_rs` gathering this sequence's state row out of the recurrent buffer so the kernel
can read the copy. At batch 1 that moves the layer's whole 3.1 MB state to select ONE
row. The kernel resolves the row itself with a single scalar load and reads in place, so
the population goes 48.0 -> 0.0 and every other population is unchanged. **Gate it to
`n_tokens == 1`**: at prefill the gather is amortised over the ubatch and worth nothing,
and the kernel's `keep_rs_t` path writes rollback snapshots into the same buffer the fold
would be reading, which moved gate-shape perplexity 2.9608 -> 2.9655 when left enabled
there. Decode-gated, both runner perplexity shapes are identical in every digit.

`0005` took the three that were pure per-head scalars. `0007` took the two `l2_norm`
launches, and the grid-ratio pricing that this file previously used to decline them was
wrong in **both** directions. The real redundancy is far worse than the "three times"
claimed here before — `gated_delta_net` runs 1536 workgroups of 4 warps, 6144 warps
against `l2_norm`'s 16, so 384x — *and* the ratio is the wrong question, because a grid
ratio prices RE-READING the producer's input and there is nothing to re-read: a gdn warp
already holds the whole 128-element q and k in `q_reg`/`k_reg` under exactly the
element-to-lane mapping `l2_norm_f32<32>` uses, and `block_reduce<SUM>` degenerates to
the same `warp_reduce_sum` at `block_size == WARP_SIZE`. The fold costs 2 warp reductions
and 2 rsqrts per warp on resident values: measured +49.5 us/token of gdn against 194.7 us
of `l2_norm` plus 96 launches, netting +0.86%. Registers went 38 -> 39 with zero spills.

**The transferable rule: price a fold by the grid ratio only when the consumer must
RE-READ the producer's input. When one consumer warp already holds the producer's entire
reduction domain in registers, the ratio is irrelevant and the cost is ALU.** The test is
`producer_reduction_domain` fits in one consumer warp. It does not reopen everything - the
`ssm_norm` over gdn's output reduces across all 128 columns of a head while a gdn warp holds
only one, so that fold stays dead by grid ratio.

See the `-funsafe-math-optimizations` warning below — it applies to any producer fold
on this backend, and `0007` needed it on every normalised value.

`0008` closed the conv-state path (get_rows + concat + cpy + ssm_conv, 4 launches per
layer) and drew the family's boundary in doing so. The full fold - recompute the 4-tap
dot and silu in the fused kernel, 4 -> 1 - measured +1.51% but moved the DECODE-shape
perplexity +0.102% while the gate shape stayed identical, the mirror image of 0006's
divergence. The ISA shows why and it is not fixable from source: under unsafe math the
compiler reassociates stock's own dot (v_fma x3 first, then x1, x2, x0), chooses that
order per function, and even varies it between two loop paths of one kernel - arithmetic
recompiled in a new function cannot promise stock's rounding. The shipped 0008 is the
placement-only variant, 3 -> 1 with the stock ssm_conv+silu consuming byte-identical
input: +0.90%, identical in every digit at both shapes. **The rule: an addressing
elision may move ARITHMETIC only if the consumer's instruction stream is unchanged
(0006, 0007 fold INTO an existing kernel); a fusion that re-implements a float
computation in a new kernel is capped by the compiler's reassociation freedom, and on
this backend that freedom is real.**

**ggml-hip here is built with `-funsafe-math-optimizations`.** A value that stock
reads through a global load is opaque to the optimizer; the same value computed
in-register is not, and the compiler will reassociate across it. This is not a
tolerance question — it silently moved perplexity by up to 0.095% on a fold whose
folded values were provably bit-identical. Re-impose the barrier explicitly.
