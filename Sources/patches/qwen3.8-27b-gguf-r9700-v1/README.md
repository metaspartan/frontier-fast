# qwen3.8-27b-gguf-r9700-v1 — patch series

The series applies to llama.cpp `2b63e0610bbc2be990ae1360d5256efcdc3f9efb`, in
filename order.

- `0001` RDNA4 5-warp block for the K-quant batch-1 matvec
- `0002` cache the q8_1 activation quantization across matvecs sharing one src1
- `0003` `rms_norm` writes the q8_1 form of the activation it produces
- `0004` the fused unary+mul gating the attention output does the same
- `0005` `gated_delta_net` computes its own gate and beta

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
   8.7%. Nobody has censused it here yet.

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

`0005` took the three that were pure per-head scalars. The two `l2_norm` launches are
the next candidates by the same rule, but they are *not* free the way `0005` was:
`l2_norm` runs one workgroup per k-head (16) against `gated_delta_net`'s 48, so folding
them recomputes each reduction three times. Price that ratio before building it, and
see the `-funsafe-math-optimizations` warning below — it applies to any producer fold
on this backend.

**ggml-hip here is built with `-funsafe-math-optimizations`.** A value that stock
reads through a global load is opaque to the optimizer; the same value computed
in-register is not, and the compiler will reassociate across it. This is not a
tolerance question — it silently moved perplexity by up to 0.095% on a fold whose
folded values were provably bit-identical. Re-impose the barrier explicitly.
