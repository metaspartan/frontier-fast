Track `ornith-1.5-35b-a3b-gguf-r9700-v1` — MTP self-speculation on a sparse trunk.

The GGUF ships a trained NextN/MTP head at `blk.40`
(`qwen35moe.nextn_predict_layers = 1`), so this track self-speculates with no
second model and pays none of the draft-prefill tax a separate drafter costs.

* `0001-mtp-autoenable.patch` — turn MTP on when the GGUF carries a NextN head,
  at depth **1**. The depth is the whole story on this trunk: see the comment in
  the patch for the measured `ms/pass` against verify width. Depth 3, which pays
  on the dense sibling of this family, measures **-22.6%** here.
* `0002-mmvq-verify-width.patch` — MMVQ/MMQ crossover and the 5-warp Q4_K/Q5_K/
  Q6_K block. Ported unchanged from `qwen3.8-27b-gguf-r9700-v1`.
* `0003-recurrent-rollback-splits.patch` — do not split the speculative prompt
  around checkpoints when the hybrid memory can already roll itself back that
  far. Ported unchanged.

Not ported, and why — recorded so the next agent does not re-measure a null:

* The reduced MTP draft head (`0003`/`0008` on the dense track) edits
  `src/models/qwen35.cpp`. llama.cpp keeps a **separate** model class for the
  MoE variant in `src/models/qwen35moe.cpp`, with its own `graph_mtp`, so those
  patches are dead code on this GGUF. Censused rather than assumed: sweeping
  `GAINZ_MTP_HEAD_K` over 0 / 98304 / 49152 / 32768 / 16384 / 8192 leaves the
  per-slice accepted-length vector and the cumulative draft time bit-identical
  arm for arm, and decode flat within 0.23%.
* Tree verify and the tree-conv patches (`0011`-`0014`) need
  `build_conv_state_tree`, which `qwen35moe.cpp` never calls — the MoE class
  has no tree path at all. Even with one, the node price here (~1.46 ms, ~10%
  of a full pass) puts tree break-even at ~0.29 accepted tokens per added node
  against a measured position-2 acceptance of 0.048.
