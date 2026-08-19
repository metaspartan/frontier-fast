# qwen3.8-27b-gguf-r9700-v1 — patch series

Record series for the R9700 (gfx1201) self-hosted runner, stacked in lexical
order onto the pinned llama.cpp `2b63e0610bbc2be990ae1360d5256efcdc3f9efb`.

## The model

Qwen3.8-27B Q4_K_M (`unsloth/Qwen3.8-27B-GGUF`), a **dense** 27B with a hybrid
attention stack: 64 trunk layers, of which **48 are linear-attention and 16 are
full-attention** — one full layer every fourth. The file also ships a trained
NextN/MTP head (`qwen35.nextn_predict_layers = 1`, tensors under `blk.64`),
which llama.cpp executes only under a speculative context.

Dense means every weight is read every token: decode is weight-bandwidth bound
(~16.74 GB/token trunk), so a win here moves fewer bytes or spends fewer
launches. Speculation is the only structural escape from the bandwidth floor,
and MTP self-speculation (draft = the model's own head, exact verification) is
legal: greedy output is unchanged, the correctness gate is perplexity
equivalence (<= 0.1% relative delta).

## Series

- **0001 — 5-warp mmvq blocks for batch-1 K-quant matvec.** RDNA4 register/
  occupancy retune of the decode matvec.
- **0002 — MTP self-speculation default-on.** Server-only default in
  `common/arg.cpp`: when the GGUF declares a NextN head, enable
  `--spec-type draft-mtp` (n_draft 3) instead of decoding one token per weight
  pass. First structural escape from the decode floor.
- **0003 — MMVQ/MMQ crossover at speculative verify widths.** At verify widths
  3–8 the batch-1 MMVQ path is the wrong tool; route to the MMQ tile.
- **0004 — MTP draft head projects the Latin prefix of the vocabulary.**
  The MTP block borrows `output.weight` (Q6_K, 248320 rows); projecting only
  the ~98304-row Latin prefix cuts the draft step's LM-head read from 1042.9 MB
  to ~412 MB with identical acceptance (max draft argmax id 88013).
- **0005 — stream-k MMQ at speculative verify widths.**
- **0006 — double-buffered J=16 y tile at verify widths.** Halves MMQ barriers
  per K iteration. 0005+0006 took the record 44.98 -> 47.56 tok/s (+49.30%).
- **0007 — tree-speculative-draft.** The NextN head draws a speculative TREE
  (level-by-level batches); default shape 2,2,1,1,0,0,1 (7 nodes, depth 4).
- **0008 — tree-verify-on-hybrid-recurrent-memory.** Verifies the whole tree in
  ONE trunk weight pass: `split_tree` emits a single-lane ubatch, the GDN tree
  kernel reads each token's state from its parent's snapshot plane, and the
  winning node's snapshot is selected via `seq_rs_select` after acceptance.
  Runner-verified machinery (E 3.53 local, decode 46.70 ranked).
- **0009 — rs-decouple.** Sizes the recurrent memory by `n_parallel` (serving
  slots) instead of `n_seq_max` (which includes the tree's branch seq ids):
  branch ids own no cells, their state lives in the primary cell's snapshot
  planes. Drops a 13-node tree's recurrent memory from 18.6 GB to 2.1 GB.
  Memory ops guard out-of-range seq ids as no-ops.
- **0010 — mmvf-16 column instantiation.** F32 float matvecs (ssm_alpha/beta)
  instantiated to 16 columns so verify widths past 8 do not fall to a 64x64x16
  rocBLAS SGEMM (measured 92.2 ms vs 2.6 ms for the vector kernel). T(w) =
  42.8 + 0.36w flat to 16 with stream-k.
- **0011 — tree-shape-13-deep3.** Default shape 4,2,2,2,2,0,0,0,0,0,0,0,0
  (13 nodes, depth 3, 8 leaves): widen level 1 (top-4 coverage) where the rank
  coverage pays most, keep depth 3 (depth-4 measured -7.8% on the ranked
  corpus; the fourth level is the wrong bet).

## Verified history (board)

- 47.56 tok/s decode (+49.30%), prefill 924.9 — 0006 (record)
- 46.89 (+47.88%) — 0005
- 44.98 (+43.90%) — 0004
- 42.47 (+38.51%) — 0003+0004
- 40.74 (+35.18%) — 0002+0003
- 28.35 tok/s — stock baseline

## Measurement discipline

- The dispatch census (~1462 per decoded token, ~2.08 µs each) bounds
  launch-structure work: ~160 launches buy 1% of decode.
- Per-launch kernel wins at batch-1 do NOT compose into the speculative
  verify path (MMQ widths), where MMVQ never fires — measured dead in
  kernel-frontier-does-not-compose-with-speculation-qwen38.
- The draft step costs ~1.1-2.5 ms (kernel + host graph rebuild + D2D copies);
  that, not the verify pass, caps chain depth (depth-4 measured -7.8%).
- The ranked corpus draw sits at the hard end of the per-prompt sign spread:
  the 7-node tree measured +2.38% local / -1.89% runner.
