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
- **0007 — greedy MTP draft via GPU argmax.** The chain draft loop took a
  tempered top-10 sample (top_k 10, temp 0.8, seeded dist) instead of the
  head's argmax, leaving rank-1 mass on the table (achieved acceptance ~0.62
  vs rank-1 coverage ~0.72). Append `ggml_argmax` to the MTP graph (over the
  mask-padded logits), expose `llama_get_logits_argmax_ith`, and take the
  argmax in the chain loop — no host logits read, no sampler, no RNG. The
  tree path keeps the sampler (needs top-k diversity per node). CPU smoke:
  acceptance 0.566 / mean len 2.70 vs 0.45-0.51 / 2.33-2.52 tempered.

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
  the 7-node tree measured +2.38% local / -1.89% runner, the 13-node tree
  -8.0% (speculative-structure-beyond-depth3-chain-is-dead-on-the-ranked-
  corpus-qwen38).
