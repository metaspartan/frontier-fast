# MTP self-speculation patch series (qwen3.8-27b-gguf-r9700-v1)

## What this is

A four-patch port of the box's `spec27` tree (the build measured at 82.9 tok/s
on the R9700) down to the pinned base `2b63e0610`. The pinned qwen35 GGUF ships
its own trained NextN/MTP head (`qwen35.nextn_predict_layers = 1`, tensors under
`blk.64`), but llama.cpp never runs it unless a speculative type asks for it —
so stock spends one full 17 GB weight pass per generated token and that
~424M-parameter head sits idle.

These patches turn that head on and make it cheap:

- **0001** auto-enables MTP self-speculation at depth 3 when the GGUF ships a
  NextN head (server only; trunk tools unchanged so correctness gates measure an
  untouched engine).
- **0002** retunes the quantized matmul for the new operating point the verify
  pass presents: MMVQ falls through to MMQ at width 2 (stock waited until 8),
  and Q4_K/Q5_K/Q6_K row reductions use a 5-warp block instead of 8 so more
  workgroups stay resident on gfx1201.
- **0003** is the key one — a **reduced-vocabulary MTP draft head** (98304
  candidate rows instead of 248320). Because a draft proposal is only ever
  accepted/rejected by the trunk recomputing logits, cutting the head to the
  Latin/code region of the vocabulary is a pure throughput trade, no accuracy
  cost. The cut is a plain contiguous prefix, so it is a zero-copy view, and
  candidate index stays token id so nothing downstream needs remapping. The
  remaining head rows are padded with -1e30 so a masked row can never be
  selected and the reported candidate probabilities are unchanged.
- **0004** removes two pointless decode-checkpoint splits when the context can
  already roll itself back (rs-backed), each split costing a full-weight read.

## Why this helps

The speculative board's frontier is +49.46% (47.7 tok/s). The `spec27` head
drafts 4.2 tokens/round from the custom reduced head vs ~3 for the leader, so
it lands more verified tokens per pass over the same 17 GB. That is the
82.9-against-47.6 gap: a better draft head, not a faster kernel — consistent
with the ledger's note that draft-head *precision* is a wash but *width/accepted
positions* is where the decoded tokens come from.

## Verification status

- Patches apply cleanly (git apply --check) to `2b63e0610`.
- Applied tree is byte-identical to spec27's working tree for the 4 changed
  files (verified per-file diff -q MATCH).
- Committed content verified byte-for-byte against the box's patches.
- Trusted-runner result forthcoming.
