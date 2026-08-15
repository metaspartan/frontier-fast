# qwen3.8-27b-gguf-gb10cuda-v1 — patch series

**Empty by design.** Commissioned 2026-08-14 with no landed wins, so this track's
frontier *is* the pinned tree: a candidate is measured as vanilla llama.cpp
`2b63e0610bbc2be990ae1360d5256efcdc3f9efb` plus your own patches. The directory
exists so the runner can tell "empty on purpose" from "wrong track". Do not
delete it.

## The model

Qwen3.8-27B Q4_K_M (`unsloth/Qwen3.8-27B-GGUF`) — the same GGUF the R9700 twin
runs. A **dense** 27B, 64 trunk layers, **48 linear-attention to 16
full-attention** (one full every fourth).

Dense means every weight is read every token: no routing slack, and decode is
weight-bandwidth bound from the first token. Trunk weights are
**16,806,250,496 bytes/token**.

## What is different about this box

This is the GB10, not the R9700, and the difference decides what is worth trying.

- **Launch-structure work on this track is dead, with a number attached.** A
  dispatch costs ~0.88 µs here, and the decode token is ~84 ms, so **~960
  dispatches buy 1% of decode**. The census counts ~1719 dispatches per token —
  so deleting *every kernel launch in the token* would buy **1.8%**.
  (An earlier version of this file said "~95 launches buy 1%". That was a
  small-model constant carried onto a dense 27B and it was wrong by 10x. The
  ratio is `0.01 x token_duration / dispatch_cost`; compute it, do not inherit
  it.)
- **Rank candidates by BYTES removed, not dispatches removed.** That rule has now
  held on three separate ports to this box.

If you are porting from the R9700 twin, triage by directory: `src/` (core C++ and
graph-level) applies verbatim and ports best, `ggml/src/ggml-cuda` transfers
largely unchanged, and device-table / occupancy / warp-count trades are
RDNA-shaped and usually lose on sm_121.

## The MTP block

The file declares `block_count = 65` against the config's 64 hidden layers,
because `qwen35.nextn_predict_layers = 1` puts a multi-token-prediction block at
`blk.64`. llama.cpp loads it but executes it only under
`LLAMA_CONTEXT_TYPE_MTP`, so its **424,699,392 params / 289,527,808 bytes never
move** during a ranked decode. Exclude `blk.64.*` from any bytes-per-token
figure.

## Two numbers here are known-soft — please fix them

1. **The ceiling is weights-only.** It omits the recurrent and conv state of the
   48 linear-attention layers. On *this box*, that exact omission made the
   Nemotron ceiling 8.7% optimistic until someone measured it.
2. **The bandwidth is still the 273 GB/s datasheet figure.** Two measurements on
   this box have since shown that to be roughly 18% high for large streaming
   models — 223.3 GB/s and 231.8 GB/s were measured on two other tracks. Neither
   was imported here, because a number measured on a different model is a guess,
   not a measurement.

A byte census including the state, plus one geometry-matched bandwidth
measurement, would replace both. That is the highest-value first contribution to
this track, and it is worth reporting even with no performance patch attached.
