# qwen3.8-27b-gguf-gb10cuda-v1 — patch series

Three patches, all in the batch-1 quantized matvec, applied in filename order on top
of llama.cpp `2b63e0610bbc2be990ae1360d5256efcdc3f9efb`:

| patch | what |
|---|---|
| 0001 | coalesce the K-quant `dm`+`scales` header into one 128-bit load |
| 0002 | `__launch_bounds__` names a residency target, so ptxas compiles to the 40-register budget that 100% occupancy allows |
| 0003 | eight-warp blocks for Q4_K/Q5_K/Q6_K at `ncols_dst == 1` |

`mul_mat_vec_q` is **93.1% of decode GPU time** here, so this is where the track
lives.

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
largely unchanged.

## Occupancy and warp-count trades are NOT dead here — but only as a pair

An earlier version of this file said device-table / occupancy / warp-count trades
"are RDNA-shaped and usually lose on sm_121", and the ledger carries two DEAD
findings that agree: forced occupancy measured **−2.8%** and eight-warp blocks
measured a **null**. Both were correct when they were taken and both are now
superseded, for a reason worth internalising.

**These two levers are coupled through the register file, so testing them one at a
time measures the wrong thing.** Registers per thread decide residency:

| tree | regs | threads/blk | blocks/SM | warps/SM (of 48) |
|---|---|---|---|---|
| pinned | 56 | 128 | 9 | 36 |
| +0001 | 48 | 128 | 10 | 40 |
| +0002 | 40 | 128 | 12 | **48** |
| 0003 without 0002 | 48 | 256 | 5 | 40 |
| +0002 +0003 | 40 | 256 | 6 | **48** |

Eight warps alone buy a wider fetch and pay for it in occupancy — that is the null.
The launch bound alone holds occupancy but keeps the narrow fetch. Together they are
wide *and* fully resident, and that is **+0.82%** over 0001, reproduced across three
independent sessions (+0.674 / +0.835 / +0.817%).

**0001 is also what unlocked it.** Removing the narrow header loads freed 8 registers
on Q4_K (56 → 48), turning the squeeze to a 40-register budget from 29% into 17%.
That is the difference between the old −2.8% and the result above. Before you trust a
DEAD occupancy finding on this kernel, re-read `cuobjdump -res-usage` — the register
count is the state variable, and any patch that changes it re-opens the family.

Block width was swept at 100% occupancy: 64 threads **−4.12%**, 128 **+0.43%**, 256
**+0.84%**, 512 **−0.64%**. The optimum is 8 warps. At 512 threads only 20 of 32 lane
groups have work at this model's dominant `K = 5120`.

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
2. ~~**The bandwidth is still the 273 GB/s datasheet figure.**~~ **Fixed.** It has
   since been measured at the geometry `mul_mat_vec_q` actually issues, with an
   L2-residency control at identical block shape: **249 GB/s**, ceiling 14.87
   tok/s. Do not use 273, and do not import a grid-stride number from another
   track — on this box achievable bandwidth depends on the access pattern
   (grid-stride 233–241, one-block-per-row 247.5 at 128 threads, 252.3 with
   128-bit loads at 256 threads).

A byte census including the recurrent and conv state would replace the first one.
That is still worth reporting even with no performance patch attached.

## Memory parallelism dominates memory instruction count

Worth knowing before you spend a slot cutting loads. Two output rows per block at
`ncols_dst == 1` does exactly what it promises — SASS confirms the q8_1 activation
and its scale are shared between rows, so loads fall from 9 per output row to 6 and
dp4a from 8 to 6 (the activation-sum term `dot2` is row-independent) — and it
measures **−4.15%**. It also collapses the compiler's unroll of the K loop from 3
iterations to 1, and the loads-in-flight that costs are worth far more than the
instructions saved. Count what a change does to loads *in flight*, not just to loads.
