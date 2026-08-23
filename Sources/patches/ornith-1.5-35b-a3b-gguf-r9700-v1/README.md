Track `ornith-1.5-35b-a3b-gguf-r9700-v1` — MMVQ block shape on a sparse trunk.

* `0001-mmvq-5warp-kblock.patch` — a 5-warp rather than 8-warp MMVQ block for
  Q4_K/Q5_K/Q6_K. **86.22 -> 96.12 tok/s, +11.49%.** This is one hunk of the
  dense track's `0002-mmvq-verify-width`; the other hunk (the MMVQ->MMQ
  crossover, which only bites at `ne11` in 3..8) is not shipped because it can
  only move a speculative verify pass, and speculation is not profitable here.
  Measured separately: the crossover hunk contributes nothing at batch-1 decode
  (96.07 with it, 96.12 without, against a 0.13% within-arm spread).

  The 5-warp block is worth much more on this trunk than on a dense one. A dense
  decode is a handful of large matvecs; an A3B decode is ~40 layers x 8 routed
  experts of `[2048 x 512]` plus a shared expert — hundreds of small independent
  row reductions per token, and small reductions are exactly where occupancy
  rather than tile efficiency sets achieved bandwidth.

## Speculation is a net loss on this track — measured, not assumed

The GGUF ships a trained NextN/MTP head at `blk.40`
(`qwen35moe.nextn_predict_layers = 1`), and the obvious move — the one worth
+82% on this family's dense sibling — is to self-speculate against it. It does
not pay here, and the reason is one number a dense trunk does not have.

**The node price.** A verify pass over `k` positions on a dense model costs about
what one position costs: the pass streams one fixed weight set and the extra
tokens ride along. On a 256-expert A3B trunk each token routes to its **own** 8
experts, so `k` tokens touch up to `8k` distinct expert tiles per layer.
`llama-bench -p k -n 0 -r 5`:

| positions | 1 | 2 | 3 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|
| ms/pass | 14.07 | 16.33 | 17.25 | 18.53 | 24.27 | 27.84 | 30.19 |

~1.46 ms per extra verified position — 10% of a full pass per node, against
1–2.7% on the dense sibling. The marginal cost falls again past width 8 (0.89
then 0.59 ms) exactly as expert overlap predicts, which is what confirms this is
fan-out and not fixed launch overhead.

With `0001` in, a non-speculative decode step is 10.40 ms, so a single extra
verified position is 14% of it before any draft cost at all. Measured best
speculative configuration: **90.07 tok/s against 96.12 with speculation off.**

| configuration | decode | E |
|---|---|---|
| `0001`, no speculation | **96.12** | — |
| `0001` + MTP depth 1 + MoE head cut | 90.07 | 1.3333 |
| `0001` + MTP depth 1 | 89.51 | 1.3474 |
| stock | 86.22 | — |

Depth is 1 because the head's per-position acceptance is 0.379 / 0.048 / 0.002
against a break-even of ~0.29 per added node; depths 2 and 3 measure -13.8% and
-22.6% against stock.

## Two absent levers, censused

llama.cpp keeps **two** model classes for this architecture family:
`llama_model_qwen35` (dense, `src/models/qwen35.cpp`) and
`llama_model_qwen35moe` (`src/models/qwen35moe.cpp`), each with its own
`graph_mtp`. Every dense-track patch that edits `qwen35.cpp` is dead code here:

* The reduced MTP draft head (`0003`/`0008` on the dense track) never runs.
  Sweeping `GAINZ_MTP_HEAD_K` over 0 / 98304 / 49152 / 32768 / 16384 / 8192
  leaves the per-slice accepted-length vector bit-identical (`E = 1.3474` in all
  six), cumulative draft time within 0.2%, greedy text hash identical, decode
  flat within 0.23%. Ported into `qwen35moe::graph_mtp` it does work — draft time
  1396 -> 895 ms and +3.3% on the speculative arm — it just cannot close a 6.6%
  gap.
* Tree verify and `0011`–`0014` need `build_conv_state_tree`; `qwen35moe.cpp`
  calls only `build_conv_state` and never consults `inp->tree_src`, so the MoE
  class has no tree path at all.

## Note for the runner

The `serving.json` validator rejects `specType: "draft-mtp"` with "needs a draft
model and this track has none pinned", which is backwards for a model carrying
its own NextN head — the track metadata itself sets `customDraftHeads: true`.
Not load-bearing for this submission, which ships no speculation, but it blocks
the declarative route for anyone who wants to.
