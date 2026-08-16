# Compounding rounds plan after the tree-verify landing (qwen3.8 r9700)

Written 2026-08-16 while submission 1786899263 ran. Base state: tree machinery
landed (+2.34% local over 47.56 chain frontier, E 2.947→3.527), pass overhead
~+7.5ms over chain eats most of the E gain. Two planned rounds, coordinator-
directed, plus the blocker that gates round 1.

## Round 1: widen the tree onto the mmvf 16-column enabler

Patch: `/home/ghost/o5mm/patchout/0005-mmvf-16-column-float-matvec.patch`
(applies after the frontier series; removes the width-9 rocBLAS cliff;
T(w) = 42.8 + 0.36w flat to 16 with stream-k).

**BLOCKER FIRST — snapshot-plane memory scales catastrophically with tree
size.** Planes = (n_nodes+1); rs rows = n_seq_max × planes where n_seq_max =
n_parallel + n_leaves. The 7-node/3-leaf tree costs 4 cells × 8 planes ×
3.07MB × 48 layers ≈ 4.6GB — fine. A 15-node tree with ~7 leaves is
8 × 16 × 3.07MB × 48 ≈ 18.9GB — OOM with a 16GB model resident. Fix before
any widening: **decouple rs cell count from n_seq_max** — branch seq ids need
no recurrent cells (they only ever touch the slot's own cell via the root
token's id list). Allocate the recurrent memory with rs_size = n_parallel and
make find_slot / seq_rm / seq_cp / seq_rs_select ignore ids >= size (they are
attention-only ids). Then 15 nodes costs 1 cell × 16 planes ≈ 2.4GB. This is
~30 lines in llama-memory-recurrent.cpp + the hybrid constructor sizing.

**Then measure overhead scaling before committing to a shape** (coordinator's
point, and mine): the +7.5ms at 7 nodes has components that scale with node
count (snapshot writes ~0.23ms/node = 147MB at 640GB/s; T slope 0.36ms/node;
GDN reloads ~0.15ms/branch point; conv gather marginal small) and components
that do not (draft levels, host bookkeeping). Sweep the EXISTING machinery +
mmvf at shapes of 8/10/12/15 nodes (env-only after the rs fix + mmvf patch,
e.g. `3,2,2,1,1,...` families) and fit marginal-ms-per-node from the measured
pass times. Honest break-even: a node pays if ΔE ≥ E×(Δms/pass_ms) ≈ 0.045 E
per 0.7ms. From conditional coverage (L1 top4 0.9095, L2 top4 0.8226):
adding L1 rank-3/4 ≈ +0.14 E each (clears), L2 extras ≈ +0.07 E (clears),
L3+ extras ≈ +0.03-0.04 (marginal). Expected landing shape ~11-13 nodes,
E ~3.9-4.1, ratio vs chain ~+8-12% if marginal costs hold. E realised has
exceeded the coverage model by ~8% since the h-routing fix, so measure, don't
trust the model's absolute level.

Note file ownership: use the mmvf patch AS A FILE in the series (0009); do not
edit mmvf/mmq sources.

## Round 2: draft-loop host time (~1.2 ms/step, ~4.8 ms/pass at depth 4)

Finding `draft-step-intercept-is-host-time-not-launch-gap`: per draft step
~46 launches / 1.23ms GPU / 0.10ms launch tax — and ~1.2ms of HOST time
(graph setup, ubatch assembly, bookkeeping) by subtraction from the 2.54ms
instrumented step. At tree depth 4 that is ~4.8ms of a ~57ms pass (~8%), and
it also lowers the marginal cost of depth, compounding with round 1's shape
arithmetic.

Attack order (measure first with host-side timers around the segments of
llama_decode in the draft context — enqueue vs init_batch vs graph-reuse check
vs sched alloc; rocprof wall math is unusable on gfx1201):
1. Graph reuse in the MTP draft context across levels/passes — tree levels
   alternate widths (2,2,1,...) which may defeat can_reuse every step; if so,
   consider padding level batches to a fixed width so the graph is reused
   (verify padding cost < rebuild cost).
2. Batch/ubatch assembly path: common_batch_add + llama_batch_allocr::init
   full validation per 1-2 token batch; a slim path for the draft context.
3. The kernel-side secondary: ~7 device copies/step in the nextn layer's
   state handling (addressing-elision family, ceiling ~0.2-0.3ms/step).

If rounds 1+2 land near arithmetic: 48.7 → mid-50s. The remaining gap to the
~68 ceiling is verify-pass efficiency at widths 9-16 (MMQ stream) + dynamic
shapes.

## Also carried

- Conv gather wants to be one kernel (~1.4ms of the current overhead).
- Dynamic tree shape (widen on hard prefixes) for the per-prompt sign problem
  (hard prompts measured 0.966/0.980 vs easy 1.05-1.07).
