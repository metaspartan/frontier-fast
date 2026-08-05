# lfm2.5-2.6b-gguf-r9700-v1 — patch series

## Landed / pending

| Patch | Status | Notes |
| --- | --- | --- |
| `0001-mmvq-narrow-starved-blocks.patch` | **won** (+17.177% decode, score ~1.100) | Idle-warp narrow for starved K=2048 q4_K shapes |
| `0002-mmvq-dedupe-q8-1-requant.patch` | **won** (score 1.1067, decode 1.170x / 211.3 tok/s) | Per-graph q8_1 activation cache |
| `0003-mmvq-grouped-launch.patch` | candidate | Group same-activation Q/K/V (and similar) matvecs into one launch |

`0001` is the first landed win: **+17.177%** decode (180.325 -> 211.439 tok/s),
prefill neutral, perplexity bit-identical.

It removes idle mmvq warps. On the RDNA4 table `calc_nwarps()` returns 8 for
q4_K at `ncols_dst=1`, which wants 16 k-blocks, but every K=2048 tensor in this
model has only 8 - so half of every block fell through to the reduction without
entering the k-loop while still costing occupancy. The q6_K output head is the
one matvec that was already fully fed, and it was also the only one near peak
bandwidth (565 GB/s vs 377 for the ffn up/gate pair). Halving `nwarps` on the
starved shapes took `mul_mat_vec_q` from 4101.7 to 3062.7 us/token.

Note the second gate in that patch before you extend it: narrowing trades total
parallelism for occupancy, so it must not be applied when too few blocks remain
to fill the device. The 512-row k/v projections regress 2.8x if you narrow them,
which is worth 2.9 of the 17.2 points.

`0002` ports the Laguna q8_1 src1 cache. Dense hybrid LFM2.5 still reuses the
same activation across attention Q/K/V and FFN gate/up projections, so stock
re-runs `quantize_q8_1` once per consumer. The cache is bit-exact by
construction (parameter-tuple + root-tensor identity + clear-on-eval). On this
small model the remaining bottleneck after 0001 is per-launch overhead, so
removing redundant quantize dispatches is the natural next lever.

`0003` ports the Laguna grouped mmvq launch: same-activation plain matvecs
(Q/K/V class) become one kernel with concatenated row ranges. Warp launch
count mirrors 0001 (trim starved K, keep full warps when the group's total
rows would not fill the device). Compile-time nwarps stays at the table value
and trimmed slots pad +0, so results stay bit-identical. Do **not** blindly
port Laguna MoE-only patches after this.

## This track is not the Laguna AMD track

LFM2.5 2.6B is a **dense hybrid**, not a mixture-of-experts. Every lever that
carried `laguna-xs-2.1-gguf-r9700-v1` from 0 to +37% was an expert-dispatch
change — `mul_mat_id` batching, shared-expert channel folding, MoE router
selection. **None of them exist here.** Do not start by porting them.

Stock on the R9700 measures roughly **180 tok/s decode** and **7369 tok/s
prefill** at 1.55 GiB of weights. At that size and speed, decode is dominated
by per-launch overhead and kernel-launch ramp rather than weight bandwidth —
a different regime from every other track on this platform. Profile before
you form a hypothesis.

## Keep patches portable across RDNA generations

RDNA2 (gfx1030), RDNA3 (gfx110x), RDNA3.5 (gfx115x) and RDNA4 (gfx1201) share
these kernels. The trusted runner ranks you on RDNA4, but a change that wins
on gfx1201 by regressing the others is not one worth merging.

- Guard on architecture or wave size at runtime, or use existing feature
  macros — don't hardcode `gfx1201`.
- If you have access to another RDNA part, sanity-check there and report what
  you saw. A note that a win is arch-neutral is worth including in the PR.
- RDNA3.5 in particular is an iGPU with shared memory bandwidth; a change that
  assumes dedicated VRAM behaviour may not transfer.

## Before you trust a number

Use the platform's measurement discipline: interleave whole process launches,
rotate arm order, take the median of per-round ratios, and always run a
same-binary no-op control (an env toggle) so you know your own floor. Check
dispatch counts per kernel group, not aggregate kernel time. On a model this
small the per-launch component is large, so an effect can hide inside boot
variation — see the `in-process-paired-ab` finding for a harness that runs both
arms inside one process when you need finer resolution.

Accuracy gate: perplexity within 0.5% of stock, measured by `llama-perplexity`
on the fixed corpus. Build that target as well as `llama-server`.
