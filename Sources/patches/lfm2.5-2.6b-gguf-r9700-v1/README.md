# lfm2.5-2.6b-gguf-r9700-v1 — patch series

**Intentionally empty.** No submission has beaten the baseline yet, so this
track's frontier *is* pinned llama.cpp b10237. Your candidate is built as the
pinned tree plus your patch, and measured against the same pinned tree. Add
yours as `0001-` when you land the first win.

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
