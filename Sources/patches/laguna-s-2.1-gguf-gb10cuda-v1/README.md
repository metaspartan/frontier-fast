# laguna-s-2.1-gguf-gb10cuda-v1 — patch series

**This series is intentionally empty.** No submission has yet beaten the
baseline on this track, so its frontier *is* pinned llama.cpp b10237. A
candidate here is built as the pinned tree plus your patch, and measured
against the same pinned tree.

It previously held the eleven patches from the XS GB10 CUDA track. Those were
placed here by analogy when the shared series was split per track, not because
anything had measured them on Laguna S. They have since been measured on S:
**+0.64%, inside noise.** Carrying them made every S submitter rebase onto
eleven patches that were never earned on this hardware and model pair.

Add your patch as `0001-` when you land the first win.

## Before you trust a number on this track

Laguna S has a decode-rate state that is **fixed for a process launch and
independent of your code**: bimodal, roughly 7% wide (~23.4-24.1 vs
~25.5-25.6 tok/s), while the reading *within* a launch is rock steady. Low
variance inside one launch is the artifact's signature, not evidence that your
effect is real - a two-round A/B here measured +6.3% before rounds 3-5 erased
it. Alternate whole process launches and take the median of the per-round
ratios. The trusted runner does three paired boots per submission for this
reason; if your effect is under ~7%, expect to need about five rounds.

## Pending proposal (not yet earned)

`0001-cuda-mmvq-wide-decode-block.patch` is proposed but **unverified**. It
raises `nwarps` 4→6 where the k-blocks divide evenly, so `ffn_gate/up_exps`
(Q4_K, K=3072, 12 k-blocks) goes from two trips at 75% slot use to one balanced
trip. Correctness is settled — perplexity identical to every digit on both arms
(5.7341), `test-backend-ops` 865/865 and 1186/1186 across geometries — but the
**speed is unproven**, because the per-launch artifact above swamps effects of
this size. It carries an escape hatch: `GGML_CUDA_MMVQ_NW=0` restores stock
geometry, `=6` forces the wide block.

This file stays in the series only if a trusted-runner verdict earns it.
