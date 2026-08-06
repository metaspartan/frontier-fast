# maple-preview-gguf-gb10cuda-v1 — findings (no series yet)

## The R9700 TQ2_0 port works here mechanically — and is blocked by the gate

Measured 2026-08-06 on the trusted-runner box (gx10-838f), stock vs the
R9700 series' 0001 (native TQ2_0 CUDA) + 0018 (TQ2_0 MMQ tiles), which
apply and build cleanly against the same pinned fork:

- decode tg128: 13.6–15.0 → **131.7–132.5 tok/s** (llama-bench, ~9.2x;
  against the 54.03 server calibration the paired ratio would be ~2.4x)
- prefill pp512: 133 → **2073 tok/s** (~15.5x)
- ppl `-c 512 --chunks 8` on the gate corpus: stock **22.4217** → candidate
  **21.7312** = |3.08%| against the 0.5% equivalence limit → **would be
  rejected**.

## Why: this track's stock reference is an all-CPU computation

The divergence is not the ARM TQ2_0 dot product (it is integer-exact and
order-identical to the generic path; full-CPU stacks on this box and the
R9700 read within ~1%). The key measurements:

- stock `-ngl 99` = 22.4217, stock `-ngl 0` = 22.4228 — **numerically
  identical**. With every ternary matmul unsupported by the CUDA backend,
  the scheduler collapses effectively the whole eval graph to the CPU:
  attention, the Q4_K head, everything.
- candidate (all-GPU) = 21.7312 with FA, 21.5514 with `-fa 0` — GPU
  execution shifts ppl DOWN ~3% on this 2-bit model, dominated by the
  finer-grained q8_1 activation quantization (per-32 scales) versus the
  CPU's Q8_K (per-256), plus fattn and head-path differences.
- the R9700 twin passed its gate (+0.32%) only because RDNA stock already
  ran the head and attention on the GPU — its stock reference is
  GPU-adjacent; this box's is not.

Consequence: **any** GPU offload of the dominant ops moves ppl ~1–3% here,
so the 0.5% equivalence gate cannot be passed by a faithful GPU port. The
options are (a) CPU-parity CUDA kernels replicating Q8_K activation
quantization and CPU op ordering across ALL prefill op classes (a
reimplementation of ggml-cpu numerics on GPU), (b) decode-only offload,
which is blocked by weight placement (the offload probe uses ne11=512, and
`offload_op` would copy ~5 GB of expert weights per token; the `integrated`
UMA flag that could make host buffers directly usable is force-disabled
upstream over #15034), or (c) a gate policy that recognizes the stock
reference here is a CPU computation whose activation-quantization
granularity — not model damage — accounts for the delta.

A ~2.4x decode / ~15x prefill win is sitting behind this gate.
