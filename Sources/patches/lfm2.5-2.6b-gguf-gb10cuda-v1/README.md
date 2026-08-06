# lfm2.5-2.6b-gguf-gb10cuda-v1 — patch series

The laguna-xs gb10cuda engine family 0001–0011, unchanged (q8_1 requant
dedupe, grouped launches with grouped-mmvq off by default, norm/rope/
set-rows groups, quantize folds, mmvf batched k-loads). No LFM-specific
patches yet — the R9700 LFM levers (mmvq narrow, pre-add-norm fold) are
RDNA geometry plays the sm_121 findings warn against porting blind.

## Measured (runner box gx10-838f, 5-round interleaved whole-process A/B)

- decode tg128: stock 74.30–75.17 → cand 74.80–75.40, median per-round
  ratio **1.0091**
- prefill pp512: 3008–3017 → 3034–3042, median ~1.0083
- ppl `-c 512 --chunks 8`: **22.7466 → 22.7466 bit-identical**

Consistent with the platform's cross-port finding (the +8.69% AMD dedupe
was worth ~0.6% here): this box is memory-latency bound, not launch bound,
and LFM's small graph gives the launch levers little to remove. Submitted
to establish the frontier on a null track; expect ~+0.8%.
