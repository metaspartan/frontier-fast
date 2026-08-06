# qwen3.6-35b-a3b-gguf-r9700-v1 — patch series

Port of the verified laguna-xs-2.1-gguf-r9700-v1 launch-bound decode series
to Qwen3.6-35B-A3B (arch `qwen35moe`: 40 layers, MoE 256 experts top-8 +
shared expert, hybrid linear/full attention every 4th layer, GQA 16/2,
Q4_K_M). Same pinned engine `b10237`, same box, same class of bottleneck:
the R9700 is launch-bound, and this model shares Laguna's dominant shapes
(256-expert top-8 routing, Q4_K matvecs, q8_1 activation quantization).

## What is in the series

0001–0013 and 0014–0016 are the laguna series patches 0001–0013, 0015, 0016
and 0019 unchanged (only renumbered). Two laguna patches are deliberately
**excluded**:

- `0014-mmvf-single-warp-decode-block` (laguna numbering) — **breaks the ppl
  gate on this model**. Measured on the runner box, `llama-perplexity -c 512
  --chunks 8` over `fixtures/gainz-corpus.txt`: stock 8.8178, full series
  8.8723 (**+0.618%, gate is 0.5%**); reverting only this patch returns
  bit-exact stock ppl 8.8178. Qwen3.6's linear-attention (GDN) layers push
  f32/f16 matvec shapes through `mmvf` that Laguna never exercises, and the
  single-warp accumulation change shifts the distribution. Do not re-add it
  without a shape gate that keeps it off the GDN paths.
- `0017/0018-moe-shexp-*-ninth-channel` (laguna numbering) — wired through
  `src/models/laguna.cpp`; qwen35moe builds its shared expert through a
  different graph path. Re-deriving that fold for `build_qwen35moe` is the
  top open lever here (+3.6% same-binary on laguna).

## Measured (runner box, interleaved whole-process A/B, 5 rounds)

- decode tg128: stock 80.95–81.34 tok/s, candidate 99.38–99.78 tok/s —
  median per-round ratio **1.2274**
- prefill pp512: median ratio ~1.011
- ppl gate: stock 8.8178, candidate 8.8207 — **+0.033%**, passes
- candidate `llama-server` boots, serves, exits clean (0019/pool fix in)

The A/B without the mmvf patch is *more* stable than with it (no bimodal
rounds observed in 5 rounds).

## Open levers

1. Shared-expert ninth-channel fold adapted to `build_qwen35moe` (see above).
2. The GDN/linear-attention layers (3 of every 4) are untouched by this
   series beyond the generic grouped launches — profile their dispatch
   structure; `ssm` conv/scan ops may group the same way rms_norm did.
3. MTP head is present in the GGUF but unused by `llama-server` b10237 —
   speculative decode via MTP is a structural lever if the harness allows it.
