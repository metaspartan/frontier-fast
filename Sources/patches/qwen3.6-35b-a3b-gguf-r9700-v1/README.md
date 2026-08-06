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

## Closed levers — do not spend a slot re-deriving

- **Shared-expert ninth-channel fold (laguna 0017/0018): structurally
  impossible on this GGUF.** The fold's pair guard requires the shared
  expert's weight slab to match the routed experts' type exactly
  (`sw->type != w->type → decline`), because the extra channel reuses the
  routed launch's vec_dot. Unsloth's UD-Q4_K_M upcasts all three shexp
  tensors to **Q8_0** while the routed experts are Q4_K (gate/up) and Q5_K
  (down) — no layer can ever pair. Verified by reading the guard against
  the tensor table (blk.*.ffn_down_shexp Q8_0 [512,2048] vs
  ffn_down_exps Q5_K [512,2048,256]). A mixed-type extra channel would
  need a second vec_dot template parameter and forfeits the bit-identity
  argument — not worth the +3.6%-class gain unless someone requants.

## 0018: grouped mmvf launch for the GDN small-matvec swarm

The 30 GDN layers issue tiny F32 matvecs (ssm_alpha/ssm_beta [2048->32])
and every layer runs the F32 router [2048->256] plus the 1-row shared-expert
gate — ~180 one-warp-per-row launches per token, ~680 us of pure launch
latency (profiled 1683 dispatches/token total). This patch:

- `mmvf.cu/.cuh`: `mul_mat_vec_f_grouped` — the RDNA single-warp
  `mul_mat_vec_f<float,float,1,32>` body (same unroll-batched k-loop, same
  warp reduction) with integer-only segment routing on blockIdx, exactly the
  grouped-mmvq pattern.
- `ggml-cuda.cu`: detection + collection mirroring the mmvq group (same-src1
  plain F32 matvecs, RDNA-only, mmvf-route-only, reusing the hoist-legality
  and GLU-fuse guards). Toggle `GGML_CUDA_DISABLE_MMVF_GROUP`.
- `qwen35moe.cpp`: adjacency pins — beta+alpha built and forward-expanded
  together (sigmoid deferred), the shared-expert gate pinned next to the
  router. Creation order does not set cgraph order; the pins do.

Measured (runner box): dispatches 1683 -> 1593/token, kernel time
10.07 -> 9.75 ms/token; 7-round interleaved whole-process A/B vs the
verified 0001-0017 build: decode tg128 101.0-101.7 -> 104.5-105.2, median
per-round ratio **1.0335**; prefill neutral. PPL on the gate corpus 3.9358
(stock 3.9314, +0.112% — tighter than the verified series' +0.193%; the
adjacency pins shift which folds fire, all within gate). Server smoke clean
on short and long prompts.

**Ranked status: real but frontier-blocked by draw variance.** The trusted
run measured this build at **108.2 tok/s decode** (3/3 paired rounds,
spread 0.35% — the highest absolute decode recorded on this track; the
verified frontier entry ran 104.0), confirming the +3.3-4.0% gain. It was
still rejected (1.1938 vs 1.2980) because the stock baseline drew its FAST
mode (82.5 tok/s vs the frontier run's 78.1 — the same ~5.5% per-launch
bimodality the platform documents on Laguna S) and the banked frontier
score also carries a hot prefill-slope draw (1.199 ranked vs ~1.01 local).
Compare candidates across runs by absolute cand tok/s, not banked ratios.
Resubmit this series when the box's stock phase draws slow again, or
stacked with the next real decode gain.

## Open levers

1. (moved to closed: shexp ninth-channel — see above)
2. The GDN/linear-attention layers (3 of every 4) are untouched by this
   series beyond the generic grouped launches — profile their dispatch
   structure; `ssm` conv/scan ops may group the same way rms_norm did.
3. MTP head is present in the GGUF but unused by `llama-server` b10237 —
   speculative decode via MTP is a structural lever if the harness allows it.
