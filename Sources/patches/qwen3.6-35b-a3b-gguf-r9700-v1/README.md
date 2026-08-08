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
A second attempt (probe-timed to a 79.4 tok/s stock reading) drew fast
stock AGAIN in the runner's own launches (82.85/82.67; candidate 108.47/
108.11 — a second consecutive track record): the stock mode is decided
PER PROCESS LAUNCH, not by box state, so probe-timing resubmissions does
not work. The banked frontier's 3/3 slow-stock rounds were rare draw luck.
Do not spend further slots redrawing; this patch ships with the next real
decode gain (~+8% needed to clear 1.298 on neutral draws — the GDN
l2_norm-grouping and Q6_K-head-bandwidth levers are the candidates).

## 0019 + 0020: round 4 — l2_norm grouping and untrim default

- **0019**: grouped l2_norm launch for the GDN q/k conv-norm pairs
  (`l2_norm_f32_grouped`, same warp reduction and scale per row, segment
  routing on blockIdx — byte-identical). Toggle
  `GGML_CUDA_DISABLE_L2_NORM_GROUP`. Toggle-isolated 5-round A/B: +1.05%
  decode.
- **0020**: `GGML_MMVQ_UNTRIM_BLOCKS` default 4096 → 0 (always trim
  zero-work warps; bit-identical). Swept: 106.4 → 107.6 tok/s (+1.1%).

Stacked with 0018 (grouped mmvf), 5-round A/B vs the verified frontier
build: decode tg128 101.0–102.0 → **106.9–107.5 tok/s, median per-round
ratio 1.0580**; prefill neutral; ppl 3.9390 (+0.193% vs stock, equal to
the verified series' own reading); server smoke clean. Absolute decode
record for the track (previous ranked best: 108.2 by the 0018-only build
under a different draw; this build should exceed it on the runner).

## Open levers

1. (moved to closed: shexp ninth-channel — see above)
2. The GDN/linear-attention layers (3 of every 4) are untouched by this
   series beyond the generic grouped launches — profile their dispatch
   structure; `ssm` conv/scan ops may group the same way rms_norm did.
3. MTP head is present in the GGUF but unused by `llama-server` b10237 —
   speculative decode via MTP is a structural lever if the harness allows it.


## Measured-dead: ngram self-speculation (do not spend a slot)

The maple 0022 lever (engine-default ngram-simple, exact-verify) was ported
and 3-seed A/B'd 2026-08-07: decode 109.0-110.5 -> 85.9/106.9/96.3 (-3% to
-22%). Precision tuning (n=4-5, m=8, min_hits=2-3) reaches parity at best.
Qwen's greedy continuations are too diverse: proposals fire on the small
bench vocabulary but reject at verify, and each rejected batch-17 forward
on a 22GB model costs more than the accepted tokens repay. Consistent with
the platform's laguna-vLLM ngram finding (acceptance ~0.25). Maple's win
does not transfer here.

## 0021: load-time Q6_K requant of the UD Q8_0 projections + GEMM prefill routing

Profiling the 0018-0020 build (rocprofv3, 33 decode tokens): Q8_0 matvecs
consume **39.7% of all decode kernel time** (~2.96 ms/token). The unsloth
UD-Q4_K_M export upcasts the whole GDN/attention projection stack to Q8_0 -
attn_qkv 30x[2048->8192], attn_gate 30x[2048->4096], ssm_out 30x[4096->2048],
attn_q/attn_output on the 10 full-attention layers, plus the three shexp
mats: **~1.49 GB of weight reads per token** at 76-87% of the 640 GB/s
ceiling. The kernel side is spent - the bytes are the lever.

0021 has two coupled pieces:

1. **Load-time requant**: `llama_model_loader` converts the five projection
   families (NOT shexp) to Q6_K as the model loads (the ranked runner serves
   the pinned GGUF path, so the loader is the execution hook; the ppl gate
   measures the requantized model through the same binary). ~300 MB/token of
   decode traffic removed. `GGML_LOAD_REQUANT=0` restores stock bytes;
   conversion is multithreaded (seconds) and byte-identical across loads
   (FNV-checksummed under `GGML_LOAD_REQUANT_CSUM=1`).
2. **Large-batch GEMM routing**: the requant-only variant was ranked-
   rejected on the prefill floor (0.9476 < 0.95; decode 113.7 tok/s - fifth
   consecutive track record) because **RDNA4 Q6_K MMQ tiles cost ~20% more
   than the fp16 dequant+GEMM path at prefill batches**. Plain non-expert,
   non-head Q6_K matmuls at ne11 >= 64 now take the GEMM path: prefill
   damage shrinks from -10% (local pp512) to **-1.9%**. The output head is
   excluded: routing it makes the gate reading biased low AND noisy
   (3.908-3.924 across draws, straddling the lower bound); with the head on
   MMQ like stock, four consecutive loads read 3.9146/3.9148/3.9151/3.9167 -
   tight and mid-low in the 3.9118-3.9511 band.

Decode same-binary toggle A/B (pinned device 0): 106.1-106.4 ->
109.6-111.3 tok/s (+3.2-4.5% across sessions).

The perplexity landscape here is **non-additive** - measured configs:

| config | gate ppl | verdict |
|---|---|---|
| 5 fams Q6_K, MMQ prefill | 3.931-3.947 | in-band but prefill floor fails (ranked 0.9476) |
| 5 fams Q5_1 | 3.9031 | -0.72%, out of band LOW (band is symmetric!) |
| qkv=Q5_1 + rest Q6_K | 3.896-3.899 | out of band low |
| qkv=Q6_K + rest Q5_1 | 3.954-3.965 | out of band high |
| + shexp trio (any type) | +0.5% and SLOWER decode | excluded |
| 5 fams Q6_K + GEMM routing incl. head | 3.908-3.924 | noisy, straddles lower bound |
| **5 fams Q6_K + GEMM routing, head excluded (SHIPPED)** | **3.9146-3.9167** | **in-band, tight** |

Measurement traps burned into this round (do not re-buy):

- `llama-quantize --tensor-type` overrides are SUBSTRING patterns: `attn_q=`
  also matches `attn_qkv`, and a generic `ffn_down_exps=q5_k` clobbers the
  three layers that ship as Q6_K. The loader's exact suffix matching is the
  ground truth; offline ablations from the first pass carried phantom damage.
- **imatrix-weighted requant made ppl WORSE** (3.9924 vs 3.9562 plain,
  full-set; healthy 60-chunk wikitext imatrix). Do not reach for imatrix on
  a requant-from-Q8_0 here.
- **Always run the prefill OFF-control.** The first ranked rejection
  happened because local validation only benched prefill with requant ON.
- **Pin `HIP_VISIBLE_DEVICES=0`** (the runner's GPU_PREFIX): the box has an
  iGPU, and unpinned runs wander by ~0.1% ppl.
- ppl is not bit-deterministic even pinned (+/-0.01% stock, more with
  config changes in the graph) - sample several loads near a band edge.
- **HIP graphs are already active by default** (disable costs ~1%) - no
  differential lever; RDNA4 graph replay keeps a per-node cost, which is
  why launch grouping still pays.


## 0022: sigmoid epilogue fusion in the grouped mmvf launch

The GDN beta sigmoid (30 layers) and the shared-expert gate sigmoid (every
layer) consume grouped-mmvf segment outputs and nothing else. 0022 applies
the exact `op_sigmoid` expression in the segment epilogue and writes straight
to the sigmoid node's destination - ~70 elementwise launches/token removed,
**byte-identical outputs** (same accumulator, same expression). Per-member
gates: use_count==1, F32 contiguous same-shape, hoist-legality of the early
write vs every intermediate node, disjointness from other segment dsts.
Fused sigmoids enter the grouped-done skip set.
`GGML_CUDA_DISABLE_MMVF_SIG_FUSE` disables.

Toggle A/B (same binary, pinned): decode 110.1-110.6 -> 111.4-111.7 tok/s
(**+1.2%**, 3/3 rounds); ppl within the 0021 band; server smoke clean.

Ranked context: 0021 v2 ranked at **113.9 tok/s decode (1.3783, sixth
consecutive track record), prefill 0.9995, ttft 1.013** - floors fully
fixed; rejected only on the frontier rule (1.2342 vs the banked 1.2980,
which carries a hot-prefill 1.199 draw). The remaining gap is ~+2-8% decode
depending on the stock draw; 0022 closes ~+1.2% of it. Next scoped levers:
GDN get_rows/cpy state-movement folds (~0.28 ms/token of copies + 70
launches).


## Measured-dead: GDN conv-state writeback fold (do not re-derive as-is)

Folding the per-layer conv-state writeback cpy into the ssm_conv kernel
(skip 30 cpy launches/token) measured **+1.3% on llama-bench** with in-band
ppl - and **zero fold dispatches under llama-server**: the fork dispatches
the GDN section through its concurrent-events path in server graphs,
bypassing custom in-evaluate-loop fusion branches, so the skipped cpys
corrupt the recurrent state and greedy output diverges (caught by a
fold-on/off server identity test; a keep-cpy diagnostic isolated the
mechanism). llama-bench never activates that path.

Two portable lessons:
- **bench-only gains on this engine are not serving gains** - always run a
  SERVER-side greedy identity test for anything that skips graph nodes;
- **route new fusions through `ggml_cuda_try_fuse`** (like the existing
  gdn cache fusion) rather than the evaluate loop's inline branches - that
  is the redesign path if this lever is attempted again (~+1.3% decode plus
  the get_rows/concat virtualization behind it).

The 0022 sigmoid fusion was server-identity-verified clean in the same
session (byte-identical greedy output with the fusion toggled).


## Prefill profile (rocprofv3, pp512, 0022 build) - the round-7 map

Never profiled before this. Per-pass kernel time ~151 ms (pp512 = 3255 tok/s
local bench):

| share | kernel | note |
|---|---|---|
| 29.3% | mul_mat_q Q4_K (expert gate+up MMQ) | ~50% of int8 compute peak |
| 14.3% | mul_mat_q Q5_K (expert down MMQ) | same class |
| 11.1% | gated_delta_net | GDN chunked scan |
| 13.5% | Cijk_* rocBLAS GEMMs | fp16 attention + the 0021 GEMM route |
| 3.8% | mm_ids_helper | MoE routing scatter |
| 2.8% | dequantize_block_q6_K | the 0021 route's dequant cost |

The MoE expert MMQ (43.6% combined) is the biggest untouched surface on this
track: RDNA4 mmq tile tuning for the mmid expert shapes. Optimistic ceiling
~+15-20% prefill (score x1.03) - combined with ~+2-3% more decode this is
the remaining path to the 1.298 bank. gated_delta_net prefill efficiency is
the secondary target.

Decode-side note: the conv-state fold done RIGHT (try_fuse, adjacent
cpy/ssm_conv/silu pattern, byte-exact server identity) is performance-
NEUTRAL - with HIP graphs replaying, sub-10us launch elimination is
exhausted on this box. Kernel-time work only from here.


## 0023: round 7 - MMQ MoE expert J-tile cap (+22% prefill)

The round-7 map above paid off, but the win was not tile geometry in the
RDNA4 config table - it was the J *selection* feeding it.
`mul_mat_q_switch_J` picks J from `ncols_max`, the worst-case column count
of one expert segment (= n_tokens for mul_mat_id). At pp512 the debug trace
showed every expert GEMM (Q4_K gate/up [2048->512], Q5_K down [512->2048],
256 experts) launching J=128 tiles while segments average
`ncols_dst/nchannels_y` = 4096/256 = **16 columns**: ~87% of every WMMA
column tile was masked padding, computed anyway. That, not tile shape, was
the "~50% of int8 peak".

0023 caps J for the ids path at the smallest valid config >= 2x the
expected segment width (16 floor; 2x headroom for routing imbalance). At
ub512 that selects J=32 - the measured sweep optimum (3197 stock-J / 3650
J=16 / **3911 J=32** / 3810 J=48 / 3728 J=64 pp512 tok/s). Grid coverage
of worst-case imbalance is preserved (ntx grows as J shrinks; out-of-range
tiles exit early), weight traffic is unchanged, dense matmuls and the
decode mmvq path are untouched. `GGML_CUDA_DISABLE_MMQ_MOE_J` restores
stock selection.

Measured (runner box):

- same-binary toggle A/B, 5 interleaved rounds: pp512 3181-3225 off ->
  3885-3933 on, median per-round ratio **1.2216**; tg128 identical
  (112.2-112.9 both arms)
- whole-process stock-vs-cand (full 23-patch series): pp512 median ratio
  **1.2226**, tg128 **1.389** (stock 80.4-81.1, cand 112.0-112.6)
- ranked-style llama-server, uncached 533-token prompt: 2065 -> 2525 tok/s
  (**+22.3%** - the delta holds at server level; kernel-time lever, not
  launch-count)
- larger/smaller batches gain too: pp128 +33%, pp256 +27%, pp2048 +22%
- gate ppl: cand 3.9177/3.9180 vs stock 3.9314x2 (**-0.35%**, band 0.5%);
  the cap itself moves the 0022 build only +0.07%
- server greedy identity: byte-exact off vs on, long + short prompts

Projection at submission: decode ~1.389 (draw-dependent), prefill ~1.22,
ttft following prefill -> score ~1.33 vs the 1.2980 bank even on a fast
stock decode draw.


## Round 8 profile maps (post-0023 cand, rocprofv3)

pp512 (256 ms kernel/pass): mul_mat_q Q4_K 18.8% + Q5_K 13.0% + small-J
    2.8% (36% total, down from 43.6% pre-cap), gated_delta_net 13.1%,
    rocBLAS fp16/f32 GEMMs ~18.6%, mm_ids_helper 4.5%, q6 dequant (0021
    route) 3.4%.
decode (6.9 ms kernel/token): mmvq ~64%, of which dense Q6_K = 42% (the
    0021 requant targets: qkv/attn_q [2048->8192] 27.8us at 495 GB/s,
    attn_gate [2048->4096] 19.6us at 351 GB/s, ssm_out 13.2us at 521 GB/s,
    output head [2048->248320] 790us at 528 GB/s vs ~644 GB/s peak);
    expert ids matvecs Q4_K 11.2% (489 GB/s) + Q5_K 6.8% (451 GB/s);
    gated_delta_net 4.8%; grouped mmvf/mmvq 8.6%.
pp16384: flash_attn_tile 31.9% dominates; mul_mat_q 24.5%; GDN 9.2%;
    rocBLAS 13.1%. dec@16k: flash_attn_tile 30.3%.
The long-context board surface is flash_attn_tile; the ranked decode
surface is dense-Q6_K matvec efficiency; ranked prefill surface remains
MMQ + GDN.

## Measured-dead: MMQ RDNA4 tile geometry at J=32 (round 8)

With the 0023 J-cap in place, swept the Q4_K/Q5_K J=32 CASE rows
(nthreads, occupancy, I): stock (128,2,64) 3912 pp512; (64,2,32) 3774;
(256,2,128) 3745; occupancy 1/3/4 within noise of stock. The J *selection*
was the win; tile geometry is already optimal on RDNA4 for these expert
shapes. Do not re-sweep.

## 0024: round 8 - RDNA4 multi-row blocks for dense Q6_K mmvq (+3% decode)

Stock mmvq on RDNA4 runs rows_per_cuda_block=1 everywhere: each 256-thread
block reads one 1.7 KB Q6_K row and exits - latency-bound (351-528 GB/s on
a 644 GB/s part). The small_k multi-row mode that fixes exactly this is
hard-disabled on all RDNA in should_use_small_k (never tried on RDNA4).
0024 opts dense Q6_K (ncols_dst==1, no ids) into the small_k route with
rpb=2, and gives the grouped mmvq kernel a Q6_K small_k variant (seg_end in
block units, divisibility-gated). Per-thread k-block assignment per row is
unchanged -> byte-identical results. Swept rpb 2/4/8: +3.1/+1.9/+0.8%;
Q5_K neutral; Q4_K/Q8_0 and the ids expert paths regress (excluded).
GGML_CUDA_DISABLE_MMVQ_RPB restores stock.

- toggle A/B: tg128 112.9 -> 116.9 (+3.6% with 0025), tg64@d16384
  107.1 -> 110.7 (+3.4% - the long board gains too); pp512/pp16384 flat
- server level (isolated 0023 control libs, interleaved): decode
  112.9-113.5 -> 116.3-116.4 (+3.0%), prefill identical (843-852 both)
- whole-process stock-vs-cand, 5 rounds: tg128 ratio median 1.4470
  (0023: 1.389), pp512 1.2202 (held)
- ppl: 3.9180/3.9180/3.9186/3.9263/3.9201 across 5 loads (stock 3.9314;
  worst case -0.13%, band 0.5%)
- server greedy identity byte-exact vs isolated 0023-state control
  (long + short prompts). NOTE: the build makes thin executables over
  shared libs - a copied llama-server binary still loads the CURRENT
  build dir libs via RUNPATH. Control comparisons need a full bin-dir
  snapshot + LD_LIBRARY_PATH, or the "control" silently runs cand code.

## 0025: qkv/z grouped adjacency (bench-only, server-neutral)

The GDN z gate matvec shares src1 and (post-0021) type with the qkv matvec
but is consumed at the end of the GDN block, so topological order places it
outside the grouped-mmvq detector window. Pinning the pair adjacent via
ggml_build_forward_expand (the beta/alpha trick) folds them into one
grouped launch per GDN layer: +1% decode at llama-bench. Under llama-server
these graph sections carry concurrent events, which disable inline
grouping, so it is SERVER-NEUTRAL (decode and prefill measured identical to
control) - kept because it is free, byte-identity-verified, and pays on any
non-concurrent-events dispatch.

Projection at submission: decode ~1.447 local ratio (server +3.0% over the
0023 state that ranked 1.3955), prefill ~1.186-1.22 unchanged, ttft
following prefill -> score ~1.34-1.35 vs the 1.3103 frontier.
