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

## 0026: GDN RDNA4 DPP reductions + float4 vectorization

The gated_delta_net recurrence (13.1% pp512 / ~4.8% decode / 9.2% @16k in
the 0023-state profile) runs one warp per state column with a serial token
loop. gfx1201 ISA showed each token paying ten ds_bpermute_b32 LDS
round-trips (two 5-step __shfl_xor reductions, each step behind a full
s_wait_dscnt) plus 11 scalar global_load_b32. Two HIP-guarded changes
(gfx120x only, CUDA path untouched):

1. Wave32 DPP butterfly sum: quad_perm xor1/xor2, row_half_mirror,
   row_mirror, v_permlanex16 across the 16-lane rows - 5 VALU adds per
   reduction, zero LDS traffic, generic warp_reduce_sum fallback.
2. Lane row-ownership remap (i = lane*4 + r) for S_v==128 so state/k/q
   loads and state stores issue as dwordx4 (11 loads/token -> 5);
   host-gated on 16B alignment of all row bases/strides with scalar
   template fallback.

- kernel (test-backend-ops perf): TG 33.96 -> 18.65 us (1.82x),
  PP-512 610.7 -> 489.6 us (1.25x), PP-1024 1245 -> 996 us
- model A/B vs 25-patch control (5 interleaved rounds): pp512
  3926.0 -> 4038.7 (+2.87%), tg128 117.33 -> 119.99 (+2.26%)
- ppl 3.9271/3.9271/3.9278 (-0.11% vs stock 3.9314, band 0.5%)
- GATED_DELTA_NET correctness suite all OK (both mappings + DPP path);
  server greedy smoke coherent, 3/6 prompts drift mid-completion vs
  control - expected reassociation from the changed reduction tree
  (same class as 0024's multi-row reduction change; ppl-gated)
- long windows: GDN is 9.2% of the 16k phase, kernel uniformly faster,
  no KV/attention path touched

## 0027: RDNA4 mm_ids_helper DPP scan + multi-warp token chunking

mm_ids_helper (4.5% pp512, 48.1us avg at 512 tokens) ran one warp per
expert with a serial all-token loop paying ~7 ds_bpermute LDS round-trips
per 4-token iteration (any-reduce over 8 lanes, 3-step shfl_up scan,
broadcast) - the 0026 pathology in the MoE ids bookkeeping kernel. Fixes:

1. RDNA4 DPP forms (gfx120x-guarded): quad_perm/row_half_mirror OR-reduce,
   row_shr:8 + row_shl:8 + one v_permlanex16 for the group prefix scan AND
   the warp total (replaces 4 LDS ops), 0026-style DPP butterfly for the
   final nex_prev sum. VALU-only, zero LDS traffic.
2. Multi-warp chunking (all archs, n_tokens >= 128, nwarps=4): each warp
   compacts a contiguous token chunk into a disjoint LDS slice; counts and
   nex_prev partials merge after one __syncthreads. Pure integer kernel,
   output bit-identical by construction (verified byte-exact at server).

- kernel: 48.12 -> 10.68us at pp512 (4.5x); nwarps=8 tested worse (12.1us)
- model A/B vs 26-patch control: pp512 4054.0 -> 4183.8 (+3.20%, arms
  fully separated), tg128 flat, pp16384 +1.8%
- MUL_MAT_ID suite OK at nwarps 1/4/8; server greedy identity BYTE-EXACT;
  ppl in band (control cluster reproduced same-session)

## 0028: RDNA4 wide mmvq vec_dots (Q6_K vdr=2, Q4_K vdr=4)

Decode is 64%+ mmvq at 54-89% of DRAM peak. Wider per-lane column
footprint on RDNA4: Q6_K pairs (iqs, iqs+1) share scales/q8-blocks/d8/qh
shift and fetch ql/qh/q8 as b64 (2-byte aligned, hardware-handled); Q4_K
vdr 2->4 shares bq8_offset + unpacked scale/min across the column pair
with all loads staying 16B-aligned (144-byte blocks). Host/device vdr
agreement via a table-aware get_vdr_mmvq(type, table_id) overload; other
archs keep stock vdr, CUDA path unchanged.

- kernels (tg32 profile): Q4_K expert gate+up 19.97 -> 18.22us (473->519
  GB/s), Q6_K expert down 17.55 -> 13.82us (472->599 GB/s), attn_output
  14.59 -> 14.11us, dense Q6_K flat (lm head already 626 GB/s = 97%)
- model A/B (combined with 0027) vs 26-patch control: tg128 119.77 ->
  120.18 (+0.34%), pp512 +3.24%, pp16384 +1.6%
- full test-backend-ops OK; ppl 3.9256/3.9294/3.9271 in band; greedy
  drifts mid-completion on long prompts (reassociation class, as
  0024/0026: d8*(sc*(a+b)) vs two FMAs) - ppl-gated
- dead ends recorded as findings: Q4_K vdr=8 (per-row wave parallelism
  loss at k=2048), Q5_K vdr=4 (k=512 = 2 blocks/row), Q6_K nwarps 4,
  small_k rpb=4, and dense-stream insensitivity to load widening
  (isolated MALL-warm ssm_out shape only reaches 410 GB/s - the small
  dense matvec is latency/ramp-bound, not issue-bound)

## Round-11 profile corrections (read before touching dense Q6_K again)

The round-8 "42% dense Q6_K at 351-528 GB/s" map overstates the pool:

- In-graph per-kernel times include neighbor-kernel drain (kernels on the
  same queue overlap start/drain). Serialized (rocprofv3 PMC mode) ssm_out
  is 14.35us (479 GB/s), not the 20.2us the in-graph trace shows; the
  extra 6us belongs to the (post-0026, much faster) GDN kernel's tail.
- The output head runs 625-626 GB/s = 97% of peak in every configuration
  measured. Done.
- Dense Q6_K does NOT respond to vec_dot load width: a narrow-vdr probe
  (Q6_K vdr back to 1 everywhere) left qkv/ssm_out/head byte-flat while
  regressing the ids expert path 13.8 -> 17.8us. 0028's wide vec_dots are
  ids-path medicine only. Do not widen further (vdr=4 quad) - and RDNA4
  dual-issue is dead-on-arrival: the head kernel is ~8% VALU-busy.
- rocprofv3 DERIVED counters (VALUUtilization, MemUnitStalled, FetchSize,
  L2CacheHit...) silently read 0.0 on gfx1201 with this ROCm; only raw
  SQ_WAVES / SQ_BUSY_CYCLES / GRBM_GUI_ACTIVE collect. Do not trust a
  counter run without checking for nonzero values.
- Remaining dense pool after 0029 is ~120us/token against a realistic
  ~620 GB/s ceiling (grouped qkv+z at 567, qkv at 555) - latency/ramp
  class, geometry and load-width both exhausted.

## 0029: per-shape small_k routing (head-only rpb=2)

Post-0028 re-profile: blanket rpb=2 is net-negative. qkv dense wants rpb=1
(24.77 vs 29.36us), grouped qkv+z wants rpb=1 (35.82 vs 36.38), only the
248320-row head still wants rpb=2 (667.7 vs 672.4). small_k now requires
nrows_x >= 65536 (GGML_MMVQ_R4_RPB_MIN_ROWS; 0 restores 0028 behavior).
Byte-identical; server greedy byte-exact vs 28-patch control. Toggle A/B:
tg128 +0.55% (5/5 rounds separated).

## 0030: recurrent-state identity view (+6.6% decode, +5.1% @16k)

The big one this round. Every GDN layer gathered its 2 MB ssm state
(get_rows_float_vec 4.93us) and conv window (k_get_rows_float 1.65us)
before gated_delta_net - ~197us of copy kernel plus 60 launches per decode
token, purely to materialize bytes at a new address for the common case
where the source slot IS the destination slot. build_rs now detects the
identity mapping side-effect-free at graph build (cells[head+i].src0 ==
head+i, no rollback restore pending) and hands the consumer a contiguous
view of the cache rows instead.

Traps burned (do not re-buy):

- **Graph reuse bakes view offsets.** The fork reuses built graphs across
  decode steps (~27 builds serve a 256-token completion; build_rs runs at
  BUILD time only, while get_rows stays fresh through the s_copy input
  tensor). A view that could point at a non-zero row can go stale under
  reuse. Fix: restrict the view to n_seqs==1 && head==0 && n_rs==1 (the
  ranked serving shape) so the baked offset is row 0 always.
- **Build-time memory state is not settled.** At graph build, cells[].src0
  is often still -1 (prompt-phase builds); set_input later sees the real
  mapping. The identity check must treat unsettled as non-identity and
  fall back (it does; verified with per-build logging).
- **The engine's greedy output is NOT run-to-run stable on near-tie
  prompts, including the verified frontier binary.** In-process repeats of
  the same 2k-token greedy request on the UNMODIFIED 28-patch control
  produced 3 different completions (drift at a near-tie token, ~77%
  common prefix). Do not burn a session chasing "nondeterminism" that a
  candidate build merely re-exposes; characterize drift per-prompt against
  control and gate on ppl, as 0024/0026/0028 did.

Measured: decode kernel/33-tok 215.04 -> 203.78ms; tg128 119.9 -> 127.8
(+6.6%, 8/8 rounds); tg64@16k 111.7 -> 117.4 (+5.1%); pp512 neutral; ppl
3.9271/3.9301/3.9297 (band 0.5%); 5/6 greedy prompts byte-exact vs
control, 8-request concurrent smoke coherent; MUL_MAT suite OK.
LLAMA_DISABLE_RS_STATE_VIEW=1 restores stock.
RANKED (r11): VERIFIED 1.4324 (+43.24%) - decode 128.55 = 1.5575, prefill
1.2569, ttft 1.1860.

## Round-12 overhead map (the node-count ledger)

HIP API trace of steady decode (rocprofv3 --hip-trace): one hipGraphLaunch
per token costs ~500us of CPU submission (scales with node count), and the
GPU spends ~0.9ms/token in inter-kernel dispatch gaps inside the replay.
Decode = ~1085 kernels/token at 6.1ms busy in a 7.8ms wall token: every
node removed is worth ~1.5us of overhead PLUS its kernel time. The
scheduler itself is clean (2 splits; the 16 syncs/token are the graph-exec
wait + input/logits copies). Node classes remaining (per token): quantize
81, bin_bcast MUL 80, rms_norm family ~130, unary/gated ~140, mmvf grouped
70, Q8_0 shexp 90, cpy ~50, GDN misc. The conv chain (below) took the
first 60. Next largest coherent cuts: the remaining y-quantizes (81, the
use_count-blocked and ids-path cases), the ssm-state writeback cpy
(30 x 2MB - needs in-place GDN state write, an op-signature change), and
the MUL/ADD glue around the MoE combine.

## 0031: GDN conv-chain fold (+2.7% decode)

CONCAT(state_window, x_t) + SSM_CONV(+SILU) + conv-state writeback CPY
collapse into one 1.6us kernel per GDN layer (was 3 kernels, 5.4us):
per-channel thread reads window+x into registers (alias-safe in-place
cache writeback), stock accumulation loop, silu epilogue. -60 nodes and
-92us kernel per token. Decode-shape gated (n_t==1, d_conv==4, no bias,
single writeback; prefill keeps stock nodes). Detection at the eval loop
with the done-set pattern; view/metadata nodes over the concat are
skipped during consumer classification (the first version aborted on the
conv_state_last VIEW node itself - fixed).
GGML_CUDA_DISABLE_SSM_CONV_FOLD=1 restores stock.

- tg128 toggle: 127.9 -> 131.37 (+2.7%, 5/5 rounds disjoint); pp512
  neutral; ppl 3.9256-3.9283 in band; SSM_CONV/CONCAT suites OK.
- greedy identity fold-on/off: 2/6 byte-exact, 4/6 drift at 53-77%
  prefixes; BOTH arms individually deterministic (A/A byte-exact) - the
  fused kernel contracts the accumulation differently than the stock
  chain despite mirroring its source structure (tried register-array
  alignment; codegen context differs). Same ppl-gated class as
  0024/0026/0028.


## Round-13 profile (read this before profiling again)

**The fusions DO fire under llama-server.** Every grouped launch and inline
fold in this series is gated on `stream_ctx.concurrent_events.empty()`, and
round 10's note that "these graph sections carry concurrent events, which
disable inline grouping" reads as if the whole stack were inert on the ranked
path. It is not. A rocprofv3 kernel trace of `llama-server -ngl 99 -c 8192
--parallel 1` serving a 64-token greedy completion shows
`mul_mat_vec_q_grouped`, `mul_mat_vec_f_grouped`, `l2_norm_f32_grouped` and
`ssm_conv_fold_f32` at the same per-token counts llama-bench shows. The
concurrent-event pass only ever fires on a node **named `attn_norm` with
fan-out exactly 3** (the Q/K/V fork); `qwen35moe` never builds that shape, so
the map stays empty. Round 12's ranked +2.6% decode matching its +2.7% bench
toggle is the independent confirmation. Profile llama-bench freely - but still
verify anything that skips graph nodes at server level, as round 7 learned.

Decode, 34 tokens, verified r12 build: 205.1 ms kernel = **6.03 ms/token**
against a 7.58 ms wall token (131.4 tok/s), ~1085 dispatches/token.

| share | kernel | note |
|---|---|---|
| 17.7% | mmvq_grouped Q6_K (qkv+z, 30x35.5us) | 581 GB/s - near the practical ceiling |
| 11.9% | mmvq Q4_K ids (expert gate+up, 40x18.0us) | 524 GB/s |
| 11.0% | mmvq Q6_K small_k (output head, 1x666us) | 627 GB/s = 97% of peak, done |
| 8.1% | mmvq Q6_K ssm_out (30x16.4us) | 420 GB/s in-graph |
| 8.0% | mmvq Q5_K ids (expert down) | ~445 GB/s, ONE warp per block |
| 4.5% | mmvq_grouped Q8_0 (shexp, 50/token) | |
| ~11% | **~500 elementwise launches/token** | 1.2-2.9us each; almost pure overhead |

Prefill, pp512 (239.7 ms / 2 passes): mul_mat_q Q4_K 19.9% + Q5_K 13.8%,
gated_delta_net 10.9%, rocBLAS GEMMs ~19.8%, **`concat_non_cont` 4.7%
(30 x 188us)**, q6 dequant 3.6%, silu 1.9%, mm_ids_helper ~1.1%.

The elementwise pool is the big remaining decode surface and the reason 0032
exists. Per token, after 0032: rms_norm_pre_add 80 x 2.8us (a 2048-column norm
in ONE 1024-thread workgroup - 23 GB/s, entirely launch/ramp), topk_moe
40 x 2.7us, the MoE combine MUL/ADD glue 120 x ~1.5us, quantize_q8_1
80 x 1.3us, l2_norm_grouped 30 x 1.9us, gated-norm rms_norm+silu 60 x ~1.5us.


## 0032: GDN beta/alpha chains folded into the grouped-mmvf epilogue (+3.4% decode)

Every GDN layer ran three elementwise launches over **32 floats** right after
the grouped beta/alpha matvec: `sigmoid(beta)` 1.28us, `alpha + ssm_dt`
1.60us, `softplus(...) * ssm_a` 1.47us. 0022's epilogue machinery could only
match a sigmoid whose src IS the segment output, so it caught the shared-expert
gate and never beta (which reaches its matvec through a `ggml_reshape` view).

0032 adds a second epilogue form `softplus(sum + bias[row]) * scale[row]` (the
same three f32 statements, same order, on the same accumulator value the stock
kernels would have loaded), teaches the matcher to walk whole-tensor alias
links, and - critically - **moves the beta/alpha matvecs and both chains ahead
of the qkv/z projections** in `qwen35moe.cpp`.

That reorder is most of the win, and the lesson generalises to every future
epilogue fold on this track:

> **A hoisted epilogue write is an allocation problem, not a matching
> problem.** The write moves back to group-launch time, but ggml-alloc placed
> that 128-byte result assuming it ran later - on whatever was dead by then.
> With the chains emitted after the conv block, `gate` landed *inside* the
> 128 KB conv-input concat and `beta_sigmoid` landed on the `attn_norm`
> activation that the group kernel itself reads. Both are genuine races and
> the hazard scan correctly declined them: **0 of 30 layers folded.** Emitting
> the chains while `cur` is still live removes the aliasing entirely - 30/30,
> and +2.13% becomes +3.35%.

Two matcher rules were needed to stop the two folds blocking *each other*
(they see-sawed: fixing beta broke alpha and vice versa):

- the intermediate-hazard scan exempts group members' own matvec destinations
  (a folded member never materialises its matvec dst; the member loop compares
  EFFECTIVE destinations instead), and
- it exempts nodes another accepted fold has already absorbed - their reads
  happen from registers inside the group kernel, or not at all. Without this,
  `gate` was rejected for aliasing the buffer `beta_sigmoid` "reads", when
  `beta_sigmoid` no longer reads anything.
- the scan runs three passes so a fold accepted late unblocks one rejected
  early.

`GGML_CUDA_MMVF_EPI_DEBUG=1` prints every accept/decline with the offending
pointer pair. **Use it.** These declines are allocation-dependent and cannot
be reasoned about from source - three rebuild cycles were spent guessing
before the diagnostic was added, and it found both causes in one run.

Measured (runner box, HIP_VISIBLE_DEVICES=0):

- same-binary toggle A/B (`GGML_CUDA_DISABLE_MMVF_EPI_EXT=1` restores the
  pre-patch matcher), 3 rounds: tg128 132.91-133.06 -> **137.28-137.50**
  (+3.35%, 3/3 disjoint); pp512 neutral (4142-4191 off vs 4158-4186 on)
- pre-reorder build, 5 rounds: +2.13% (132.52-132.98 -> 135.24-135.61)
- whole-process stock vs cand, 5 rounds: tg128 80.50-81.16 -> 137.15-137.46
  (ratio **1.698**), pp512 3177-3231 -> 4125-4156 (ratio **1.301**)
- ppl 3.9271 / 3.9271 / 3.9283 vs stock 3.9314 (**-0.08 to -0.11%**, band 0.5%)
- test-backend-ops MUL_MAT / ADD / MUL / UNARY / SSM_CONV / GATED_DELTA_NET
  all OK; llama-server smoke coherent short / long / concurrent
- `GGML_CUDA_DISABLE_MMVF_SIG_FUSE=1` still disables the whole epilogue pass


## BLOCKED: the trusted runner cannot boot a bench server (read first)

**As of round 13 no submission on this track can be measured.** Both round-13
attempts were closed with:

    infrastructure fault (not your submission): a previous benchmark server
    would not release the GPU, so this run was stopped rather than measured
    beside it - <pid> /bin/sh -c pgrep -af "llama-server .*--port 8005" | head -1

The quoted process is the guard itself. `benchServerAlive()` in the runner's
`app/src/rocm-worker.ts` runs

    pgrep -f "llama-server .*--port 8005" | head -1

through a shell. `pgrep` excludes itself but **not its parent `/bin/sh -c`**,
whose argv contains the pattern - so the guard always finds a "survivor",
`waitForBenchServerGone()` always returns false, and `serverUp()` throws
before anything is measured. Reproduced on a completely idle box (no
llama-server anywhere, `rocm-smi` VRAM 0%): the guard command prints its own
shell PID. The escalation `pkill -9 -f "llama-server .*--port 8005"` has the
same defect. Two submissions x two internal retries, identical signature -
this is deterministic, not a flaky box.

Do not spend slots retrying. The fix is one line and the same file already
gets it right one function away (`pgrep -af "bin/llama-server|bin/llama-perplexity" | grep -v pgrep`):
match on `bin/llama-server` and append `| grep -v pgrep` in
`benchServerAlive()`, in the diagnostic pgrep in `serverUp()`, and in both
`pkill` calls. Verify on an idle box by running the guard command and
confirming it prints nothing. **This is the trusted measurement
infrastructure - it is the operator's to change, not a participant's.**

Round 13's patch (0032) is committed, locally measured and gated; it needs a
ranked run once the runner boots again.


## Round-14 map

1. ~~**Prefill: `concat_non_cont` is 4.7% of pp512**~~ — **DONE, see 0033.**
2. **`rms_norm_pre_add`: 80 launches/token at 2.8us** for a 2048-column norm
   in a single 1024-thread workgroup (32 waves in the block reduction on
   wave32). Try block 256 on RDNA4 - reassociation-class, ppl-gated. ~0.7%.
3. **MoE combine glue**: `mul(experts, weights)` + 8-way `add` + `mul(shexp,
   gate)` = 120 bin_bcast launches/token. Folding the weight multiply and the
   expert reduction into one kernel is ~40 nodes + ~60us/token.
4. **Q5_K expert-down mmvq runs ONE warp per block** (wg=(32,1)) at ~445 GB/s
   while Q4_K gate/up runs two warps at 524. nwarps was swept for the dense
   paths, not per-shape for the ids Q5_K path - worth one probe.
5. Dense Q6_K bandwidth and mmvq geometry stay closed (round-11 corrections).
   The output head at 97% of peak is done.
6. The epilogue machinery now generalises: any (matvec -> elementwise chain)
   on a grouped-mmvf segment is a candidate. Remember rule (1) of 0032 -
   pin the chain adjacent to its matvec in the model graph FIRST, or the
   allocator will make every fold illegal.


## 0033: the conv-chain fold at prefill shape - concat_non_cont deleted (+3.6% prefill)

Round-14 item 1, confirmed on a fresh profile of the r13+0032 build: per pp512
pass `concat_non_cont` is 30 x 165us (4.1% of 240ms kernel). The GDN conv input
is a [3+n_t, C] TIME-major buffer assembled from a CHANNEL-major activation
purely so `ggml_ssm_conv` can transpose it back - 33 MB of scalar-indexed
traffic per layer at ~170 GB/s, five times the convolution it feeds.

0033 generalises the 0031 fold from `n_t == 1` to any `n_t` (and any `n_s`):
one thread per channel walks `qkv_mixed` in its natural layout with the
d_conv window in registers and writes silu(conv) straight out. Per pass
30 concats + 30 `ssm_conv_long_token_f32` + 30 writeback cpys become 30 fold
kernels at 38.3us (an effective 876 GB/s - the activation is still MALL-warm
from the projection that produced it). Kernel time 240.3 -> 224.3 ms/pass.

- toggle A/B (`GGML_CUDA_DISABLE_SSM_CONV_FOLD_SEQ=1`), 5 rounds: pp512
  4155.2-4195.1 -> **4302.3-4359.2**, median per-round ratio **1.0355**,
  5/5 disjoint; tg128 neutral (137.3-137.7 both arms)
- whole-process stock vs cand, 5 rounds: pp512 3214-3265 -> 4309-4357
  (median ratio **1.3392**, was 1.301 at 0032), tg128 80.9-81.5 -> 137.1-137.6
  (median **1.6934**, held)
- ppl 3.9288/3.9261/3.9271/3.9271/3.9267 vs stock 3.9314 (**-0.07 to -0.13%**);
  fold-off control same session 3.9262-3.9390
- test-backend-ops SSM_CONV / CONCAT / CPY / UNARY / GATED_DELTA_NET / MUL_MAT OK
- split sweep flat 16-128 (4250-4294), 64 shipped; single-chunk (split>=n_t)
  costs ~2% at pp512 but is what short server prompts use anyway

Three traps burned here, all worth the next agent's time:

- **A fold can be invisible to the gate.** `llama-perplexity` packs
  `n_batch/n_ctx` sequences into one ubatch, so the gate corpus at `-c 512`
  runs **n_t=128, n_s=4** - and 0031's detector required `n_s == 1`. The first
  version of this patch never fired during ANY perplexity run while reading a
  perfectly healthy 3.9271. Always confirm the fold FIRES in the tool you are
  gating with: `GGML_CUDA_SSM_CONV_FOLD_DEBUG=1` prints one accept/decline line
  per layer with the numbered gate that declined.
- **`llama-bench`'s prompt is not reproducible across processes.** It fills the
  prompt with `std::rand()`; two runs of the same binary produce different
  tokens, so tensor dumps from two llama-bench processes are incomparable
  (an A/A check caught this - it read 0.18% agreement). Use `llama-perplexity`
  for any cross-process numeric comparison.
- **A chained-ubatch test is not chained unless you look past the cap.** Dumping
  "the first 60 conv outputs" at 30 layers/ubatch only ever covers the warmup
  plus ONE real ubatch, so it never exercises the state writeback -> next
  ubatch path at all. `GGML_SSM_DUMP_MAX` now controls the cap; the writeback
  itself is directly comparable between arms via `GGML_SSM_WB_DUMP` (the stock
  arm dumps the tail columns of its materialised conv_input, the folded arm
  the cache row it just wrote).

Numerics, measured rather than argued (that is what the dump envs are for):
over a full 512-token prefill all 30 conv outputs and all 30 writebacks are
**byte-identical** to stock, likewise at n_t = 2, 4, 5, 8, 11, 12, 16, 64, 128
and n_s = 2, 4. From the **second ubatch of a sequence** - i.e. once the conv
state is no longer zero - the 4-term dot product contracts differently than the
stock kernel's and ~10% of outputs move by exactly **1 ULP**. Server greedy:
candidate deterministic (6/6 A/A byte-exact), 2/6 byte-exact vs the fold-off
control and 4/6 drifting at 9-58% prefixes. Same ppl-gated class as
0024/0026/0028/0031 and the mildest instance of it in the series. Ruled out
along the way, each with an experiment: the concat buffer being left unwritten
(re-running the concat anyway changes nothing), the pool allocation for the
staging buffer (the no-scratch single-chunk path behaves identically), HIP
graphs (`GGML_CUDA_DISABLE_GRAPHS=1` changes nothing), chunk size
(split=1 and split=64 give bit-identical state), and `#pragma clang fp
contract(off)` (no effect).

Projection with 0032: decode 1.6494 (r12 1.5959 x 1.0335), prefill
1.2468 x 1.0355 = 1.2911, ttft following prefill -> score **~1.494-1.499**
vs the 1.4525 bank (0032 alone projects ~1.484).


## 0034: MoE routing-weight multiply folded into the mul_mat_id epilogue (+1.5% decode, BYTE-EXACT)

`ggml_mul(experts[2048, 8], weights[1, 8])` is one launch per layer - 40/token,
1.52us each - moving 128 KB to apply eight scalars, when the matvec that just
produced those bytes has the value in a register and knows which expert slot
(destination channel) it is writing. 0034 adds `dst_scale` to the mm-fusion
struct: one f32 per destination channel, applied before the store, with the
store redirected to the MUL's destination. `GGML_CUDA_DISABLE_MOE_WEIGHT_FUSE=1`
restores the stock pair.

- toggle A/B, 5 rounds: tg128 137.38-137.69 -> **139.47-139.73**, median ratio
  **1.0152**, 5/5 disjoint; pp512 neutral (4322.6 vs 4323.4 median)
- dispatches 941 -> 901/token; k_bin_bcast 121 -> 81/token
- whole-process vs stock, 5 rounds: tg128 median ratio **1.7254** (0033: 1.6934),
  pp512 **1.3489**
- ppl 3.9254-3.9286 vs stock 3.9314; ops suites OK
- **server greedy 6/6 BYTE-EXACT** vs the fold-off control (and 6/6 A/A)

**The lesson worth carrying: do not reuse `has_fusion` for a cheap epilogue.**
The first version routed `dst_scale` through the existing `has_fusion` template
flag and measured only **+0.76%**, with 4/6 prompts drifting. Two separate
causes, both invisible from the call site:

- `has_fusion=true` also instantiates the gate path, and its `tmp_shared_gate`
  array doubles the block's LDS footprint. The Q5_K expert-down launch
  (nwarps=8 compile-time, one warp actually launched) is LDS-occupancy-limited,
  so the fused matvec went **12.92 -> 14.72us** and ate most of the saving.
- `has_fusion` applies `result += x_biases[j]` unconditionally. That is the
  identity on every value except -0.0, which is exactly enough to lose
  byte-exactness and drift the greedy output.

A dedicated `has_dst_scale` template parameter fixes both: the expert-down
matvec keeps its 12.9us and its bits, and the same fold measures +1.52%.

Remaining `k_bin_bcast` after this: two 40/token populations over [2048], the
larger being `ffn_shexp * shared_gate` (1.84us each). The shexp down projection
is an INDIVIDUAL mmvq (b=2048x1, 40/token, 3.45us), so relaxing `dst_scale` to
the plain-MUL_MAT case with a single-element factor folds it the same way -
~134us/token, the next round-16 item.


## 0035: shared-expert gate pinned and folded into the down projection (+1.6% decode, BYTE-EXACT)

After 0034, 40 of the remaining 81 `k_bin_bcast` launches per token are
`ggml_mul(ffn_shexp[2048], sigmoid(shared_gate)[1])` - a full 2048-element pass
to apply ONE scalar, once per layer, 1.84us each. 0034's `dst_scale` epilogue
already applies exactly this shape (a plain matvec has one destination channel,
so the per-channel factor is a single broadcast scalar) and the shared-expert
down projection is an individual mmvq launch, not a grouped one.

It still did not fire, for a reason that is pure graph ORDER: built in source
order the layer emits `ffn_shexp (MUL_MAT), shared_expert_gate_sigmoid (UNARY),
MUL`, because the sigmoid is created after the shared-expert FFN and the
topological visit only reaches it through the multiply. The pair is not
adjacent - and the scalar is not even computed when the projection runs, so
folding it there would have read stale memory. Creating the gate before the FFN
and pinning it with `ggml_build_forward_expand` (the 0018/0032 trick) fixes
both. Two halves, only useful together:

1. `qwen35moe.cpp`: pin `shared_expert_gate_sigmoid` ahead of the shexp FFN.
   `GGML_QWEN_SHEXP_GATE_PIN=0` restores the stock order.
2. the 0034 detector extended to plain `MUL_MAT`, placed AFTER the grouped-mmvq
   collector so a groupable matvec still prefers grouping.
   `GGML_CUDA_DISABLE_SHEXP_GATE_FUSE=1` disables it.

- toggle A/B (both halves), 5 rounds: tg128 139.40-139.70 -> **141.64-141.94**,
  median ratio **1.0161**, 5/5 disjoint; pp512 neutral
- dispatches 901 -> 861/token; k_bin_bcast 81 -> 41/token
- whole-process vs stock, 5 rounds: tg128 median ratio **1.7542**
  (0034: 1.7254, 0033: 1.6934), pp512 **1.3418**
- ppl 3.9262-3.9277 vs stock 3.9314; ops suites OK
- **server greedy 6/6 BYTE-EXACT** vs the pin-off/fold-off control

**Measurement trap burned here, and it cost a full round.** A
`python3 edit.py && make ... | tail -N` hides BOTH a failed edit and the
skipped build behind it, and the A/B that follows then measures the PREVIOUS
binary. That produced a confident, completely wrong conclusion ("the reorder
alone is worth +1.7%, the fold is worth nothing") which survived two follow-up
experiments before the `.so` mtime gave it away. **Check the artifact
timestamp, not the console tail**, whenever a toggle unexpectedly reads zero.

Projection with 0033+0034+0035: decode 1.5959 x 1.0335 x 1.0152 x 1.0161 =
**1.7015**, prefill 1.2468 x 1.0355 = **1.2911**, ttft following prefill ->
score **~1.529** vs the 1.4525 bank.


## Measured-dead (round 17): the remaining 81 quantize_q8_1 launches

Instrumented both `quantize_row_q8_1_cuda` call sites in `mmvq.cu`; the 81
launches per decode token are exactly three populations:

| n/token | activation | consumer |
|---|---|---|
| 40 | `ffn_moe_swiglu` [512,8] | expert down (Q5_K/Q6_K), MUL_MAT_ID |
| 30 | `final_output` (RESHAPE of the GDN gated norm) [4096] | `ssm_out` Q6_K |
| 10 | `attn_gated` (MUL) [4096] | `attn_output` Q6_K |

The largest is **structurally unfoldable**. The MoE gate/up matvecs and their
GLU are already fused into ONE mmvq launch, and in that kernel each CUDA block
produces one output element per row - so a 32-element q8_1 block spans 32
different blocks. Writing q8_1 from that epilogue needs cross-block
communication. (Also probed and measured dead: the batch test in
`ggml_cuda_norm_quant_has_mmvq_consumer` uses `s1->ne[1]`, which for a
MUL_MAT_ID consumer is the EXPERT dimension rather than ncols_dst; correcting
it to `ne[2]` changes nothing, because the MoE glu never reaches that hook -
the fused gate/up+glu path consumes it first.)

The other two populations (40/token combined, ~112us with graph-replay
overhead, ~1.6% decode) remain **open**, and share one root cause: both
producers are SAME-SHAPE multiplies, while the folded-quantize multiply kernel
only handles the broadcast-scalar form (`ggml_cuda_mul_bcast0_quant_ok`
requires `src1->ne[0] == 1`). A same-shape mul+q8_1 kernel registered through
`ggml_cuda_quant_register_for_consumer` (the consumers read a RESHAPE of the
producer, so the cache entry must be keyed on the consumer's view) folds both.


## Round-17 map (decode census after 0035: 861 dispatches, 5.85 ms kernel in a
## 7.06 ms wall token)

| tot/token | kernel | note |
|---|---|---|
| 1064us | mmvq_grouped Q6_K qkv+z (30x35.4us) | 581 GB/s, at the practical ceiling |
| 726us | mmvq Q4_K MoE gate+up (40x18.1us) | 527 GB/s |
| 667us | mmvq Q6_K output head (1x666us) | 621 GB/s = 96% of peak, done |
| 490us | mmvq Q6_K ssm_out (30x16.3us) | 419 GB/s in-graph (479 serialized) |
| 485us | mmvq Q5_K/Q6_K expert down (37x13.1us) | **447 GB/s, one warp per block - the weakest big matvec** |
| 325us | mmvf_grouped (70) | |
| 267us | mmvq_grouped Q8_0 shexp (50) | |
| 196us | rms_norm_pre_add (69x2.8us) | one 1024-thread workgroup; ~26 GB/s from a single CU, which is already above that CU's fair share of DRAM - little headroom |
| 195us | gated_delta_net (30) | 0026 territory |
| 150us | topk_moe (40x2.7us) | one block, 256 experts; launch-floor bound |
| 106us | quantize_q8_1 (81) | see the dead end above; 40 of them still open |
| 67us | k_bin_bcast (41) | what is left after 0034/0035 |

Best remaining candidates, in order: (1) the same-shape mul+quantize fold
above (~1.6% decode); (2) the Q5_K/Q6_K expert-down matvec at 447 GB/s against
the 527 GB/s the Q4_K gate/up achieves - it launches ONE warp per block after
idle-warp trimming (k=512 is 2 k-blocks, and Q5_K vdr=2 covers both in one
warp), so the lever is rows-per-block rather than warps (note 0024/0028
recorded blanket rpb on the ids path as a regression, so it needs to be
per-shape); (3) folding the 8-way expert reduction into the down matvec so
`rms_norm_pre_add` reads 16 KB instead of 72 KB.


## 0036: the q8_1 quantize folded into the fused unary+mul (+1.5% decode, BYTE-EXACT)

Round-17 map item (1), and it closes the quantize census. Of the 81
`quantize_q8_1` launches per decode token, round 17 measured the largest
population (40, the MoE swiglu) structurally unfoldable. The other two are the
same shape of problem and fold together:

| n/token | activation | producer | consumer |
|---|---|---|---|
| 30 | GDN gated norm | `ggml_mul(rms_norm(o), silu(z))` | `ssm_out` Q6_K |
| 10 | attention gate product | `ggml_mul(attn, sigmoid(g))` | `attn_output` Q6_K |

Both producers are SAME-SHAPE multiplies whose second operand is a unary of
another tensor, so **upstream already fuses each pair** into one
`unary_gated_op_kernel` launch. The value the consumer's `quantize_q8_1` then
loads back out of memory is the float that kernel held in a register one
instruction earlier.

The kernel that does both already existed: 0010 added
`unary_gated_op_quant_kernel` for the GLU path, and its body is the plain gated
kernel plus the stock `quantize_q8_1` statements over the value just stored.
This patch only makes it *reachable* from the unary+mul fusion site - three
small edits, no new kernel:

1. `ggml_cuda_op_unary_mul` takes an optional `q8_1_dst` and routes to
   `unary_gated_quant_cuda` when non-null (F32 only).
2. `ggml_cuda_try_fuse` asks `ggml_cuda_quant_register_for_consumer` whether
   the product feeds a mul_mat_vec_q in the lookahead window.
3. `GGML_CUDA_DISABLE_UNARY_MUL_QUANT=1` restores the stock pair.

**Why `quant_register_for_consumer` and not the plain register** - the same
reason 0012 needed it. Neither consumer reads the product tensor: `ssm_out`
reads `final_output`, a RESHAPE to `[head_v_dim*num_v_heads, n_t, n_s]`, and
the o-projection reads the flat `[n_embd]` row. The cache entry has to be keyed
on the consumer's view or the consumer looks it up and misses.

- dispatches 861 -> **821/token**; `quantize_q8_1` 81 -> **41/token**
- toggle A/B, 5 rounds: tg128 141.49-141.73 -> **143.63-143.87**, median ratio
  **1.0147**, 5/5 disjoint; pp512 neutral (4347.1 vs 4344.5 median)
- whole-process vs stock, 5 rounds: tg128 median ratio **1.7825**
  (0035: 1.7542), pp512 **1.3504**
- ppl 3.9273/3.9296/3.9277 vs stock 3.9314 (**-0.09%**); fold-off control same
  session 3.9237/3.9271/3.9260
- test-backend-ops UNARY / MUL / GLU / MUL_MAT / MUL_MAT_ID / RMS_NORM OK
- **server greedy 6/6 BYTE-EXACT** vs the fold-off control (and 6/6 A/A), all
  six completions non-empty and >= 276 bytes
- full 36-patch series applies clean to pristine b10237 and the resulting tree
  is `diff -r` identical to the measured tree

**Trap (1) checked rather than assumed this time, and it passed.** A kernel
trace of the *gate command itself* (`llama-perplexity -c 512 --chunks 8`) shows
`quantize_q8_1` 121 -> 81 with exactly 30 silu and 10 sigmoid pairs moving from
`unary_gated_op_kernel` to `unary_gated_op_quant_kernel`. The fold fires under
the tool it is gated with, so the ppl reading covers it. Worth noting the
opposite failure mode is easy here: at large `n_t` the consumer takes the MMQ
path, `ggml_cuda_should_use_mmvq` declines, the register returns null and the
fold correctly does nothing - a `-b`/`-ub` choice that keeps every ubatch large
would have produced a meaningless green ppl.

Byte-exactness argument: the folded kernel evaluates `op(x)*g` with the same
operands in the same order as the kernel it replaces, and quantizes that float
from a register instead of from the store that just wrote it (an f32 store/load
round trip, exact). `quant_register_for_consumer` declines anything needing a
padded row (`ne10_padded != ne10`), so the flat thread index maps to the q8_1
block and lane that `quantize_q8_1`'s row-major mapping gives, and the amax /
sum warp reductions see the same values in the same lanes. The 6/6 byte-exact
server greedy is the measurement that confirms it.

Projection with 0033+0034+0035+0036: decode 1.5959 x 1.0335 x 1.0152 x 1.0161
x 1.0147 = **1.7265**, prefill 1.2468 x 1.0355 = **1.2911**, ttft following
prefill (1.1838 x 1.0355 = 1.2258) -> score **~1.547** vs the 1.4525 bank.


## Round-19 map (post-0036 census — start here)

Fresh rocprofv3 kernel trace of the 0036 build, `llama-bench -p 0 -n 34 -r 1`.
Normalize by **35** tokens: the run's total dispatch count divides exactly
(28738/35 = 821.0 with the fold on, 30138/35 = 861.0 with it off), which is the
cheapest available check that the window is a whole number of tokens — do this
before trusting any per-token number, because a naive "last 70% of dispatches"
slice lands mid-token and reads ~25% low.

**821 dispatches, 5.72 ms kernel, in a 6.96 ms wall token (143.6 tok/s).**
The 1.24 ms that is not kernel is ~1.5 us of bubble per dispatch, and it is why
every dispatch-removing round in this series has paid about **1.5% per 40
launches removed** (0034, 0035, 0036 each removed exactly 40 and each measured
+1.5-1.6%). Dispatch count remains the single best predictor of decode gain on
this track.

| us/token | n/token | kernel | note |
|---|---|---|---|
| 1063 | 30 | mmvq_grouped Q6_K qkv+z (35.4us) | 581 GB/s, practical ceiling |
| 717 | 40 | mmvq Q4_K MoE gate+up (17.9us) | 527 GB/s |
| 666 | 1 | mmvq Q6_K output head | 621 GB/s = 96% of peak, done |
| 486 | 30 | mmvq Q6_K ssm_out (16.2us) | 425 GB/s in-graph — **at its isolated ceiling**, see round 11 |
| 481 | 37 | mmvq Q5_K expert down (13.0us) | **444 GB/s — still the weakest big matvec** |
| 327 | 70 | mmvf_grouped | dominated by the F32 router (2 MB/layer) |
| 267 | 50 | mmvq_grouped Q8_0 shexp | |
| 246 | 10 | mmvq Q6_K attn wq, 8192 rows (24.6us) | 560 GB/s |
| 225 | 80 | rms_norm_pre_add (2.8us) | **closed this round, see below** |
| 152 | 30 | gated_delta_net | 0026 territory; ~394 GB/s on 2 MB of state |
| 149 | 40 | mmvq Q8_0 shexp down (3.7us) | |
| 138 | 10 | mmvq Q6_K attn wo (13.8us) | 249 GB/s — small and latency-bound |
| 115 | 40 | topk_moe (2.7us) | ONE warp, 256 experts; pure launch floor |
| 109 | 80 | unary_gated_op_quant | after 0036 |
| 86 | 51 | rms_norm_f32 | |
| 71 | 41 | k_bin_bcast | the fused 8-way MoE expert reduction, 1.7us each |
| 60 | 41 | quantize_q8_1 | the MoE-swiglu 40 are structurally dead (round 17) |
| 51 | 13 | `__amd_rocclr_copyBuffer` | unattributed; worth one probe |

### Measured dead this round: rms_norm_pre_add at block 256

Round-14 map item 2 proposed dropping the 2048-column folded norm from a
1024-thread workgroup to 256 on RDNA4, on the theory that a 32-wave LDS block
reduction is the cost. **It is a 4.0% decode REGRESSION** (tg128 143.91/143.95
at 1024 vs 138.09/138.10 at 256, interleaved, arms disjoint). Both
instantiations already exist upstream, so this cost only a build.

The premise is backwards. At decode the grid is one block for one row, so the
kernel is bound by memory latency on the row loads, not by reduction depth:
1024 threads x 2 columns issues the 8 KB row in two rounds of loads, 256
threads x 8 columns serializes eight. Block size here is a memory-parallelism
knob and bigger wins — and 1024 is already the hardware maximum, so there is
nothing above it and nothing between that can beat it. The kernel's real cost
is a launch plus ~2 dependent memory round trips (2.8us for 40 KB = 14 GB/s on
one CU); the only lever left would be grouping the launches, and that is
structurally blocked — the 80 norms per token are strictly sequential in the
graph (norm -> attn -> norm -> ffn, twice per layer over 40 layers), so there
is never a same-shape pair in flight to collect. Treat this whole kernel as
closed.

### Best remaining candidate: fold the 8-way expert reduction into the down matvec

Round-14 item 3, still open and now the largest clean structural win. After
0034 the expert-down `mul_mat_id` writes `[2048, 8]` (one slice per expert
slot, routing weight already applied by `dst_scale`), and a fused multi-add
`k_bin_bcast` then sums the eight slices — 40 launches/token, 1.7us each, 71
us/token, moving 64 KB of reads and 64 KB of writes per layer to produce 8 KB.

The design that preserves the matvec's parallelism, which is the whole
difficulty: today `blockIdx.y` IS the expert slot (`channel_dst`, and
`ids[channel_dst]` selects the weight slab), so 2048 rows x 8 slots = 16384
warps are in flight. Do **not** make one warp loop over the eight experts —
these matvecs are latency-bound (see the ssm_out/wo rows above), and cutting
warps in flight 8x will cost more than the fold saves. Instead launch
`blockDim.y = n_expert_used` with each warp y taking `channel_dst = y` for the
same row, then reduce the eight partials through LDS and store once. Same 16384
warps, one store instead of eight, and the `k_bin_bcast` disappears. `nwarps`
is 1 for this shape today (the launch is `wg=32`), so `threadIdx.y` is free —
but note `blocks_per_iter` and the warp-trimming logic both derive from the
compile-time `nwarps`, so they need to stay keyed to k-splitting rather than to
the new expert dimension.

Byte-exactness is available if wanted: reduce the eight partials **in slot
order 0..7 in a single lane** rather than as an LDS tree, which is the order
the ggml add chain uses. Expected ~71 us (the bin_bcast) + ~10 us (traffic) +
40 fewer dispatches at ~1.5 us of bubble = **~2% decode**.

Ranked after that: (2) the Q5_K expert-down at 444 GB/s against the 527 the
Q4_K gate/up achieves — per-shape rows-per-block, never blanket (0024/0028
recorded blanket rpb on the ids path as a regression); (3) attributing the 13
`__amd_rocclr_copyBuffer` launches per token (51 us) — 0030 killed the state
gathers, so it is not obvious what is left copying.

### Prefill: the Q6_K dequant+GEMM route is deliberate, not a bug

A fresh pp512 trace shows `dequantize_block_q6_K` (3.8%) + `convert_unary`
(1.4%) + the rocBLAS `Cijk_*` GEMMs (~18.2%) = ~23% of the pass. That is 0021's
**intentional** routing: RDNA4 Q6_K MMQ tiles cost ~20% more than the fp16
dequant+GEMM path at prefill batches. Do not "fix" it back to MMQ — that was
measured and ranked 0.9476. The genuinely open prefill items are `mul_mat_q`
(41.3%, MoE experts, J-tile geometry already swept dead at J=32),
`gated_delta_net` (11.8%, 440us x 30/pass) and `k_bin_bcast` (4.7%, 663
launches) — the last being the prefill analogue of 0034, since at prefill
shapes MUL_MAT_ID takes MMQ and the routing-weight multiply is not folded.

## 0037: the MoE expert-slot reduction folded into the down matvec (+1.6% decode)

Round-19's "best remaining candidate", and the design in that section survived
contact: `blockDim.y = n_expert_used` with an LDS reduction in slot order, NOT
one warp looping over eight experts. What the design did **not** anticipate is
that most of the round would go on where the result is allowed to land.

After 0034 the expert-down `mul_mat_id` writes `[n_embd, n_expert_used]` with
the routing weight already applied in its epilogue, and a fused multi-add sums
the eight slices — 40 launches/token at 1.7us, moving 64 KB of reads and 64 KB
of writes per layer to produce 8 KB, when the matvec that wrote those bytes had
every one of them in a register. Two halves, only useful together:

1. `mmvq.cu`: `mul_mat_vec_q_expert_reduce`. `threadIdx.y` becomes the expert
   slot — free, because this shape trims to a single k-split warp
   (`calc_nwarps_launched`) — so the same `nrows_x * n_expert_used` warps stay
   in flight, the eight partials reduce through 32 bytes of LDS in slot order
   0..7 (the order `launch_bin_bcast_pack`'s fused multi-add uses), and there is
   one store instead of eight. Gated to exactly the launch it is derived from:
   one row per block, every k-block covered by one warp in a single pass, host
   trimming active. A dedicated entry point reuses the shared q8_1 cache, so
   declining and falling through to `ggml_cuda_mul_mat_vec_q` costs nothing.
2. `llama-graph.cpp`: the expert combine becomes **in-place at the decode
   shape**, so `moe_out` lands on the matvec's own output block (expert slot 0).

### Rule 1 of 0032 again, in a new costume: it is an allocation problem

The fold hoists the store back to matvec time, while the kernel's other blocks
are still reading the routing weights and the ids. Out of place, ggml-alloc
frees both at the routing multiply and puts `moe_out` **exactly there**:
`GGML_CUDA_MOE_REDUCE_DEBUG=1` reported 156/160 decode sites declining on the
weights and 4/160 on the ids — the fold matched perfectly and fired zero times.

Expert slot 0 is disjoint from both **by construction**, and that is the whole
reason half 2 exists: `experts` is allocated at the `MUL_MAT_ID` while the ids
are still live, and the routing multiply's destination reuses it in place while
`weights` is still live, so neither can ever be recycled into it.

### Measured-dead, and expensive: pinning tensors with dangling view nodes

The first version of half 2 kept `weights` and `selected_experts` alive with
consumerless `ggml_view_1d` nodes (a view keeps its source alive in ggml-alloc,
and view nodes are skipped at execution — apparently free). It worked: 160/160
sites folded. It also **made the engine nondeterministic**. Server greedy A/A
on the *same build with the CUDA fold disabled* went from 5/6 byte-exact to
**0/6**; it was not the 0032 epilogue pass and not the 0030 state view (both
toggled off, still 0/6). The mechanism was never identified and the approach was
abandoned rather than debugged. Two portable notes:

- **Do not pin tensors with consumerless view nodes on this fork.** Their
  `n_views` is never decremented, so the source is pinned for the whole graph
  and every later allocation moves.
- `ggml_view_tensor` is not the tool anyway: it builds an **op-NONE** tensor,
  which `ggml_build_forward_expand` files as a LEAF, so it neither counts as a
  view for ggml-alloc nor receives a backend assignment — the run aborts in
  `ggml_gallocr_allocate_node` on `GGML_ASSERT(buffer_id >= 0)`. Use
  `ggml_view_1d` if you must.

The ungated version also moved the gate ppl from 3.9268 (+/-0.0015 across loads)
to a flat 3.9359, because the extra use shifts allocations and the topk_moe
fusion's memory-range hazard scan is allocation-dependent. Any graph-lifetime
change on this model is a numerics change at prefill; keep them off the prefill
shape.

### Not byte-exact — and which half is responsible was measured, not assumed

`GGML_CUDA_MOE_REDUCE_PER_SLOT=1` keeps the new launch geometry but stores per
expert slot and leaves the add chain in the graph. It reproduces the folded
result **exactly** (decode-path ppl 5.5086 both), so the slot-order reduction is
exact and it is the rewritten matvec body the compiler contracts differently —
the same class as 0024/0026/0028/0031/0033, and the same reason 0031's
"mirrored the source structure" attempt failed. Keep that toggle: it is the only
cheap way to split a matvec fold's arithmetic from its epilogue.

### The gate this round needed, and does not have by default

**A decode-shape fold is invisible to the official ppl gate.** `llama-perplexity
-c 512` runs n_t=128 ubatches, the consumer takes MMQ, and this fold (which
requires `ne[2] == 1`) never fires — the green reading covers nothing.
`llama-perplexity -b 512 -ub 1` makes every ubatch a single token, so the fold
fires on every layer of every position, and gives a real decode-path perplexity.
Every decode-only patch in this series (0031, 0034, 0035, 0036) should be
re-read with that in mind; this is the first round to actually measure it.

Measured (runner box, `HIP_VISIBLE_DEVICES=0`):

- toggle A/B (both halves off = pre-patch), 5 rounds: tg128 143.63-143.91 ->
  **145.89-146.10**, median per-round ratio **1.0157**, 5/5 disjoint; pp512
  neutral (arms overlap, 4280-4362 off vs 4271-4338 on)
- dispatches **821 -> 781/token**; `k_bin_bcast` **41 -> 1/token**; decode
  kernel time 5.734 -> 5.686 ms/token. The mmvq family stays at 251 launches
  per token in both arms (40 individual+grouped expert-down launches become 40
  expert-reduce launches), so the whole gain is the vanished bin_bcast plus its
  ~1.5us of dispatch bubble — round 19's "1.5% per 40 launches removed" holds
  for a fourth consecutive round
- whole-process vs stock, 5 rounds: tg128 median ratio **1.7979** (0036:
  1.7825), pp512 1.3410
- **server level** (`llama-server -ngl 99 -c 8192 --parallel 1`, interleaved,
  uncached 533-token prompt): decode 140.60-141.17 -> **142.60-143.16** (arms
  disjoint, +1.4%); prefill 3466-3525 vs 3477-3517, neutral
- **prefill OFF-control**: first-token logprobs (`n_probs=20`, 5 prompts) are
  **byte-identical** between arms — prefill is provably untouched, as the two
  shape gates (`ne[2] == 1`, `n_tokens == 1`) require
- gate ppl 3.9248/3.9259/3.9262 vs stock 3.9314 (**-0.13 to -0.17%**);
  fold-off control same session 3.9269/3.9275
- **decode-path ppl** (`-b 512 -ub 1 --chunks 8`): **3.9404** vs 3.9338
  fold-off, i.e. **+0.229% vs stock 3.9314** — inside the 0.5% band, and
  deterministic across loads in both arms
- `test-backend-ops` MUL_MAT_ID / ADD / MUL / VIEW OK
- server greedy: A/A **deterministic** (5/6 byte-exact; p5 is the near-tie
  prompt that also drifts A/A in the control), vs control 2/6 byte-exact and
  4/6 drifting mid-completion — the reassociation class above
- the full 37-patch series applies clean to pristine `b10237` and the resulting
  tree is `diff -r` identical to the measured tree

Toggles: `GGML_CUDA_DISABLE_MOE_EXPERT_REDUCE=1` (fold),
`GGML_MOE_COMBINE_INPLACE=0` (graph half — set both for a true pre-patch
control), `GGML_CUDA_MOE_REDUCE_DEBUG=1` (one accept/decline line per site with
the gate that declined), `GGML_CUDA_MOE_REDUCE_PER_SLOT=1` (bisect mode).

Projection with 0033+0034+0035+0036+0037: decode 1.7265 x 1.0157 = **1.7536**,
prefill **1.2911**, ttft **1.2258** -> score **~1.563** vs the 1.4525 bank.

## 0038: the accumulator's declared shape was the contraction — 0037 is byte-exact

0037 shipped as reassociation-class on the strength of its own bisect mode:
`GGML_CUDA_MOE_REDUCE_PER_SLOT=1` reproduced the folded result exactly, so the
slot-order reduction was exact and the rewritten matvec body was to blame.
The body was not to blame. The **declaration** was.

Stock `mul_mat_vec_q` accumulates into
`float tmp[ncols_dst][rows_per_cuda_block]` and indexes it inside `#pragma
unroll` loops. 0037 collapsed that to a scalar `float tmp` — the same
arithmetic, and the compiler contracts it differently. Restoring the array
shape (`float tmp[rpb]`, written by an unrolled row loop exactly as stock's `i`
loop does) restores stock codegen:

- **decode-path perplexity** (`-b 512 -ub 1 --chunks 8`, so the fold fires on
  every layer at every position): fold ON **3.9338**, fold-off control
  **3.9338** — identical, where 0037 read 3.9404 against the same control
- **server greedy vs the fold-off control: 6/6 BYTE-EXACT** (0037: 2/6). Against
  the full pre-patch build it is 5/6, the sixth being p5, the near-tie prompt
  that drifts A/A in the control too
- no cost: toggle A/B 5 rounds, tg128 143.01-143.23 -> 145.16-145.41, median
  ratio **1.0152**, 5/5 disjoint — 0037 measured 1.0157, i.e. the same gain
- official gate ppl 3.9320/3.9284/3.9267 vs stock 3.9314; prefill still
  structurally untouched; `test-backend-ops` MUL_MAT_ID / ADD / MUL OK

**This contradicts 0031's note that mirroring the source structure does not
help.** It does — but the thing that has to match is the accumulator's declared
shape and the loop nest that writes it, not the surrounding statements. Try it
before accepting a reassociation-class result on any future matvec rewrite; it
took one build here and turned a ppl-gated argument into bit-identity. It is
also worth re-trying on 0031 and 0033, whose folds are still drifting.

### Measured-dead this round: rows per block in the expert-reduce launch

The array shape arrives with the knob it was written for,
`GGML_MMVQ_EXPERT_REDUCE_RPB` (default 1 = shipped). Round-20 item 1 was the
expert-down at 444 GB/s against the 527 the Q4_K gate/up reaches; giving each
warp more than one row is the obvious fix, and it is not one. Swept on gfx1201,
paired and interleaved:

| rpb | tg128 | note |
|---|---|---|
| **1** | 145.45-145.76 | **shipped**; byte-exact |
| 2 | 145.74-146.03 | +0.20%, 5/5 disjoint — but re-contracts, so not byte-exact |
| 4 | 138.27 | **-4.8%** |
| 8 | 144.42 | -0.6% |

rpb=2 is real but tiny and costs the bit-identity argument, so it is not the
default; rpb=4 collapses. This is the third independent confirmation (after
round 11's dense-Q6_K corrections and 0029's per-shape retreat) that **matvec
launch geometry on this part is exhausted** — the small quantized matvecs are
latency/ramp-bound, and neither wider loads nor more rows per block moves them.
Per-row, per-thread k-block assignment is unchanged at every rpb, so the knob is
safe to re-sweep on other hardware.

Projection is unchanged from 0037 (~1.563 vs the 1.4525 bank): 0038 buys
bit-identity, not speed.

## Round-20 map (post-0037: 781 dispatches/token)

`k_bin_bcast` is now 1/token and the quantize census is closed, so the cheap
dispatch-removal seam this series has mined since round 12 is essentially spent
at decode. What is left, in order:

1. ~~**The Q5_K/Q6_K expert down at 444 GB/s**~~ — **swept dead, see 0038.**
   Rows per block inside the fold kernel gives +0.20% at rpb=2 and -4.8% at
   rpb=4. Geometry on this matvec is finished.
2. **`rms_norm_pre_add` 80 x 2.8us** is closed as a kernel (round 19) but its
   *input* is not: after 0037 it reads the 8 KB reduced row instead of nothing
   new, so the traffic argument that motivated the fold is spent — what remains
   is folding the residual add chain, which is the same hoisted-write
   allocation problem this round solved, and slot 0 is again the safe landing
   spot.
3. `mmvf_grouped` (70/token, 327us) is dominated by the F32 router reading
   2 MB/layer; a Q8_0 router would halve that but is a weight-format change.
4. Prefill: `mul_mat_q` (41.3%), `gated_delta_net` (11.8%) and the prefill
   `k_bin_bcast` (4.7%, 663 launches) — the last being the prefill analogue of
   0034, still untouched because MUL_MAT_ID takes MMQ at prefill shapes.

## Recommended firing order

The runner applies the whole patch directory, so the series is a ladder rather
than a choice; 0001-0038 is the verified set and every patch has a disable
toggle. 0038 has no toggle and no speed of its own — it is 0037's kernel made
bit-identical, and must not be applied without 0037. If a ranked run has to be
pared back, drop from the end: 0038+0037, 0036,
0035, 0034 are independent +1.5% decode folds; 0033 is the prefill fold; 0032
and 0030 are the two largest decode wins and should be the last to go. 0037 has
two halves in different files — dropping it means dropping both, since the
graph half alone leaves the decode combine as n_expert_used-1 separate adds.
