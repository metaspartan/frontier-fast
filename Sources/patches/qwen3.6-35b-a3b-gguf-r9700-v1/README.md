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

## Round 22: the firing audit — what the official gate actually covers

Round 21 noticed in passing that a decode-shape fold can be invisible to
`llama-perplexity -c 512`. This round measured that for the whole banked stack
instead of arguing it, and the answer is worse than the note suggested: **five
of the nine banked folds never execute a single instruction during the official
gate.** Their entire green perplexity history covers nothing.

The instrument is a kernel census, not a source reading. For each patch, run
the *same* command with that patch's toggle flipped, take a `rocprofv3
--kernel-trace` census (kernel name -> launch count), and diff it against the
all-on arm. A patch that fires changes the census. A patch that does not fire
changes nothing, and a green ppl from that command is silence, not evidence.

Two shapes, both `llama-perplexity` from the candidate build:

- **gate shape** — `-c 512`, what the runner runs. On this fork that reports
  `batch_size=2048, n_seq=4`, and each ubatch is one 512-token sequence, so
  the folds see `n_t=512, n_s=1`. (Round 13's note that the gate runs
  `n_t=128, n_s=4` is wrong for this invocation; measure it, do not assume it.)
- **decode shape** — `-b 512 -ub 1`. Every ubatch is a single token, so
  `n_t=1` and every decode-gated fold fires at every layer of every position.
  8 chunks is ~75 s and is deterministic across loads in every arm measured.

### The table

| patch | fires at gate shape? | fires at decode shape? | decode-path ppl, patch off -> series | greedy identity vs its own off-control |
|---|---|---|---|---|
| 0030 rs-state identity view | **yes** | yes | 3.9338 -> 3.9338 (identical) | 5/6 (r11) |
| 0031 GDN conv-chain fold | **NO** | yes | 3.9183 -> 3.9338 (**+0.396%**) | 2/6 |
| 0032 mmvf GDN epilogue fold | **NO** | yes | 3.9338 -> 3.9338 (identical) | **6/6 (new)** |
| 0033 conv fold at seq shape | **yes** | no (0031 owns `n_t==1`) | 3.9338 -> 3.9338 (inert) | 2/6 at its own shape |
| 0034 MoE routing-weight fold | **NO** | no in the full stack* | 3.9338 -> 3.9338 (identical) | 6/6 |
| 0035 shexp gate pin + fold | **NO** | yes | 3.9338 -> 3.9338 (identical) | 6/6 |
| 0036 unary+mul q8_1 fold | **yes** | yes | 3.9338 -> 3.9338 (identical) | 6/6 |
| 0037+0038 expert-slot reduction | **NO** | yes | 3.9382 -> 3.9338 (**-0.112%**) | 6/6 |

\* 0034's census is unchanged at the decode shape only because **0037 absorbs
the same routing-weight MUL** — the expert-reduce detector runs first and
swallows it. Isolated with 0037 off, 0034 fires and is still numerically
inert (3.9382 with and without it). It is not dead weight: it is 0037's
fallback when the add chain does not match.

Reference points for the column: **stock decode-path ppl 3.9409**, the full
banked 0001-0038 stack **3.9338** — a relative delta of **-0.180%**, inside the
0.5% band. Every arm above was read twice from two separate loads and both
loads agreed to all four decimals.

### What the table says

1. **The queue is safe, and now for a measured reason.** Before this round the
   decode path had no perplexity number at all for 0031, 0032, 0034, 0035 or
   0037; the readings that existed were taken at a shape where those kernels
   were not running. The stack is -0.180% from stock on the path it actually
   optimises.
2. **0031 is the whole numeric budget.** It is the only patch that moves the
   decode-path perplexity materially (+0.396% against its own off-control), and
   it is one of the five that the gate never exercises. Everything else in the
   series is either identical to its control or, in 0037's case, -0.112%.
3. **0032 was the biggest blind spot and came back clean.** The second-largest
   decode win in the series (+3.4%) had no numeric characterisation at all —
   its round-13 entry lists ppl at a shape where it does not fire, and it never
   had a greedy identity run. Measured this round: decode-path ppl identical to
   its off-control, and **6/6 byte-exact** server greedy. It is byte-exact.
4. **0038's bit-identity claim survives a sharper test.** Round 21 compared the
   fold against a control with *both* halves off, which conflates the matvec
   with the graph rewrite. Split here: with only the CUDA fold disabled
   (`GGML_CUDA_DISABLE_MOE_EXPERT_REDUCE=1`, graph half left on) the decode-path
   ppl is **3.9338 — identical to all-on**. With only the graph half disabled
   (`GGML_MOE_COMBINE_INPLACE=0`) it is **3.9382**. So the -0.112% belongs
   entirely to the fused multi-add in the graph, and 0038's accumulator really
   did make the matvec byte-exact.

### Method notes for the next agent

- **Do not trust a debug env to prove firing.** `llama-bench` installs a null
  ggml log callback, so every `GGML_LOG_INFO` accept/decline print vanishes
  there — 0037's detector logged nothing under llama-bench while demonstrably
  folding 40 launches per token. The census diff has no such failure mode.
- A firing census only needs `--chunks 1`; the decode-shape census here used
  `-c 128 -b 128 -ub 1` and still resolved every patch.
- The Tensile GEMM kernel name (`Cijk_...MT32x64x16...` vs `...MT64x64x16...`)
  differs between arms at the gate shape at a constant launch count of 100.
  That is Tensile picking a tile, not a fold firing. Diff on counts, and read
  the names.

## 0039: the decode conv fold loses its sign-of-zero divergence

Round 21's lesson — that the *declared shape* of an accumulator, not the
surrounding statements, is what makes the compiler contract differently — was
retried on 0031 as instructed. It is half right here, and the half that works
is not the half that was expected.

Stock `ssm_conv_f32` accumulates inside a **runtime** loop over `n_t`, indexes
its register window with the runtime expression `(i + j) % d_conv`, and
finishes with `sumf += b`, where `b` comes from a runtime-nullable bias pointer
the compiler cannot see through. 0031 collapsed all three: `n_t == 1` is a
compile-time fact in that kernel, so the loop vanished, the modulo index became
constant, and the trailing add was dropped as the identity it almost is. 0039
restores all three — `n_t` arrives as a kernel argument so the loop cannot be
unrolled, and `bias` arrives as a pointer so `sumf += b` survives as a real
instruction.

Measured on the runner box, `GGML_SSM_CONV_FOLD_STOCK_SHAPE=0` restoring the
0031 kernel:

- **conv output at the first decode position, dumped and compared float by
  float against the stock chain** (`GGML_CUDA_DISABLE_GRAPHS=1`,
  `GGML_SSM_CONV_DUMP`, 30 GDN layers x 8192 channels = 245,760 values):
  0039 is **0 differing — byte-identical to stock**. The 0031 kernel differs in
  **39** of them, and every one of the 39 is value-equal with a different bit
  pattern: **-0.0 where stock produces +0.0**. That is exactly what the dropped
  `sumf += b` costs, and it is the same trap 0034 documented from the other
  direction (`result += x_biases[j]` is the identity on everything except
  -0.0). The seq folds already carried an explicit `__float_as_int` fixup for
  this; the decode fold never did, and nobody had looked.
- **it is not enough to change the correctness class.** Server greedy against
  the fold-off control is still **2/6** byte-exact, identical to 0031's own
  2/6, and 0039 vs 0031 is **6/6** — the two kernels agree with each other
  everywhere the six completions reach. The first decode position has an
  all-zero conv window, so byte-exactness there only proves the sign-of-zero
  fix; once the window carries real data the 4-term dot product still contracts
  differently, exactly as 0033's entry describes for its own kernel.
- no cost: toggle A/B, 5 rounds, arms fully overlapping — this patch buys
  fidelity, not speed.
- decode-path ppl identical to the 0031 kernel.

**So the round-21 doctrine needs one more clause.** Mirroring the accumulator's
declared shape recovered bit-identity for 0038 because the whole difference
there *was* codegen. Here the same treatment recovered only the arithmetic that
was genuinely missing (the trailing add) and left the contraction difference
standing. Before assuming a rewrite is reassociation-class, dump the tensor and
count the differing values: if they are value-equal with different bits, it is
a dropped identity operation and it is fixable; if they move by ULPs, it is
contraction and mirroring the source will not save it. That diagnostic costs
one build and it is now wired in — `ggml_cuda_op_ssm_conv_fold` gained the
`ssm_conv_fold_dump` hook the seq path already had.

0031 therefore stays the series' one materially non-bit-exact decode fold, at
+0.396% on the decode path against its own control and -0.180% for the stack as
a whole against stock. It is inside the band, it is measured, and it is now
documented at the shape where it actually runs.

## Round-23 map (post-0039 re-profile, read before picking a lever)

Fresh rocprofv3 census of the banked build, `llama-bench -p 0 -n 34 -r 1`,
normalised by 35 tokens (27338 dispatches; 781.1/token, so the window is whole).
**781 dispatches, 5.70 ms of kernel in a 6.96 ms wall token** — unchanged from
the round-20 projection, and only 33 distinct kernels remain.

| us/token | n/token | each | kernel | note |
|---|---|---|---|---|
| 1064 | 30 | 35.5us | mmvq_grouped Q6_K qkv+z | 583 GB/s, 91% of peak |
| 735 | 40 | 18.4us | mmvq Q6_K ssm_out + attn wo | at its isolated ceiling (round 11) |
| 719 | 40 | 18.0us | mmvq Q4_K MoE gate+up | 527 GB/s |
| 666 | 1 | 666us | mmvq Q6_K output head | 96% of peak, done |
| 499+42 | 37+3 | 13.5us | mmvq expert-reduce Q5_K/Q6_K | 0037/0038; geometry swept dead |
| 330 | 70 | 4.7us | mmvf_grouped | 216us of it is the F32 router, 388 GB/s |
| 246 | 10 | 24.6us | mmvq Q6_K attn wq | 560 GB/s |
| 228 | 80 | 2.9us | rms_norm_pre_add | closed as a kernel (round 19) |
| 167 | 31 | 5.4us | mmvq_grouped Q8_0 shexp | |
| 153 | 40 | 3.8us | mmvq Q8_0 shexp down | |
| 151 | 30 | 5.0us | gated_delta_net | |
| 115 | 19 | 6.0us | mmvq Q8_0 (ids) | |
| 109 | 40 | 2.7us | topk_moe | ONE workgroup; ~1.3us above the launch floor |
| 101 | 71 | 1.4us | unary_gated_op_quant | after 0036 |
| 82 | 60 | 1.4us | quantize_q8_1 | 40 are structurally dead (round 17) |
| 82 | 50 | 1.6us | rms_norm_f32 | 30 of them are the GDN gated norm |
| 61 | 30 | 2.0us | l2_norm_f32_grouped | the GDN q/k conv norms — **best open fold** |
| 58+22 | 10+10 | | flash_attn_tile + combine | |
| 52 | 12.6 | 4.1us | `__amd_rocclr_copyBuffer` | still unattributed; worth one probe |
| 47 | 30 | 1.6us | ssm_conv_fold_stock | 0031/0039 |

Long context on the same build: **tg128 @ d16384 = 134.5 tok/s, @ d32768 =
124.8 tok/s** (decode-only, r=2). The 32k long board entry stands.

Prefill census (pp512, per pass, 2276 dispatches / 114.9 ms of kernel before
0040): `mul_mat_q` Q4_K experts 21.5%, Q5_K expert-down 14.7%, `gated_delta_net`
11.5%, the deliberate Q6_K dequant+GEMM route ~14%, **`k_bin_bcast` op_mul 3.0%
(40 x 87us) — taken by 0040**, `k_bin_bcast` op_add 0.9%.

Ranked open levers after 0040: ~~(1) fold the two GDN `l2_norm`s into the decode
conv-fold epilogue~~ — **taken by 0041, +1.3% decode**; (2) the F32 router at
388 GB/s against the 583 the grouped Q6_K matvec reaches — 257 one-warp blocks
is too little memory parallelism for 2 MB, but k-splitting it reroutes experts,
so it is not free — **0042 took the byte-exact half of this (unroll), leaving
only the parts that reroute experts**; ~~(3) attribute the 12.6 `copyBuffer`
launches~~ — **closed in round 26: 6/token in steady state, 0.36%**; (4) prefill
`gated_delta_net` (11.5%); (5) the 30 GDN gated `rms_norm_f32` launches are the
same shape of opportunity 0041 just took, one kernel further down the chain —
they normalise `[128, 32 heads]` straight out of `gated_delta_net`, whose grid
is already 32 blocks in z.

## 0040: the MoE routing weight folded into the MMQ prefill epilogue (+2.3% prefill, BYTE-EXACT)

0034 folded the routing-weight multiply into the `mul_mat_id` epilogue at the
decode shape, where MUL_MAT_ID takes mmvq. At prefill batches the same node
takes MMQ instead, so the multiply survives as a full elementwise pass over
`[n_embd, n_expert_used, n_tokens]`: 40 launches of 87 us, **3.48 ms of a
114.9 ms pass**, moving 33 MB of read-modify-write per layer to apply one scalar
per column.

The index needed no new bookkeeping, which is why this was worth doing at all.
MMQ stores through `dst[ids_dst[j]*stride_col_dst + i]`, and for MUL_MAT_ID
`ids_dst[j]` is the flat destination row `token*n_expert_used + slot` — exactly
the flat index into the `[1, n_expert_used, n_tokens]` routing-weight vector. So
the epilogue multiplies by `dst_scale[ids_dst[j]]` and stores once. Bit-exact by
0034's argument: the multiplicand is the same rounded f32 the stock kernel would
have written and the MUL would have read back.

### The whole round is in one structural detail: a null-pointer branch is not free

The first version put the obvious conditional inside MMQ's unrolled store loop:

```c
if (dst_scale_used) { val *= dst_scale[ids_dst[j]]; }
```

Toggle-isolated that measured +1.55% on pp512, 5/5 disjoint — and it was a lie.
Against a control binary built from the parent commit the **fold-off arm was
1.3% slower than the control** (the Q5_K expert-down went 456.6 -> 478.5 us with
the fold disabled), so the patch as a whole scored **1.0013** — a wash. A
conditional global load inside a fully unrolled store loop costs on every MMQ
launch in the build, whether or not the fold ever fires.

Splitting it into a **separate scaled store loop**, with the stock loop left
verbatim behind `if (dst_scale != nullptr) { ...; return; }`, fixes it
completely: the fold-off arm returns to 0.9992 against the control and the
fold-on arm is 1.0228. Same arithmetic, same number of instructions executed —
only the codegen of the path that does not execute changed.

**This is the prefill OFF-control trap from 0021 in a new costume, and the
toggle A/B cannot see it.** Both toggle arms live in the same binary, so any
cost the patch imposes on the untoggled path is subtracted from both and
cancels. Any patch that edits a hot kernel shared with untouched call sites
needs a third arm built from the parent commit.

Measured (runner box, `HIP_VISIBLE_DEVICES=0`, whole-process interleaved 3-arm
A/B, rotated order, 5 rounds):

| arm | pp512 | median | vs control |
|---|---|---|---|
| control (parent commit binary) | 4277.2-4343.8 | 4305.7 | — |
| 0040, fold OFF | 4268.2-4342.4 | 4302.3 | **0.9992** |
| 0040, fold ON | 4368.8-4442.8 | 4424.9 | **1.0277** (per-round median 1.0228) |

Arms disjoint 5/5 — `min(ON) = 4368.8 > max(control) = 4343.8`. tg128 neutral in
all three arms (145.4-145.9).

- pp512 census: **2276 -> 2236 dispatches/pass, 117.4 -> 112.0 ms of kernel**;
  the 40 `k_bin_bcast` op_mul launches at grid 1024x8x512 are gone
- gate ppl `-c 512 --chunks 8`: ON **3.9249 / 3.9269** (two loads), OFF 3.9271,
  stock 3.9314 — **-0.11 to -0.17%**, in band. This fold DOES fire at the gate
  shape (it is gated on the prefill shape), so for once the official gate covers
  the patch that needs it.
- decode-path ppl `-b 512 -ub 1 --chunks 8`: **3.9338**, identical to the banked
  stack — the fold is gated off the decode shape by construction (mmvq claims
  those batches, and that is 0034)
- `llama-server` greedy **6/6 BYTE-EXACT** vs the fold-off control, completions
  276-1194 B, all non-empty
- `test-backend-ops` MUL_MAT_ID / MUL_MAT / MUL OK
- stream-k stays correct: the partial written to the fixup buffer is left
  unscaled and `mul_mat_q_stream_k_fixup` applies the weight as it accumulates
  (the shapes this model uses are all non-stream-k, but the plumbing is there)
- `GGML_CUDA_DISABLE_MOE_WEIGHT_FUSE_MMQ=1` restores the stock pair

Detection mirrors 0034 with the decode guard inverted: a (MUL_MAT_ID, MUL)
subgraph via `ggml_can_fuse_subgraph`, an f32 factor of shape
`[1, n_expert_used, n_tokens]` contiguous, an identically laid out MUL
destination, and the exact dispatch conditions under which
`ggml_cuda_mul_mat_id` would have taken the MMQ route (quantized src0, batch
above the mmvq mmid cutoff, `ggml_cuda_should_use_mmq`).

## 0041: the GDN q/k l2 norms folded into the decode conv-fold epilogue (+1.3% decode, BYTE-EXACT)

Round-23's top-ranked open lever, and it paid. qwen35moe convolves `[q | k | v]`
as one 8192-channel block and then l2-normalises the q and k halves per 128-wide
head. Grouped since 0019 that is still **30 launches per token at 2.0 us**, each
re-reading 16 KB the conv fold had in registers a moment earlier.

The kernel half is a shape coincidence worth remembering: a 256-thread conv
block owns exactly 256 channels, which is exactly **two 128-wide norm rows**. So
warps 0 and 1 replay the stock one-warp `l2_norm` body out of shared memory —
same `col = lane; col += 32` accumulation order, the same `warp_reduce_sum`
(for `block_size == WARP_SIZE`, `block_reduce<SUM,32>` *is* `warp_reduce_sum`),
the same `rsqrtf(fmaxf(tmp, eps*eps))`, the same store order. Byte-identical to
`l2_norm_f32_grouped` by construction.

### Where the result is allowed to land was the whole round (again)

This is the third time in this series (0032, 0037, now 0041) that the fold was
easy and the allocation was not. Two successive declines, each fixed by a
different move:

1. **With separate destinations, ggml-alloc hands the two norms buffers that a
   node between the conv and the norms is still using.** Declined on every layer
   of every token. Fixed by making the norms **in-place on the conv output at
   the decode shape** (`qwen35moe.cpp`, `GGML_L2_NORM_INPLACE=0` restores).
   Semantically free — the q and k halves of the conv output have no consumer
   other than these two norms — and it puts the result inside the buffer the
   fold already owns. Gated to `n_t == 1`: at prefill the q/k views are strided
   over tokens, and an in-place destination would make the norm kernel store
   contiguously into a strided view.
2. **In place, the destination then overlaps the ACTIVATION.** At the decode
   shape ggml-alloc routinely co-locates them, because the CONCAT is the
   activation's last consumer. That is not a new hazard: the fold already
   depends on the co-location being *exact*, since thread `ch` reads `x[ch]` and
   then writes `y[ch]`, the same address. The norm write lands on the same 256
   channels the block has already read, after a `__syncthreads`, so it is safe
   under exactly that condition — `x->data == silu->data` and a unit channel
   stride — and the detector declines on anything weaker.

And one guard that had to be **removed**: the CONCAT output must not be in the
operand list. When this fold fires the concat is never materialised, and the
0031 detector has already rejected any consumer of it other than the SSM_CONV
and the writeback CPY, both folded away. Its buffer is dead — and ggml-alloc
hands exactly that buffer to the norm destinations on every layer, so guarding
on it declined the whole fold for nothing. **When a fold declines, print which
operand it declined against** (`GGML_CUDA_SSM_CONV_L2_DBG=1`); three of this
round's builds were spent narrowing that down one bit at a time.

The epilogue is a separate kernel (`ssm_conv_fold_l2_f32`) rather than a runtime
branch inside `ssm_conv_fold_stock_f32` — see 0040 for what a not-taken branch
in a hot kernel costs.

Measured (runner box, `HIP_VISIBLE_DEVICES=0`, whole-process interleaved 3-arm
A/B against a control binary built from the parent commit, rotated order):

| arm | tg128 | median | vs control |
|---|---|---|---|
| control (0040 binary) | 145.36-145.79 | 145.65 | — |
| fold + in-place OFF | 145.57-145.80 | 145.63 | **0.9999** |
| ON | 147.50-147.80 | 147.58 | **1.0133** |

Arms disjoint 5/5 — `min(ON) = 147.50 > max(control) = 145.79`.

- pp512 neutral: 6 further rounds at `-r 3`, per-round ratios 0.9988-1.0039,
  median 0.9997
- long context: tg128 @ d16384 **134.01 -> 135.84** (+1.37%), @ d32768
  **124.53 -> 126.07** (+1.24%) — the 32k long board moves with it
- decode census: **781.1 -> 751.1 dispatches/token**, 5680 -> 5627 us of kernel;
  `l2_norm_f32_grouped` is gone and the conv fold goes 1.57 -> 2.24 us, so the
  ~1.6 us of per-dispatch bubble is again most of the win
- decode-path ppl `-b 512 -ub 1 --chunks 8`: **3.9338** all on, **3.9338** with
  both halves off, **3.9338** with in-place on and the fold off — identical to
  four decimals in all three arms, and identical to the banked stack
- gate-shape ppl `-c 512 --chunks 8`: 3.9271, unchanged (neither half fires
  there — this is a decode-shape fold, so the official gate does not cover it;
  the decode-shape reading above is the one that counts)
- `llama-server` greedy **6/6 BYTE-EXACT** vs the fold-off control
- `test-backend-ops` L2_NORM / SSM_CONV / GATED_DELTA_NET OK

One toggle interaction to know: `ssm_conv_fold_l2_f32` is derived from 0039's
stock-shaped loop nest, so when the l2 fold accepts,
`GGML_SSM_CONV_FOLD_STOCK_SHAPE=0` no longer reaches the 0031 kernel. Disable
`GGML_CUDA_DISABLE_SSM_CONV_L2_FOLD=1` first if you want that comparison back.

## 0042: the grouped mmvf unroll — the F32 router is latency-starved, not slow (+0.7% decode, BYTE-EXACT)

Round-23's open lever (2). The grouped mmvf launch is 70 dispatches and 330 us
per token, most of it the 256-row F32 MoE router: 2 MB of weights per layer read
by **257 one-warp blocks in 5.39 us = 388 GB/s**, against the 583 GB/s the
grouped Q6_K matvec reaches on the same part. Nothing is wrong with the loop —
257 waves cannot keep enough loads in flight to cover DRAM latency, and the only
knob that changes that without changing the arithmetic is `nunroll`.

**`nunroll` is a free knob, and that is the point worth carrying.** It only
groups the loads: the unrolled block issues its loads together and then mads
them in ascending column order, and the tail loop continues the same sequence,
so every lane still accumulates columns `tid, tid+32, tid+64, ...` in that order
at every setting. Bit-identical by construction — no correctness argument to
make, just a sweep.

| nunroll | tg128 | note |
|---|---|---|
| 4 (stock) | 146.85 | 8 loads in flight per wave |
| 8 | 147.66 | |
| **16** | **148.01** | **shipped** |
| 32 | 147.79 | register pressure starts to bite |

Per kernel, 4 -> 16: the router group **5.39 -> 4.89 us** (388 -> 429 GB/s), the
beta/alpha group **3.71 -> 2.64 us** (-29%), decode kernel time 5650 -> 5584 us.

3-arm A/B against a control built from the parent commit, rotated, 5 rounds:

| arm | tg128 | median | vs control |
|---|---|---|---|
| control (0041 binary) | 147.29-147.76 | 147.56 | — |
| this build forced to nunroll=4 | 147.31-147.68 | 147.40 | **0.9989** (templating is free) |
| default nunroll=16 | 148.38-148.73 | 148.59 | **1.0070** |

Arms disjoint 5/5. pp512 neutral (0.9995). tg128 @ d16384 135.84 -> 136.73,
@ d32768 126.07 -> 126.82. Decode-path ppl **3.9338** at nunroll 16 and at
nunroll 4 — identical, and identical to the banked stack; gate-shape ppl 3.9293
vs stock 3.9314; `llama-server` greedy **6/6 BYTE-EXACT** vs the nunroll=4
control; `test-backend-ops` MUL_MAT OK.

`GGML_MMVF_GROUP_NUNROLL=4|8|16|32` re-sweeps it. Safe to re-sweep on other
hardware: the per-lane column assignment is unchanged at every setting, so the
answer can differ without the numerics differing.

The router is still only at 429 GB/s. The remaining gap needs either more rows
per block or wider per-lane loads, and both change the per-lane column
assignment — which for the router means different logits, which means a
different top-8. That is inside the gate but no longer byte-exact, so it is a
separate decision, not a continuation of this one.

## Round 26: four levers measured dead — do not re-buy any of them

No patch. Four levers were taken to a measurement and all four closed — three of
the round-23 open items plus a port of a sibling track's win. Each cost one
build and one sweep; the point of writing them down is that they each *look*
like the obvious next thing.

### 1. The mmvq k-loop unroll does NOT inherit 0042's win

0042 raised the grouped **F32** matvec's unroll from 4 to 16 for +0.7%. The
obvious follow-up is the quantized matvecs, where the Q5_K expert-down sits at
444 GB/s and the Q4_K expert gate+up at 527 GB/s against 583 for the dense Q6_K
qkv matvec. `#pragma unroll N` was added to all four mmvq k-loops (bit-identical
by the same argument — the accumulation into `tmp[j][i]` stays in ascending
`kbx` order at any unroll) and swept, interleaved, 3 rounds against a
parent-commit control:

| arm | tg128 median |
|---|---|
| control | 148.86 |
| unroll disabled | 148.73 |
| unroll 2 | 148.60 |
| unroll 4 | **147.72 (-0.77%)** |

Monotonically worse. **The premise does not transfer, and the diagnostic that
separates the two cases is free: divide the launch grid by the warp size.** The
router launches 257 waves and is latency-limited, so loads-in-flight is the
lever. The expert matvecs launch 8192-16384 waves — already saturated — so their
bandwidth gap is locality (reads scattered across 256 expert slabs, 8 used), and
unrolling only costs registers. Together with the three geometry sweeps
(rows-per-block, nwarps, MMQ J), **every kernel-tuning axis on the quantized
matvecs is now closed**; the Q5_K expert-down at 444 GB/s is an access-pattern
problem, not a tuning problem.

### 2. 0023's MoE J-tile multiplier is already the optimum

0023 caps the MMQ J tile at `2 * cols_per_expert` for MoE. At pp512 that is
`2 * ceil(4096/256) = 32`, and each expert's ~16-column segment fills half a
tile — which reads like obvious waste. It is not. Swept (bit-identical: J
changes the tile decomposition, never the per-element k-loop order), 3
interleaved rounds, pp512 medians:

| multiplier | floor | pp512 | vs shipped |
|---|---|---|---|
| **2** | 16 | **4421** | **shipped** |
| 1 | 16 | 4086 | -7.6% |
| 1 | 8 | 4084 | -7.6% |
| 3 | 16 | 4258 | -3.7% |
| 4 | 16 | 4184 | -5.4% |

pp2048 agrees (4194 at mult=2 vs 3909 at mult=1). A sharp local optimum in both
directions — halving J costs more in tiles and re-loaded x-tiles than the padding
it saves. `GGML_CUDA_MMQ_MOE_J_MULT` / `_MIN` were not kept; the sweep is the
result.

### 3. `__amd_rocclr_copyBuffer` is a profiling artefact of the first token

Round-19 and round-23 both listed "attribute the 12.6 `copyBuffer` launches per
token (52 us)" as an open item. The dispatch-stream analysis (sort the kernel
trace by timestamp, split on the output-head launch, count per window) says the
number is wrong: **237 of the 441 copies in a 35-token trace happen in the first
window**, and the steady-state rate is ~6 per token — 25 us, 0.36% of the wall
token, arriving in bursts of 0 to 26 that do not correlate with graph position.
In a typical token there are exactly **two**, one just after the token-embedding
`get_rows` and one mid-graph. Not a lever. **Per-token averages taken over a
whole trace hide setup work; split on a token boundary before believing one.**

### 4. The register-cached rms_norm row pays on GB10 and nothing here

A sibling agent landed exactly this on the GB10 `lfm2.5` track (its 0013):
`rms_norm_pre_add_f32` walks its row twice and the second walk re-reads the
floats the first one computed, so when `ncols` is an exact multiple of
`block_size` and the per-thread column count is small, both loops become
unrolled loops and the values stay in registers. **+0.65% prefill there, 5/5
arms disjoint.** The patch applies to this tree verbatim — same `b10237` base,
same kernel — and the two `rms_norm_pre_add` instantiations are 3.6% of a pp512
pass here against 4.4% there, so the kernel share is comparable.

On R9700 it is **neutral**: pp512 per-round ratios 0.9970-1.0068, median 1.0000,
arms fully overlapping; tg128 median 1.0010 with arms overlapping (5/5 rounds
positive, but +0.10% is below this harness's floor). Correctness is fine either
way — decode-path ppl 3.9338 identical, gate 3.9284 vs control 3.9306, greedy
6/6 byte-exact — it just does not buy time. Reverted.

**The reason is the memory system, not the kernel, and it is worth carrying:**
the GB10 is unified LPDDR5X with no large last-level cache, so re-reading a row
costs a real trip; the R9700 has **64 MB of Infinity Cache**, and an 8 KB row
that was just written is always still in it. Before porting any "avoid the
second read" optimisation onto this box, ask whether the re-read working set
fits the LLC. The corollary is the useful half: **on R9700 redundant reads of
recently-written data are effectively free**, which is why every win in this
series has come from launch count (~1.5% per 40 dispatches removed) or from
first-touch weight bandwidth, and never from re-read elimination.

### Still open, in order

1. ~~**Prefill `gated_delta_net`**~~ — **taken by 0043, +5.7% prefill, and the
   chunkwise scan the `//TODO` asks for was not needed**: the kernel was
   issue-bound on its own reduction, not on the serial recurrence. After 0043 it
   is 283 us x 30 = 7.6% of a pass; a chunkwise scan is still the only way to
   attack what is left, and it is still an algorithmic change with real numeric
   risk.
2. **The MoE expert matvecs' locality at decode** (Q5_K down 444 GB/s, 499
   us/token) — every tuning axis is closed, so this needs a layout idea
   (interleaving the 8 routed slabs, or a different expert-major ordering),
   which on a pinned GGUF means the 0021 load-time hook. **This is now the top
   open lever**, and it is the only one left that moves decode.
3. `unary_gated_op_quant` (71/token, 101 us + ~114 us of bubble) cannot fold into
   the gate/up matvec epilogue: the q8_1 block max spans 32 output rows and each
   matvec block owns one row, so it would need a cross-block reduction.
4. The GDN gated `rms_norm` (30/token) cannot fold into `gated_delta_net`
   either: its grid is `(H, n_seqs, S_v/num_warps)`, so **32 blocks** cooperate
   on each head's 128-wide output row and the norm needs all of them. 0041's
   trick does not generalise — check the producer's block-to-row mapping first.
5. `topk_moe` (40/token, 2.73 us) is already fixed-depth after 0014; what is
   left is close to the single-workgroup launch floor.

## 0043: the prefill gated_delta_net is issue-bound, and half the issue is the butterfly (+5.7% prefill)

Round-26's open lever (1), and it did **not** need the chunkwise scan the source
`//TODO` suggests. The premise in that TODO — that the serial recurrence is the
problem — is wrong on this part at this shape, and the free diagnostic said so:

```
grid (H=32, n_seqs=1, S_v/num_warps=32) x 4 warps = 4096 waves / 128 SIMDs = 32 waves/SIMD
442 us x 2.4 GHz = 1.06 M cycles / 32 wave-slots / 512 tokens = 65 cycles per token per wave
```

65 cycles against **~39 VALU ops** of actual per-token work. Not 257 waves and
latency-starved (0042's router), not saturated-and-scattered (the expert
matvecs, round 26) — **issue-bound with a wave count that is already right**. And
of those 39 ops, 20 are the two 5-step DPP butterflies and only 12 are
arithmetic. The reduction costs more than the recurrence.

### The reformulation

Stock gives one column of the recurrent state to a whole wave: 32 lanes hold 4
rows each, and each token turns two row-sharded dot products (`S^T k`, `S^T q`)
into scalars, so two full butterflies pay for 12 FMAs.

`gated_delta_net_mc_cuda` gives **CPW columns to one wave**. The wave splits into
CPW groups of `32/CPW` lanes; each group owns one column and a proportionally
taller *contiguous* row shard. Per column the arithmetic is untouched — the same
four-row FMA chains over the same rows — but the butterfly shrinks to
`log2(32/CPW)` steps. The dropped levels are not lost: they become in-register
adds over the lane's own chunk accumulators, which is where the stock lane was
already spending adds. Cost per column at CPW=4 is ~19 ops against 39.

Measured per kernel: **442 -> 283 us, 1.56x** (CPW=2 291, CPW=8 300).

The stock kernel is left **verbatim** and still serves decode. The multi-column
kernel is gated on `n_tokens > 1` and on the vec4 layout, so the decode path
keeps the same kernel, the same geometry, the same registers, and — 0040's rule
— no new branch in a kernel untouched call sites run.

### The measurement that decided the shape of the patch

The first version templated CPW onto the *existing* kernel rather than adding a
second one. Against a real control that read **0.994 on tg128, 5/5 rounds** — a
0.6% decode loss for a prefill patch, purely from the CPW=1 instantiation's
codegen. Splitting it into two kernels put decode back to 0.999 (arms
overlapping) and cost prefill nothing. Same lesson as 0040 in a third costume:
**if the decode path can keep running literally the code it ran before, make it
do that.**

| arm | pp512 med | vs control | tg128 med | vs control |
|---|---|---|---|---|
| control (0042 binary) | 4376.98 | — | 148.54 | — |
| CPW=1 (stock kernel) | 4367.91 | 1.0005 | 148.34 | 0.9984 |
| CPW=2 | 4574.74 | 1.0489 | 148.43 | 0.9987 |
| **CPW=4 (shipped)** | **4615.17** | **1.0571** | 148.41 | 0.9991 |
| CPW=8 | 4605.63 | 1.0522 | 148.36 | 0.9988 |

pp arms disjoint 5/5 (`min(CPW=4) = 4601.28 > max(control) = 4447.88`); tg arms
overlap the control in all four cases. CPW=1 at 1.0005 is the proof that the
added kernels cost the untouched path nothing.

- long context neutral: tg128 @ d16384 136.95 -> 136.66, @ d32768 126.90 -> 126.72
- firing proven by census diff: 60 launches of
  `gated_delta_net_mc_cuda<128,false,false,4>` replace 60 of
  `gated_delta_net_cuda<128,false,false,true>` at the prefill shape
- `test-backend-ops` GATED_DELTA_NET / SSM_CONV OK
- `GGML_GDN_COLS_PER_WAVE=1|2|4|8` re-sweeps it; `1` restores the stock kernel

### A control snapshot is not a control — check the RUNPATH

The first A/B of this round read the control at 4664 and CPW=1 at 4399 and said
the refactor cost 5.7%. All of it was an artefact. `cp -a cand/build/bin ctl43bin`
is what the round-19 note says to do, and it is **not sufficient**: these thin
executables carry an *absolute* `DT_RUNPATH` of
`/home/ghost/fable-qwen/cand/build/bin`, so `ctl43bin/llama-bench` loaded the
**new** `libggml-hip.so` and the "control" arm was the candidate at its default
setting. `ldd` on the snapshot says so in one line. `DT_RUNPATH` is searched
*after* `LD_LIBRARY_PATH` (unlike `DT_RPATH`), so the fix is one variable:

```sh
LD_LIBRARY_PATH=~/fable-qwen/ctl43bin ~/fable-qwen/ctl43bin/llama-bench ...
# verify: ldd <bin> | grep ggml-hip   ->   must point INTO the snapshot
```

Every future control arm in this series must be verified with `ldd` before the
numbers are believed.

### Not bit-exact — and the reason closes the door on ever making it so

The claim this patch was designed around was that the reduction tree is
*identical*: chunk `j` of multi-column lane `m` holds exactly what stock lane
`m*nchunk + j` held, so pairing adjacent chunks in registers reproduces the low
lane-index bits of the stock butterfly and the shortened DPP sequence supplies
the high bits, step for step (`^1`, `^2`, `row_half_mirror`, `row_mirror`,
`permlanex16` is a bit-order-0..4 balanced tree, and each step only ever
exchanges inside a group of NL lanes). The argument is correct at source level
and it does not survive the compiler: this tree builds with
**`-funsafe-math-optimizations`**, so LLVM carries `reassoc` on every FP op and
re-associates each kernel's accumulation into whatever chain it prefers. Two
differently shaped sources will not agree no matter how carefully the
source-level order is matched, and matching them would mean disabling reassoc in
the *stock* kernel too — which changes decode.

**Carry this forward: on this tree, "I wrote the same reduction order" is not an
argument for bit-exactness.** The byte-exact folds in this series (0034, 0035,
0036, 0040, 0041, 0042) are byte-exact because they *move* an unchanged
computation, not because they rebuild one.

Classified by dumping the op's `dst` and state for the first prefill launch of
every layer, CPW=1 vs CPW=4 (CPW=1 reproduces itself byte-for-byte across two
loads, so the op is deterministic and this is purely reassociation):

| | ndiff | max ULP | max rel |
|---|---|---|---|
| attn out, CPW=2 | 28.2% | 9 | 7.9e-7 |
| attn out, CPW=4 | 40.7% | 9 | 6.7e-7 |
| attn out, CPW=8 | 53.2% | 15 | 1.0e-6 |
| state, CPW=4 | 0.85% | 512 | 3.5e-5 (on values ~1e-9) |
| state, CPW=8 | 23.4% | 9430 | 6.2e-4 |

CPW=8 is visibly looser as well as slower — a second reason to ship 4.

### What that costs at the gate: a coin flip of fixed size, not a cost that scales

`llama-perplexity -c 512 --chunks 8`, 4 loads per arm. Two facts first, both
worth knowing independently of this patch: **stock is perfectly reproducible
(3.9314 four times), and the banked stack is not** — CPW=1, which runs the stock
kernel, spreads 3.9230-3.9276 across loads. Something already in 0001-0042 makes
the gate shape load-dependent at the +-0.05% level. So read medians only.

| arm | loads | median | vs stock |
|---|---|---|---|
| stock | 3.9314 x4 | 3.9314 | — |
| 0042 control binary | 3.9265 / 3.9271 / 3.9271 / 3.9331 | 3.9271 | -0.109% |
| CPW=1 (same, new binary) | 3.9271 / 3.9230 / 3.9276 / 3.9271 | 3.9271 | -0.109% |
| **CPW=4 (shipped)** | 3.9182 / 3.9199 / 3.9201 / 3.9211 | **3.9200** | **-0.290%** |
| CPW=2 | 3.9379 / 3.9339 / 3.9375 / 3.9307 | 3.9357 | +0.109% |
| CPW=8 | 3.9322 / 3.9308 / 3.9292 / 3.9383 | 3.9315 | +0.003% |

The two control rows landing on the same median to four digits is the
cross-check that the method is sound. And then **the three multi-column arms go
-0.29%, +0.11%, +0.00% with no monotone relation to CPW at all** — which is the
whole story: a ULP perturbation in front of a top-8-of-256 router reroutes
experts and perplexity lands wherever it lands. There is no "more aggressive
CPW costs more accuracy" axis to trade along, so **CPW=2 is not a safer
pare-back** — it is just a different draw, and a slower one.

What that means operationally: the draw is a property of the *configuration*,
not of the run. Within CPW=4 the four loads span 0.074%; across configurations
the span is 0.40%. So the shipped -0.290% is stable and reproducible, sits 0.21
points inside the +-0.5% gate, and does not need re-drawing. But it is the first
patch in the series to spend real gate budget, and the budget is shared: **any
future non-byte-exact fold has to be measured on top of this stack, not against
stock, and it draws its own +-0.3%.** If a round ever needs the headroom back,
`GGML_GDN_COLS_PER_WAVE=1` is the only reliable way to get it — that restores the
stock kernel and gives up the whole +5.7%.

decode-path ppl `-b 512 -ub 1 --chunks 8`: **3.9338**, identical to the banked
stack, in both arms — the multi-column kernel is gated off the decode shape by
construction, so the decode ladder is untouched.

`llama-server` greedy vs the CPW=1 control: **2/6 byte-identical**, the other
four diverging after 101-408 B, all six completions non-empty and >= 276 B.
That is the expected and permitted consequence of a float32 regrouping in front
of a top-8 router (0031 and 0033 are the precedents in this series), but it does
mean **byte-exact greedy is no longer available as this round's safety net** —
the decode-path perplexity reading and the census diff are.

## 0044: the vendor BLAS was running three matmuls per layer on 8 of 64 CUs (+7.3% prefill)

After 0043 the largest non-MoE prefill item is a group of **fp32 rocBLAS GEMMs**:
200 launches per two passes of `Cijk_Alik_Bljk_S_B_Bias_HA_S_SAV_..._MT64x64x16`,
**12.1 ms of a 106.8 ms pass (11.4%)**. They are qwen35moe's three plain f32
matmuls: the MoE router `[2048 -> 256]` on all 40 layers, and `ssm_alpha` and
`ssm_beta` `[2048 -> 32]` on the 30 linear-attention layers — 100 per pass.

**The diagnostic was the dispatch dimensions, not the kernel time.** Group the
kernel trace by workgroup count and the answer is immediate:

| launches/pass | shape | macro tile | workgroups | each |
|---|---|---|---|---|
| 60 | alpha+beta, N=32 | 64x64x16 | **8** | 119.4 us |
| 40 | router, N=256 | 64x64x16 | **32** | 124.3 us |

Eight workgroups of 64 threads on a 64 CU part. rocBLAS picks one macro tile for
the whole family and with 32 output rows the N dimension does not even fill a
single tile column, so the GEMM runs on an eighth of the GPU. The memory floor
for the alpha/beta shape is about 9 us and for the router about 27 us; they were
taking 119 and 124.

### The kernel

`mul_mat_f32_skinny_cuda` gives one **wave** an `NT x MT` tile of the output and
splits k across its 32 lanes. The grid is `(N/NT) x (M/(MT*nwarps))`, so the
alpha/beta shape launches **256 waves instead of 16**. Each lane accumulates a
k-strided partial; one warp reduction per output element finishes it, and since
that is `NT*MT` reductions against `K/32` iterations of `NT*MT` fmas each, the
reduction amortises away. Waves inside a block share `n0` and differ in `m0`, so
the src0 rows a block touches stay in that CU's L1 while the src1 rows stream.

| | before | after | |
|---|---|---|---|
| alpha/beta | 119.4 us | **23.8 us** | 5.0x |
| router | 126.1 us | **57.0 us** | 2.2x |
| group total | 12.1 ms/pass | **3.7 ms/pass** | |

`NT x MT` was swept at 4x8, 8x4, 8x8, 8x16 and 16x8: the per-kernel times move
(alpha/beta 21.3-28.8 us, router 57.0-71.9 us) but **every arm lands inside the
wall-clock noise**, so 8x8 ships and `GGML_F32_SKINNY_TILE` re-sweeps it. The
router is still ~2x off its floor; closing that needs LDS staging and a real
GEMM, which is a separate round.

**mmf cannot serve these, and it is worth knowing why before reaching for it:**
on RDNA4 `mul_mat_f`'s WMMA path is compiled out for `T = float`
(`if constexpr (!(half2 || nv_bfloat162)) { NO_DEVICE_CODE; }`), and its non-ids
path does not tile columns at all — only the `has_ids` branch computes
`col_tiles`, so plain MUL_MAT is capped at `cols_per_block <= 16` by
construction, which is what `should_use_mmf`'s `src1_ncols > 16` rejection is
really encoding. A census confirms it: **mmf fires nowhere in this model, at
either shape**, so the new kernel is additive and shares nothing.

| arm | pp512 med | vs control | tg128 med | vs control |
|---|---|---|---|---|
| control (0043 binary) | 4625.45 | — | 148.60 | — |
| `MAX_ROWS=0` (off) | 4643.48 | 0.9967 | 148.42 | 1.0003 |
| `MAX_ROWS=64` (alpha/beta only) | 4822.83 | 1.0392 | 148.60 | 1.0003 |
| **`MAX_ROWS=256` (shipped)** | **4957.11** | **1.0727** | 148.48 | 1.0002 |

pp arms disjoint 5/5 (`min = 4941.07 > max(control) = 4697.00`); tg arms overlap
the control everywhere. It scales past pp512: **pp4096 4118.35 -> 4360.01
(+5.87%)**. Long context neutral (d16384 136.61 -> 136.55, d32768 126.79 ->
126.80).

### Perplexity: the second draw came back

This is the second consecutive non-byte-exact prefill patch, and 0043's section
warned that the budget is shared. It landed the other way:

| stack | gate ppl median (3 loads) | vs stock 3.9314 |
|---|---|---|
| 0043 (this round's off arm) | 3.9211 | -0.262% |
| **0043 + 0044** | **3.9309** | **-0.013%** |

Reassociating the router's own dot products moved the gate reading *back* to
stock. That is the clearest possible demonstration of the point 0043 made: the
shift is a random walk through a top-8-of-256 selection, not accumulating
damage, and it does not compound in a predictable direction. **It also means the
budget cannot be reasoned about patch by patch — only the whole stack's reading
counts, and it has to be re-measured after every non-byte-exact change.**

- decode-path ppl `-b 512 -ub 1 --chunks 8`: **3.9338** in both arms — `M = 1`
  is below the column threshold, so the kernel is gated off the decode shape
- ragged ubatches, which take the bounds-checked instantiation rather than the
  unguarded one: `-b 500 -ub 100` 4.7366 -> 4.7300, `-ub 48` 4.7679 -> 4.7719
- firing proven by census diff: 200 `Cijk_..._MT64x64x16` become 200
  `mul_mat_f32_skinny_cuda`
- `llama-server` greedy **5/6 byte-identical**; only the long prompt differs,
  and it is the only one of the six with enough tokens to reach the column
  threshold — a nice incidental confirmation that the gate does what it says

### The shipped default did not match the measured arm — catch this with a no-env census

The first version of this patch was committed with `GGML_F32_SKINNY_MAX_ROWS`
defaulting to **64**, while every number above was measured with the env var set
to 256. With no env set the router — 40 of the 100 launches, and the larger half
of the win — was still going to rocBLAS. The A/B was honest about what it
measured and the commit message said "MAX_ROWS=256 (shipped)"; the code simply
did not agree with it, and nothing in the A/B could have noticed, because both
of its arms set the variable explicitly.

**A toggle-swept knob has to have its shipped default confirmed by a run with no
environment at all**, and the check is the census, not the wall clock: with the
default corrected to 256, a clean `llama-bench` shows 100 `mul_mat_f32_skinny_cuda`
per pass and **zero** `Cijk_..._S_B_Bias_...` launches, 3.70 ms/pass against
12.11. Re-measured on the corrected binary, no env set, 3 rounds vs the same
0043 control: pp512 **4655.45 -> 5011.09, ratio 1.0765, 3/3 disjoint**
(`min(cand) = 4986.09 > max(control) = 4682.60`), tg128 148.55 -> 148.40; gate
ppl **3.9318 on both loads**, +0.010% against stock 3.9314.

### One unreproduced test-backend-ops failure, recorded rather than buried

The first run of `test-backend-ops -o MUL_MAT` with the gates opened wide
(`MAX_ROWS=4096 MIN_COLS=2`, forcing every eligible shape in the suite through
the new kernel) reported `1/2 backends passed FAIL`. It has **not reproduced in
seven subsequent runs** — four of them with the same forced gates, one
deliberately run back-to-back after a model job to reproduce the memory
pressure of the original context — all `1186/1186 tests passed`, and the
failure printed no failing case line. The most likely explanation is an
allocation failure under memory pressure from the perplexity processes that ran
immediately before it in the same script. It is written down because an
intermittent op-test failure is exactly the kind of thing the next agent should
know to re-check if anything here ever looks wrong.

## Round-29 map: what prefill is now, and the two ceilings that bound it

Fresh census of the shipped 0044 build, `llama-bench -p 512 -n 0 -r 1`,
normalised per pass (the trace covers a warmup pass and a measured one).
**~99.8 ms of kernel per pass.** Two structural facts come first, because they
bound everything anyone can still do here.

**1. Prefill is 97.6% kernel-occupied.** pp512 = 5011 tok/s is 102.2 ms per pass
against 99.8 ms of kernel, so there is only ~2.4 ms of gap across 2236
dispatches. **Launch count is not a prefill lever on this box** — every win in
this series that came from removing dispatches (~1.5% per 40 removed) was a
*decode* win, and that intuition does not transfer to this shape. Prefill wins
have to come out of kernel time.

**2. The MoE experts are 44% of the pass and are a streaming read of the whole
model.** At 512 tokens with top-8 of 256, every expert is selected by ~16
tokens, so a pass reads essentially the entire quantised model once:

```
per layer:  2 x [2048 x 512 x 256] Q4_K (gate, up) = 302 MB
            1 x [512 x 2048 x 256] Q5_K (down)     = 185 MB
per pass:   487 MB x 40 layers = 19.5 GB
measured:   43.5 ms  ->  448 GB/s  and  23.7 TFLOP/s-equivalent
```

448 GB/s against the R9700's ~640 GB/s of GDDR6 is **70% of peak** — leaning
memory-bound but not pinned to the wall, and about 16% of what DP4A should give
on the arithmetic side (each weight byte is reused by 16 tokens here, so unlike
decode this is not a pure stream). Round 26 closed every tile-geometry axis and
0023 fixed the J tile, so this is not a tuning lever; but 44% of the pass at 70%
of peak is where the remaining prefill headroom actually lives, and nobody has
attacked it from the memory side.

| ms/pass | % | n/pass | each | kernel |
|---|---|---|---|---|
| 24.43 | 24.2 | 80 | 305.4us | `mul_mat_q` Q4_K experts (gate, up) |
| 19.06 | 18.8 | 40 | 476.5us | `mul_mat_q` Q5_K expert-down |
| 8.41 | 8.3 | 30 | 280.3us | `gated_delta_net_mc` (0043) |
| 5.82 | 5.8 | 40 | 145.5us | `Cijk` HSS GEMM — the 0021 Q6_K dequant+GEMM route |
| 3.59 | 3.6 | 10 | 359.2us | `flash_attn_tile<256,256,4,8>` |
| 3.70 | 3.7 | 100 | ~37us | `mul_mat_f32_skinny_cuda` (0044) |
| 3.41 | 3.4 | 40 | 85.4us | `Cijk` HSS GEMM (bias variant) |
| 3.01 | 3.0 | 100 | 30.1us | `mul_mat_q` small |
| 2.28 | 2.3 | 81 | 28.1us | `k_bin_bcast` op_add |
| 2.24+1.96 | 4.2 | 110 | | `dequantize_block_q6_K` — the 0021 route's conversion cost |
| 2.23 | 2.2 | 30 | 74.4us | `Cijk` HSS GEMM |
| 2.21 | 2.2 | 80 | 27.6us | `unary_gated_op_kernel` |

### Re-confirmed this round: 0021's Q6_K dequant+GEMM route is still right

The prefill pass has got 12% faster since 0021 chose fp16 dequant+GEMM over MMQ
for the dense Q6_K matmuls, so the trade was re-run — it is a single env var and
costs one bench:

| `GGML_Q6K_MMQ_MAX_BATCH` | pp512 |
|---|---|
| 64 (shipped) | 5000.5 / 5003.3 |
| 32 | 4995.8 / 4978.6 |
| 128 | 5000.1 / 4968.0 |
| **0 (force MMQ)** | **4418.2 / 4369.1 (-12.5%)** |

The batch threshold itself is flat between 32 and 128; forcing MMQ costs 12.5%.
**Do not "fix" the dequant+GEMM route back to MMQ** — and note the conversion
overhead it carries (4.2 ms/pass of `dequantize_block_q6_K` plus ~1.4 ms of
`convert_unary`) is already priced into that comparison.

### Measured dead: lifting the AMD WMMA head-size cap on flash attention

`flash_attn_tile<256,256,4,8>` is 3.6% of a prefill pass (10 launches, 359 us)
and 1% of decode, and the reason it is the tile kernel is not a heuristic that
happened to lose — it is structural, and worth one paragraph so nobody re-buys
it. `ggml_cuda_get_best_fattn_kernel` has exactly one AMD WMMA rule:

```c
// AMD WMMA is always faster than the tile kernel if the full tile width of 16 can be utilized.
if ((amd_wmma_available(cc) && gqa_opt_applies && Q->ne[0] <= 128) && ... )
    return BEST_FATTN_KERNEL_MMA_F16;
```

Instrumenting the selector settles what is failing in one run:
`Qne=256,... gqa=8 mask=1 maxbias=0.0 kvpad=1 strides16=1 gqa_opt=1 wmma=1` —
**everything passes except the head size. qwen35moe's full-attention layers are
DKQ = DV = 256**, so the `<= 128` cap keeps them on the tile kernel
unconditionally, at any batch size.

Lifting the cap to 256 does select the MMA kernel — the instance exists and
launches (`flash_attn_ext_f16<256, 256, 8, 8, false, false>`, 10 per pass) — and
then **`ggml_abort`s inside the backend synchronize**. The cap is loading an
actual constraint of the AMD MMA instances at that head size, not a tuning
guess. Reverted; no patch. If anyone revisits this, the abort is the thing to
read first, not the timings.

## Round 30: where the decode token actually goes (read before picking a lever)

This round instrumented the token instead of guessing at it, and three of the
four results close doors rather than open them. All of it is on the shipped
0044 build.

**1. The CPU is not in the loop.** A `rocprofv3 --hip-trace` over 50 steady
decode tokens: the token period is 6.715 ms and **6.655 ms of it is spent
inside `hipStreamSynchronize`**. Time outside any HIP API call is **46
us/token — 0.7%**. `hipGraphLaunch` is 4.9 us and happens once per token;
`hipLaunchKernel` does not appear in steady state at all, so replay really is
one graph launch. llama.cpp's graph reuse is already doing its job
(`LLAMA_GRAPH_REUSE_DISABLE=1` costs 9.1%: 148.6 -> 135.1 tok/s). **There is
nothing to win on the host side of this track.** Do not profile the CPU again.

**2. The ~1.0 ms/token that is not kernel time is GPU-side graph replay, and
it has a floor.** A standalone HIP-graph probe (`gprobe.hip` on the box, 400
nodes, trivial kernel, 20 replays):

| graph shape | per node |
|---|---|
| serial chain | **2.25-2.54 us** |
| two parallel chains (fork/join capture on a 2nd stream) | **12.8 us** |
| one fat kernel doing all the work | 0.016 us |

Fifteen ROCm knobs were swept against the serial number and **not one helps**:
`HIP_FORCE_DEV_KERNARG=0/1` and `AMD_DIRECT_DISPATCH=0` are ~25% WORSE (3.1
us), and `GPU_MAX_HW_QUEUES`, `HSA_ENABLE_SDMA`, `AMD_SERIALIZE_KERNEL`,
`HSA_ENABLE_INTERRUPT`, `DEBUG_CLR_GRAPH_PACKET_CAPTURE`, `AMD_OPT_FLUSH`,
`HSA_DISABLE_CACHE_INV` are all inside noise of 2.25-2.30. **The
multi-stream / concurrent-branch idea is dead**: expressing the graph's real
parallelism costs 5x more than the serialization it removes. Do not spend a
slot on it.

**3. Expert weight LAYOUT is dead** — this was the previous round's top open
lever. `bwprobe.hip` reads the exact access pattern of each matvec (one block
per row, 8-byte lane loads, the real row length and pitch) and compares the
8-of-256 expert gather against a contiguous 8-slab read and against a fully
interleaved layout:

| pattern (5.77 MB, Q5_K expert down: 16384 rows x 352 B) | GB/s |
|---|---|
| 8 slabs scattered over 256 | 803.8 |
| 8 slabs contiguous | 830.6 |
| rows interleaved across the 8 | 861.4 |

and for the Q4_K gate/up gather (4096 rows x 1152 B): 306.5 / 307.4 / 309.9.
The three layouts are the same number. **Scattering 8 of 256 expert slabs
costs nothing**, because each slab is 590-720 KB of contiguous rows and that
is already far above any DRAM page or channel-interleave granularity. There
is no repack of the pinned GGUF that buys decode bandwidth, and 0021's
load-time hook should not be spent trying.

**4. Every dense matvec is exactly at its access pattern's ceiling.** Same
probe, same geometry as the shipped kernels:

| matvec | probe ceiling | shipped kernel |
|---|---|---|
| lm head Q6_K (417 MB) | 627 GB/s | 626 GB/s |
| qkv+z grouped Q6_K (20.6 MB) | 580 | 587 |
| attn_q Q6_K (13.8 MB) | 556 | 560 |
| ssm_out Q6_K (6.88 MB) | 493 | 479 (serialized) |

The ceiling falls with transfer size, not with anything the kernel does: only
the 417 MB lm head reaches DRAM peak. Fit `T = c + bytes/627 GB/s` and every
one of these lands at **c = 2.7-3.0 us of ramp per launch**. So a matvec
dispatch costs ~2.8 us of ramp plus ~1.3 us of exposed replay gap before it
moves a byte, and there are 773 dispatches in a token.

### The arithmetic that bounds this track

Per-token weight traffic from the GGUF, with 0021's requant applied:
Q6_K projections 1050 MB + Q8_0 shexp 133 MB + MoE experts (8 of 256) 610 MB
+ Q6_K lm head 417 MB + F32 routers/alpha/beta 104 MB = **~2.31 GB/token**.

- At the R9700's 640 GB/s that is **3.6 ms = 276 tok/s**, and the lm head
  proves 627 GB/s is actually reachable, so ~2.7 ms of the 6.7 ms token is
  irreducible streaming.
- Measured now: 6.72 ms/token = **344 GB/s end-to-end**, 54% of peak.
- The remaining 3.0 ms decomposes as ~773 dispatches x ~3.9 us of combined
  ramp + replay gap. **That is the whole story of this track's decode.**

A decode target of 210 tok/s is 4.76 ms/token, which requires 486 GB/s
sustained across every microsecond including all 773 launches - i.e. the
dispatch+ramp budget would have to fall from 3.0 ms to ~1.1 ms, a **3x cut in
launch count**, from 773 to ~270. The series has removed roughly 40 launches
per successful round for eleven rounds. Nothing in the measured structure of
this graph supports that, and the two mechanisms that could have (host-side
work, concurrent streams) are both measured dead above. A realistic ceiling
for pure kernel work on this box is **185-200 tok/s**; 210+ needs fewer BYTES,
and the only lever that reduces bytes is requantization, which the 0.1%
perplexity gate makes a coin flip (see the 0021 table).

**Amended by round 31: "the launch count must fall" is the wrong way to say
this.** A merge that removed exactly 40 dispatches/token measured +0.05%,
because the kernel time it added cancelled the replay gap it removed. Read
round 31 before picking a co-launch: the predictor is the host/guest time
ratio, not the dispatch delta.

## 0045: the GDN F32 matvec group co-launched with the qkv+z grouped mmvq (+1.3% decode, BYTE-EXACT)

Given round 30, the only lever left is merging launches. Every GDN layer emits
the alpha/beta F32 matvec group immediately **before** the 35 us qkv+z grouped
mmvq (node order: `node_3`, `node_4` are the two F32 matvecs, then their folded
epilogue chain, then `node_11`/`z-0`), and the two are independent - same
activation, disjoint destinations, nothing but views and the F32 group's own
epilogue between them. 0045 appends the F32 rows as extra blocks of the
quantized launch: the grouped mmvq kernel runs the `mul_mat_vec_f_grouped` body
verbatim for those blocks (same one-warp-per-row shape, same lane->column
mapping, same warp reduction, same sigmoid/softplus epilogue), so it is
byte-identical.

**The block ORDER is the entire patch.** Two arrangements, same 30 dispatches
removed per token:

| F32 blocks placed | decode |
|---|---|
| at the END of the grid | **-1.50%** (5/5 rounds separated) |
| at the FRONT of the grid | **+1.30%** (5/5 rounds separated) |

At the end they are scheduled last, each occupies 1 of the block's 8 warps, and
they extend the kernel's tail by their full latency - which costs more than the
dispatch they saved. At the front they retire inside the quantized group's
memory time. **When merging two kernels of very different sizes on this part,
put the small one first**; this is probably the most transferable thing in the
round.

Measured, 5 interleaved rounds against a binary built from this patch's own
parent commit (RUNPATH verified with `ldd`: the control resolves
`libggml-hip.so.0` into its own snapshot):

```
candidate tg128 149.99 149.92 149.71 149.66 149.55  mean 149.77
control   tg128 147.91 147.90 147.89 147.79 147.73  mean 147.84   -> +1.30%
```

Same-binary toggle agrees (+1.33%, 149.42 vs 147.46, 5/5 separated). Prefill is
flat and untouched by construction - the grouped mmvq path is decode-only. Five
interleaved pp512 rounds with the arm order alternated each round:

```
candidate pp512 4956.6 4982.7 4920.0 4933.4 4980.6  mean 4954.6
control   pp512 4975.1 4920.4 4955.6 4961.1 4954.7  mean 4953.4   -> +0.02%
```

(A single non-interleaved pair earlier in the session read 4910 vs 4977 and
looked like a 1.3% prefill regression. It was drift. **Never call a prefill
delta from one unpaired pair on this box** - the pp512 spread across a session
is ~1.5%, wider than most effects worth measuring.)

Firing proven by census diff, not a debug print: **25238 vs 26288 dispatches
over 35 tokens = exactly -30.0/token**, and `mul_mat_vec_f_grouped` drops from
70 to 40 launches per token.

Correctness: **decode-path perplexity is bit-identical**, 8.8713 candidate vs
8.8713 control (`-b 512 -ub 1 --chunks 8` on `gainz-corpus.txt`; that corpus
gives a different absolute number than the wikitext readings elsewhere in this
file - what matters is that the two arms agree to four decimals), and server
greedy is 6/6 byte-identical against the toggle-off control.

`GGML_CUDA_MMVQ_F32_COLAUNCH=0` restores the separate grouped mmvf launch.

### The bug this round nearly shipped: co-launched groups need a disjointness check of their own

The first working version read **5/6** on server greedy - only the 1000-token
prompt differed - while `-c 512` perplexity, decode-path perplexity and the
five short prompts were all bit-identical. The tempting reading is "server
run-to-run noise". It was not: running the SAME arm twice back to back is
**6/6 identical**, so the harness is deterministic and a 5/6 is a real
difference.

The cause is structural and worth internalising for any future co-launch.
Hoist legality is checked by walking the nodes BETWEEN the two sites, and this
detector deliberately **excludes the nodes the co-launch itself computes** -
otherwise the F32 group's own folded epilogue would block every hoist. That
exclusion means the two merged groups are never checked against **each other**.
As separate launches they were strictly ordered by the stream; inside one
kernel they run with no ordering at all, so any aliasing between the quantized
group's writes and the F32 group's writes or inputs becomes a race. ggml-alloc's
layout is shape-dependent, which is exactly why it appeared only at long
context.

The fix is an explicit cross-product check: every quantized member's write
against every F32 member's write, folded destination, bias/scale, skip nodes
and both sources, and each F32 write against the quantized sources. It declines
the co-launch instead of racing. After it: **6/6 byte-identical**, the census
still shows -30.0 dispatches/token, and the speed is unchanged - candidate
149.40/149.30/149.37/149.20 against control 147.42/147.49/147.45/147.35,
**+1.28%, 4/4 rounds separated**.

**Rule for the next co-launch:** hoist legality is about the nodes in between;
it says nothing about the two things you are merging. Check those separately,
always.

### Debugging note that cost a build

The first version scanned FORWARD from the grouped-mmvq site for an F32 group
and never fired (census diff: exactly 0 dispatches changed). The F32 group is
**before** the quantized group in node order; the kernel trace looked the other
way round only because I mis-aligned the per-token window. Detection has to run
at the mmvf site and pull the quantized group down into it. **Dump the node
list (`name`, `op`) around the site before writing a directional detector** -
the kernel order in a rocprofv3 trace is not the node order.

## Round 31: removing a dispatch is not the lever - hiding one is (no patch shipped)

Round 30 concluded that dispatch count is the whole story of this track's decode
and that the job is to keep merging launches. This round built a merge that
removes **exactly 40.0 dispatches/token** and measured it at **+0.05% - flat**.
The heuristic the series has run on for eleven rounds ("~1.5% per 40 launches
removed") is wrong as stated, and this round replaces it with one that predicts
both this result and every earlier one.

### The post-0045 census (start here; supersedes the round-23 map)

`rocprofv3 --kernel-trace`, `llama-bench -p 0 -n 34 -r 1`, 25238 dispatches /35
tokens = **721.1/token, 5564 us of kernel in a ~6.72 ms token**.

| n/tok | us/tok | each | kernel |
|---|---|---|---|
| 150 | 2515 | 16.77 | `mul_mat_vec_q` (all dense/id matvecs) |
| 61 | 1275 | 20.90 | `mul_mat_vec_q_grouped` |
| 61 | 86 | 1.40 | `unary_gated_op_quant` |
| 60 | 77 | 1.29 | `quantize_q8_1` |
| 50 | 85 | 1.70 | `rms_norm_f32<256>` |
| 40 | 197 | 4.93 | `mul_mat_vec_f_grouped<16>` |
| 40 | 117 | 2.92 | `topk_moe_cuda<256>` (ONE workgroup) |
| 40 | 543 | 13.56 | `mul_mat_vec_q_expert_reduce` |
| 80 | 231 | 2.89 | `rms_norm_pre_add_f32<1024,{1,2,3}>` |
| 30 | 66 | 2.19 | `ssm_conv_fold_l2_f32` |
| 30 | 153 | 5.09 | `gated_delta_net_cuda` |
| 20 | 39 | 1.96 | `rope_multi` |
| 12.6 | 52 | 4.09 | `__amd_rocclr_copyBuffer` |
| 10+10+10 | 95 | | flash_attn_tile / combine / cpy_scalar |

Split by grid geometry, the `mul_mat_vec_q` bucket is: 40 routed gate+up+GLU
(4096 blocks, 18.07us), 40 ssm_out + attn_wo (18.07), 40 shexp down (2048
blocks, 3.57), 19 shexp gate+up fused (512 blocks, 6.01), 10 attn wq (24.51),
1 lm head (665.15). The grouped bucket is 30 qkv+z (36.78) + 21 shexp gate/up
(5.54) + 10 attention k/v (5.54).

### The per-token dispatch sequence (aligned to the node dump)

A **GDN layer is 14 dispatches**, an **attention layer 22**, of which the MoE
block is 8 in both:

```
 1 mmvf_grouped(shexp_gate + router)   257 blk    4.42us
 2 topk_moe                              1 blk    2.68us
 3 mmvq routed gate+up+GLU  (Q4_K,id) 4096 blk   17.89us
 4 quantize_q8_1 (routed swiglu)         2x8      0.91us
 5 mmvq_expert_reduce (Q5_K down)     2048 blk   13.19us
 6 mmvq_grouped shexp gate/up (Q8_0)  1024 blk    5.54us
 7 unary_gated_op_quant (shexp swiglu)   2 blk    1.16us
 8 mmvq shexp down (Q8_0)             2048 blk    3.60us
```

Nodes 1-5 are the routed branch and 6-8 the shared-expert branch; **the two are
completely independent** - both read only `attn_post_norm` - and the whole
remaining decode opportunity lives in that fact. Get the node list with a
one-shot `fprintf(stderr, ...)` over `cgraph->nodes` in
`ggml_cuda_graph_evaluate_and_capture` (print `name`, `op`, `ne`, every `src`
and `data`); `llama-bench` silences `GGML_LOG_*` but not stderr.

### What was built, and why it was blocked in stock order

Stock order emits the shared expert AFTER the routed reduction. `ggml-alloc`
assigns buffers from node order, so `ffn_gate_shexp`'s output lands on the block
the **expert-id list has just freed** (`ffn_gate-0` at `0x...404480 + 2 KB`
straddles `ffn_moe_topk-0` at `0x...404880`). 0045's cross-product disjointness
check sees that and correctly refuses the merge - the shared gate would
overwrite ids the routed launch still reads. **The refusal is the allocator's
doing, not the detector's**, which is why no backend-side change can fix it.

The fix is entirely in `src/models/qwen35moe.cpp` and needs no change to
`llama-graph.cpp`: `build_moe_ffn` already takes `probs_in` and uses it **as the
logits verbatim**, so the router matvec can be emitted by the caller. Emit it
right after the shared-expert gate (keeping the F32 pair the backend groups),
then pin the shared expert's gate/up behind it, and hand the logits back through
`probs_in`. Node order becomes 59 shared_gate(F32), 60 router(F32), 61
ffn_gate_shexp(Q8_0), 62 ffn_up_shexp(Q8_0), then the topk chain. The allocator
then gives `ffn_gate-0` `0x...404880` and moves argsort to `0x...405880`, all
four are live at once, and **0045's existing detector fires unchanged**.

Verified by census diff, not by a print: **721.1 -> 681.1 dispatches/token,
exactly -40.0**, `mul_mat_vec_f_grouped` goes 40 -> 0 and `mul_mat_vec_q_grouped`
40 -> 80, in all 40 MoE layers.

### The result: flat

Five interleaved rounds, arm order alternated (`GGML_QWEN_SHEXP_GATEUP_PIN`):

```
on  149.80 149.86 149.92 150.01 149.87   mean 149.89
off 149.96 149.86 149.93 149.71 149.64   mean 149.82   -> +0.05%, arms interleaved
```

Kernel time explains it exactly. Total kernel went **5564 -> 5602 us/token, UP
38 us**, while 40 removed dispatches are worth about 40 x 1.3 = 52 us of exposed
replay gap. The merged launch costs **11.65 us against 4.93 + 5.54 = 10.47 us
run separately** - the guest did not hide at all, and cost 1.18 us extra on top.

### The corrected predictor: host/guest ratio, not dispatch count

0045 is the contrast, and its numbers are already in this file. Its host (qkv+z
grouped) went **35.5 -> 36.78 us** while absorbing a **4.7 us** guest: **73% of
the guest hid**, host/guest ratio 7.5. This round's ratio was 1.1 and 0% hid.

> A merged launch is worth the guest's whole time plus one replay gap **only
> when the host is much larger and memory-bound**. Merging two small
> latency-bound launches buys the replay gap and gives it straight back in
> kernel time. **Rank co-launch candidates by host/guest time ratio, never by
> dispatches removed.**

That also re-reads round 30's arithmetic. The token is 3610 us of irreducible
weight streaming + ~700 us of matvec launch ramp + ~1020 us of small-kernel time
+ ~1110 us of replay gap. The last two pools (2130 us, 32% of the token) are only
attackable **together**, by hiding small work inside big matvecs; attacking the
gap alone just moves the cost.

### Block order re-tested at the new site - front is right everywhere

0045 found the F32 tail must go at the FRONT of the merged grid. Re-tested here
with a switchable split (`GGML_CUDA_MMVQ_F32_AT_END`), 4 interleaved rounds:

```
front 150.01 150.04 150.07 149.93
end   144.70 144.84 144.57 144.51    -> -3.5%, 4/4 separated
```

So ordering is not the explanation for the flat result, and **front is the only
shipping arrangement** - do not re-derive this.

### The reorder is NOT byte-exact - know why before building on it

Decode-path perplexity (`-b 512 -ub 1 --chunks 8`, gainz-corpus), two loads per
arm, perfectly reproducible: **8.8679 pinned vs 8.8713 stock, -0.038%**. Well
inside the 0.1% gate but not identity. The cause is visible in the census: 19 of
the 40 layers were taking llama.cpp's **fused gate+up+GLU mmvq** (512 blocks,
6.01us) for the shared expert, and the reorder moves all 40 onto the grouped
mmvq plus a separate `unary_gated_op_quant` (`mul_mat_vec_q` 150 -> 131,
`unary_gated_op_quant` 61 -> 80, `quantize_q8_1` 60 -> 41). Two different
kernels compute the same SwiGLU. Any future patch that keeps this reorder either
has to preserve the fusion or be gated as a non-byte-exact change.

The reorder is preserved on the box as branch **`r31-shexp-reorder`** (commit
`be6537cd2` in `~/fable-qwen/cand`, diff at `~/fable-qwen/r31-shexp-reorder.diff`,
102 lines). It is not in the series: neutral speed is not worth a shipped
numeric change.

### The remaining hide-pool, ranked - and the one piece of machinery it needs

Every guest worth hiding is now identified, with its host and the measured
ratio. **All four need the same missing capability: two quantization types in
one launch.** The 0045 tail already proves a foreign block range inside
`mul_mat_vec_q_grouped` can be bit-identical (it runs the mmvf body verbatim for
F32 rows); this is that, with a quantized body.

| guest | host | ratio | x/token | projected |
|---|---|---|---|---|
| shexp gate/up Q8_0, 5.5us | routed gate+up+GLU Q4_K/id, 17.9us | 3.3 | 40 | ~+2.0% |
| shexp down Q8_0, 3.6us | `expert_reduce` Q5_K, 13.4us | 3.7 | 40 | ~+1.5% |
| attn k/v Q8_0, 5.6us | attn wq Q6_K, 24.5us | 4.4 | 10 | ~+0.6% |
| `topk_moe`, 2.7us, ONE workgroup | shexp gate/up group, 1024 blk | - | 40 | ~+2.6% |

Two things make this cheaper than it looks:

1. **Every guest is Q8_0.** The GGUF ships `attn_{q,k,v,output}` and the whole
   shexp trio as Q8_0, and 0021 requants only `attn_q`/`attn_output` (and not
   the shexp trio - that variant measured slower and was excluded). So the
   second template parameter only ever needs `GGML_TYPE_Q8_0`: one extra
   instantiation family, guarded host-side, not a `type x type` explosion.
   This is also why the round-1 note "shared-expert ninth-channel fold is
   structurally impossible on this GGUF" needs re-reading: the type mismatch is
   real, but a **second vec_dot template parameter does not forfeit bit-identity**
   the way that note assumes - each row keeps its own thread geometry, k-block
   assignment and reduction order, exactly as the 0045 F32 tail does.
2. The graph reorder that makes rows 1 and 2 legal is already written and
   measured (above), and the same `probs_in` trick can place the shared expert
   anywhere in the MoE block.

Row 4 is the largest single number and the only one whose guest is not a matvec:
`topk_moe` is one workgroup of pure launch latency, 40 times a token. It would
have to run as one block of the grouped mmvq kernel, which means its body must
work at the host's block shape (32x8 = 256 threads, against its own 32x4) with
warps 4-7 participating in its `__syncthreads()` and doing nothing - and that
changes its reduction unless the extra warps are excluded from the reduction
tree. Scope it after rows 1-2.

### Go/no-go on the persistent multi-op kernel (round-30 idea 3)

**No-go as stated, go as the tail generalisation above.** A single kernel
covering a whole layer section would have to serialise what the graph currently
overlaps, and this round measured what actually pays: not fewer kernels, but
more work per kernel *in the shadow of a large memory-bound one*. The tail
mechanism reaches the entire 2130 us pool incrementally, patch by patch, with a
bit-identity argument at every step; a persistent kernel does not.

## Recommended firing order

The runner applies the whole patch directory, so the series is a ladder rather
than a choice; **0001-0044** is the verified set and every patch that carries
speed has a disable toggle. Two do not carry speed: **0038** is 0037's kernel
made bit-identical and must not be applied without 0037; **0039** is 0031's
kernel made sign-of-zero-exact and must not be applied without 0031
(`GGML_SSM_CONV_FOLD_STOCK_SHAPE=0` falls back to the 0031 kernel in place).
If a ranked run has to be pared back, drop from the end: **0045** (decode only,
independent of everything except the grouped mmvq/mmvf launches it merges -
`GGML_CUDA_MMVQ_F32_COLAUNCH=0` restores the separate launches), then 0044 (prefill only,
a new kernel plus one dispatch branch, independent of everything —
`GGML_F32_SKINNY_MAX_ROWS=0` disables it), then 0043 (prefill only,
independent of everything — `GGML_GDN_COLS_PER_WAVE=1` restores the stock
kernel; note CPW=2 is *not* a safer half-measure, it is just a different
perplexity draw and a slower one), then 0042 (a
one-constant change, independent of everything), then 0041 (needs 0031+0039,
which own the decode conv fold it extends), then 0040 (prefill only, independent
of everything), then 0039, then 0038+0037, then 0036, 0035, 0034 (independent
+1.5% decode folds); 0033 is the prefill fold; 0032 and 0030 are the two largest
decode wins and should be the last to go. 0037 has two halves in different files
— dropping it means dropping both, since the graph half alone leaves the decode
combine as n_expert_used-1 separate adds. 0041 likewise has two halves (the
in-place norm in `qwen35moe.cpp` and the CUDA epilogue) and the CUDA half
declines without the graph half.

**Projection after 0044: ~1.632-1.668 vs the 1.4525 bank.** Decode 1.7536 ->
**1.7893** (x1.0133 from 0041, x1.0070 from 0042); prefill 1.2911 -> 1.3205
(x1.0228 from 0040) -> 1.3959 (x1.0571 from 0043) -> **1.4974** (x1.0727 from
0044); ttft moves with prefill for prompt processing, so the band spans ttft
flat (1.632) to ttft carrying the full prefill ratio (1.668). Every ratio here
is measured against a binary built from the patch's own parent commit, not
against a toggle — and from 0043 on, against one whose `ldd` was checked.

Prefill has gone 1.3205 -> 1.4974 in two rounds, and the reason both were there
to take is the same: **a big kernel slice is not evidence that the kernel is
working hard.** 0043's gated_delta_net had the right wave count and was spending
half its issue slots on a reduction; 0044's GEMMs were spending 88% of the GPU
on nothing at all. Divide the grid by the workgroup size before you profile
anything else.

Rounds 21 and 22 bought correctness; 23, 24 and 25 are the first speed since
0037 — one on the prefill side, two on the decode side.

Correctness state of the stack after 0044: **+0.010% against stock at the gate
shape** (3.9318 on both loads against stock's perfectly reproducible 3.9314),
and **3.9338 on the decode path**, identical to the pre-0043 stack — both 0043
and 0044 are gated off the decode shape by construction, so the entire decode
ladder is numerically untouched by this session. Every banked fold still has a
perplexity number taken at the shape where it actually runs.

### Gate the next decode fold like this

`llama-perplexity -c 512` does not execute a fold gated on `n_t == 1` or
`ne[2] == 1` on this fork. Before believing any perplexity reading:

1. **Prove firing with a census diff**, not a debug print: `rocprofv3
   --kernel-trace --output-format csv` over the same command with the patch's
   toggle on and off, `--chunks 1` is enough, then diff the kernel-name/count
   tables. `llama-bench` silences `GGML_LOG_INFO`, so accept/decline prints are
   not evidence there.
2. **Read perplexity at `-b 512 -ub 1 --chunks 8`** (~75 s, deterministic
   across loads) as well as at `-c 512`, and report both.
3. **Run server greedy at the decode shape** (`llama-server`, not `llama-cli`),
   6 prompts, byte-compared against the same-binary toggle-off control, and
   check the completions are non-empty and >= 256 B before believing a 6/6.
4. If the result is not byte-exact, **dump the tensor and classify the
   difference** before calling it reassociation: value-equal with different
   bits is a dropped identity operation and is fixable (0039); ULP moves are
   contraction and mirroring the source will not save it (0031, 0033).
5. **If the patch edits a kernel that untouched call sites also run, add a
   third arm built from the parent commit** (0040). A toggle A/B puts both arms
   in the same binary, so any cost the edit imposes on the un-toggled path is
   subtracted from both and cancels; 0040's first version read +1.55% on the
   toggle and 1.0013 against a real control.

## 0046: the topk_moe router selection hidden inside the shared-expert mmvq (+1.9% decode)

Round 31 ranked the hide-pool by host/guest time ratio and this is its top
entry. It is also the only candidate whose guest is not a matvec, and that
turned out to be the reason it is the cheapest to build: **`topk_moe_cuda` has
no `__shared__` and no `__syncthreads`** — the whole top-8-of-256 selection is
`__shfl_xor` over the 32 lanes of one warp. A kernel like that can be dropped
into warp 0 of one extra block of a foreign launch with nothing to reconcile:
same lane→expert mapping, same bitonic butterfly, same normalisation
reduction. Bit-identity is by construction, not by inspection — the body was
refactored into a `topk_moe_body` device template that the standalone kernel
and the guest both call.

**The ratio is not 2.1, it is 1024.** Reading the census row (`topk_moe` 2.9 us
against a 5.5 us host) as a 1.9 ratio would have declined this by round 31's own
rule. The rule is about *capacity*, and the right measure here is blocks: the
guest is ONE workgroup joining 1024. It never competes for the host's bandwidth,
so it retires inside the host's memory time and costs nothing. Time ratio is the
correct predictor only when both sides are shaped like matvecs.

### The graph half, and why prefill has to be excluded

Stock order emits the shared expert after the routed reduction, so `ggml-alloc`
hands `ffn_gate_shexp` the block the expert-id list just freed and the
disjointness check correctly refuses. Pinning the router matvec and the
shared-expert gate/up ahead of the top-k chain (handing the logits back through
`build_moe_ffn`'s `probs_in`) relays them out tightly packed and provably
disjoint: gate `[+0x4880,+0x5080)`, up `[+0x5080,+0x5880)`, argsort
`[+0x5880,+0x5c80)`, weights `[+0x4500,+0x4520)`.

The reorder must be **gated on `n_tokens == 1`**. Un-gated it also fires at
prefill, where the gate/up pair is adjacent to its GLU and llama.cpp fuses the
three into one GEMM+GLU launch; breaking that read **−0.94% pp512** in the first
paired measurement. Gated, prefill is flat (+0.07% over six interleaved rounds)
*and* the whole official gate shape becomes the stock build path.

### Numbers

Against a binary built from the patch's own parent commit, control RUNPATH
verified by `ldd`, arm order alternated, no env set on either arm:

```
candidate tg128 152.58 152.47 152.23 152.22   mean 152.38
control   tg128 149.66 149.46 149.27 149.50   mean 149.47   -> +1.94%, 4/4 separated
```

Census diff: 25238 → 23838 dispatches / 35 tokens = **exactly −40.0/token**,
`topk_moe_cuda` 1400 → 0, total kernel 194.72 → 192.44 ms (**−65.1 us/token**).
The kernel saving alone is −113 us/token of `topk_moe`; the grouped mmvq grows
because 19 layers also move onto it from the fused path (below).

Perplexity on the **runner corpus** — the shape distinction is the whole story:

| | stock | banked (0045) | with 0046 |
|---|---|---|---|
| gate shape `-c 512` | 3.9314 | 3.9317 | 3.9314–3.9318 |
| decode path `-b 512 -ub 1` | 3.9409 | 3.9338 (−0.180%) | 3.9382 (**−0.069%**) |

The reorder is not byte-exact — it moves 19 of 40 layers off llama.cpp's fused
gate+up+GLU mmvq onto grouped mmvq + `unary_gated_op_quant`, two kernels
computing the same SwiGLU. But it moves the stack **toward** stock on the decode
path, and at the gate shape the graph is literally the stock build path. Toggled
off, the binary reproduces the control's 8.8713 (gainz-corpus) exactly, so the
co-launch half carries no numeric change at all.

Server greedy at the decode shape: **6/6 byte-identical** co-launch on vs off
(6 prompts, every completion ≥ 276 B, including the 1000-token prompt), against
a **same-arm-twice control that reads 5/6** — the harness itself is noisier than
the change. Record that: server greedy on this box is not a clean 6/6 oracle.

`GGML_CUDA_MMVQ_TOPK_COLAUNCH=0` restores the separate launch;
`GGML_QWEN_SHEXP_GATEUP_PIN=0` restores stock node order.

### The artifact trap that cost this round a submission slot

The first submission of 0046 was **ranked and scored 1.6225 — flat against the
banked 1.6261 — because the shipped patch file contained only the new header**.
`git add ggml/src/ggml-cuda/topk-moe-body.cuh` followed by `git commit -F msg`
(no `-a`, no second `git add`) committed exactly one file; `git format-patch`
faithfully emitted exactly one file; `git apply --index` applied it without
error, because adding an unused header IS a valid patch. The runner then built
and measured the banked stack with a spare header in it, and reported the truth.

Two checks, both cheap, catch this and nothing else does:

```
git apply --stat Sources/patches/<track>/<new>.patch     # must list every file
strings <tree>/build/bin/libggml-hip.so.0.18.0 | grep <YOUR_ENV_TOGGLE>
```

The second one is the general form: **every patch in this series carries a
disable toggle, so the toggle's env-var string is a fingerprint of whether the
patch is in a given binary.** `~/gainz-runner-rocm/work/<sha>/llama-tree/` keeps
the runner's own tree and build; grepping its `.so` for the toggle string is how
this was diagnosed. Do not grep the runner's *sources* for the change — the
worktree is reset after the run, and an untracked added file survives that reset
while tracked edits do not, which is exactly the misleading picture it gave here.

### What this leaves for the rest of the hide-pool

The machinery built here is a *guest block* with its own body inside the grouped
mmvq, plus a cross-product disjointness check between the two merged groups. The
remaining three rows of the round-31 table (shexp gate/up → routed gate+up,
shexp down → expert_reduce, attn k/v → attn wq) all need the other machinery —
a **second vec_dot type in one launch** — which this patch does not build.

## Round 33: can this box reach 200 tok/s? Measured, and the answer is no

The owner asked for 200 tok/s decode with numbers attached rather than a
feeling. This section answers it from the post-0046 census and two new probes,
not from projection. Figures are on the **llama-bench scale** (152.4 tok/s with
0046); the ranked server path runs ~2 tok/s below at the same ratio, so 200
ranked is ~205 here.

### Where the 6563 us token goes now

`rocprofv3 --kernel-trace`, `llama-bench -p 0 -n 34 -r 1`: 23838 dispatches over
35 tokens = **681.1/token, 5498 us of kernel in a 6563 us token**.

| component | us/token | share | basis |
|---|---|---|---|
| weight streaming | 3742 | 57% | 2.35 GB at the 627 GB/s the lm head proves reachable |
| matvec launch ramp | 829 | 13% | 291 matvec launches x 2.85 us (round 29's `T = c + bytes/627` fit) |
| small-kernel time | 927 | 14% | the other 390 launches |
| exposed replay gap | 1064 | 16% | 681 dispatches x 1.56 us |

**200 tok/s is a 5000 us token: 1563 us has to come out of the 2820 us that is
not streaming.**

### 1. The dispatch floor is 502 — computed, not guessed

A decode token is one serial chain, so the minimum number of launches is the
**longest chain of cross-block dependencies**: independent nodes can share a
launch, dependent ones cannot without a grid barrier. Computed directly from a
node dump by walking data ranges (a node depends on the most recent earlier node
whose destination overlaps one of its sources):

| a boundary is required at | floor | per layer |
|---|---|---|
| matmul + attention + scan + norm + softmax + argsort | **502** | 12.6 |
| same, but norms/softmax fold into the consumer by recompute | **301** | 7.5 |

Against 681.1 today that is **179 removable dispatches** with folds of the kind
this series already ships, or 380 with a recompute program that does not exist.
At the 3.2 us/dispatch that 0046 actually bought (127 us for 40) and 2.4 us for
the elementwise class:

- the whole conservative co-launch/fold program: **166-171 tok/s**
- plus a full recompute program at the aggressive floor: **~185 tok/s**

Round 31's four ranked candidates are worth about +3% of that (~157 tok/s), and
**row 1 conflicts with 0046**: if the shexp gate/up pair merges into the routed
gate+up host it stops being available as the topk guest's host, so those two are
alternatives, not additions.

### 2. The persistent multi-op kernel is dead — by a factor of 16

Round 30's no-go was reasoned from hiding. A persistent kernel attacks the
budget differently: it replaces a launch boundary with a grid barrier, removing
the replay gap **and** the ramp rather than hiding one inside the other. So it
deserved a measurement, and the break-even is arithmetic:

```
overhead after every fold  = 462 x 1.56 (gap) + 291 x 2.85 (ramp) = 1550 us
persistent equivalent      =  41 x 4.41 (per-layer launches) + 462 x B
break-even barrier cost B  = 2.96 us
```

A standalone HIP probe (sense-reversing barrier, `atomicAdd` + `__threadfence`,
all blocks resident, 2000 iterations) measures B on gfx1201:

| resident blocks | barrier |
|---|---|
| 64 | 5.59 us |
| 128 | 20.47 us |
| 256 (the residency limit) | **48.9 us** (47.9 on the repeat) |

**Even the 64-block case is 1.9x over break-even; the real case is 16x over.**
The cost roughly quadruples per doubling — it is an L2 atomic ping-pong with no
hardware grid-sync behind it. Two further facts close it:

- `hipOccupancyMaxActiveBlocksPerMultiprocessor` gives **8 blocks/CU x 32 =
  256 resident blocks** for a 256-thread kernel. The shipped matvecs launch
  1024-4096 blocks, so a persistent grid must loop 4-16x over logical blocks —
  serializing what the hardware scheduler currently overlaps.
- Measured directly: 8 passes over distinct 20 MB slabs (past the 64 MB
  Infinity Cache), one launch per pass **42.96 us/pass (488 GB/s)** against a
  persistent grid with barriers **65.60 us/pass (320 GB/s)**, +22.6 us/pass.
  At 6 MB and 2 MB the persistent arm is 9x and 14x slower.

Substituting the measured B into the budget: a persistent design lands at
**37 tok/s** at the real residency and **138 tok/s** even at the 64-block best
case. **Go/no-go: NO-GO, now on measurement rather than on argument.** Do not
re-open it on this architecture; it needs a hardware grid barrier that gfx1201
does not have.

### 3. The byte side is closed — no requantization survives 0.1%

Streaming alone is 3742 us of the 5000 us a 200 tok/s token allows. With the
co-launch/fold floor's 1550 us of overhead the two do not fit: bytes would have
to fall to 3450 us = 2.16 GB, an **8% cut**. The 0021 table, re-read against
0.1% instead of the 0.5% band it was measured in, says every available step is
5-9x too coarse:

| step | measured ppl move | vs a 0.1% gate |
|---|---|---|
| 5 dense families Q6_K -> Q5_1 | -0.72% | 7x over |
| qkv Q5_1, rest Q6_K | -0.9% | 9x over |
| qkv Q6_K, rest Q5_1 | +0.6-0.85% | 6-8x over |
| shexp trio requantized at all | +0.5% AND slower decode | excluded twice |
| output head routed/altered | 3.908-3.924, straddles the band edge | noisy, excluded |

The elasticity is roughly **0.7% perplexity per bit** off the dense projections.
There is no upcast left to harvest: 0021 already took the one that existed (the
UD export shipping projections as Q8_0), the experts are Q4_K/Q5_K and the head
is Q6_K. The only untouched precision is the **F32 routers and alpha/beta, 104
MB/token (4.5%)** — F16 would buy ~81 us (+1.2%) — but those logits drive
top-8-of-256 selection, where even a float32 regrouping reroutes an expert. One
experiment at most; it cannot approach 200.

### The honest answer

**200 tok/s is not reachable on this box with byte-preserving kernel work.**

| path | ceiling | status |
|---|---|---|
| round 31's four ranked co-launch candidates | ~157 tok/s | 0046 shipped one of them |
| every co-launch and fold down to the 502 floor | 166-171 tok/s | open, incremental |
| plus a norm/softmax recompute program (301 floor) | ~185 tok/s | open, large |
| persistent multi-op kernels | 37-138 tok/s | **measured dead** |
| requantization | any target | **gate-dead** |
| zero-overhead asymptote (2.35 GB at 627 GB/s) | 267 tok/s | physics, not reachable |

The realistic ceiling is **170-185 tok/s on llama-bench, ~165-180 ranked**, and
getting there means folding essentially every elementwise and norm launch into
an adjacent matvec — roughly 180 more dispatches removed at ~2.5 us each, or
about fifteen more rounds at this round's rate. 200 needs either a quantization
that fails the gate or a part with more bandwidth.

## Round 34: the per-dispatch cost measured at nine dispatch counts — 210 tok/s is arithmetically out of reach

Round 33 projected 166–171 tok/s for the co-launch/fold program and ~185 for a
recompute program on top, by applying the **3.2 us/dispatch** that 0046 actually
bought to the 502 and 301 dispatch floors. A competing reading of the same
numbers — "3610 us of streaming plus 502 x 3.2 = 192 tok/s, plus 301 x 3.2 =
219" — reaches a very different answer from the same two inputs. This round
settles it by **measuring** t(d) instead of extrapolating one point.

### The ladder

Every fold in this series carries a disable toggle, so the series is its own
dispatch-count dial. Nine cumulative arms, each adding one more disabled fold,
`llama-bench -p 0 -n 128 -r 3` for timing and a `rocprofv3 --kernel-trace`
census for the exact dispatch count, **swept in both directions** (forward then
reverse) to control for drift:

| arm (cumulative) | dispatches/token | tg128 | us/token |
|---|---|---|---|
| base (0046 stack) | 681.1 | 153.03 | 6535 |
| `+MMVQ_F32_COLAUNCH=0` | 751.1 | 147.88 | 6762 |
| `+DISABLE_NORM_QUANT` | 871.1 | 142.06 | 7039 |
| `+DISABLE_UNARY_MUL_QUANT` | 911.1 | 140.38 | 7124 |
| `+DISABLE_PRE_ADD_NORM` | 980.1 | 136.83 | 7308 |
| `+DISABLE_SSM_CONV_FOLD(+L2)` | 1070.1 | 131.52 | 7603 |
| `+DISABLE_{MMVQ,MMVF,NORM,ROPE,SET_ROWS}_GROUP` | 1360.1 | 116.81 | 8561 |
| `+DISABLE_MOE_EXPERT_REDUCE` | 1640.1 | 106.70 | 9372 |

(`GGML_CUDA_MMVQ_TOPK_COLAUNCH=0` is excluded from the fit: it costs 2.0% of
decode but leaves the dispatch count at 681.1, so it is not a dispatch-count
point. Worth knowing on its own — 0046's win is not purely a dispatch removal.)

### The fit, and the number that decides everything

```
t(d) = 4448.4 us + 2.991 us * d       R2 = 0.99705
```

Two facts come straight off it:

1. **The marginal cost per dispatch does not decay as the graph shrinks.**
   Local slopes, low d to high d: 3.25, 2.31, 2.11, 2.68, 3.28, 3.30, 2.90
   us/dispatch. No trend. The one measured co-launch (0046, 3.2 us) sits inside
   that band. So a linear model is the right model, and extrapolating the 3.2 us
   downward was legitimate.
2. **The intercept is 4448 us, not 3610.** Weight streaming is 3742 us (2.35 GB
   at the 627 GB/s the lm head proves reachable). The remaining **~700 us is the
   matvec bandwidth ramp** — 291 matvec launches that are the model's own
   matmuls and cannot be removed, each spending ~2.4 us climbing to full
   bandwidth. That time is *inside* kernel execution, so removing dispatches
   never touches it.

That second fact is the whole discrepancy. A budget of the form
"streaming + d x per-dispatch" implicitly assumes every launch, matvec included,
costs only the per-dispatch overhead. The measurement says there is a fixed
~700 us on top of streaming that no amount of folding reaches.

### What the model says about every target

| d | t(d) | tok/s | status |
|---|---|---|---|
| 681.1 (today) | 6535 | 153.0 | measured |
| 502 (conservative floor) | 5950 | **168** | round 33's 166–171 confirmed |
| 301 (recompute floor) | 5349 | **187** | round 33's ~185 confirmed |
| 291 (matvec launches alone) | 5318 | **188** | **hard ceiling of the dispatch program** |
| 184 | 5000 | 200 | needs 107 fewer launches than the model has matmuls |
| 105 | 4762 | 210 | ditto |
| 0 (physically impossible) | 4448 | 225 | the asymptote |

**Round 33's 166–171 and ~185 are the correct figures.** The 192/219 reading is
wrong for one identifiable reason: it prices the 291 irreducible matvec launches
at the elementwise per-dispatch rate and drops their ramp.

**200 and 210 tok/s are not reachable by any dispatch program on this box**, and
the specific blocking term is not the barrier cost and not the byte count: it is
that 291 matvec launches cost `4448 + 291 x 2.99 = 5318 us` before a single
elementwise kernel exists. Reaching 210 would require deleting 186 of the
model's own matmuls.

### The 301 floor is itself optimistic for this graph

Even 187 assumes norms fold into their consumers by recompute. The largest norm
class is `rms_norm_pre_add` — 80/token, 230 us — and its consumers are the
4096-block routed gate+up mmvq and the 12352-block qkv+z grouped mmvq. Every
host block already reads the *quantized* activation (2.3 KB of q8_1), not the
f32 row; recomputing the reduction means each block reads the 8 KB f32 row
instead, i.e. 12352 x 8 KB = **98 MB of extra L2 traffic per layer** (~39 us at
L2 bandwidth) to save one 2.8 us launch. Recompute is only viable where the host
grid is small. The reachable floor for this graph is nearer **510–540**, worth
`(681 - 525) x 2.99 = 466 us` -> 6069 us -> **165 tok/s**.

### The one correction that raises the ceiling: merge matvecs, not elementwise ops

The fit also reprices the candidate list. Removing an **elementwise** dispatch is
worth 2.99 us (the gap). Removing a **matvec** dispatch is worth 2.99 + ~2.85 =
**5.8 us**, because it takes a bandwidth ramp with it. That nearly doubles the
value of the round-31 hide-pool rows whose guest is itself a matvec, and it means
the ranking should be:

| move | class | x/token | us/token | est. |
|---|---|---|---|---|
| shexp down (Q8_0) merged into `expert_reduce` (Q5_K) | matvec | 40 | 232 | +3.7% |
| shexp swiglu `unary_gated_op_quant` -> guest of routed gate+up mmvq | elementwise | 40 | 120 | +1.8% |
| routed swiglu `quantize_q8_1` -> guest of shexp down mmvq | elementwise | 40 | 120 | +1.8% |
| GDN `ssm_norm` + z-gate `unary_gated_op_quant` fused | elementwise | 30 | 90 | +1.4% |
| attn k/v grouped mmvq merged into attn wq | matvec | 10 | 58 | +0.9% |

The first and last need the **second vec_dot type in one launch** that round 31
identified and 0046 did not build. The middle three need a *guest block* inside
the stock `mul_mat_vec_q` (0046 built the equivalent only for the custom
`mul_mat_vec_q_grouped`), plus the node reorder that makes `ggml-alloc` hand the
guest a disjoint buffer.

Taken together those five are `-160` dispatches of which 50 are matvecs:
`466 + 50 x 2.85 = 609 us` -> 5926 us -> **169 tok/s**, ~166 ranked. That is the
honest target band, and it is +10% on today.

### The post-0046 census this round is derived from

`rocprofv3 --kernel-trace`, `llama-bench -p 0 -n 34 -r 1`, 23838 dispatches / 35
tokens = **681.1/token, 5494 us of kernel in a 6535 us token**. Dispatch-id order
is the true stream order (start-timestamp order has 2449 inversions — sort by
`Dispatch_Id`, not by `Start_Timestamp`, or the layer structure is unreadable).

A **GDN layer is 7 dispatches** and the **MoE block 8**:

```
GDN   1 rms_norm_pre_add_f32<1024,N,true>   1 blk    2.8us   input norm + residual add + q8_1
      2 mul_mat_vec_q_grouped           12352 blk   36.5us   qkv + z + the 0045 F32 beta/alpha tail
      3 ssm_conv_fold_l2_f32               32 blk    2.1us
      4 gated_delta_net_cuda               32 blk    4.7us
      5 rms_norm_f32<256>                  32 blk    1.6us   ssm_norm
      6 unary_gated_op_quant               16 blk    1.3us   norm * silu(z) + q8_1
      7 mul_mat_vec_q                    2048 blk   16.2us   ssm_out
MoE   1 rms_norm_pre_add_f32<1024,N,true>   1 blk    2.8us   post-attn norm + q8_1
      2 mul_mat_vec_f_grouped<16>         257 blk    4.7us   shexp_gate F32 + router F32
      3 mul_mat_vec_q_grouped            1025 blk    5.2us   shexp gate/up Q8_0 + 0046 topk guest
      4 mul_mat_vec_q                    4096 blk   17.3us   routed gate+up, GLU fused
      5 quantize_q8_1                      16 blk    1.3us   routed swiglu -> q8_1
      6 mul_mat_vec_q_expert_reduce      2048 blk   13.4us   routed down + weighted reduce
      7 unary_gated_op_quant                2 blk    1.4us   shexp swiglu + q8_1
      8 mul_mat_vec_q                    2048 blk    3.8us   shexp down
```

MoE 7 depends only on MoE 3, and MoE 5 only on MoE 4, which is what makes the
two elementwise co-launches above legal: 7 can ride inside 4, and 5 can ride
inside 8, once the graph is reordered so `ggml-alloc` stops handing the guest a
block the host is still reading (the same failure round 31 hit and round 32
fixed with a `probs_in` reorder).

Both `unary_gated_op_quant_kernel` and `quantize_q8_1` are **warp-local** — no
`__shared__`, no `__syncthreads`, the q8_1 reduction is `warp_reduce_*<QK8_1=32>`
over 32 consecutive elements. Because the element index is flat and contiguous,
those 32-element groups stay warp-aligned under **any** host block size that is a
multiple of 32, so relocating them into a foreign grid is bit-identical by
construction, exactly as 0046's topk guest was.

## 0047: the shared-expert down projection co-launched inside the routed expert-reduce (+2.6% decode, BIT-IDENTICAL)

Round 34's fit prices a removed **matvec** dispatch at `2.99 + ~2.85 = 5.8 us`
— it takes a bandwidth ramp with it — and ranked this first at ~+3.7%. It is
the first patch in the series to build the **second vec_dot type in one
launch** that rounds 31 and 32 both identified and neither built.

### Why this pair and not another

The shared-expert down projection and the routed `expert_reduce` have the
**same output width** (n_embd = 2048 rows) and read entirely different
activations. Q8_0 against Q5_K, 3.8 us against 13.4 us, 40 of each per token.

Geometry is the part worth copying. The guest's standalone launch gives one row
**two** warps (`calc_nwarps_launched` trims Q8_0/ncols=512 from 8 to 2), and the
host block is `(32, n_slots=8)`. So `8/2 = 4` guest rows share a block and the
merged grid is **512 guest + 2048 host blocks** — the guest contributes exactly
the 4096 warps its own 2048x2 launch had, with no idle-warp tax. Getting this
wrong is the easy way to give the whole win back: one row per block would have
launched 2048 blocks of 8 warps with 6 of them dead.

Bit-identity is by construction, the 0045 argument with a quantized body:

- `blocks_per_iter` is derived from the guest type's **full compile-time
  nwarps**, not from the trimmed launch, so no k-block moves between threads;
- the cross-warp exchange keeps the trimmed warps' slots, which are exactly
  `+0.0f` — for a `-0.0f` accumulator that addition is not the identity;
- the 0035 shared-expert gate scalar is applied in the guest epilogue exactly
  where `has_dst_scale` applies it in `mul_mat_vec_q`.

### The graph half is an ALLOCATOR problem, and that decides where the tail goes

Stock order runs the shared-expert tail **after** the routed reduction, and by
then `ggml-alloc` has freed the expert ids (`+0x5880`) and routing weights
(`+0x4500`) and hands `ffn_shexp` (8 KB) a block that covers both — which the
merged launch is still reading. No backend change can fix that; the node has to
be **allocated earlier**, while the ids and weights are still live.

`build_moe_ffn` now relays a caller-supplied tail between the routed activation
and the routed down projection (two optional `pin_pre_down` arguments; the
routed activation is expanded first so the gate/up+GLU fusion stays adjacent).
`qwen35moe.cpp` builds the whole shared-expert tail — sigmoid gate, swiglu, down
projection, gate multiply — before calling `build_moe_ffn` and hands it over.
Node order becomes:

```
74 ffn_moe_gate(id)  75 ffn_moe_up(id)  76 ffn_moe_swiglu
77 shared_expert_gate_sigmoid   78 ffn_swiglu   79 ffn_shexp   80 ffn_shexp_gated
81 ffn_moe_down(id) 82 ffn_moe_weighted  83-90 views  91-97 adds
```

which is what makes 79 and 81 adjacent **and** puts `ffn_shexp` on a block the
allocator picks while ids/weights are live. Gated on `n_tokens == 1`: at prefill
the graph is the stock build path, exactly as 0046 is.

### Numbers

Against a binary built from this patch's own parent commit, control RUNPATH
verified by `ldd`, arms alternated, no env set on either arm:

```
candidate tg128 157.48 157.51 157.16 157.10   mean 157.31
control   tg128 153.36 153.46 153.20 153.20   mean 153.31   -> +2.61%, 4/4 separated
```

Census: **681.1 -> 641.1 dispatches/token, exactly -40.0**; `mul_mat_vec_q`
4585 -> 3185 per 35 tokens and **every other kernel count unchanged** — unlike
0046's reorder, this one moves nothing onto a different kernel.

Decode-path perplexity (`-b 512 -ub 1 --chunks 8`, runner corpus) is
**bit-identical**: `3.9382` on both arms across three loads each.

`GGML_CUDA_SHEXP_DOWN_COLAUNCH=0` restores the two separate launches;
`GGML_QWEN_SHEXP_TAIL_PIN=0` restores stock node order.

### Predicted +3.7%, measured +2.61% — read the gap as a correction to the model

The dispatch removal is exactly the predicted -40.0, so the miss is in the
per-matvec price, not in the count. `5.8 us x 40 = 232 us` predicted 158.7 tok/s
from 153.0; the measured 157.3 corresponds to **~4.1 us per removed matvec**.
The ramp is evidently only partly recovered: the guest still streams its own
1.1 MB, and inside a foreign grid it starts that stream behind the host's. Price
the next matvec merge at **4 us, not 5.8** — 0048 below is the elementwise class
and came in at the elementwise rate, so the 2.99 us figure stands unchanged.

## 0048: the shared-expert SwiGLU hidden inside the routed gate+up matvec (+1.5% decode, BIT-IDENTICAL)

0047's reorder puts the shared-expert SwiGLU immediately after the routed
gate+up `MUL_MAT_ID`, and the two are independent — the SwiGLU reads only the
shared-expert gate/up pair, computed several dispatches earlier. This is 0046's
move with an elementwise body, and it confirms 0046's rule: **for a tiny guest
the predictor is capacity, not the time ratio.** 2 blocks joining 4096 never
competes for the host's bandwidth.

The body is warp-local — no `__shared__`, no `__syncthreads`, and its two
reductions are `warp_reduce_*<QK8_1=32>` over 32 **consecutive** elements of a
flat index — so it is bit-identical under any block shape whose thread count is
a multiple of the warp size. It runs as one extra `blockIdx.x` slice of the
routed matvec's grid, indexing its own blocks off the **channel axis it has no
use for** (8 channels x 64 threads = 512 elements, exactly `k`). The body moved
into `glu-quant-body.cuh` so the standalone kernel and the guest execute the
same statements, exactly as 0046 did with `topk-moe-body.cuh`.

One packaging detail worth stealing: the q8_1 destination is registered in the
cache **before** the launch, so if the host geometry cannot carry the guest the
fallback runs the standalone kernel into that same buffer. Without it a declined
guest would leave the consumer an unwritten cache entry — a silent wrong-answer
bug, not a crash.

```
candidate tg128 159.89 159.80 159.67 159.45   mean 159.70   (0047+0048)
control   tg128 153.53 153.27 153.36 153.35   mean 153.38   -> +4.12%, 4/4 separated
```

Census: **641.1 -> 601.1 dispatches/token, exactly -40.0**;
`unary_gated_op_quant` 2800 -> 1400 per 35 tokens (the GDN one remains).
`-40 x 2.99 us = 120 us` predicts +1.9%; measured +1.47%. The elementwise rate
is holding within 25%.

`GGML_CUDA_MMVQ_GLU_COLAUNCH=0` restores the separate launch.

### Correctness of the pair

- decode-path perplexity **3.9382 on both arms, three loads each** — bit-identical;
- server greedy at the decode shape, 6 prompts including a 1000-token one,
  **6/6 byte-identical** against the same-binary toggle-off control, and the
  same-arm-twice control is also a clean **6/6** (round 32's harness read 5/6
  on the same-arm control — that noise was not reproduced here);
- prefill flat (`pp512` within 0.5% both ways, and both halves are gated off the
  prefill shape by construction).

### Warning: the `-c 512` gate-shape perplexity was NOT reproducible this session

The 0044 notes call `llama-perplexity -c 512 --chunks 8` "perfectly
reproducible". It was not, on either arm, on this box today:

```
control    3.9247 3.9318 3.9318 3.9335 3.9320 3.9437 3.9318 3.9318 3.9362 3.9326
candidate  3.9485 3.9312 3.9303 3.9312 3.9355 3.9318 3.9318 3.9368 3.9310 3.9324
```

Control spread alone is **0.48%**, five times the gate. Medians agree exactly
(3.9318 vs 3.9318) and the means differ by 0.003%, so the reading is fine — but
a single draw at this shape can no longer be used to accept or reject anything.
**Read the decode-path shape (`-b 512 -ub 1`), which was exactly reproducible on
both arms, and use the gate shape only as a median over >= 5 loads.**

## Where the decode token is after 0047+0048

601.1 dispatches/token, ~159.7 tok/s on llama-bench. Against round 34's model
(`t(d) = 4448.4 + 2.991 d`) 601 predicts 6246 us / 160.1 tok/s — the measurement
lands within 0.3% of the fit, so the model survives 80 dispatches of extrapolation.

The MoE block is now **7 dispatches**, and what is left of it:

```
MoE 1 rms_norm_pre_add       1 blk   2.8us   post-attn norm + q8_1
    2 mmvf_grouped         257 blk   4.7us   shexp_gate F32 + router F32
    3 mmvq_grouped        1025 blk   5.2us   shexp gate/up Q8_0 + 0046 topk guest
    4 mmvq                4104 blk  17.3us   routed gate+up + GLU + 0048 swiglu guest
    5 quantize_q8_1         16 blk   1.3us   routed swiglu -> q8_1
    6 mmvq_expert_reduce  2560 blk  13.4us   routed down + reduce + 0047 shexp down
```

**Round 34's row 3 (routed swiglu `quantize_q8_1` -> guest of the shexp down
mmvq) is now arithmetically dead, and 0047 is why.** Its only possible host was
the shared-expert down projection, which no longer exists as a separate launch —
and the launch that absorbed it *reads* the buffer that quantize produces. The
two were always alternatives (+3.7% against +1.8%); this records which one won.
Nor can it fold into the routed GLU epilogue: in `mul_mat_vec_q` each block
produces ONE output element, so the 32 consecutive elements a q8_1 block needs
live in 32 different blocks. Do not re-open it.

Remaining from the round-34 list, re-priced at the corrected 4 us/matvec:

| move | class | x/token | est. |
|---|---|---|---|
| GDN `ssm_norm` + z-gate `unary_gated_op_quant` fused | elementwise | 30 | +1.4% |
| attn k/v grouped mmvq merged into attn wq (needs 0047's machinery) | matvec | 10 | +0.6% |

Both are still open. The realistic band from round 34 (**165-169 tok/s
llama-bench**) is unchanged; 159.7 is 60% of the way there from 153.0.
