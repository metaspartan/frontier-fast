# qwen3.6-35b-a3b-gguf-gb10cuda-v1 — patch series

Port of the laguna-xs gb10cuda engine series (0001–0011, sm_121-tuned:
q8_1 requant dedupe, grouped launches with grouped-mmvq off by default —
it loses on this device — norm/rope/set-rows groups, quantize folds, mmvf
batched k-loads) plus the generic topk-moe sorted-list selection (0012,
qwen35moe routes top-8 of 256 like Laguna). All patches unchanged from
their verified sources; 0012 renumbered from the qwen r9700 series' 0014.

## Measured (runner box gx10-838f, 5-round interleaved whole-process A/B)

- decode tg128: stock 40.97–41.26 → candidate 42.04–42.29 tok/s, median
  per-round ratio **1.0272**
- prefill pp512: 803.5 → 805.8 (~1.003)
- ppl `-c 512 --chunks 8` on the gate corpus: **3.9322 → 3.9322,
  bit-identical** (the series is bit-exact by construction on this model)
- candidate `llama-server` boots, serves coherent greedy text, exits clean

## Notes

- The GDN linear-attention layers (30 of 40) put much more traffic through
  `mul_mat_vec_f` than Laguna does — 0011 (batched k-loads) is doing real
  work here.
- **Grouped-mmvf measured-dead here (do not spend a slot):** the R9700's
  +3.35% lever (mul_mat_vec_f_grouped + qwen35moe adjacency pins) was
  ported (arch guard relaxed, one-warp grouped kernel) and toggle-A/B'd on
  the runner box 2026-08-07: decode median ratio ~1.005 over 5 interleaved
  rounds, prefill neutral, ppl 3.9367 (+0.11%, gate-fine). Removing ~90
  launches/token buys half a percent on a memory-latency/occupancy-bound
  box — consistent with the grouped-mmvq history on sm_121.


- **ngram self-speculation measured-dead here too** (2026-08-07): decode
  41.7 -> 37.5 (-10%), mirroring the R9700 twin. Same acceptance economics.

## 0013: load-time Q6_K requant of the UD Q8_0 projections (GB10 port)

Port of the R9700 lever (see that track's README for the full derivation).
The UD export's Q8_0 projection upcasts are ~1.49 GB of weight reads per
decoded token; the loader requantizes the five families to Q6_K at load.
On the GB10's LPDDR5 the byte cut pays MORE than on the R9700:

- decode 42.0-42.25 -> 44.4-44.5 tok/s (**+5.7%**, 3/3 same-binary toggles)
- prefill 806 -> 773 tok/s local (-4.0%; MMQ path - the RDNA4 GEMM routing
  is **HIP-guarded off** here because sm_121 tensor-core Q6_K MMQ measures
  FASTER than dequant+GEMM, the opposite of RDNA4)
- gate ppl: cand 3.9306 vs stock 3.9322 (**-0.04%**)
- server smoke clean; GGML_LOAD_REQUANT=0 restores stock bytes

The R9700 ranked run measured about HALF the local pp512 prefill delta, so
the ranked prefill floor has comfortable margin. The 0022-class sigmoid
fusion does NOT port here (needs the grouped-mmvf launch, which is
measured-dead on sm_121).


## 0014: Turing mmvq parameter table for sm_121

nsys decode profile (post-0013): the MoE expert matvecs run at **35-40% of
achievable bandwidth** (Q4_K gate+up fused ~99 GB/s, Q5_K down ~80 GB/s)
while the dense Q6_K mats sit at the ~240 GB/s ceiling - LPDDR5 latency
wants more ILP per warp. sm_121 was falling to the GENERIC table (4
warps/row); the Turing table (2 warps for K-quants) fits better: decode
44.4-44.5 -> **45.2-45.4 tok/s (+1.9%)**, ppl byte-unchanged for the gate
(mmvq is decode-only; the ppl path is prefill-shaped MMQ).

Remaining headroom on this track: the expert matvecs are still far off the
ceiling - a custom mmid config (more rows per block / deeper unroll for the
[2048->512]x8 and [512->2048]x8 shapes at ncols_dst=1) is the scoped next
lever, worth up to ~+15% if they reach dense-mat efficiency.


## 0015: dedicated SM121 mmvq table (deep rows for the expert shapes)

Replaces 0014's Turing aliasing with a real SM121 table: nwarps=2 for
K-quants (as 0014) plus **rows_per_block = 2*nwarps under small_k** - four
independent dot products per thread off shared activation loads. All the
MoE expert shapes take small_k at nwarps=2. Dense stays 1 row/block (the
grouped-mmvq launch statically asserts blockIdx == row - the build fails
loudly if violated, which is how this was discovered).

45.2-45.4 -> **45.56-45.97 tok/s (+0.9%)**; ppl byte-unchanged. Cumulative
stack vs stock toggle: **+9.4%** (42.0 -> 45.97). Expert matvecs still have
headroom to the ~240 GB/s dense ceiling; next steps: rows=8 sweep, or a
dedicated expert-batch kernel that walks all 8 experts per block.


## 0016: cap the MMQ MoE J tile at 64 on sm_121 (prefill)

Port-with-inversion of the R9700 round-7 J-cap (its 0023). The RDNA4
near-segment-width rule (J=32) **loses 9%** on sm_121 - tensor-core MMQ
wants wide column tiles - but stock J=128 overshoots too. The healthy-box
target sweep (16/32/48/56/64/72/80/96/112/128 -> 1597/2101/2272/2375/2370/
2358/2352/2350/2328/2297 pp512 tok/s, stock 2297-2308) peaks at J=56-64;
the patch caps the mmid path at the smallest valid config >= 64,
arch-guarded to sm_121, `GGML_CUDA_DISABLE_MMQ_MOE_J` restores stock.

- pp512 toggle A/B, 5 interleaved rounds: 2250-2292 -> 2350-2372, median
  per-round ratio **1.0332**; 2-round confirm +2.2%
- decode unchanged (tg64 72.4-72.8 both arms; expert decode is mmvq)
- gate ppl 3.9306 on and off, byte-identical (-0.04% vs stock)
- server greedy identity byte-exact; server uncached prefill 941 -> 987 tok/s

## 0017: the sm_121 wide Q4_K mmvq vec_dot (+1.77% decode)

Cross-port of the lfm2.5 gb10cuda 0012 lever, and **the correction to this
README's "Q6_K mmvq load-path engineering is FLAT" entry below**. That entry
is right about Q6_K and right about why — `block_q6_K` is 210 bytes, so
`ql`/`qh` are 2-byte aligned and the load *cannot* widen — but its conclusion
("issue-side MLP is NOT the limiter") generalised one quant type too far.
Q4_K's block is 144 bytes with `qs` at offset 16, and a row is a whole number
of blocks, so `qs` is 16-byte aligned and the weight fetch *can* become an
8-byte `int2`. When it does, it pays.

`get_vdr_mmvq(type, table_id)` returns 4 for Q4_K under the sm_121 table
0014/0015 already established, and the sm_121 `vec_dot_q4_K_q8_1` covers the
column pair `(iqs, iqs+2)`: the pair shares `bq8_offset`, the unpacked
scale/min pair and `d8`, so that work happens once instead of twice. Only the
**device** kernels route through the table-aware overload — the host
`should_use_small_k` / `mmvq_may_use_small_k` predicates keep the stock vdr on
purpose, so 0015's `rows_per_block` tuning for the expert shapes does not move
at the same time and this stays a single-variable change. nwarps is untouched
(this table already sets 2 for K-quants), so `blocks_per_iter` simply doubles
from 4 to 8 and the k-loop makes half as many passes.

Measured on the runner box (vllm container parked), 5 interleaved
whole-process rounds against the 16-patch build:

| round | 0016 tg128 | 0017 tg128 | 0016 pp512 | 0017 pp512 |
| --- | --- | --- | --- | --- |
| 1 | 71.21 | **72.43** | 2340.99 | 2340.83 |
| 2 | 71.06 | **72.50** | 2349.25 | 2335.82 |
| 3 | 71.27 | **72.47** | 2342.99 | 2339.35 |
| 4 | 71.05 | **72.63** | 2320.56 | 2352.00 |
| 5 | 71.26 | **72.39** | 2337.57 | 2346.99 |

decode median ratio **1.0177**, 5/5 arms disjoint; pp512 overlapping in both
directions, i.e. neutral (prefill takes MMQ and never enters this kernel).

- **firing proof.** nwarps does not change here, so block dims do not move and
  a name/shape census cannot see the patch. The register census can: the Q4_K
  `mul_mat_vec_q` goes **48 → 56 registers/thread** at identical grid, block
  and launch count, and every Q5_K / Q6_K / Q8_0 row is untouched.
- gate ppl `-c 512 --chunks 8` **bit-identical 3.9306** both arms — the gate's
  shapes never enter mmvq, so that reading covers nothing. Decode-path ppl
  (`-b 512 -ub 1 --chunks 8`) **3.9143 → 3.9237, +0.240%**, inside the band.
  Quote both.
- **server greedy** (`llama-server`, 6 prompts, `temperature 0`, non-emptiness
  asserted — 6/6 completions, 3789 and 3774 bytes): **2/6 byte-exact, 4/6
  diverge mid-completion** (first divergence at bytes 471, 127, 135, 262).
  Reassociation class, the same as every wide/multi-row matvec rewrite in this
  family: the wide call moves an addition out of the warp reduction tree and
  into a lane. Perplexity is the arbiter and it is in band.
- the full 17-patch series applies clean to pristine `b10237` and the tree is
  `diff -r` identical to the measured tree.

Projection: decode `1.0177` on top of the bank → score **~1.055** against the
**1.0435** bank.

### What this says about the remaining pool

A decode census (7 tokens, nsys) counts per token: **104 Q6_K**, **100 Q8_0**,
**40 Q4_K**, **37 Q5_K** mmvq launches. Q4_K is the *smallest* of the four
pools, which is exactly why this lever is worth 1.8% here and 7.8% on the
LFM track where Q4_K is 71% of decode. The next candidate is **Q5_K**:
`block_q5_K` is 176 bytes with `qs` at offset 48, both multiples of 16, so the
same `int2` cast is legal. Check the expert `ncols_x` before building it — the
R9700 twin measured Q5_K `vdr = 4` dead at k=512, where a row is only 2
k-blocks and most lanes go idle. Q6_K is structurally closed (see below), and
Q8_0 already uses wide accessors.

## 0018: sm_121 one-warp block for the Q5_K expert-down shape (+0.32% decode)

The answer to 0017's "next candidate is Q5_K" question above, and it is not
the `int2` widening. **Q5_K `vdr = 4` is dead-by-construction here**, for a
reason the shape census makes exact: this model's only Q5_K tensors are the
37 `ffn_down_exps` at `[512 x 2048] x 256`, so `ncols_x` = 512 = **two**
Q5_K superblocks per row. A single warp at the stock `vdr = 2` already
covers `vdr*warp_size/qi` = 2 blocks, so there is no k left to widen into —
`vdr = 4` would give the row 8 block-slots for 2 blocks and idle 48 of 64
lanes. This is the same k=512 wall the R9700 twin measured, confirmed by
static shape analysis rather than re-bought on the runner.

The real defect at this shape is the opposite of a load-width problem. The
sm_121 table gives K-quants `nwarps = 2`, so `blocks_per_iter` = 4 against
`blocks_per_row_x` = 2 and **half of every thread block has no k-block to
read for the whole kernel**. Upstream's `small_k` fires on exactly this
condition but only compensates with more rows — it never shrinks the block,
which is why 0015's deep rows paid and the follow-up rows = 8 sweep was
neutral. 0018 runs the shape as a one-warp block that keeps the same 4-row
tile (`calc_nwarps` -> 1 for Q5_K, `calc_rows_per_block` keeps 4 rows when
the small_k block is one warp, and the host `should_use_small_k` predicate —
which assumes `nwarps > 1` because only then can lanes idle — also fires for
a one-warp config whose single iteration already covers the row).

- **firing proof (nsys geometry census)**: `mul_mat_vec_q<(ggml_type)13, 1,
  false, true>` block `32 2 1` -> `32 1 1` at the same grid `512 8 1`, every
  other kernel identical. Its GPU time 12.36 -> 11.59 ms (**-6.3%**)
- decode tg128 72.33/72.41/72.51/72.30/72.53 -> 72.56/72.63/72.74/72.85/
  72.81, **5/5 rounds disjoint, median ratio 1.0032**
- prefill pp512 neutral; **ppl bit-identical** (gate 3.9306 = 3.9306,
  decode-path `-b 512 -ub 1 --chunks 16` 3.9657 = 3.9657) — the idle lanes
  contributed only exact zeros to the cross-warp reduction, so the summation
  order never moved
- `GGML_CUDA_DISABLE_SM121_ONE_WARP_ROW` restores the two-warp config

## 0019: sm_121 mmvq k-loop unroll 4 (+0.71% decode)

The decode census puts **~50% of decode kernel time in Q6_K dense matvecs**
(attn_qkv 18.2%, the `[2048 x 248320]` output head 13.0%, attn_out 7.7%,
attn_gate 7.0%, shexp 3.0%) at 170-200 GB/s against this box's 251.6 GB/s
mmvq-geometry probe. The "Q6_K load-path is FLAT" entry below is still
right that no wider *call* helps — but it tested wider calls, not more
*iterations in flight*. At `vdr = 1` a Q6_K lane issues one ql, one qh and
two q8_1 loads per k-loop iteration and then waits, and since
`blocks_per_row_x` is a runtime value nvcc leaves the loop rolled.

4 is the optimum and the shapes say why: at `ncols_x` = 2048 with `vdr = 1`,
`nwarps = 2`, `blocks_per_iter` = 2 against `blocks_per_row_x` = 8 is
**exactly 4 iterations**, so unroll 4 turns the dominant Q6_K kernels into
one straight-line body with four independent load chains and no loop.

| unroll | tg128 | vs 0018 |
|---|---|---|
| 2 | 72.47-72.67 | neutral |
| **4** | **73.01-73.28** | **+0.71%, 8/8 rounds disjoint over two sessions** |
| 8 | 71.50-71.82 | -1.5% (register pressure) |
| 16 | 56.33-56.48 | -22% (spills) |

Accumulation order into `tmp[][]` is unchanged, so the result is bit-exact:
gate ppl 3.9306 and decode-path 16-chunk ppl 3.9657, both identical to the
0018 control. Prefill neutral. `GGML_CUDA_DISABLE_SM121_MMVQ_UNROLL`
restores the rolled loop.

**Firing order** (each measured against a binary built from its own parent):
0013 requant +5.7% -> 0014 Turing table +1.9% -> 0015 deep rows +0.9% ->
0016 MMQ J=64 +3.3% prefill -> 0017 wide Q4_K +1.77% -> 0018 Q5_K one-warp
+0.32% -> 0019 k-loop unroll 4 +0.71%.

## Measured dead ends (healthy box, 2026-08-08, do not re-buy)

- **sm_121 deep-row tile for the Q4_K expert gate/up shape is NEGATIVE**:
  the natural sequel to 0018 — at 0017's `vdr = 4` the Q4_K exps
  (`[2048 x 512] x 256`, `blocks_per_row_x` = 8 = `blocks_per_iter`) consume
  the row in one iteration, so rows are the only remaining source of
  per-lane MLP. Routing the ids matvecs of that shape into `small_k`
  (rows_per_block 1 -> 4) **fired exactly as intended** — nsys shows the
  Q4_K kernel grid `512 8 1` -> `128 8 1` — and cost decode: tg128 median
  72.24 vs 72.74 for the 0018 control (**-0.7%**), that kernel's own GPU
  time 20.87 -> 20.98 ms. ppl byte-identical (correct, just slower).
  Together with the rows = 8 result below this closes the rows dimension in
  both directions: on sm_121 the deep tile only ever pays when it is
  *rescuing idle lanes*, never as added ILP on a block that is already full.
- **Dense Q6_K multi-row mmvq (R9700 0024 port) is NEGATIVE on sm_121**:
  opting dense Q6_K (ncols_dst==1, no ids) into the small_k route, swept
  rows_per_block 2 and 4 at nwarps=2, same-binary toggle A/B 5 interleaved
  rounds at 2405 MHz: rpb=2 tg64 72.51 -> 70.51 (-2.8%), rpb=4 72.47 ->
  69.67 (-3.9%), monotonic 1>2>4. ppl byte-identical 3.9306 all arms
  (correct, just slower). Exact inversion of the RDNA4 +3.0% verified win:
  dense large-K matvecs on LPDDR5 hide latency with many resident blocks,
  not per-thread ILP; deep rows only pay for the tiny-K expert shapes
  (blocks_per_row ~2) that the 0015 small_k config already covers.
- **Expert small_k rows=8 (4*nwarps) is flat-to-negative**: the 0015
  README's "rows=8 sweep" next step, measured cross-binary vs the 0016
  control, 5 rounds: rows4 tg64 median 72.43 vs rows8 72.12 (-0.4%). The
  shipped 2*nwarps depth is the optimum; sm_121 mmvq geometry is closed in
  both directions (see also grouped-mmvq/mmvf history).
- **CUDA graphs already capture at decode — no launch-overhead pool**:
  MUL_MAT_ID only blocks capture when src0 is unquantized or
  ne[2] > mmvq_mmid_max (decode has ne[2]=1, quantized experts).
  GGML_CUDA_DISABLE_GRAPHS=1 A/B, 3 rounds: graphs-on tg64 72.32-72.56 vs
  off 71.61-71.92 — capture is live and worth its ~1.1%; decode is
  DRAM-side, not launch-side.

- **Q6_K mmvq load-path engineering is FLAT (+-0.1%)**: vdr=2 and vdr=4
  variants (2/4 adjacent quant ints per vec_dot call, shared scale/ds/offset
  loads, exact integer dp4a combine) and `__ldcs` evict-first streaming on
  ql/qh all measured neutral in interleaved whole-process tg32 A/B (6-round
  ldcs median +0.07%). block_q6_K is 210 B = 2-byte aligned, so wide
  vectorized or cp.async loads are structurally impossible; issue-side MLP
  is NOT the limiter - the kernel rate is DRAM-side-determined.
  **Amended by 0017**: the first two clauses hold, the last does not
  generalise. It is Q6_K's 210-byte block that closes the lever, not the
  hardware. Q4_K's 144-byte block admits the same widening and it is worth
  +1.77% decode; Q5_K's 176-byte block should admit it too.
- **Output-head requant is decode-NEUTRAL**: head Q6_K->Q5_K (68 MB/token
  cut, ppl +0.40% - inside band but thin margin) and Q6_K->Q5_1 (36
  MB/token, ppl -0.33%) both measured +0.0-0.2% tg32. The byte cut is
  exactly eaten by the target types' lower dense-shape kernel efficiency
  (Q6_K dense mmvq runs near the LPDDR5 ceiling; Q5_K/Q5_1 dense do not).
  Byte cuts only pay when the destination kernel is at least as efficient.

## 0020: the recurrent state served as a cache view, not a gather (+3.82% decode)

Cross-port of the r9700 track's 0030, and the first patch in this series that
came from that track's *core* work rather than its backend work. It touches no
backend code at all — `src/llama-graph.*`, `src/llama-memory-recurrent.*`,
`src/models/delta-net-base.cpp`, `src/models/qwen35moe.cpp` — which is why it
transfers between HIP and CUDA unchanged.

Thirty of this model's forty layers are gated-delta-net. Each one materialises
its recurrent state before `gated_delta_net` runs: `build_rs` gathers the ssm
state and `build_conv_state` gathers the conv window, two `get_rows` launches
per layer per decoded token. That gather exists so seq-copy migration and
rollback-snapshot restore can redirect a slot. In steady single-slot decode its
index vector is the identity, and the copy moves bytes to a new address purely
so the consumer can read them from there.
`llama_memory_recurrent_context::s_copy_main_is_identity()` detects that at
graph-build time without `s_copy()`'s side effects, and `build_rs` hands the
consumer a contiguous `ggml_view_2d` of the cache rows instead. The view is
restricted to `n_tokens == 1 && n_seqs == 1 && head == 0 && n_rs == 1`. The
`n_seqs`/`head`/`n_rs` part is the sibling track's: llama.cpp reuses built graphs
across decode steps and a view bakes its offset at build time, so at row 0 a
reused graph can never go stale, and that is the `--parallel 1` ranked shape. The
`n_tokens == 1` clause is added here so the view is taken only on the
decode-shape build it was derived for and is provably inert at every
prompt-shaped build, including the long-context evaluation windows. It costs
nothing: the gain is a decode gain and decode is `n_tokens == 1`, at short and
long context alike. `LLAMA_DISABLE_RS_STATE_VIEW=1` restores stock.

**Why this one ports when the launch-structure family did not.** This README
already records grouped-mmvq as off by default, grouped-mmvf as measured dead
(+0.5% for ~90 launches removed), and CUDA graph capture as live at decode with
no launch-overhead pool behind it. All three say the same thing: on sm_121 a
removed dispatch is worth almost nothing, because the replay gap the HIP track
recovers is not there to recover. 0020 is not a launch-count patch. It removes
a **read plus a write of the entire recurrent state on 30 layers of every
token** — bytes, on a 121.6 GiB LPDDR5 part whose cold DRAM rate is 200-205
GB/s. That is the property to test a candidate port against on this device: not
how many dispatches it deletes, but how many bytes.

Same-binary toggle, 5 interleaved whole-process rounds against the 19-patch
build:

| round | on tg128 | off tg128 | on pp512 | off pp512 |
| --- | --- | --- | --- | --- |
| 1 | 75.74 | 72.97 | 2355.60 | 2356.87 |
| 2 | 75.57 | 73.09 | 2355.79 | 2338.26 |
| 3 | 75.78 | 72.86 | 2351.35 | 2371.15 |
| 4 | 75.78 | 73.00 | 2351.90 | 2336.70 |
| 5 | 75.57 | 72.91 | 2319.05 | 2358.87 |

decode median per-round ratio **1.0380**, 5/5 rounds disjoint (min-on 75.57 >
max-off 73.09); pp512 overlaps in both directions, i.e. neutral — the view is
not taken on any prompt-shaped build by construction.

Perplexity is **identical to four decimal places at every measured shape**, view
on and view off: gate `-c 512 --chunks 8` **3.9306** (equal to the parent
frontier's recorded value), decode shape `-c 512 -b 512 -ub 1 --chunks 8`
**3.9237**, the 16384-token window of the held-out long corpus **5.9375**, and
the 32768-token window **6.9450**. The view and the gather deliver the same bytes
to the same consumer, so this is structural rather than a lucky draw, and this
patch spends none of the long-context budget.

For the next reader: this series' own 32k reading of **6.9450** against stock's
**6.9341** (+0.157%) is entirely patch **0013**'s load-time requant —
`GGML_LOAD_REQUANT=0` on the same binary returns exactly 6.9341, while disabling
the MMQ J-cap instead leaves 6.9450 unchanged. It is a byte-level weight change,
so no kernel rewrite recovers it, and it is the accuracy side of a trade that
buys +5.7% decode. Neither 0020 nor 0021 moves it.

## 0021: the q8_1 quantize folded into the fused unary+mul (+0.44% decode)

Cross-port of the r9700 track's 0036, unmodified — the patch carries no arch
guard, and 0009 already provides the mul-fold hook it extends. Same-binary
toggle (`GGML_CUDA_DISABLE_UNARY_MUL_QUANT=1`), 5 interleaved rounds on top of
0020: tg128 75.79/75.83/75.85/75.85/75.96 -> control 75.41/75.49/75.52/
75.57/75.67, median ratio **1.0044**, 5/5 disjoint (min-on 75.79 > max-off
75.67); pp512 neutral. Gate ppl **3.9306** both arms.

It is small here for the same reason the rest of the launch-structure family is
small: an nsys decode census of the 0020 build puts only 108 us/token in
`k_bin_bcast op_mul` (82 launches) and 225 us/token in `quantize_q8_1` (125
launches) out of a ~14 ms token, so the entire pool this patch can reach is
about 2.4%. It is banked because it is byte-exact and toggleable, not because
it is a lever.

### Decode census of the 0020 build (nsys, share of kernel time)

| share | n/token | kernel |
| --- | --- | --- |
| 48.3% | 107 | `mul_mat_vec_q` Q6_K dense (qkv/z, ssm_out, attn_out, lm head) |
| 14.1% | 41 | `mul_mat_vec_q` Q4_K MoE gate+up (ids) |
| 8.7% | 38 | `mul_mat_vec_q` Q5_K expert down (one-warp, 0018) |
| 5.3% | 144 | `mul_mat_vec_f` F32 — the GDN alpha/beta/router/shexp-gate swarm |
| 3.5% | 41 | `mul_mat_vec_q` Q8_0 (ids) |
| 2.0% | 31 | `gated_delta_net_cuda` |
| 1.5% | 125 | `quantize_q8_1` |
| 0.7% | 82 | `k_bin_bcast op_mul` |

Half the decode token is still dense Q6_K matvec, which 0014-0019 have already
worked over; the F32 swarm is the largest remaining *count*, and grouping it is
the lever this README records as measured dead here.

### Prefill census of the same build (nsys, share of kernel time)

| share | n/pass | kernel |
| --- | --- | --- |
| 26.6% | 160 | `mul_mat_q` Q4_K experts |
| 17.0% | 74 | `mul_mat_q` Q5_K expert down |
| 12.4% | 220 | `mul_mat_q` Q6_K (J=128 path) |
| 12.3% | 60 | `gated_delta_net_cuda` |
| 4.6% | 220 | `k_bin_bcast op_mul` |
| 2.3% | 60 | `concat_non_cont` |
| 1.8% | 60 | `ssm_conv_long_token_f32` |

The two GDN prefill items the r9700 track attacked are both live here and both
unported: its 0043 (multi-column waves in `gated_delta_net`, +5.7% prefill
there) addresses the 12.3% row, and its 0033 (conv-chain fold at prefill shape,
which deletes `concat_non_cont`, +3.6% prefill there) addresses the 2.3% + 1.8%
pair. Neither applies cleanly on top of this series — both need their
`ggml-cuda.cu` hunks rebased — and 0043 is explicitly not bit-exact on the HIP
track, so it spends perplexity budget that 0020 and 0021 leave untouched. They
are the scoped next levers for prefill.

## The roofline, measured rather than assumed (2026-08-14)

Every headroom claim on this track has been computed against **273 GB/s**,
which is the DGX Spark datasheet number, and against a bytes/token figure the
board itself labels `estimated-from-architecture`. Both were measured on the
box before 0022 was designed, because 0022's whole argument is the size of the
gap between what a kernel gets and what the DRAM will give.

### What the DRAM actually delivers

A standalone CUDA program, cudaEvent-timed, no profiler, pseudo-random
contents, one block per row exactly as `mul_mat_vec_q` launches:

| working set | geometry | 64 thr/blk | 128 thr/blk | 256 thr/blk |
| --- | --- | --- | --- | --- |
| 417 MB (the output head, exact) | 248320 x 1680 B | **231.8** | 242.7 | 244.1 |
| 5.28 GB | 3145728 x 1680 B | 238-239 | 250-253 | — |
| 13.8 MB (L2-resident control) | 8192 x 1680 B | 1306-1363 | — | — |

The last row is the residency check that replaces a miss counter (this box
denies GPU performance counters to non-root, so `ncu` cannot run): a working
set inside the 24 MB L2 reads 5.6x faster, so the 417 MB and 5.28 GB rows are
DRAM. Zeroed and pseudo-random buffers read the same, so there is no
compression artefact either way.

**Sustained streaming read on this GB10 is 232-252 GB/s, not 273.**

### What the token actually reads

Byte census from the GGUF tensor table, matched launch-for-launch against an
nsys decode trace of the 21-patch frontier build (103 decoded tokens, every
kernel group's launch count divides exactly):

| MB/token | us/token | GB/s | matvec group |
| --- | --- | --- | --- |
| 550.5 | 2565 | 214.6 | Q6_K attn_qkv x30 + attn_q x10 `[2048 x 8192]` |
| 417.2 | 2053 | 203.2 | Q6_K output head `[2048 x 248320]` |
| 377.5 | 2008 | 188.0 | Q4_K ffn_gate+up_exps x40 (fused) `[2048 x 512] x8` |
| 275.3 | 1423 | 193.5 | Q6_K ssm_out x30 + attn_output x10 `[4096 x 2048]` |
| 213.4 | 1237 | 172.5 | Q5_K ffn_down_exps x37 `[512 x 2048] x8` |
| 206.4 | 981 | 210.4 | Q6_K attn_gate x30 `[2048 x 4096]` |
| 111.4 | 589 | 189.3 | Q8_0 shexp gate+up x40 (fused) + attn_k/v x20 |
| 83.9 | 369 | 227.1 | F32 ffn_gate_inp router x40 `[2048 x 256]` |
| 44.6 | 267 | 166.9 | Q8_0 ffn_down_shexp x40 `[512 x 2048]` |
| 20.6 | 120 | 172.3 | Q6_K ffn_down_exps x3 `[512 x 2048] x8` |
| 15.7 | 111 | 142.3 | F32 ssm_alpha/ssm_beta x60 `[2048 x 32]` |
| 0.3 | 49 | 6.7 | F32 ffn_gate_inp_shexp x40 |
| **2316.8** | **11772** | **196.8** | **aggregate matvec** (431 launches/token) |

Add the recurrent state (30 GDN layers x 2.097 MB, read and written: 125.8 MB),
the conv state (2.9 MB), the KV cache of the 10 attention layers at 512 context
(10.5 MB), the logits write (1.0 MB) and the norm weights (17.7 MB), and the
token reads **2474.8 MB**, against the board's estimate of 2777.9 MB. The
estimate is close to the *stock* byte count; 0013 already removed about 290 MB
of it.

Matvec is **88.6%** of GPU-busy time (11772 of 13277 us/token, where GPU-busy
is the union of kernel intervals and lands within 0.7% of the unprofiled wall
clock).

### The ceiling those two numbers imply

At 2474.8 MB/token the frontier's 74.59 tok/s is **184.6 GB/s** end to end:

| against | GB/s | ceiling | headroom over 74.59 |
| --- | --- | --- | --- |
| datasheet (what the board uses) | 273.0 | 110.3 | +47.9% |
| best sustained streaming read | 250.0 | 101.0 | +35.4% |
| head geometry at 64 thr/blk (the kernel's own block) | 231.8 | 93.7 | +25.6% |
| the best rate any real matvec here reaches | 214.6 | 86.7 | +16.3% |
| the current aggregate matvec rate | 196.8 | 79.5 | +6.6% |

The published 98.27 tok/s ceiling survives this, but by cancellation rather
than by being right: 273 GB/s overstates the bandwidth by about 9% and
2777.9 MB overstates the bytes by about 12%. Recomputed honestly the ceiling is
99-101 tok/s.

The reachable part is the narrow band. Every large matvec is already within
5-15% of what its own access pattern sustains, so the difference between the
top and bottom rows of that table is not engineering headroom - it is the cost
of reading a 210-byte Q6_K block with 4-byte loads, which this track has
measured dead twice. 0022 takes the one part of the gap that is a launch
parameter rather than a load width.

### Correction to an earlier entry

The dead end recorded as "the 149 GB/s profile figure was measured under nsys
overhead" is wrong about the mechanism. Node-level graph tracing costs 5.2% of
wall clock (74.79 -> 70.91 tok/s) but **does not inflate kernel durations**:
run with `GGML_CUDA_DISABLE_GRAPHS=1` under a plain kernel trace, which costs
0.96%, the same kernels report the same medians to within 1% - the output head
reads 2045.63 us one way and 2044.29 us the other. The overhead lands in the
gaps between nodes. Kernel times from an nsys census may be used directly; what
must not be used directly is the *sum* of them, which overcounts by about 8%
because adjacent CUPTI ranges overlap by roughly 1 us of drain.

## 0022: sm_121 wide row block for the dense Q6_K matvecs (+1.2% decode)

Dense Q6_K is 1449 of the 2317 MB of matvec weights a token reads. The table
above says the group runs at 202-215 GB/s and the probe says its own geometry
sustains 232 GB/s at 64 threads per block and 243 at 128. Q6_K's 210-byte block
forbids a wider load, so the remaining way to get more narrow loads in flight
against one row is more lanes on the row: the SM121 mmvq table gives K-quants
two warps, and this gives Q6_K four.

`rows_per_block` does not move - `calc_rows_per_block` returns 1 for every
`ncols_dst == 1` shape that is not `small_k`, and Q6_K never takes `small_k` on
this model - so the grid is unchanged and only the block grows. That is the
difference from the whole-table nwarps sweep recorded as neutral in the dead
ends above: raising nwarps for the entire K-quant row also raised the expert
shapes to `rows_per_block = 2*nwarps = 8`, which the same section separately
measures at -0.4%, so the two were being netted against each other.

Firing proof (nsys, grids identical, blocks doubled): head 2044 -> 1977 us,
ssm_out 36.99 -> 35.26 us, qkv 66.50 -> 66.61 us, attn_gate 33.54 -> 33.73 us,
and every Q4_K/Q5_K/Q8_0/F32 launch unchanged.

Three sessions, two independently built shared libraries with distinct md5
selected by `LD_LIBRARY_PATH`, counterbalanced order, whole-process launches,
`llama-bench tg128 -r 2`:

| session | control | wide | median ratio |
| --- | --- | --- | --- |
| A, 6 rounds | 75.92-76.13 | 76.64-77.10 | 1.0123 |
| B, 6 rounds | 75.91-76.15 | 76.93-77.14 | 1.0131 |
| C, 5 rounds | 75.90-76.09 | 76.76-76.92 | 1.0120 |

17 of 17 rounds disjoint. pp512 overlaps in both directions over 6 rounds
(2339-2355 control, 2347-2358 wide): decode-shape mmvq is not on the prefill
path, which runs MMQ. Gate ppl `-c 512 --chunks 8` reads **3.9306** on five
consecutive loads of each arm, equal to the parent frontier's recorded value.

Eight warps was swept in the same harness and is an occupancy cliff:
72.73-72.93 against 75.91-76.15, **-4.2%**. Four is the setting.
`GGML_CUDA_DISABLE_SM121_WIDE_ROW_BLOCK` restores two warps.

## 0023: guard the folded add chain against a cross-block overlap

A correctness guard, credited to the round-51 work on the sibling track
`qwen3.6-35b-a3b-gguf-r9700-v1`, carried here because this series ships the
same 0006 fold unguarded. It carries no performance claim.

0006's `rms_norm_pre_add` issues one block per row: block r writes `add_dst`
row r in loop 1 and `dst` row r in loop 2, with only a block-scope
`__syncthreads()` between them, and reads the add operand rows in loop 1.
Nothing orders one block's loop 2 against another block's loop 1, so a write
range belonging to block r must not intersect a read range belonging to a
different block. The existing check tests `dst` against the stored add result
only. Since every writer and reader is contiguous and the same shape, their row
grids are congruent, and an overlap is safe only when the two tensors start at
exactly the same address - the in-place residual add the kernel already
assumes. On the sibling track 9 of 79 folds per graph overlap with bases 128
bytes apart, and the fold there produces a 0.107% spread over six distinct
perplexity values.

**On this track the guard declines nothing.** `GGML_RMS_PRE_ADD_AUDIT=1` dumps
each candidate's ranges and the verdict:

| shape | folds seen | declined |
| --- | --- | --- |
| `-c 512 --chunks 8` | 79 at nrows 2, 632 at nrows 512 | 0 |
| `-c 2048 --chunks 2` | 79 at nrows 2, 632 at nrows 512 | 0 |
| `-c 16384` long window | 79 at nrows 2, 2528 at nrows 512 | 0 |
| `-c 512 -b 512 -ub 1` | 828 at nrows 1 | 0 |

and the gate reads 3.9306 on five loads guarded and 3.9306 on five unguarded.
This track's allocator does not produce the sibling's offset pair, so the fold
is correct here by accident of layout rather than by construction; the guard
makes it correct by construction at zero declined folds and therefore zero
cost. `GGML_CUDA_PRE_ADD_NORM_UNSAFE=1` drops it.
