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

## Measured dead ends (healthy box, 2026-08-08, do not re-buy)

- **Q6_K mmvq load-path engineering is FLAT (+-0.1%)**: vdr=2 and vdr=4
  variants (2/4 adjacent quant ints per vec_dot call, shared scale/ds/offset
  loads, exact integer dp4a combine) and `__ldcs` evict-first streaming on
  ql/qh all measured neutral in interleaved whole-process tg32 A/B (6-round
  ldcs median +0.07%). block_q6_K is 210 B = 2-byte aligned, so wide
  vectorized or cp.async loads are structurally impossible; issue-side MLP
  is NOT the limiter - the kernel rate is DRAM-side-determined.
- **Output-head requant is decode-NEUTRAL**: head Q6_K->Q5_K (68 MB/token
  cut, ppl +0.40% - inside band but thin margin) and Q6_K->Q5_1 (36
  MB/token, ppl -0.33%) both measured +0.0-0.2% tg32. The byte cut is
  exactly eaten by the target types' lower dense-shape kernel efficiency
  (Q6_K dense mmvq runs near the LPDDR5 ceiling; Q5_K/Q5_1 dense do not).
  Byte cuts only pay when the destination kernel is at least as efficient.
