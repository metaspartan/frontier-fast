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
