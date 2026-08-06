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
- The R9700 grouped-mmvf lever (small F32 matvec pairs) is untested on
  sm_121; the grouped-mmvq experience says do not assume it transfers.
