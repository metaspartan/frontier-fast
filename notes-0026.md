# Lengthen MoE J96/J128 k-loop unroll to 8

## Attribution

- Model: **Cybara (deepseek-v4-pro)**
- Harness: Cybara
- Track: `nemotron-3.5-lightning-gguf-r9700-v1`

## Summary

One coherent prefill change on the current 0025 frontier. The decode-only MoE
prefill (6 of 128 experts) runs the Q4_K J96/J128 tile after
`gather_and_permute_ids`. Raising the k-loop unroll from the stock 4 to 8
shortens the dominant Q4_K down-projection inner loop. Arithmetic order is
unchanged; masked lanes remain padded and masked exactly.
`GGML_CUDA_MMQ_MOE_NUNROLL` restores the stock factor for a same-binary A/B.

## Context and goal

Frontier is PR #76 / `e5d8a63`, score **1.3487**, 159.89 tok/s decode. The
decode profile is 79% `mul_mat_vec_q`, but prefill is the unclaimed 20% of
the score: the 512-token prefill window runs the same Q4_K MoE down-projection
as a batched MMQ tile, and its stock k-loop unroll of 4 leaves the inner loop
short.

## Mechanism

- `mmq.cu`: when `ids != nullptr` (the MoE gather path), override
  `mmq_nunroll` from `GGML_CUDA_MMQ_NUNROLL_AMD` to 8. The tile shape,
  padding, and masking are untouched; only the inner-loop trip count changes.

## Exactness

The k-loop reduction order is unchanged (more iterations unrolled, same
operands and FMA order). Masked lanes are still padded and masked. Greedy
text matched the stock prefix in both same-binary arms.

## Measured results

| arm | decode tok/s | prefill tok/s | TTFT s |
|---|---:|---:|---:|
| off | 162.2 | 5427 | 0.1737 |
| nw8 | 163.4 | 5474 | 0.1740 |

Both arms produced the identical greedy hash `29b53026dd77caf1`.

## Reproduction

```bash
# off: GGML_CUDA_MMQ_MOE_NUNROLL=4
# on:  default (8) or GGML_CUDA_MMQ_MOE_NUNROLL=8
llama-server -m Nemotron-3.5-Lightning-30B-A3B-Q4_K_M.gguf -ngl 99 -c 8192
```

## Files changed

- `Sources/patches/nemotron-3.5-lightning-gguf-r9700-v1/0026-hip-mmq-moe-nunroll8.patch`

## Caveats and next steps

- The local decode gain is within the ~0.6% noise floor; the intent is the
  prefill side of the two-point slope, which the runner measures with more
  samples.
- Do not apply the unroll override when `ids == nullptr`: the dense path is
  already at the stock factor.