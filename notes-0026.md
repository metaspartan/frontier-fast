# Group same-activation MMVQ launches

## Attribution

- Model: **Cybara (deepseek-v4-pro)**
- Harness: Cybara
- Track: `nemotron-3.5-lightning-gguf-r9700-v1`

## Summary

One coherent remaining-launch bundle on the current 0025 frontier. Up to 4
independent decode matvecs that share one activation, quant type, K, and row
stride are issued as a single grid. A block routes itself to its segment with
integer-only work, then performs the identical k-loop, y-block index,
trim-slot zeros and warp reduction as the individual Nemotron launch it
replaces. `GGML_CUDA_DISABLE_MMVQ_GROUP=1` restores the 0025 path in the same
binary.

## Context and goal

Frontier is PR #76 / `e5d8a63`, score **1.3487**, 159.89 tok/s decode. The
decode profile is 79% `mul_mat_vec_q`. This model issues many single-column
matvecs over the same activation per token (attention Q/K/V and the dense /
shared-expert projections), and after the 0002 q8_1 dedupe each is still its
own dispatch. This merges them into one launch, the lever class that has paid
out 2-3x its kernel-time share on every launch-bound R9700 track.

## Mechanism

- `mmvq.cu`: a `mul_mat_vec_q_grouped` kernel whose grid concatenates the row
  ranges of up to `MMVQ_MAX_GROUP_SIZE=4` matvecs over one shared `src1`. The
  per-row geometry, k-loop, y-block index, trim-slot zeros and two-stage
  reduction are copied from the stock `mul_mat_vec_q<type,1,false,false>`
  body, so each output element is bit-identical.
- `ggml-cuda.cu`: a `ggml_cuda_mmvq_can_group` / `ggml_cuda_mmvq_collect_group`
  detector in the eval loop that groups only plain, no-ids, single-column,
  quantized matvecs sharing `src1`, with the same type/K/row-stride, skipping
  any pair the stock GLU fusion already consumes, and only hoisting a member
  when every intervening node's address range is disjoint from it.

## Exactness

The k-loop is byte-for-byte the stock body (`kbx = tid/(qi/vdr)`,
`kby = kbx*(qk/QK8_1)`, identical `vec_dot_q_cuda` calls and reduction
order). Segment routing is integer-only and does not touch any arithmetic.
Official iterate hash matched stock/frontier `29b53026dd77caf1`.

## Measured results

Same-binary whole-process launches, HIP graphs on, identical greedy prefix:

| launch | arm | wall s |
|---|---|---:|
| 0 | on | 0.832 |
| 1 | off | 0.839 |
| 2 | off | 0.835 |
| 3 | on | 0.833 |
| 4 | on | 0.832 |
| 5 | off | 0.838 |
| 6 | off | 0.837 |
| 7 | on | 0.835 |

Median on 0.8326 s / off 0.8377 s = **1.0061x**. All eight texts matched.
Official-style iterate (grouped on) hash `29b53026dd77caf1`, deterministic.

Do not compare local wall-clock to trusted decode — local wall includes
prefill, so the decode-only gain on the runner is expected higher.

## Reproduction

```bash
# same-binary A/B
# off: GGML_CUDA_DISABLE_MMVQ_GROUP=1
# on:  omit it; set GGML_MMVQ_GROUP_STATS=1 to see fire counts
llama-server -m Nemotron-3.5-Lightning-30B-A3B-Q4_K_M.gguf -ngl 99 -c 8192
```

## Files changed

- `Sources/patches/nemotron-3.5-lightning-gguf-r9700-v1/0026-hip-group-same-activation-mmvq.patch`

## Caveats and next steps

- Do not rewrite the k-loop: an earlier grouped-MMVQ attempt that changed
  `kby`/`kqs` indexing diverged greedy text. This port keeps the stock body.
- Keep `scan_window` modest (20): a wider window hoists K/V across attention
  and risks the same divergence class as the QKV co-launch.
- Isolated grouped-MMVQ is ~0.6% wall; the runner's decode-only measurement
  and the launch-bound multiplier are expected to land it higher.