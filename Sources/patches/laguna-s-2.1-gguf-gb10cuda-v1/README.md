# laguna-s-2.1-gguf-gb10cuda-v1 — patch series

Applied in order against pinned llama.cpp **b10237**
(`2b63e0610bbc2be990ae1360d5256efcdc3f9efb`).

| # | Patch | Measured on this track |
| --- | --- | --- |
| 0001 | `cuda-mmvq-group-same-activation-matvecs` | **+1.868% decode** (in-process paired A/B, 12 rotated cycles, no-op floor 0.99994) |

The series previously held the eleven patches from the XS GB10 CUDA track.
Those were placed here by analogy when the shared series was split per track,
not because anything had measured them on Laguna S. They were then measured on
S: **+0.64%, inside noise**, and removed.

## What 0001 does

Laguna S decode issues several quantized matvecs over the same activation
vector — `attn_q`/`attn_k`/`attn_v` all read the post-`attn_norm` hidden state,
and the shared-expert gate/up projections both read the post-`ffn_norm` one.
0001 merges them into a single launch whose grid concatenates their row ranges.
Per-row arithmetic is unchanged, so outputs are byte-identical.

It makes **no ggml graph change** — the merge happens entirely inside the CUDA
backend, by issuing later members from the position of the first.

## Two things this track has now established about that lever

**1. It is not launch-count reduction.** Merging the *large* `attn_q`
projection in with K and V helps further (+1.868%) than merging only the small
ones (+1.109%). GB10 decode is 93.4% kernel time and removing launches is worth
about 0.6% here. The win is that one larger grid reaches a memory rate that
several smaller grids do not: measured per-launch, `attn_q` (9216 rows, 30.1 MB)
runs at 216 GB/s while `attn_k` (1024 rows, 3.34 MB) manages 159 GB/s and
`attn_gate` (72 rows) only 37 GB/s.

**2. Segment selection must not use a runtime index.** The natural form

```cuda
seg = ...;  vx = args.vx[seg];
```

makes nvcc give the kernel an **80-byte stack frame** — it materialises the
whole kernel-parameter struct in per-thread local memory so it can be indexed.
That is ~10 kB of local traffic per block, and it *inverts the result*:

| | mode 1 (small only) | mode 2 (all eligible) |
| --- | --- | --- |
| runtime index — `STACK:80` | **−4.90%** | **−3.80%** |
| unrolled constant index — `STACK:0` | **+1.109%** | **+1.868%** |

Same binary, same harness, same day. Every stock `mul_mat_vec_q` instantiation
has `STACK:0`; the grouped kernel was the only one in `libggml-cuda.so` with a
stack frame. **`cuobjdump -res-usage <lib> | grep -A1 <kernel>` is the check** —
run it on any new CUDA kernel here before concluding anything from a timing.
The earlier −15% attributed to this lever on the XS GB10 track is very likely
the same spill, not a property of the device.

## Before you trust a number on this track

Laguna S has a decode-rate state that is **fixed for a process launch and
independent of your code**: bimodal, roughly 7% wide (~23.4-24.1 vs
~25.5-25.6 tok/s), while the reading *within* a launch is rock steady. Low
variance inside one launch is the artifact's signature, not evidence that your
effect is real — a two-round A/B here measured +6.3% before rounds 3-5 erased
it.

Do not fight it with more boots. Make your knob **re-readable at runtime** and
run both arms inside ONE process on the same pages, bracketed by the reference
arm and rotated. `GGML_CUDA_MMVQ_GROUP_FILE` in 0001 is an example: it re-reads
the policy once per graph evaluation. That protocol gave a same-binary no-op
floor of 0.99994 and resolved a 1.1% effect with a spread of 0.0035 — on a box
where whole-process A/B could not resolve 7%. Both A/B runs quoted above ran in
a single server launch each; the two runs happened to land in *different*
clusters (ref 25.45 vs 23.98 tok/s) and still agreed on the arm ratios.

## Knobs in 0001

- `GGML_CUDA_MMVQ_GROUP=0` — disable grouping (same-binary A/B control).
- `GGML_CUDA_MMVQ_GROUP_FILE=<path>` — re-read the policy from a file at every
  graph evaluation: `0` off, `1` small members only, `2` (default) any eligible.
- `GGML_MMVQ_GROUP_STATS=1` — log a line whenever the effective policy or the
  groups it forms change. With the shipped default on Laguna S this reports
  `mode=2 groups=93 members=229`.

## What is left on the table

`attn_gate` does **not** currently join the K/V group — only 3-member groups
form (`attn_q`, `attn_k`, `attn_v`), plus 2-member shared-expert pairs. Its
launch is the least efficient in the whole decode (72 rows, 0.235 MB, 6.3 µs,
37 GB/s) and it is rejected by the hoist-legality range check, not by type or
shape. Diagnosing that is worth roughly another 0.3 ms/token.

Larger remaining targets, from a per-launch nsys census (25 decode steps):

| Bucket | ms/token | achieved | ideal at 217 GB/s | waste |
| --- | --- | --- | --- | --- |
| Q4_K routed experts | 11.7 | 177 GB/s | 9.54 | **2.16 ms** |
| BF16 routed experts | 7.93 | 190 GB/s | 6.96 | 0.97 ms |
| shared expert gate/up/down | 3.03 | ~155 GB/s | 2.17 | 0.86 ms |
| attn q/o projections | 12.2 | ~217 GB/s | — | ~0 |

The Q4_K expert bucket is the biggest prize, but its obvious angle —
`q4k-expert-thread-utilisation` — is already recorded **dead** (both fixes
measured +0.006% and −0.072%).
