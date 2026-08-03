# Model optimizations

This directory is for model-specific inference optimizations. Participants can implement:
- Custom decode kernels
- Speculative decoding config
- Attention optimizations
- Memory layout improvements

All changes must preserve exact greedy output.

## Laguna-S-2.1-NVFP4: re-derivation of the MoE decode M-sub-tile

`Sources/kernels/gainz_kernels/moe_tile.py` was derived and verified on
Laguna-XS-2.1. Laguna-S-2.1 is the same NVFP4 MoE architecture but not the
same shapes, so the derivation is re-done here at S's config rather than
assumed to carry over.

| | Laguna-XS-2.1 | Laguna-S-2.1 |
|---|---|---|
| layers | 40 | 48 |
| hidden_size | 2048 | 3072 |
| num_experts | 256 | 256 |
| num_experts_per_tok | 8 | 10 |
| moe_intermediate_size | 512 | 1024 |
| MoE GEMM1 (N, K) | (1024, 2048) | (2048, 3072) |
| MoE GEMM2 (N, K) | (2048, 512) | (3072, 1024) |
| weight scale group_size | 16 | 16 |

### Why the sub-tile is sound at S

The soundness argument is a property of `moe_align_block_size`, not of the
model: it pads each routed expert's token list up to `BLOCK_SIZE_M` and packs
that expert's real rows at the FRONT of its block. With a forward carrying at
most `M` tokens, no expert can be routed more than `M` rows, so under the
guard `M <= 16` every block holds at most 16 valid rows and rows 16..63 are
padding whose loads were already `other=0.0`-masked and whose stores were
already `mask`-ed off.

The larger `top_k` (10 vs 8) does not weaken this: `top_k` sets how many
*blocks* exist (one expert each), not how many rows any one expert receives.
At serial decode, S routes 1 token to 10 experts, so 10 blocks each hold
exactly 1 valid row out of 64 — the padding fraction is if anything slightly
worse than XS, i.e. there is more dead work to remove.

`get_default_config` under `VLLM_BATCH_INVARIANT=1` returns the same pinned
`BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32, GROUP_SIZE_M=8` for S as
for XS, so the sub-tile applies unchanged, and every prefill-shaped forward
still falls through to the stock launcher.

### Measured at S shapes

Offline, in the pinned `vllm/vllm-openai:v0.25.1` container on the GB10, with
the stock launcher and the sub-tiled launcher driven from the SAME
`moe_align_block_size(bs=64)` output and the same inputs, comparing the uint16
bit patterns of both fused GEMMs' bf16 outputs at M in {1, 2, 4, 8, 16} across
4 seeds (E=256, top_k=10, hidden=3072, moe_intermediate=1024):

- **bitwise identical in all 20 cases, both GEMMs.**
- M=1 timing, CUDA events, 300 iterations, per MoE layer pair:
  stock 0.4757 ms -> sub-tile 0.4308 ms (**1.1044x**).

S's MoE is a larger share of its step than XS's, but its dense/attention bf16
work grows too, so the end-to-end decode gain is left to the runner to
measure rather than extrapolated here.

## Negative result: grouped `b_scale` loads are NOT value-preserving

Recorded so it is not retried. Rationale looked airtight and the isolated-GEMM
evidence was perfect, but the engine disagrees.

A k-iteration spans `BLOCK_SIZE_K_PACKED=16` packed weight bytes while one
fp8-e4m3 scale covers `group_size_packed=8` of them, so only 2 distinct scale
columns exist per k-iteration and the stock kernel materialises each of them 8
times. Because `BLOCK_SIZE_K_PACKED` is an exact multiple of
`group_size_packed`, the stock column index rewrites exactly to
`NUM_SCALE_GROUPS * k + j // group_size_packed`, so loading the
`[BLOCK_SIZE_N, 2]` tile of distinct scales and `broadcast_to`-replicating it
back to `[BLOCK_SIZE_N, 16]` reproduces the identical fp32 value in every
lane.

Evidence for:
- Isolated GEMM, both fused GEMMs, uint16 bit patterns, M in {1,2,4,8,16} x 4
  seeds, at BOTH XS and S shapes: **0 bit differences**.
- 1.278x further on the XS MoE layer pair (0.0937 -> 0.0733 ms at M=1),
  1.032x on the S pair (0.4308 -> 0.4175 ms).

Evidence against (three-boot engine A/B, all arms with
`--num-gpu-blocks-override` pinned so every arm reported an identical
329,326-token KV cache, control arm proving the harness noise floor is zero):

| arm | teacher-forced vs control | 128-token greedy decode vs control |
|---|---|---|
| kernels, grouped scale OFF (merged frontier) | 0/128 | byte-identical |
| kernels, grouped scale ON | 15/128, first at 59 | diverges at char 58 |

Same container, same flags, same package, one environment variable apart. The
values fed to `tl.dot` are provably identical, so the difference must come
from the layout change that `reshape`/`broadcast_to` forces on the `b`
operand altering how Triton lowers the `tl.dot` accumulation — a
reduction-order effect that an output-level GEMM comparison at these shapes
does not expose.

Lesson: an isolated-GEMM bitwise check is necessary but NOT sufficient. Any
change that alters the *layout* of a `tl.dot` operand — even one that provably
preserves every value — has to clear a full engine A/B with the KV cache
pinned before it can be trusted.

### Pinning the KV cache is required for a usable local A/B

Without `--num-gpu-blocks-override`, consecutive boots on this box land on
different KV cache sizes (measured: 711,287 / 709,361 / 709,665 tokens across
three back-to-back boots) and that difference alone flips tokens: a provable
no-op arm scored 9/128 against the control. Pinning the block count made every
arm report 329,326 tokens and dropped the no-op arm to 0/128. Pin it, or the
local gate cannot resolve anything.
