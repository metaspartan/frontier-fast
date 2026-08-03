"""Pipeline-stage tuning for the NVFP4 MoE emulation kernel launches.

Scope discipline (learned from the rejected BLOCK_SIZE_M submission): this
patch changes NO dispatch shape.  ``moe_align_block_size`` inputs, EM, the
grid, every constexpr block size (BLOCK_SIZE_M/N/K, GROUP_SIZE_M) and all
tensor shapes are exactly the stock batch-invariant values.  The only
change is an explicit ``num_stages`` (and the default ``num_warps=4``) on
the kernel launch: how the compiled kernel overlaps its loads with
compute, never what it computes and never how work is laid out across
programs.

Verified bitwise on this GB10 in the pinned vllm/vllm-openai:v0.25.1
container at exact Laguna XS 2.1 shapes (E=256, top_k=8, hidden=2048,
moe_intermediate=512), stock BLOCK_SIZE_M=64 config, uint16 views of both
fused GEMM outputs:

  - decode  M=1  : num_stages in {1, 2, 3} and default — all identical
  - prefill M=512: num_warps in {4, 8} x num_stages in {2, 3, 4} — all
    identical

Measured medians (CUDA events, per MoE layer, w13+w2, stock config):

  decode  M=1  : default 0.1143 ms -> num_stages=1: 0.1096 ms
  prefill M=512: default 4.4120 ms -> num_stages=2: 4.2840 ms

Projected over the 39 MoE layers: ~+0.65% decode (0.18 ms of a ~28.4 ms
step) and ~+3% TTFT (5.0 ms of ~161 ms) — both inside the calibration
band, above the ~±0.6% noise floor in combination.

This module replaces ``invoke_fused_moe_nvfp4_emulation_kernel`` in the
emulation experts module with a copy of the original vLLM v0.25.1 launch
logic that adds ``num_warps=4`` (the default) and ``num_stages=1`` for
decode-shaped calls (<= 16 A-rows: the w13 GEMM sees M=1 and the w2 GEMM
M*top_k=8 at serial decode) or ``num_stages=2`` for larger calls.
"""

import torch

import vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe as _emu
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger("gainz_kernels.moe_stages")

_SMALL_ROWS_MAX = 16
_SMALL_NUM_STAGES = 1
_LARGE_NUM_STAGES = 2
_NUM_WARPS = 4


def _patched_invoke(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    B_scale: torch.Tensor,
    act_global_scale: torch.Tensor,
    w_global_scale: torch.Tensor,
    topk_weights: "torch.Tensor | None",
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict,
    compute_type: tl.dtype,
):
    """Original launch logic + explicit num_warps/num_stages."""
    assert B_scale is not None and B_scale.ndim == 3

    N = B.size(1)
    K = A.size(1)

    M = A.size(0)
    num_tokens = M * top_k

    EM = sorted_token_ids.size(0)
    if A.size(0) < config["BLOCK_SIZE_M"]:
        EM = min(
            sorted_token_ids.size(0),
            A.size(0) * top_k * config["BLOCK_SIZE_M"],
        )

    grid = lambda META: (  # noqa: E731
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    num_stages = _SMALL_NUM_STAGES if M <= _SMALL_ROWS_MAX else _LARGE_NUM_STAGES

    _emu.fused_moe_nvfp4_emulation_kernel[grid](
        A,
        B,
        C,
        B_scale,
        w_global_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        EM,
        num_tokens,
        A.stride(0),
        A.stride(1),
        # B is [E, N, K//2]: swap N and K strides so kernel indexes [K, N].
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        # B_scale is [E, N, K//group]: swap N and K strides likewise.
        B_scale.stride(0),
        B_scale.stride(2),
        B_scale.stride(1),
        block_k_diviable=K % config["BLOCK_SIZE_K"] == 0,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        group_size=16,
        BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
        GROUP_SIZE_M=config["GROUP_SIZE_M"],
        num_warps=_NUM_WARPS,
        num_stages=num_stages,
    )


def install() -> None:
    """Rebind the emulation module's kernel launcher to the staged version."""
    if _emu.invoke_fused_moe_nvfp4_emulation_kernel is _patched_invoke:
        return
    _emu.invoke_fused_moe_nvfp4_emulation_kernel = _patched_invoke
    logger.info(
        "gainz_kernels: NVFP4 MoE pipeline-stage tuning installed "
        "(num_stages %d for <=%d-row calls, %d otherwise; num_warps %d)",
        _SMALL_NUM_STAGES,
        _SMALL_ROWS_MAX,
        _LARGE_NUM_STAGES,
        _NUM_WARPS,
    )
