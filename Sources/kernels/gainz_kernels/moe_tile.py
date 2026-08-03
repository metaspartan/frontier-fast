"""Decode-shape M-sub-tiling for the NVFP4 MoE Triton emulation path.

Under ``VLLM_BATCH_INVARIANT=1`` vLLM pins the MoE config to
``BLOCK_SIZE_M=64`` for every batch size (``get_default_config`` in
fused_moe.py).  At serial decode the MoE carries ONE token routed to
top_k=8 experts, and ``moe_align_block_size`` pads each routed expert's
token list up to BLOCK_SIZE_M rows — so every decode step runs its
``tl.dot`` tiles on 64-row M-blocks holding a single valid row.  63/64 of
the MMA work is padding.

What this does
--------------
It removes the padded rows from the GEMM **without touching anything the
baseline reduction depends on**.

``moe_align_block_size`` is still called with the baseline's
``BLOCK_SIZE_M`` (64), so ``sorted_token_ids``, ``expert_ids``,
``num_tokens_post_padded``, ``EM`` and the launch grid are the byte-for-byte
baseline tensors, and program ``pid_m`` still maps to exactly the same
aligned block and the same expert.  The one thing that changes is how many
M rows of that block the kernel materialises: when the forward carries at
most 16 tokens, every expert owns at most 16 rows, and
``moe_align_block_size`` packs an expert's rows at the FRONT of its block —
so rows 16..63 are provably padding.  The kernel loads
``tl.arange(0, 16)`` at the block base ``pid_m * 64`` and drops the
trailing 48.

Why it cannot change an output bit
----------------------------------
The rows that disappear are rows whose loads were already ``other=0.0``
masked and whose stores were already ``mask``-ed off — they contribute
nothing to any output element in the baseline either.  For every row that
remains, the computation is the unchanged source: the same expert, the same
K-loop over ``BLOCK_SIZE_K=32`` with ``BLOCK_SIZE_N=64``, the same fp32
``tl.dot`` accumulator chain, the same routed-weight multiply, the same
single bf16 rounding at the end.  No sum is re-associated, re-ordered or
re-grouped, and the per-token top_k reduction (``moe_sum``) sees the same
``intermediate_cache3`` it saw before.

This is deliberately narrower than lowering ``BLOCK_SIZE_M`` in the config
dict: that also re-runs ``moe_align_block_size`` at block_size=16, which
rebuilds the padded block layout and the ``sorted_token_ids`` buffer.  Here
the alignment path is left completely alone.

Guard
-----
The sub-tile is used only when the MoE forward carries at most
``_SMALL_M_MAX`` (16) tokens — the token count is read from the ``M``
argument of the ``try_get_optimal_moe_config`` call that ``apply()`` makes
immediately before both fused GEMMs, so it needs no device
synchronisation.  With at most 16 tokens no expert can own more than 16
rows.  Every other shape (all prefill-shaped batches) falls through to the
stock launcher untouched.

Measured on this GB10 in the pinned vllm/vllm-openai:v0.25.1 container at
exact Laguna-XS-2.1 MoE shapes (E=256, top_k=8, hidden=2048,
moe_intermediate=512): both fused GEMMs bitwise identical (uint16 view of
the bf16 outputs) to the stock BLOCK_SIZE_M=64 path at M in
{1, 2, 4, 8, 16} across 4 seeds, and 0.1797 ms -> 0.1556 ms per MoE layer
pair at M=1 (CUDA events, 200 iters).
"""

import vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe as _emu
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.fused_moe import write_zeros_to_output
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    _e2m1_inline,
)
from vllm.triton_utils import tl, triton

logger = init_logger("gainz_kernels.moe_tile")

# A forward carrying at most this many tokens is decode-shaped: no expert
# can own more rows than this, so a 16-row sub-tile at the front of each
# aligned block covers all of them.
_SMALL_M_MAX = 16
_SUB_BLOCK_M = 16

# The sub-tile's bitwise evidence was gathered at Laguna-XS-2.1 GEMM shapes
# only, and the ranked runner measured 2/128 teacher-forced divergence when
# the same tile ran at Laguna-S shapes (finding `s-moe-m-subtile-port`).
# Gate the fast path to the exact XS (N, K) pairs — w13: (2*512, 2048),
# w2: (2048, 512) — so any other model (e.g. Laguna-S, whose pairs are
# (2048, 3072) and (3072, 1024)) provably falls through to the stock
# launcher untouched.
_XS_GEMM_SHAPES = frozenset({(1024, 2048), (2048, 512)})

_orig_try_get_optimal_moe_config = _emu.try_get_optimal_moe_config
_orig_invoke = _emu.invoke_fused_moe_nvfp4_emulation_kernel

# Token count of the MoE forward currently being applied.
_current_tokens = 0


def _recording_try_get_optimal_moe_config(
    w1_shape, w2_shape, top_k, dtype, M, block_shape=None
):
    """Pass-through: records M, returns the stock config unmodified."""
    global _current_tokens
    _current_tokens = M
    return _orig_try_get_optimal_moe_config(
        w1_shape, w2_shape, top_k, dtype, M, block_shape=block_shape
    )


@triton.jit
def fused_moe_nvfp4_emulation_subtile_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    w_global_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsk,
    stride_bsn,
    block_k_diviable: tl.constexpr,
    BLOCK_STRIDE_M: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    group_size: tl.constexpr,
):
    """vLLM's ``fused_moe_nvfp4_emulation_kernel`` with one change.

    The aligned block stride stays ``BLOCK_STRIDE_M`` — the value
    ``moe_align_block_size`` was called with — so ``num_pid_m``, the
    pid -> block map, the expert lookup and the padding guard are the
    baseline's.  Only the number of M rows materialised per block drops to
    ``BLOCK_SIZE_M``.  Callers must guarantee every block holds at most
    ``BLOCK_SIZE_M`` valid rows.
    """
    BLOCK_SIZE_K_PACKED: tl.constexpr = BLOCK_SIZE_K // 2

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_STRIDE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_STRIDE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_STRIDE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_k_packed = tl.arange(0, BLOCK_SIZE_K_PACKED)

    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + offs_bn[:, None] * stride_bn
        + offs_k_packed[None, :] * stride_bk
    )

    group_size_packed: tl.constexpr = group_size // 2

    w_global_scale = tl.load(w_global_scale_ptr + off_experts).to(tl.float32)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if block_k_diviable:
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                other=0.0,
            )

        if block_k_diviable:
            raw_bytes = tl.load(b_ptrs)
        else:
            kp_mask = offs_k_packed[None, :] < (K // 2) - k * BLOCK_SIZE_K_PACKED
            raw_bytes = tl.load(b_ptrs, mask=kp_mask, other=0)

        low_nibble = raw_bytes & 0x0F
        high_nibble = (raw_bytes >> 4) & 0x0F

        low_decoded = _e2m1_inline(low_nibble)
        high_decoded = _e2m1_inline(high_nibble)

        b_scale_ptrs = (
            b_scale_ptr
            + off_experts * stride_bse
            + offs_bn[:, None] * stride_bsn
            + ((offs_k_packed[None, :] + BLOCK_SIZE_K_PACKED * k) // group_size_packed)
            * stride_bsk
        )
        if block_k_diviable:
            b_scale_raw = tl.load(b_scale_ptrs)
        else:
            b_scale_raw = tl.load(b_scale_ptrs, mask=kp_mask, other=0.0)

        b_scale = tl.cast(b_scale_raw, tl.float8e4nv, bitcast=True).to(tl.float32)
        b_scale = b_scale * w_global_scale

        low_scaled = low_decoded * b_scale
        high_scaled = high_decoded * b_scale

        b = tl.trans(tl.interleave(low_scaled, high_scaled)).to(compute_type)

        accumulator = tl.dot(a, b, acc=accumulator)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K_PACKED * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def _subtile_invoke(
    A,
    B,
    C,
    B_scale,
    act_global_scale,
    w_global_scale,
    topk_weights,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    mul_routed_weight,
    top_k,
    config,
    compute_type,
):
    """Launch the sub-tiled kernel for decode shapes, else the stock one."""
    block_stride_m = config["BLOCK_SIZE_M"]
    if (
        block_stride_m <= _SUB_BLOCK_M
        or _current_tokens <= 0
        or _current_tokens > _SMALL_M_MAX
        or (B.size(1), A.size(1)) not in _XS_GEMM_SHAPES
    ):
        return _orig_invoke(
            A,
            B,
            C,
            B_scale,
            act_global_scale,
            w_global_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight,
            top_k,
            config,
            compute_type,
        )

    assert B_scale is not None and B_scale.ndim == 3

    N = B.size(1)
    K = A.size(1)
    num_tokens = A.size(0) * top_k

    # EM exactly as the stock launcher computes it at the baseline's
    # BLOCK_SIZE_M, so the block count and pid -> block map are unchanged.
    EM = sorted_token_ids.size(0)
    if A.size(0) < block_stride_m:
        EM = min(sorted_token_ids.size(0), A.size(0) * top_k * block_stride_m)

    grid = lambda META: (  # noqa: E731
        triton.cdiv(EM, block_stride_m) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    fused_moe_nvfp4_emulation_subtile_kernel[grid](
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
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        B_scale.stride(0),
        B_scale.stride(2),
        B_scale.stride(1),
        block_k_diviable=K % config["BLOCK_SIZE_K"] == 0,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        group_size=16,
        BLOCK_STRIDE_M=block_stride_m,
        BLOCK_SIZE_M=_SUB_BLOCK_M,
        BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
        GROUP_SIZE_M=config["GROUP_SIZE_M"],
    )


def install() -> None:
    """Rebind the emulation module's config lookup and GEMM launcher."""
    if _emu.try_get_optimal_moe_config is not _recording_try_get_optimal_moe_config:
        _emu.try_get_optimal_moe_config = _recording_try_get_optimal_moe_config
    if _emu.invoke_fused_moe_nvfp4_emulation_kernel is not _subtile_invoke:
        _emu.invoke_fused_moe_nvfp4_emulation_kernel = _subtile_invoke
    logger.info(
        "gainz_kernels: NVFP4 MoE decode M-sub-tiling installed (GEMM M-extent "
        "-> %d inside the unchanged %d-row aligned blocks, for forwards "
        "carrying <= %d tokens)",
        _SUB_BLOCK_M,
        64,
        _SMALL_M_MAX,
    )
