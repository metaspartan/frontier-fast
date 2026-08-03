"""Decode-shape M-tile right-sizing for the NVFP4 MoE Triton emulation path.

Target: under ``VLLM_BATCH_INVARIANT=1`` the batch-invariant default MoE
config pins ``BLOCK_SIZE_M=64`` for every batch size (see
``get_default_config`` in vllm's fused_moe.py).  During serial decode the
MoE sees ONE token routed to top_k=8 experts, and ``moe_align_block_size``
pads every routed expert's token list up to BLOCK_SIZE_M rows — so each
decode step runs its ``tl.dot`` tiles on 64-row M-blocks that contain a
single valid row.  63/64 of the MMA work is padding.

The fix: when the MoE forward carries at most ``_SMALL_M_MAX`` tokens,
override ``BLOCK_SIZE_M`` down to 16 (the ``tl.dot`` minimum) for the
NVFP4 emulation experts' config lookup.  Alignment, grid and both fused
GEMMs then use 16-row blocks: 4x less padded MMA compute per step.

Why this cannot change any output bit:

- ``BLOCK_SIZE_M`` (and GROUP_SIZE_M) only select WHICH program computes a
  given output row and how much padding rides along.  The value of output
  element c[m, n] is the K-loop of ``tl.dot`` over a[m, :] and the
  dequantized b[:, n] with ``BLOCK_SIZE_N=64 / BLOCK_SIZE_K=32`` unchanged
  — the reduction path, operand values, fp32 accumulation order and final
  bf16 rounding are all identical.  Padding rows are masked out of loads
  and stores.
- Verified empirically on this GB10 in the pinned vllm/vllm-openai:v0.25.1
  container at exact Laguna XS 2.1 shapes (E=256, top_k=8, hidden=2048,
  moe_intermediate=512): bitwise equality (uint16 view of the bf16
  outputs) of both fused GEMMs between BLOCK_SIZE_M=64 and 16 at M in
  {1, 4} and =32/16 at M=512, across 5 random seeds.

Measured on the same silicon (CUDA events, 300 iters, per MoE layer):
  M=1:   BM=64 0.1285 ms  ->  BM=16 0.0996 ms   (saves 0.0289 ms/layer)
Projected over 39 MoE layers on a ~28.4 ms decode step: ~ +4.1% decode.
Prefill (M=512) keeps the stock config: BM=16 was measured SLOWER there
(5.10 ms vs 4.49 ms/layer) and BM=32's ~1% gain is inside the noise floor,
so shapes above the threshold are left completely untouched.

Scope: only the emulation module's own ``try_get_optimal_moe_config``
binding is patched, so the plain TritonExperts path and every other user
of that helper resolve the original function.
"""

import vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe as _emu
from vllm.logger import init_logger

logger = init_logger("gainz_kernels.moe_tile")

# Decode-shaped forwards only: the ranked serial harness decodes one
# sequence (M=1); anything up to 16 tokens still profits (verified M=4).
# Larger (prefill-shaped) batches keep the stock batch-invariant config.
_SMALL_M_MAX = 16
_SMALL_BLOCK_SIZE_M = 16

_orig_try_get_optimal_moe_config = _emu.try_get_optimal_moe_config


def _patched_try_get_optimal_moe_config(
    w1_shape,
    w2_shape,
    top_k,
    dtype,
    M,
    block_shape=None,
):
    config = _orig_try_get_optimal_moe_config(
        w1_shape, w2_shape, top_k, dtype, M, block_shape=block_shape
    )
    if M <= _SMALL_M_MAX and config.get("BLOCK_SIZE_M", 0) > _SMALL_BLOCK_SIZE_M:
        # Copy before overriding: never mutate a shared config dict.
        config = dict(config, BLOCK_SIZE_M=_SMALL_BLOCK_SIZE_M)
    return config


def install() -> None:
    """Rebind the emulation module's config lookup to the M-aware version."""
    if _emu.try_get_optimal_moe_config is _patched_try_get_optimal_moe_config:
        return
    _emu.try_get_optimal_moe_config = _patched_try_get_optimal_moe_config
    logger.info(
        "gainz_kernels: NVFP4 MoE decode M-tile right-sizing installed "
        "(BLOCK_SIZE_M -> %d for token counts <= %d)",
        _SMALL_BLOCK_SIZE_M,
        _SMALL_M_MAX,
    )
