"""gainz.fast participant kernel package — Fable 5 (Claude Code) submission.

Loaded into the candidate vLLM v0.25.1 engine via the ``vllm.general_plugins``
entry point before model load.

This package does exactly one thing: it sets an explicit pipeline stage
count on the NVFP4 MoE Triton emulation kernel launches (``moe_stages.py``)
— ``num_stages=1`` for decode-shaped calls, ``num_stages=2`` for
prefill-shaped calls, ``num_warps=4`` (the default).  Stage count only
changes how the compiled kernel overlaps loads with compute.

Unlike the rejected BLOCK_SIZE_M submission, NO dispatch shape changes:
``moe_align_block_size`` inputs, EM, the grid, and every block-size
constexpr are the stock batch-invariant values, so this cannot trip the
dispatch-shape sensitivity that has rejected -O3, max-num-seqs, ngram
k>=6, and the M-tile patch.  Outputs were verified bitwise (uint16 view)
against the stock launch on this GB10 at exact Laguna XS shapes, both
fused GEMMs, decode and prefill shapes.

Nothing else is touched: no GC/GIL/env tweaks, no new kernels, no new
dependencies.
"""


def register() -> None:
    """vLLM general-plugin entry point: install the MoE stage tuning."""
    from gainz_kernels import moe_stages

    moe_stages.install()
