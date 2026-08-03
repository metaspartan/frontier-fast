"""gainz.fast participant kernel package — Fable 5 (Claude Code) submission.

Loaded into the candidate vLLM v0.25.1 engine via the ``vllm.general_plugins``
entry point before model load.

This package does exactly one thing: for the NVFP4 MoE Triton EMULATION
experts (the path the batch-invariant engine uses on GB10) it drops the
padded M rows out of the two fused decode GEMMs, WITHOUT touching the
block alignment that the expert reduction is built on.

``moe_align_block_size`` is still called with the baseline
``BLOCK_SIZE_M=64``, so ``sorted_token_ids``, ``expert_ids``,
``num_tokens_post_padded``, ``EM``, the launch grid and the
program -> block -> expert mapping are the baseline tensors bit for bit.
Only the M-extent materialised inside each block shrinks 64 -> 16, which
is sound whenever the forward carries at most 16 tokens: no expert can
then own more than 16 rows, and ``moe_align_block_size`` packs an
experts rows at the front of its block, so rows 16..63 are provably
padding whose loads were zero-masked and whose stores were masked off.

Every surviving row is computed by the unchanged source: same expert,
same K-loop, same fp32 accumulation order, same routed-weight multiply,
same single bf16 rounding. No sum is re-associated or re-ordered. See
``moe_tile.py`` for the derivation and the measured bitwise evidence.

Nothing else is touched: no GC/GIL/env tweaks (a previous submission
showed those can destabilize the correctness gate), no new dependencies,
and every non-decode shape falls through to the stock launcher.
"""


def register() -> None:
    """vLLM general-plugin entry point: install the MoE M-sub-tiling."""
    from gainz_kernels import moe_tile

    moe_tile.install()
