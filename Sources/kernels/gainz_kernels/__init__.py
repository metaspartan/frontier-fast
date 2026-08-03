"""gainz.fast participant kernel package — Fable 5 (Claude Code) submission.

Loaded into the candidate vLLM v0.25.1 engine via the ``vllm.general_plugins``
entry point before model load.

This package removes provably-dead work from the NVFP4 MoE Triton
EMULATION experts (the path the batch-invariant engine uses on GB10),
WITHOUT touching the block alignment that the expert reduction is built
on.  Two cuts, both strictly value-preserving:

1. The padded M rows are dropped from the two fused decode GEMMs.
2. Each NVFP4 weight scale is loaded, bitcast and rescaled once per
   k-group instead of once per packed weight byte.

``moe_align_block_size`` is still called with the baseline
``BLOCK_SIZE_M=64``, so ``sorted_token_ids``, ``expert_ids``,
``num_tokens_post_padded``, ``EM``, the launch grid and the
program -> block -> expert mapping are the baseline tensors bit for bit.
Only the M-extent materialised inside each block shrinks 64 -> 16, which
is sound whenever the forward carries at most 16 tokens: no expert can
then own more than 16 rows, and ``moe_align_block_size`` packs an
experts rows at the front of its block, so rows 16..63 are provably
padding whose loads were zero-masked and whose stores were masked off.

The second cut exploits that one k-iteration spans 16 packed weight
bytes while a single fp8-e4m3 scale covers 8 of them: only 2 distinct
scale columns exist per iteration, and the stock kernel materialises
each of them 8 times. The distinct columns are loaded once and
``broadcast_to``-replicated. A broadcast copies a value, so every lane
of the reconstructed scale tile holds the identical fp32 datum the stock
kernel held there.

Every surviving row is computed by the unchanged source: same expert,
same K-loop, same fp32 accumulation order, same routed-weight multiply,
same single bf16 rounding. No sum is re-associated or re-ordered. See
``moe_tile.py`` for the derivations and the measured bitwise evidence at
both Laguna-XS-2.1 and Laguna-S-2.1 MoE shapes.

Nothing else is touched: no GC/GIL/env tweaks (a previous submission
showed those can destabilize the correctness gate), no new dependencies,
and every non-decode shape falls through to the stock launcher.
"""


def register() -> None:
    """vLLM general-plugin entry point: install the MoE M-sub-tiling."""
    from gainz_kernels import moe_tile

    moe_tile.install()
