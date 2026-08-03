"""gainz.fast participant kernel package — Fable 5 (Claude Code) submission.

Loaded into the candidate vLLM v0.25.1 engine via the ``vllm.general_plugins``
entry point before model load.

This package does exactly one thing: for the NVFP4 MoE Triton EMULATION
experts (the path the batch-invariant engine uses on GB10), it right-sizes
the fused kernel's M-tile at decode shapes — ``BLOCK_SIZE_M`` 64 -> 16 when
the MoE forward carries at most 16 tokens (see ``moe_tile.py``).  At serial
decode each routed expert block holds a single valid token padded to
BLOCK_SIZE_M rows, so the stock 64-row tile spends 63/64 of its MMA work on
padding; 16-row tiles cut that 4x.  BLOCK_SIZE_N/BLOCK_SIZE_K and therefore
the entire K-reduction path of ``tl.dot`` are untouched, and the outputs
were verified bitwise identical on this GB10 across shapes and seeds.
Prefill-shaped batches (> 16 tokens) keep the stock config bit-for-bit.

Nothing else is touched: no GC/GIL/env tweaks (a previous submission showed
those can destabilize the correctness gate), no new kernels on the ranked
path, no new dependencies.
"""


def register() -> None:
    """vLLM general-plugin entry point: install the MoE M-tile override."""
    from gainz_kernels import moe_tile

    moe_tile.install()
