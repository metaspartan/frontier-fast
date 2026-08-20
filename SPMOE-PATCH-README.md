# SP-MoE draft-stage expert-prefetch — implementation patch (qwen3.6-35b-a3b)

## What this does
In `qwen35moe.cpp` `build_layer_ffn`, when `GAINZ_EXPERT_PREFETCH` is set AND the
context is a speculative MTP/D-STAGE DRAFT context (`LLAMA_CONTEXT_TYPE_MTP`),
widen the routed-expert count to `n_expert_used + 2` for the DRAFT forward only.

Why this is SP-MoE and correctness-safe:
- SP-MoE: acquire MORE expert rows than the verify round strictly needs, during
  the cheap non-gated draft pass, so those HBM rows are resident when the
  verify GEMM runs — prefetching expert weight bandwidth.
- DraftExpert: the self-drafter loads a broader routed set -> higher acceptance.
- Safety: only the TRUNK (verify) output is correctness-gated; drafts only get
  accepted/rejected. Widening draft coverage cannot change greedy output.

## Env + scope
- `GAINZ_EXPERT_PREFETCH=1` to enable; unset = byte-identical stock.
- Applied only when `cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP` (draft), so
  bare decode and prefill are unaffected. Trunk verify unchanged.

## Status
AUTHORED. NOT yet applied/built/measured. Requires a ~40-60 min HIP build on
rebase27 (pinned 2b63e0610) + a sanctioned GPU window (suspend endpoint, take
lock, build, measure qwen3.6+DFlash at depth 6 with prefetch on vs off, restore).
This is the one remaining lever that could plausibly exceed the 233.3 frontier.
