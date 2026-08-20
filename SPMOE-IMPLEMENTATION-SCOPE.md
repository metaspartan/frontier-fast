# qwen3.6-35b-a3b R9700 — SP-MoE expert-prefetch implementation scope (ready-to-fire)

## Objective
Beat the 233.3 tok/s spec frontier / 162.2 tok/s kernel frontier on
qwen3.6-35b-a3b (R9700) via **expert prefetch during the cheap draft step**
(SP-MoE, arXiv 2510.10302): while the DFlash drafter proposes, load the
top-r routed experts for the NEXT verify round so their HBM fetch is hidden.

## Exact edit point (verified in source, pinned base 2b63e0610)
`src/models/qwen35moe.cpp` — `build_layer_ffn` (line ~497):
```cpp
ggml_tensor * moe_out = build_moe_ffn(cur,
    layer.ffn_gate_inp, layer.ffn_up_exps, layer.ffn_gate_exps, layer.ffn_down_exps,
    nullptr,
    n_expert, n_expert_used,
    LLM_FFN_SILU, true, hparams.expert_weights_scale,
    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
    nullptr, layer.ffn_gate_up_exps,
    layer.ffn_up_exps_s, layer.ffn_gate_exps_s, layer.ffn_down_exps_s);
```
The MoE gating (top-k of `ffn_gate_inp` → `n_expert_used` used experts) happens
inside `build_moe_ffn`. That is where the drafter's next-round expert selection
would be captured BEFORE the verify GEMM so those weights start loading during
the draft forward.

## Concrete design (patch sketch)
1. In the speculative path only (not bare decode), hoist the gating: compute
   `ggml_top_k(softmax(ffn_gate_inp * cur))` one step early and pass the chosen
   expert ids to the next `build_moe_ffn` so the expert GEMM reads begin during
   the draft/decode overlap.
2. Wrap the expert `ggml_get_rows`/`ggml_moe` downstream so the prefetched
   expert weight rows are resident when the verify GEMM lands.
3. Env-gated (e.g. `GAINZ_EXPERT_PREFETCH=1`) and only on `draft-dflash` +
   `draftMax=6` (the measured best depth), so correctness/bare paths are unchanged.
4. Gate the delta: llama.cpp tracks MLX/CUDA MoE subtree correctly; expect
   `--n-gpu-layers 99` plus the normal ppl-equivalence gate (target exact).

## Why it can beat both frontier axes
- Kernel frontier 162.2 removes expert-launch work but NOT the expert-weight
  HBM fetch (still paid once per routed expert).
- DFlash depth-6 measured 118 tok/s, dead alone — it adds draft tokens but does
  not hide the expert fetch.
- SP-MoE prefetch + top-r coverage hides that fetch AND raises acceptance by
  exposing more routed experts during draft — a different, compounding axis.

## Status
SCOPED + READY. Written up 2026-08-20. Requires: PEP's sanctioned suspend/restore
window (endpoint down ~1 hr for HIP build + ABBA vs 233.3), then submit.
Box: /home/ghost/rebase27 (clean pinned base, build tree present) is the target
worktree.
