#!/usr/bin/env bash
# Apply the SP-MoE gate to qwen35moe.cpp via python (robust, no fragile git diff).
# Idempotent: safe to run twice. Verifies at the end.
set -uo pipefail
F=/home/ghost/rebase27/src/models/qwen35moe.cpp
echo "=== backup ==="
cp "$F" "$F.spmoe.bak"
echo "=== apply via python ==="
python3 - <<'PY'
f='/home/ghost/rebase27/src/models/qwen35moe.cpp'
s=open(f).read()

marker='    GGML_ASSERT(model.layers[il].ffn_gate_inp != nullptr);'
gate='''
    // SP-MoE-style draft-stage expert prefetch. Env-gated, applies only on the
    // speculative MTP draft context: widen routed-expert count so the top
    // experts' rows are resident for the verify round. Draft output is never
    // gated (only trunk verify output is), so greedy output is unchanged.
    const bool draft_prefetch = getenv("GAINZ_EXPERT_PREFETCH") != nullptr &&
                                cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP;
    const int64_t n_used = draft_prefetch ? (n_expert_used + 2) : n_expert_used;
'''
if 'GAINZ_EXPERT_PREFETCH' not in s:
    assert marker in s, "marker not found"
    s = s.replace(marker, marker + gate, 1)
    assert 'GAINZ_EXPERT_PREFETCH' in s, "insert failed"
else:
    print('gate already present; skipping insert')

# switch call site n_expert_used -> n_used in build_layer_ffn only
before='''            nullptr,
            n_expert, n_expert_used,
            LLM_FFN_SILU, true,'''
after='''            nullptr,
            n_expert, n_used,
            LLM_FFN_SILU, true,'''
n=0
while before in s:
    s=s.replace(before, after, 1); n+=1
print('call-site replacements:', n)

open(f,'w').write(s)
PY
echo "=== verify ==="
grep -n 'GAINZ_EXPERT_PREFETCH' "$F"
grep -n 'n_expert, n_used,' "$F"
echo "=== APPLE_APPLIED ==="
