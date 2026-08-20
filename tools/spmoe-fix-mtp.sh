#!/usr/bin/env bash
# Fix SP-MoE: revert the MTP/nextn call-site @~694 back to n_expert_used
# (that function has NO n_used declaration). Keep trunk @~516 patched.
set -uo pipefail
F=/home/ghost/rebase27/src/models/qwen35moe.cpp
python3 - <<'PY'
f='/home/ghost/rebase27/src/models/qwen35moe.cpp'
lines=open(f).read().split('\n')
# revert line 694 (1-indexed) to n_expert_used (trunk at 516 keeps n_used)
idx=693  # 0-indexed 694
print('before:', lines[idx].strip())
lines[idx]=lines[idx].replace('n_expert, n_used,', 'n_expert, n_expert_used,')
print('after :', lines[idx].strip())
open(f,'w').write('\n'.join(lines))
PY
echo "=== verify ==="
grep -n 'n_expert, n_used\|n_expert, n_expert_used,' "$F"
