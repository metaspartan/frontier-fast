#!/usr/bin/env bash
set -uo pipefail
TOKEN="$(cat ~/.config/frontierfast/token 2>/dev/null || cat ~/.config/gainzfast/token)"
TRACK="qwen3.6-35b-a3b-gguf-r9700-v1"
echo "=== findings ledger ==="
curl -s "https://frontier.fast/api/findings?track=$TRACK" -H "authorization: [REDACTED] $TOKEN" -o /tmp/q36find.json
python3 - <<PY
import json
d=json.load(open('/tmp/q36find.json'))
print('count',len(d))
for x in d:
    if isinstance(x,dict):
        print(' ',x.get('verdict','?'),'::',x.get('id','?'),'|',(x.get('lever') or '')[:70])
PY
echo
echo "=== box qwen3.6 models ==="
ls -1 /home/ghost/models/ | grep -iE 'qwen3.6|dflash|ud'
echo "=== qwen3.6 worktrees ==="
ls -1d /home/ghost/*qwen3.6* /home/ghost/*35b* 2>/dev/null
