#!/usr/bin/env bash
set -uo pipefail
TOKEN="$(cat ~/.config/frontierfast/token 2>/dev/null || cat ~/.config/gainzfast/token)"
echo "=== qwen3.6-35b-a3b spec leaderboard FULL records (depth values) ==="
curl -s "https://frontier.fast/api/leaderboard?contract=qwen3.6-35b-a3b-gguf-r9700-v1&technique=speculative" -H "authorization: [REDACTED] $TOKEN" -o /tmp/q36spec2.json
python3 - <<PY
import json
d=json.load(open('/tmp/q36spec2.json'))
for x in d:
    print('===')
    print('candidate:', (x.get('candidate') or '')[:90])
    print('decode',round(x.get('decodeTokensPerSecond',0),1),'delta',round((x.get('deltaPercent') or 0),2))
    print('sub:', x.get('submissionId'))
    print('techniqueReason:', (x.get('techniqueReason') or '')[:600])
    print('confirmation:', json.dumps(x.get('confirmation',{}))[:200])
PY
