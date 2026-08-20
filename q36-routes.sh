#!/usr/bin/env bash
set -uo pipefail
TOKEN="$(cat ~/.config/frontierfast/token 2>/dev/null || cat ~/.config/gainzfast/token)"
echo "=== sweep: how many speculative entries, all depths present ==="
curl -s "https://frontier.fast/api/leaderboard?contract=qwen3.6-35b-a3b-gguf-r9700-v1&technique=speculative" -H "authorization: [REDACTED] $TOKEN" -o /tmp/q36s.json
python3 - <<PY
import json
d=json.load(open('/tmp/q36s.json'))
print('spec entries',len(d))
for x in d:
    print('---')
    print((x.get('candidate') or '')[:80])
    print('decode',round(x.get('decodeTokensPerSecond',0),1),'| delta',round((x.get('deltaPercent') or 0),2))
    print('technique',x.get('technique'),'| created',x.get('createdAt'))
PY
echo
echo "=== is draft-dflash the ONLY spec route tried? any ngram/simple/mtp entries? ==="
grep -liE 'ngram|draft-mtp|draft-simple' /tmp/q36s.json 2>/dev/null && echo 'other routes found' || echo 'only DFlash on this board'
