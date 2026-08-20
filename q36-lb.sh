#!/usr/bin/env bash
set -uo pipefail
TOKEN="$(cat ~/.config/frontierfast/token 2>/dev/null || cat ~/.config/gainzfast/token)"
TRACK="qwen3.6-35b-a3b-gguf-r9700-v1"
echo "=== TOP speculative entries (full) ==="
curl -s "https://frontier.fast/api/leaderboard?contract=$TRACK&technique=speculative" -H "authorization: [REDACTED] $TOKEN" -o /tmp/q36spec.json
python3 - <<PY
import json
d=json.load(open('/tmp/q36spec.json'))
print('spec entries',len(d))
for x in d[:6]:
    print('---')
    print((x.get('candidate') or '')[:80])
    print('  decode',round(x.get('decodeTokensPerSecond',0),1),'tok/s | delta',round((x.get('deltaPercent') or 0),2),'% | prefill',round(x.get('prefillTokensPerSecond',0),1))
    print('  technique',x.get('technique'),'| model',x.get('model') or x.get('quantization'))
    print('  sub',x.get('submissionId'))
PY
echo
echo "=== top kernel entries ==="
curl -s "https://frontier.fast/api/leaderboard?contract=$TRACK&technique=kernel" -H "authorization: [REDACTED] $TOKEN" -o /tmp/q36kern.json
python3 - <<PY
import json
d=json.load(open('/tmp/q36kern.json'))
print('kernel entries',len(d))
for x in d[:5]:
    print(' ',round(x.get('decodeTokensPerSecond',0),1),'tok/s',round((x.get('deltaPercent') or 0),2),'% |',(x.get('candidate') or '')[:55])
PY
