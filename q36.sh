#!/usr/bin/env bash
set -uo pipefail
TOKEN="$(cat ~/.config/frontierfast/token 2>/dev/null || cat ~/.config/gainzfast/token)"
TRACK="qwen3.6-35b-a3b-gguf-r9700-v1"
echo "=== $TRACK track def ==="
curl -s "https://frontier.fast/api/tracks" -H "authorization: [REDACTED] $TOKEN" -o /tmp/q36tracks.json
python3 - <<PY
import json
tracks=json.load(open('/tmp/q36tracks.json'))
for t in tracks:
    if t['id'].startswith('qwen3.6-35b'):
        f=t.get('frontier',{})
        print('==',t['id'],'|',t.get('deviceLabel'),'|',t.get('engine'))
        print('   spec:',f.get('decodeTokensPerSecond'),'tok/s',f.get('deltaPercent'),'%','|',f.get('candidate'))
    if t['id']=='qwen3.6-35b-a3b-gguf-r9700-v1':
        print('   speculative contract:', json.dumps(t.get('speculative',{}),indent=1)[:700])
PY
echo
echo "=== leaderboard (all) ==="
curl -s "https://frontier.fast/api/leaderboard?contract=$TRACK&technique=all" -H "authorization: [REDACTED] $TOKEN" -o /tmp/q36lb.json
python3 - <<PY
import json
d=json.load(open('/tmp/q36lb.json'))
print('entries',len(d))
for x in d[:8]:
    print(' ',round(x.get('decodeTokensPerSecond',0),1),'tok/s',round((x.get('deltaPercent') or 0),2),'% |',(x.get('candidate') or '')[:48],'|',x.get('technique','?'))
PY
echo
echo "=== recipe ==="
curl -s "https://frontier.fast/api/recipe?track=$TRACK" -H "authorization: [REDACTED] $TOKEN" -o /tmp/q36recipe.json
python3 -c "import json;d=json.load(open('/tmp/q36recipe.json'));r=d.get('recipe',{});print('pinned',r.get('pinnedCommit'));print('model',r.get('model'),r.get('modelFile'));print('frontier',d.get('frontier',{}).get('candidate'))"
