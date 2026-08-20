#!/usr/bin/env bash
set -uo pipefail
TOKEN="$(cat ~/.config/frontierfast/token 2>/dev/null || cat ~/.config/gainzfast/token)"
TRACK="qwen3.6-35b-a3b-gguf-r9700-v1"
echo "=== speculative serving.json example (exact ranked path) ==="
curl -s "https://frontier.fast/api/tracks" -H "authorization: [REDACTED] $TOKEN" -o /tmp/q36t.json
python3 - <<PY
import json
tracks=json.load(open('/tmp/q36t.json'))
for t in tracks:
    if t['id']=='qwen3.6-35b-a3b-gguf-r9700-v1':
        s=t.get('speculative',{})
        print('method',s.get('method'))
        print('howToRun:')
        for h in s.get('howToRun',[])[:3]:
            print('  ---'); print(' ',h[:500])
        print('draftModel:', repr(s.get('draftModel')))
        print('caveat:', (s.get('caveat') or '')[:200])
PY
echo
echo "=== box draft models sizes ==="
ls -la /home/ghost/models/Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf /home/ghost/models/Qwen3.6-35B-A3B-DFlash-Q4_K_M.fixed.gguf /home/ghost/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf 2>/dev/null | awk '{print $5, $9}'
echo "=== qwen3.6 worktrees + builds on box ==="
ls -1d /home/ghost/*qwen3.6* /home/ghost/*35b* /home/ghost/*a3b* 2>/dev/null
