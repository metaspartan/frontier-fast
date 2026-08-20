#!/usr/bin/env bash
# HONEST status probe — no GPU seizure. Verify current state + whether any
# lighter untested spec axis exists (ngram/simple/mixed drafting) that could
# beat 233.3 without a 1hr kernel build.
set -uo pipefail
echo "=== endpoint (should be active; box untouched) ==="
systemctl --user is-active gainz-qwen-endpoint.service
cat /var/tmp/gainz-gpu.lock/owner 2>/dev/null
echo
echo "=== dflash engine spec types ==="
/home/ghost/dflash2full/build/bin/llama-server --help 2>&1 | grep -A1 'spec-type'
echo
echo "=== qwen3.6 speculative contract (other routes?) ==="
TOKEN="$(cat ~/.config/frontierfast/token 2>/dev/null || cat ~/.config/gainzfast/token)"
curl -s "https://frontier.fast/api/tracks" -H "authorization: [REDACTED] $TOKEN" | python3 -c "import json,sys
for t in json.load(sys.stdin):
  if t['id']=='qwen3.6-35b-a3b-gguf-r9700-v1':
    s=t.get('speculative',{})
    for h in s.get('howToRun',[]): print('howToRun:', h[:400]); print()
"
