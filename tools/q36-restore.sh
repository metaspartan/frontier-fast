#!/usr/bin/env bash
# Restore after q36-d6-bench.sh (room for it to have ended without restoring).
set -uo pipefail
pkill -f '8096' 2>/dev/null; sleep 3
rm -rf /var/tmp/gainz-gpu.lock
systemctl --user start gainz-qwen-endpoint.service
sleep 30
echo "=== verify ==="
systemctl --user is-active gainz-qwen-endpoint.service
pgrep -af llama-server | grep -v pgrep | head -1
cat /var/tmp/gainz-gpu.lock/owner 2>/dev/null
/opt/rocm/core-7.14/bin/rocm-smi --showmeminfo vram 2>/dev/null | grep -i 'GPU\[0\]'
curl -s -m 6 http://127.0.0.1:8080/v1/models | head -c 40
echo "=== RESTORED ==="
