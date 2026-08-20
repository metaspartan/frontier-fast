#!/usr/bin/env bash
set -uo pipefail
echo "=== stop any 8093 dflash server ==="
pkill -f '8093' 2>/dev/null
sleep 3
echo "=== release lock + restore endpoint ==="
rm -rf /var/tmp/gainz-gpu.lock
systemctl --user start gainz-qwen-endpoint.service
sleep 30
echo "=== verify restore ==="
systemctl --user is-active gainz-qwen-endpoint.service
pgrep -af llama-server | grep -v pgrep | head -1
cat /var/tmp/gainz-gpu.lock/owner 2>/dev/null
/opt/rocm/core-7.14/bin/rocm-smi --showmeminfo vram 2>/dev/null | grep -i 'GPU\[0\]'
curl -s -m 6 http://127.0.0.1:8080/v1/models | head -c 50
echo "=== RESTORED ==="
