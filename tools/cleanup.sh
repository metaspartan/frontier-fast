#!/usr/bin/env bash
# Clean any stray processes (the cause of the infra-fault rejection), then
# report back so a clean resubmit can happen.
set -uo pipefail
echo "=== kill all stray llama-server/bench processes (not the endpoint's own) ==="
# Kill any of OUR bench/SP-MoE servers; leave the endpoint supervisor's spec27 server.
pkill -f 'rebase27/build/bin/llama-server' 2>/dev/null
pkill -f 'spmoe' 2>/dev/null
pkill -f 'orch' 2>/dev/null
pkill -f '8099' 2>/dev/null
pkill -f '8101' 2>/dev/null
sleep 3
echo "=== remaining llama-server processes ==="
pgrep -af llama-server | grep -v pgrep
echo "=== endpoint + lock + VRAM ==="
systemctl --user is-active gainz-qwen-endpoint.service
cat /var/tmp/gainz-gpu.lock/owner 2>/dev/null
/opt/rocm/core-7.14/bin/rocm-smi --showmeminfo vram 2>/dev/null | grep -i 'GPU\[0\]' | grep -i used
echo "=== CLEAN_CHECK_DONE ==="
