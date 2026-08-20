#!/usr/bin/env bash
set -uo pipefail
TOKEN="$(cat ~/.config/frontierfast/token 2>/dev/null || cat ~/.config/gainzfast/token)"
echo "=== box: which DFlash-enabled engine + fixed draft model ==="
echo "--- engines ---"
ls -1d /home/ghost/dflash2full /home/ghost/dflash2w /home/ghost/llama.cpp-dflash-0b1bad1 2>/dev/null
echo "--- draft GGUFs + sizes ---"
ls -la /home/ghost/models/Qwen3.6-35B-A3B-DFlash* 2>/dev/null | awk '{print $5, $9}'
echo "--- is there a dflash-enabled llama-server binary already built? ---"
find /home/ghost -maxdepth 3 -name llama-server -path '*build*' -newermt '2026-08-15' 2>/dev/null
echo "--- serving.json on track (ranked draft-dflash path) ---"
sed -n '1,60p' /c/Users/Carsen/frontier-fast/Sources/runner/serving.json 2>/dev/null
