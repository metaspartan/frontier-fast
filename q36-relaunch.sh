#!/usr/bin/env bash
cd /home/ghost || exit 1
pkill -f '8093' 2>/dev/null
sleep 2
setsid bash -c "
  exec /home/ghost/dflash2full/build/bin/llama-server \
    -m /home/ghost/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
    -md /home/ghost/models/Qwen3.6-35B-A3B-DFlash-Q4_K_M.fixed.gguf \
    --spec-type draft-dflash --spec-draft-n-max 6 \
    -ngl 99 -c 4096 --parallel 1 --port 8093 --host 127.0.0.1 --alias qwen3.6-35b \
    > /home/ghost/q36-df-kern3.log 2>&1
" < /dev/null > /dev/null 2>&1 &
echo "launch_pid=$!"
sleep 40
echo "=== server log ==="
grep -E 'n_max=|block_size=|listening|error|exiting|model loaded' /home/ghost/q36-df-kern3.log | tail -8
echo "=== port 8093 ==="
curl -s -m 4 http://127.0.0.1:8093/v1/models | head -c 120
