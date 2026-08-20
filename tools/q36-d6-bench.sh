#!/usr/bin/env bash
# qwen3.6-35b-a3b R9700 — self-contained DFlash depth-6 bench with lock+restore.
# Delivered via git pull (reliable transport). Run on the box as root of a
# sanctioned suspend/restore window.
set -uo pipefail
M=/home/ghost/models
BIN=/home/ghost/dflash2full/build/bin/llama-server
PORT=8096

echo "=== take lock ==="
rm -rf /var/tmp/gainz-gpu.lock; mkdir -p /var/tmp/gainz-gpu.lock
echo "spmoe-bench $$" > /var/tmp/gainz-gpu.lock/owner
echo "locked_by=$(cat /var/tmp/gainz-gpu.lock/owner)"

echo "=== preflight ==="
"$BIN" --help 2>&1 | grep -oE 'draft-dflash' | head -1

echo "=== launch qwen3.6 target+draft depth 6 port $PORT ==="
pkill -f "$PORT" 2>/dev/null; sleep 2
setsid bash -c "
  exec $BIN \
    -m $M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
    -md $M/Qwen3.6-35B-A3B-DFlash-Q4_K_M.fixed.gguf \
    --spec-type draft-dflash --spec-draft-n-max 6 \
    -ngl 99 -c 4096 --parallel 1 --port $PORT --host 127.0.0.1 --alias qwen3.6-35b \
    > /home/ghost/q36-d6.log 2>&1
" < /dev/null > /dev/null 2>&1 &

echo "waiting for server..."
for i in $(seq 1 40); do curl -s -m 2 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { echo server_up; break; }; sleep 3; done
echo "=== server log ==="
grep -E 'n_max=|block_size=|model loaded|listening|error|exiting' /home/ghost/q36-d6.log | tail -8

PROM="Write a detailed technical paragraph about the history of computing, microprocessors and semiconductor manufacturing."
echo "=== bench 4x 128-token ==="
for i in 1 2 3 4; do
  printf '{"model":"qwen3.6-35b","prompt":"%s","max_tokens":128,"temperature":0}' "$PROM" > /tmp/q36b.json
  S=$(date +%s.%N); R=$(curl -s -m 300 http://127.0.0.1:$PORT/v1/completions -H 'content-type: application/json' --data @/tmp/q36b.json); E=$(date +%s.%N)
  EL=$(echo "$E - $S" | bc)
  N=$(echo "$R" | python3 -c "import json,sys
try:
 print(json.load(sys.stdin)['usage']['completion_tokens'])
except Exception: print(-1)")
  CPS=$(echo "$N $EL" | awk '{if($2>0)printf "%.1f",$1/$2; else print 0}')
  echo "run=$i toks=$N cps=$CPS"; sleep 1
done
echo "=== spec stats ==="
grep -E 'draft acceptance|eval time' /home/ghost/q36-d6.log | tail -4
echo "=== RESULT_DONE ==="
