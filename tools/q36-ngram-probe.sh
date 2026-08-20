#!/usr/bin/env bash
# HONEST alternative-axis probe: measure Q3.6 ngram-cache draft (a WHOLLY
# different spec path from DFlash) on the R9700, to see if ANY alternative must
# be lower/higher than 233.3. Box: sanctioned suspend/measure/restore. NOT an
# SP-MoE blind build. ngram needs no draft weights.
set -uo pipefail
M=/home/ghost/models
BIN=/home/ghost/dflash2full/build/bin/llama-server
PORT=8098
echo "=== take lock + suspend endpoint ==="
systemctl --user stop gainz-qwen-endpoint.service 2>/dev/null; sleep 4
rm -rf /var/tmp/gainz-gpu.lock; mkdir -p /var/tmp/gainz-gpu.lock
echo "q36-ngram $$" > /var/tmp/gainz-gpu.lock/owner
echo locked_by=$(cat /var/tmp/gainz-gpu.lock/owner)
echo "=== launch qwen3.6 target, ngram-cache spec, port $PORT ==="
pkill -f "$PORT" 2>/dev/null; sleep 2
setsid bash -c "exec $BIN -m $M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --spec-type ngram-cache -ngl 99 -c 4096 --parallel 1 --port $PORT --host 127.0.0.1 --alias qwen3.6-35b > /home/ghost/q36-ngram.log 2>&1" < /dev/null > /dev/null 2>&1 &
for i in $(seq 1 40); do curl -s -m2 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { echo server_up; break; }; sleep 3; done
PROM="Write a detailed technical paragraph about the history of computing, microprocessors and semiconductor manufacturing."
for i in 1 2 3; do
  printf '{"model":"qwen3.6-35b","prompt":"%s","max_tokens":128,"temperature":0}' "$PROM" > /tmp/q36n.json
  S=$(date +%s.%N); R=$(curl -s -m300 http://127.0.0.1:$PORT/v1/completions -H 'content-type: application/json' --data @/tmp/q36n.json); E=$(date +%s.%N)
  N=$(echo "$R"|python3 -c "import json,sys
try: print(json.load(sys.stdin)['usage']['completion_tokens'])
except: print(-1)")
  EL=$(echo "$E - $S"|bc); CPS=$(echo "$N $EL"|awk '{if($2>0)printf "%.1f",$1/$2; else print 0}')
  echo "run=$i toks=$N cps=$CPS"; sleep 1
done
grep -E 'draft acceptance|eval time' /home/ghost/q36-ngram.log | tail -3
echo "=== NEW_ALT_AXIS_RESULT ==="
