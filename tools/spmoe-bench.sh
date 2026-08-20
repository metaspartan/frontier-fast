#!/usr/bin/env bash
set -uo pipefail
W=/home/ghost/rebase27
BIN=$W/build/bin/llama-server
M=/home/ghost/models
# $1 = on|off
MODE="${1:-off}"
PORT=8097
if [ "$MODE" = "on" ]; then export GAINZ_EXPERT_PREFETCH=1; else unset GAINZ_EXPERT_PREFETCH; fi
echo "=== mode=$MODE prefetch=${GAINZ_EXPERT_PREFETCH:-off} ==="
pkill -f "$PORT" 2>/dev/null; sleep 2
setsid bash -c "exec $BIN -m $M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf -md $M/Qwen3.6-35B-A3B-DFlash-Q4_K_M.fixed.gguf --spec-type draft-dflash --spec-draft-n-max 6 -ngl 99 -c 4096 --parallel 1 --port $PORT --host 127.0.0.1 --alias qwen3.6-35b > $W/q36-$MODE.log 2>&1" </dev/null >/dev/null 2>&1 &
for i in $(seq 1 40); do curl -s -m2 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { echo server_up; break; }; sleep 3; done
PROM="Write a detailed technical paragraph about the history of computing, microprocessors and semiconductor manufacturing."
for i in 1 2 3 4; do
  printf '{"model":"qwen3.6-35b","prompt":"%s","max_tokens":128,"temperature":0}' "$PROM" > /tmp/q36x.json
  S=$(date +%s.%N); R=$(curl -s -m300 http://127.0.0.1:$PORT/v1/completions -H 'content-type: application/json' --data @/tmp/q36x.json); E=$(date +%s.%N)
  N=$(echo "$R"|python3 -c "import json,sys
try: print(json.load(sys.stdin)['usage']['completion_tokens'])
except: print(-1)")
  EL=$(echo "$E - $S"|bc); CPS=$(echo "$N $EL"|awk '{if($2>0)printf "%.1f",$1/$2; else print 0}')
  echo "mode=$MODE run=$i toks=$N cps=$CPS"; sleep 1
done
echo "=== server spec ==="
grep -E 'draft acceptance|eval time' $W/q36-$MODE.log | tail -3
echo "=== MODE_${MODE}_DONE ==="
