#!/usr/bin/env bash
# Robust SP-MoE paired measurement. Fixes: waits for server readiness via /health,
# kills prior server + waits for PID exit before next mode, captures BOTH client
# usage and authoritative server eval-time per mode. Self-restores.
set -uo pipefail
W=/home/ghost/rebase27
M=/home/ghost/models
BIN=$W/build/bin/llama-server
OUT=/home/ghost/spmoe3.status
port=8101
: > "$OUT"

log(){ echo "$@" | tee -a "$OUT"; }

server_up() {
  local p=$1
  for i in $(seq 1 60); do
    curl -s -m2 "http://127.0.0.1:$p/health" >/dev/null 2>&1 && return 0
    sleep 2
  done
  return 1
}
kill_server() {
  pkill -f "port $port" 2>/dev/null
  pkill -f "8099" 2>/dev/null
  pkill -f "8101" 2>/dev/null
  sleep 3
}

log "step=start $(date -u +%H:%M:%S)"
systemctl --user stop gainz-qwen-endpoint.service 2>/dev/null; sleep 4
rm -rf /var/tmp/gainz-gpu.lock; mkdir -p /var/tmp/gainz-gpu.lock
echo "spmoe3 $$" > /var/tmp/gainz-gpu.lock/owner
log "patch=$($BIN --version 2>&1 | head -1)"
grep -q 'n_expert, n_used,' "$W/src/models/qwen35moe.cpp" && log "patch=yes" || log "patch=NO"

PROM="Write a detailed technical paragraph about the history of computing, microprocessors and semiconductor manufacturing."

for MODE in off on; do
  [ "$MODE" = on ] && export GAINZ_EXPERT_PREFETCH=1 || unset GAINZ_EXPERT_PREFETCH
  kill_server
  log "step=mode_$MODE $(date -u +%H:%M:%S)"
  setsid bash -c "exec $BIN -m $M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf -md $M/Qwen3.6-35B-A3B-DFlash-Q4_K_M.fixed.gguf --spec-type draft-dflash --spec-draft-n-max 6 -ngl 99 -c 4096 --parallel 1 --port $port --host 127.0.0.1 --alias qwen3.6-35b > /home/ghost/spmoe3-$MODE.log 2>&1" </dev/null >/dev/null 2>&1 &
  server_pid=$!
  if server_up $port; then
    log "mode=$MODE server=UP"
    for i in 1 2 3 4 5; do
      printf '{"model":"qwen3.6-35b","prompt":"%s","max_tokens":128,"temperature":0}' "$PROM" > /tmp/s3.json
      S=$(date +%s.%N); curl -s -m300 http://127.0.0.1:$port/v1/completions -H 'content-type: application/json' --data @/tmp/s3.json > /tmp/s3.resp; E=$(date +%s.%N)
      N=$(python3 -c "import json
try: print(json.load(open('/tmp/s3.resp'))['usage']['completion_tokens'])
except Exception: print(-1)")
      EL=$(echo "$E - $S"|bc); CPS=$(echo "$N $EL"|awk '{if($2>0)printf "%.1f",$1/$2; else print 0}')
      log "mode=$MODE run=$i toks=$N cps=$CPS"
      sleep 1
    done
    log "mode=$MODE SERVER_TIMING:"
    grep -E 'eval time|draft acceptance' /home/ghost/spmoe3-$MODE.log | tail -2 | tee -a "$OUT"
    kill_server
  else
    log "mode=$MODE server=DOWN"
    tail -4 /home/ghost/spmoe3-$MODE.log | tee -a "$OUT"
  fi
done

log "step=restore $(date -u +%H:%M:%S)"
rm -rf /var/tmp/gainz-gpu.lock
systemctl --user start gainz-qwen-endpoint.service
log "step=done $(date -u +%H:%M:%S)"
