#!/usr/bin/env bash
# Clean SP-MoE measurement orchestrator WITH the ctx_other draft-launch fix.
# Suspends endpoint, launches DFlash with the draft directly (fixed config),
# benches OFF/ON on the genuinely-rebuilt SP-MoE binary, captures SERVER
# eval-time tok/s (authoritative), restores.
set -uo pipefail
W=/home/ghost/rebase27
M=/home/ghost/models
BIN=$W/build/bin/llama-server
OUT=/home/ghost/spmoe2.status
: > "$OUT"

echo "step=start $(date -u +%H:%M:%S)" >> "$OUT"
systemctl --user stop gainz-qwen-endpoint.service 2>/dev/null
sleep 4
rm -rf /var/tmp/gainz-gpu.lock; mkdir -p /var/tmp/gainz-gpu.lock
echo "spmoe2 $$" > /var/tmp/gainz-gpu.lock/owner

# confirm binary is rebuilt+patched
echo "bin=$($BIN --version 2>&1 | head -1)" >> "$OUT"
grep -q 'n_expert, n_used,' "$W/src/models/qwen35moe.cpp" && echo "patch=yes" >> "$OUT" || echo "patch=NO" >> "$OUT"

for MODE in off on; do
  if [ "$MODE" = on ]; then export GAINZ_EXPERT_PREFETCH=1; else unset GAINZ_EXPERT_PREFETCH; fi
  echo "step=mode_$MODE $(date -u +%H:%M:%S)" >> "$OUT"
  pkill -f '8099' 2>/dev/null; sleep 3
  # launch target+draft; keep the DFlash draft resident via single process
  setsid bash -c "exec $BIN -m $M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf -md $M/Qwen3.6-35B-A3B-DFlash-Q4_K_M.fixed.gguf --spec-type draft-dflash --spec-draft-n-max 6 -ngl 99 -c 4096 --parallel 1 --port 8099 --host 127.0.0.1 --alias qwen3.6-35b > /home/ghost/spmoe2-$MODE.log 2>&1" </dev/null >/dev/null 2>&1 &
  # wait
  up=0
  for i in $(seq 1 40); do curl -s -m2 http://127.0.0.1:8099/v1/models >/dev/null 2>&1 && { up=1; break; }; sleep 3; done
  echo "mode=$MODE server_up=$up" >> "$OUT"
  if [ "$up" = 1 ]; then
    # 5 decodes; read usage from a temp file (robust)
    PROM="Write a detailed technical paragraph about the history of computing, microprocessors and semiconductor manufacturing."
    for i in 1 2 3 4 5; do
      printf '{"model":"qwen3.6-35b","prompt":"%s","max_tokens":128,"temperature":0}' "$PROM" > /tmp/s2.json
      S=$(date +%s.%N)
      curl -s -m300 http://127.0.0.1:8099/v1/completions -H 'content-type: application/json' --data @/tmp/s2.json > /tmp/s2.resp
      E=$(date +%s.%N)
      N=$(python3 -c "import json
try: print(json.load(open('/tmp/s2.resp'))['usage']['completion_tokens'])
except Exception: print(-1)")
      EL=$(echo "$E - $S"|bc); CPS=$(echo "$N $EL"|awk '{if($2>0)printf "%.1f",$1/$2; else print 0}')
      echo "mode=$MODE run=$i toks=$N cps=$CPS" >> "$OUT"
      sleep 1
    done
  fi
  # authoritative server timing
  grep -E 'draft acceptance|eval time' /home/ghost/spmoe2-$MODE.log 2>/dev/null | tail -3 >> "$OUT"
  pkill -f '8099' 2>/dev/null
done

echo "step=restore $(date -u +%H:%M:%S)" >> "$OUT"
rm -rf /var/tmp/gainz-gpu.lock
systemctl --user start gainz-qwen-endpoint.service
echo "step=done $(date -u +%H:%M:%S)" >> "$OUT"
