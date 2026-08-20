#!/usr/bin/env bash
# Orchestrate: take GPU, build SP-MoE (fixed), bench on/off, write results, restore.
# Run DETACHED. Write progress to /home/ghost/orch.status. Safe restore inside.
set -uo pipefail
W=/home/ghost/rebase27
M=/home/ghost/models
OUT=/home/ghost/orch.status
: > "$OUT"
echo "step=start $(date -u +%H:%M:%S)" >> "$OUT"

echo "step=suspend $(date -u +%H:%M:%S)" >> "$OUT"
systemctl --user stop gainz-qwen-endpoint.service 2>/dev/null
sleep 4
rm -rf /var/tmp/gainz-gpu.lock; mkdir -p /var/tmp/gainz-gpu.lock
echo "orch $$" > /var/tmp/gainz-gpu.lock/owner

echo "step=build $(date -u +%H:%M:%S)" >> "$OUT"
cd "$W" || { echo "abort cd build" >> "$OUT"; exit 1; }
export HIPCXX=/opt/rocm/core-7.14/lib/llvm/bin/clang
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_COMPILER=$HIPCXX -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  > /home/ghost/orch-cmake.log 2>&1
echo "cmake $?" >> "$OUT"
cmake --build build -j 32 > /home/ghost/orch-build.log 2>&1
echo "build $?" >> "$OUT"
ls -la build/bin/llama-server >> "$OUT" 2>&1

echo "step=bench $(date -u +%H:%M:%S)" >> "$OUT"
cd /home/ghost
for MODE in off on; do
  if [ "$MODE" = on ]; then export GAINZ_EXPERT_PREFETCH=1; else unset GAINZ_EXPERT_PREFETCH; fi
  pkill -f '8096' 2>/dev/null; sleep 2
  setsid bash -c "exec $W/build/bin/llama-server -m $M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf -md $M/Qwen3.6-35B-A3B-DFlash-Q4_K_M.fixed.gguf --spec-type draft-dflash --spec-draft-n-max 6 -ngl 99 -c 4096 --parallel 1 --port 8096 --host 127.0.0.1 --alias qwen3.6-35b > /home/ghost/orch-$MODE.log 2>&1" </dev/null >/dev/null 2>&1 &
  for i in $(seq 1 40); do curl -s -m2 http://127.0.0.1:8096/v1/models >/dev/null 2>&1 && break; sleep 3; done
  PROM="Write a detailed technical paragraph about the history of computing and microprocessors."
  BEST=0
  for i in 1 2 3; do
    printf '{"model":"qwen3.6-35b","prompt":"%s","max_tokens":128,"temperature":0}' "$PROM" > /tmp/o.json
    S=$(date +%s.%N); curl -s -m300 http://127.0.0.1:8096/v1/completions -H 'content-type: application/json' --data @/tmp/o.json >/tmp/o.resp; E=$(date +%s.%N)
    N=$(python3 -c "import json
try:print(json.load(open('/tmp/o.resp'))['usage']['completion_tokens'])
except:print(-1)")
    EL=$(echo "$E - $S"|bc); CPS=$(echo "$N $EL"|awk '{if($2>0)printf "%.1f",$1/$2; else print 0}')
    echo "mode=$MODE run=$i cps=$CPS" >> "$OUT"
    [ "${CPS%.*}" -gt "${BEST%.*}" ] 2>/dev/null && BEST=$CPS
    sleep 1
  done
  grep -E 'draft acceptance|eval time' /home/ghost/orch-$MODE.log | tail -2 >> "$OUT"
  pkill -f '8096'
done

echo "step=restore $(date -u +%H:%M:%S)" >> "$OUT"
rm -rf /var/tmp/gainz-gpu.lock
systemctl --user start gainz-qwen-endpoint.service
echo "step=done $(date -u +%H:%M:%S)" >> "$OUT"
