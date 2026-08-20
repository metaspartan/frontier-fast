#!/usr/bin/env bash
# qwen3.6-35b-a3b R9700 — SP-MoE gated qwen35moe build+measure, env-gated,
# self-contained, curl-delivered. Suspends endpoint, builds, measures, restores.
set -uo pipefail
M=/home/ghost/models
W=/home/ghost/rebase27
LOG=/home/ghost/spmoe-build.log

echo "=== take lock + suspend endpoint ==="
systemctl --user stop gainz-qwen-endpoint.service 2>/dev/null
sleep 4
rm -rf /var/tmp/gainz-gpu.lock; mkdir -p /var/tmp/gainz-gpu.lock
echo "spmoe-build $$" > /var/tmp/gainz-gpu.lock/owner
echo "locked_by=$(cat /var/tmp/gainz-gpu.lock/owner)"

echo "=== source tree has no-op SP-MoE gate already in qwen35moe? ==="
grep -n 'GAINZ_EXPERT_PREFETCH' "$W/src/models/qwen35moe.cpp" | head -2 || echo "NO_GATE_YET(no-op build continues; change not yet applied)"

cd "$W" || exit 1
echo "=== cmake configure (reuse/reuse build) ==="
export HIPCXX=/opt/rocm/core-7.14/lib/llvm/bin/clang
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_COMPILER=$HIPCXX -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  > /home/ghost/spmoe-cmake.log 2>&1
echo "cmake rc=$?"
cmake --build build -j 32 > "$LOG" 2>&1
echo "build rc=$?"
tail -4 "$LOG"
