#!/usr/bin/env bash
set -uo pipefail
M=/home/ghost/models
W=/home/ghost/rebase27
LOG=/home/ghost/spmoe-build.log
echo "=== take lock + suspend endpoint ==="
systemctl --user stop gainz-qwen-endpoint.service 2>/dev/null; sleep 4
rm -rf /var/tmp/gainz-gpu.lock; mkdir -p /var/tmp/gainz-gpu.lock
echo "spmoe-build $$" > /var/tmp/gainz-gpu.lock/owner
echo "locked_by=$(cat /var/tmp/gainz-gpu.lock/owner)"
cd "$W" || exit 1
echo "=== confirm patch applied ==="
grep -c GAINZ_EXPERT_PREFETCH src/models/qwen35moe.cpp
echo "=== build (rebase27, HIP) ==="
export HIPCXX=/opt/rocm/core-7.14/lib/llvm/bin/clang
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_COMPILER=$HIPCXX -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  > /home/ghost/spmoe-cmake.log 2>&1
echo "cmake rc=$?"
cmake --build build -j 32 > "$LOG" 2>&1
echo "build rc=$?"
tail -4 "$LOG"
echo "=== BUILD_DONE ==="
