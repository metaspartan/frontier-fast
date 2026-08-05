#!/usr/bin/env bash
set -euo pipefail

TRACK="${GAINZ_TRACK:-$(python3 -c 'import json;print(json.load(open("benchmark.json"))["defaultTrack"])' 2>/dev/null || echo laguna-xs-2.1-nvfp4-gb10-v1)}"

echo "gainz.fast setup"
echo "================"

# Check bun
if ! command -v bun &>/dev/null; then
  echo "Installing bun..."
  curl -fsSL https://bun.sh/install | bash
  export PATH="$HOME/.bun/bin:$PATH"
fi

echo "Installing dependencies..."
bun install

echo ""
echo "Verifying the track contract..."
bun test
bun run Sources/cli.ts tracks

echo ""
echo "Active track: $TRACK"
echo "  (override with GAINZ_TRACK=<id>; the live contract is always"
echo "   authoritative: curl -s https://gainz.fast/api/tracks)"

# Engine-specific prerequisites. Which one applies is decided by the track's
# quantization, the same rule Sources/runner/engine.ts uses.
case "$TRACK" in
  *gguf*)
    echo ""
    echo "Engine: llama.cpp. You need the pinned tree built with your track's series:"
    echo "  git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp"
    echo "  git checkout 2b63e0610bbc2be990ae1360d5256efcdc3f9efb   # b10237"
    echo "  for p in ../Sources/patches/$TRACK/*.patch; do git apply \"\$p\"; done"
    echo "Build llama-perplexity as well as llama-server — it is the accuracy gate."
    echo "Exact build flags: curl -s \"https://gainz.fast/api/recipe?track=$TRACK\""
    echo "Then serve:  ./build/bin/llama-server -m <gguf> -ngl 99 -c 8192 --parallel 1"
    echo "The benchmark drives GAINZ_BASE_URL (default http://127.0.0.1:8080/v1)."
    ;;
  *mlx*)
    echo ""
    echo "Engine: MLX (in-process, no server)."
    command -v python3 >/dev/null || echo "  WARNING: python3 not found"
    python3 -c 'import mlx_lm' 2>/dev/null \
      && echo "  mlx-lm: installed" \
      || echo "  mlx-lm: MISSING — run: uv venv .venv && . .venv/bin/activate && uv pip install mlx-lm"
    echo "  harness: tools/mlx_bench.py (set GAINZ_PYTHON to pick a different interpreter)"
    echo "  engine rebuilds (Sources/mlx-engine-patches/) need cmake, ninja and Xcode."
    ;;
  *)
    echo ""
    echo "Engine: vLLM. Serve the pinned image with deterministic kernels:"
    echo "  docker run --gpus all --ipc=host -e VLLM_BATCH_INVARIANT=1 -p 8000:8000 \\"
    echo "    vllm/vllm-openai:v0.25.1 --model <model> --max-model-len 8192"
    echo "VLLM_BATCH_INVARIANT=1 is REQUIRED — without it greedy output is not reproducible."
    echo "The benchmark drives GAINZ_BASE_URL/VLLM_BASE_URL (default http://127.0.0.1:8000/v1)."
    ;;
esac

echo ""
echo "Setup complete. Run: GAINZ_TRACK=$TRACK ./benchmark.sh --local-iterate"
