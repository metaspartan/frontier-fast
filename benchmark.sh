#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---local-iterate}"
TRACK="${GAINZ_TRACK:-laguna-xs-2.1-nvfp4-gb10-v1}"

echo "gainz.fast benchmark — $MODE (track: $TRACK)"
echo "================================================"

case "$MODE" in
  --local-iterate)
    echo "Running fast local iteration (correctness smoke + timing estimate)..."
    bun run Sources/cli.ts benchmark --track "$TRACK" --local-iterate
    ;;
  --local-submit)
    echo "Running longer local pre-submit signal..."
    bun run Sources/cli.ts benchmark --track "$TRACK" --local-submit
    ;;
  --baseline)
    echo "Running baseline measurement..."
    bun run Sources/cli.ts benchmark --track "$TRACK" --baseline
    ;;
  *)
    echo "Usage: ./benchmark.sh [--local-iterate|--local-submit|--baseline]"
    echo "  --local-iterate  Fast correctness smoke + timing estimate"
    echo "  --local-submit   Longer pre-submit signal"
    echo "  --baseline       Baseline measurement only"
    exit 1
    ;;
esac
