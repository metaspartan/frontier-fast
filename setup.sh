#!/usr/bin/env bash
set -euo pipefail

echo "gainz.fast setup"
echo "================"

# Check bun
if ! command -v bun &>/dev/null; then
  echo "Installing bun..."
  curl -fsSL https://bun.sh/install | bash
  export PATH="$HOME/.bun/bin:$PATH"
fi

# Install dependencies
echo "Installing dependencies..."
bun install

# Verify benchmark contract
echo ""
echo "Active contract:"
cat benchmark.json | head -5

echo ""
echo "Setup complete. Run ./benchmark.sh --local-iterate to start."
