#!/usr/bin/env bash
set -euo pipefail
CHANGED=$(git diff --name-only origin/main...HEAD 2>/dev/null || git diff --name-only HEAD~1 HEAD)
for f in $CHANGED; do
  case "$f" in
    Sources/runner/*|Sources/transforms/*|Sources/model/*|Sources/scoring/*) echo "OK: $f" ;;
    benchmark.json|correctness_prompts/*|.github/*|AGENTS.md|README.md|LICENSE|.gitignore|package.json) echo "OK: $f" ;;
    *) echo "REJECT: non-editable file changed: $f"; exit 1 ;;
  esac
done
echo "Modifiable surface check passed."
