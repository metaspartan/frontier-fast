#!/usr/bin/env bash
# Reject changes outside the participant-editable surface declared in
# benchmark.json. Infrastructure paths (workflows, contracts, fixtures,
# tools) are maintainer-only: set GAINZ_ALLOW_INFRA=1 on trusted main-branch
# CI to permit them.
set -euo pipefail
CHANGED=$(git diff --name-only origin/main...HEAD 2>/dev/null || git diff --name-only HEAD~1 HEAD)
for f in $CHANGED; do
  case "$f" in
    Sources/runner/*|Sources/transforms/*|Sources/model/*|Sources/scoring/*|Sources/kernels/*) echo "OK: $f" ;;
    benchmark.json|correctness_prompts/*|.github/*|tools/*|Tests/*|AGENTS.md|CLAUDE.md|README.md|LICENSE|.gitignore|.gitattributes|package.json|setup.sh|benchmark.sh|.claude/*|.cursor/*|.codex/*|.opencode/*|.gainz/*|Sources/types.ts|Sources/contracts.ts|Sources/scoring.ts|Sources/cli.ts|Sources/runner/index.ts)
      if [ "${GAINZ_ALLOW_INFRA:-0}" = "1" ]; then echo "OK (infra): $f"; else echo "REJECT: non-editable file changed: $f"; exit 1; fi ;;
    *) echo "REJECT: non-editable file changed: $f"; exit 1 ;;
  esac
done
echo "Modifiable surface check passed."
