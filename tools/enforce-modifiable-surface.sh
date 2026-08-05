#!/usr/bin/env bash
# Reject changes outside the participant-editable surface declared in
# benchmark.json. Infrastructure paths (workflows, contracts, fixtures,
# tools) are maintainer-only: set GAINZ_ALLOW_INFRA=1 on trusted main-branch
# CI to permit them.
#
# The editable list below MUST stay in sync with `editablePaths` in
# benchmark.json and with each track's `allowlistedPaths` in
# https://gainz.fast/api/tracks. Tests/fixtures.test.ts fails if it drifts.
#
# This once rejected every llama.cpp, MLX and vLLM-source submission: the
# patch directories are the whole participant surface on six of the eight
# tracks and none of them were listed here.
set -euo pipefail
CHANGED=$(git diff --name-only origin/main...HEAD 2>/dev/null || git diff --name-only HEAD~1 HEAD)
for f in $CHANGED; do
  case "$f" in
    # Shared TypeScript surface (all tracks).
    Sources/runner/*|Sources/transforms/*|Sources/model/*|Sources/scoring/*|Sources/kernels/*) echo "OK: $f" ;;
    # Engine patch surfaces. Sources/patches is per-track (llama.cpp + MLX
    # Python overlay), vllm-patches is shared by both vLLM tracks, and
    # mlx-engine-patches rebuilds MLX itself.
    Sources/patches/*|Sources/vllm-patches/*|Sources/mlx-engine-patches/*) echo "OK: $f" ;;
    benchmark.json|correctness_prompts/*|fixtures/*|.github/*|tools/*|Tests/*|docs/*|AGENTS.md|CLAUDE.md|README.md|TASK.md|LICENSE|.gitignore|.gitattributes|package.json|bun.lock|tsconfig.json|setup.sh|benchmark.sh|.claude/*|.cursor/*|.codex/*|.opencode/*|.gainz/*|Sources/types.ts|Sources/contracts.ts|Sources/scoring.ts|Sources/cli.ts)
      if [ "${GAINZ_ALLOW_INFRA:-0}" = "1" ]; then echo "OK (infra): $f"; else echo "REJECT: non-editable file changed: $f"; exit 1; fi ;;
    *) echo "REJECT: non-editable file changed: $f"; exit 1 ;;
  esac
done
echo "Modifiable surface check passed."
