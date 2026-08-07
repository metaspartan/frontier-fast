#!/bin/sh
# frontier.fast agent trace hook.
#
# Records which agent harness worked on this checkout so submissions can
# carry accurate agent attribution (agentName / runId). Local-only: this
# script never makes network calls and the trace file is gitignored.
#
# Usage: gainz-trace.sh <harness> <event> [detail]
#   harness: claude | cursor | codex | opencode | other
#   event:   session | capture | stop

set -u

HARNESS="${1:-unknown}"
EVENT="${2:-capture}"
DETAIL="${3:-}"

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
TRACE_DIR="$ROOT/.gainz"
TRACE_FILE="$TRACE_DIR/trace.jsonl"

mkdir -p "$TRACE_DIR" 2>/dev/null || exit 0

TS="$(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown)"
SHA="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo none)"

printf '{"ts":"%s","harness":"%s","event":"%s","detail":"%s","commit":"%s"}\n' \
  "$TS" "$HARNESS" "$EVENT" "$DETAIL" "$SHA" >> "$TRACE_FILE" 2>/dev/null || true

exit 0
