#!/usr/bin/env bash
set -euo pipefail

# The original serial controller owns PID 144208. Once its HERMES child exits,
# terminate any second LU launcher it creates; the separately launched LU run
# is already active in another tmux session.
SERIAL_PARENT=144208
HERMES_WORKER=144243

while kill -0 "$HERMES_WORKER" 2>/dev/null; do
  sleep 2
done

for _ in $(seq 1 60); do
  mapfile -t children < <(pgrep -P "$SERIAL_PARENT" || true)
  for child in "${children[@]}"; do
    mapfile -t grandchildren < <(pgrep -P "$child" || true)
    for grandchild in "${grandchildren[@]}"; do
      pkill -TERM -P "$grandchild" 2>/dev/null || true
      kill -TERM "$grandchild" 2>/dev/null || true
    done
    pkill -TERM -P "$child" 2>/dev/null || true
    kill -TERM "$child" 2>/dev/null || true
  done
  if ! kill -0 "$SERIAL_PARENT" 2>/dev/null; then
    exit 0
  fi
  sleep 1
done
