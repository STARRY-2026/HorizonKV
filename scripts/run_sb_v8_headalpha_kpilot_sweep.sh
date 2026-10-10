#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"

mkdir -p "$LOG_DIR"
for k_delta in 4 8; do
    echo "Starting Head-Alpha k pilot with k_delta=$k_delta"
    K_DELTA="$k_delta" "$SCRIPT_DIR/run_sb_v8_headalpha_kpilot.sh" \
        > "$LOG_DIR/v8_headalpha_failure12_kd${k_delta}.log" 2>&1
    echo "Completed Head-Alpha k pilot with k_delta=$k_delta"
done
