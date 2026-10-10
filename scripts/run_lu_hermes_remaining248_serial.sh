#!/usr/bin/env bash
set -euo pipefail

cd /home/yujun/program/hermes

PYTHON=/home/yujun/.conda/envs/hermes-qwen/bin/python
ROOT=results/myobservations/native_k1000_remaining248_20260916
ANNO="$ROOT/anno_remaining248.json"
LOG_DIR=logs/native_k1000_remaining248_20260916
STATUS="$LOG_DIR/serial.status"
LU_SCORE=results/myobservations/lu_kv_offline_50v_20260913/lu_raw_blend05_clip075_150_runtime_score.csv
HERMES_SCORE=results/observations/memory_core_budget_scores/uniform.csv
LU_TAG=lu-raw-blend05-clip075-150-native-remaining248-k1000
HERMES_TAG=hermes-uniform-native-k1000-remaining248

mkdir -p "$ROOT" "$LOG_DIR"
"$PYTHON" scripts/build_streamingbench_complement.py \
  data/streamingbench/streamingbench_realtime_local.json \
  data/streamingbench/streamingbench_realtime_half_userpaths.json \
  "$ANNO"

printf 'starting old LU %s\n' "$(date -u +%FT%TZ)" > "$STATUS"
if bash scripts/run_native_k1000_budget.sh \
  "$ANNO" "$LU_SCORE" "$LU_TAG" lu_old_remaining248_k1000; then
  printf 'complete old LU %s\n' "$(date -u +%FT%TZ)" >> "$STATUS"
else
  code=$?
  printf 'failed old LU exit_code=%s %s\n' "$code" "$(date -u +%FT%TZ)" >> "$STATUS"
  exit "$code"
fi

printf 'starting HERMES %s\n' "$(date -u +%FT%TZ)" >> "$STATUS"
if bash scripts/run_native_k1000_budget.sh \
  "$ANNO" "$HERMES_SCORE" "$HERMES_TAG" hermes_uniform_remaining248_k1000; then
  printf 'complete HERMES %s\n' "$(date -u +%FT%TZ)" >> "$STATUS"
else
  code=$?
  printf 'failed HERMES exit_code=%s %s\n' "$code" "$(date -u +%FT%TZ)" >> "$STATUS"
  exit "$code"
fi
