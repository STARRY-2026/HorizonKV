#!/usr/bin/env bash
set -euo pipefail

cd /home/yujun/program/hermes
ROOT=results/myobservations/lu_kv_balanced100v_20260915
LOG_DIR=logs/lu_kv_offline_balanced100v_20260915
mkdir -p "$ROOT" "$LOG_DIR"

/home/yujun/.conda/envs/hermes-qwen/bin/python scripts/build_lu_balanced_calibration.py \
  > "$LOG_DIR/build_manifest.log" 2>&1

printf 'starting %s\n' "$(date -u +%FT%TZ)" > "$LOG_DIR/run.status"
export PYTHONPATH=/home/yujun/program/hermes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DECORD_EOF_RETRY_MAX=40960

if /home/yujun/.conda/envs/hermes-qwen/bin/python -u head_analysis/lu_kv_offline.py \
  --calibration "$ROOT/calibration_100v.json" \
  --exclude-annotation data/streamingbench/streamingbench_realtime_half_userpaths.json \
  --output "$ROOT/traces" \
  --num-videos 100 \
  --questions-per-video 2 \
  --min-time 0 \
  --max-time 1000000000 \
  --seed 20260915 \
  --recent 32 \
  --budgets 1000 6000 \
  > "$LOG_DIR/offline.log" 2>&1; then
  printf 'complete %s\n' "$(date -u +%FT%TZ)" >> "$LOG_DIR/run.status"
else
  exit_code=$?
  printf 'failed exit_code=%s %s\n' "$exit_code" "$(date -u +%FT%TZ)" >> "$LOG_DIR/run.status"
  exit "$exit_code"
fi
