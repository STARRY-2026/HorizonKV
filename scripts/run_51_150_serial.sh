#!/usr/bin/env bash
set -uo pipefail

cd /home/yujun/program/hermes
export PYTHONPATH=/home/yujun/program/hermes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DECORD_EOF_RETRY_MAX=40960

ANNO=results/myobservations/native_k1000_51_150/anno_51_150.json
LOG_DIR=logs/native_k1000_51_150
mkdir -p "$LOG_DIR"

echo "start $(date --iso-8601=seconds)" > "$LOG_DIR/serial.status"

echo "[1/2] v8 native k1000 51-150 $(date --iso-8601=seconds)" >> "$LOG_DIR/serial.status"
bash scripts/run_native_k1000_budget.sh \
  "$ANNO" results/myobservations/relocatev3/v8_blend3_scores.csv \
  v8-native-k1000-51-150 v8_native_k1000_51_150

echo "[2/2] hermes uniform native k1000 51-150 $(date --iso-8601=seconds)" >> "$LOG_DIR/serial.status"
bash scripts/run_native_k1000_budget.sh \
  "$ANNO" results/observations/memory_core_budget_scores/uniform.csv \
  hermes-uniform-native-k1000-51-150 hermes_uniform_native_k1000_51_150

echo "done $(date --iso-8601=seconds)" >> "$LOG_DIR/serial.status"
