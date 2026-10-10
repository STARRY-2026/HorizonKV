#!/usr/bin/env bash
set -euo pipefail

cd /home/yujun/program/hermes
export PYTHONPATH=/home/yujun/program/hermes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DECORD_EOF_RETRY_MAX=40960

PYTHON=/home/yujun/.conda/envs/hermes-qwen/bin/python
ANNO=results/myobservations/native_k500_first150/anno_first150.json
LOG_DIR=logs/native_k2000_first150
STATUS="$LOG_DIR/serial.status"
mkdir -p "$LOG_DIR"

COMMON=(
  --model qwen2.5_vl_7b
  --dataset streamingbench
  --anno_path "$ANNO"
  --num_chunks 1
  --devices 0
  --sample_fps 0.5
  --kv_size 2000
  --compress_mode hermes
  --kv_head_ragged_prefill true
  --kv_head_budget_scheme sparsemm
  --kv_head_budget_sparsemm_ratio 0.2
  --kv_head_budget_sparsemm_window_size 32
  --kv_head_budget_union_cap_ratio 1.0
  --debug false
)

fail() {
  rc=$?
  printf 'failed rc=%s at %s\n' "$rc" "$(date --iso-8601=seconds)" > "$STATUS"
  exit "$rc"
}
trap fail ERR

printf 'running hermes at %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
"$PYTHON" video_qa/run_infer.py "${COMMON[@]}" \
  --kv_head_budget_scores results/observations/memory_core_budget_scores/uniform.csv \
  --output_tag hermes-uniform-native-k2000-first150 \
  > "$LOG_DIR/hermes_uniform_native_k2000_first150.log" 2>&1

printf 'running v8 at %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
"$PYTHON" video_qa/run_infer.py "${COMMON[@]}" \
  --kv_head_budget_scores results/myobservations/relocatev3/v8_blend3_scores.csv \
  --output_tag v8-native-k2000-first150 \
  > "$LOG_DIR/v8_native_k2000_first150.log" 2>&1

printf 'done at %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
