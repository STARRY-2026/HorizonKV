#!/usr/bin/env bash
set -euo pipefail

cd /home/yujun/program/hermes
export PYTHONPATH=/home/yujun/program/hermes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DECORD_EOF_RETRY_MAX=40960

PYTHON=/home/yujun/.conda/envs/hermes-qwen/bin/python
ANNO=results/myobservations/native_k1000_51_150/anno_51_150.json
LU_SCORE=results/myobservations/lu_kv_offline_50v_20260913/lu_raw_k2000_resolved_all_blend05_clip075_150_runtime_score.csv
LOG_DIR=logs/lu_k2000_51_150
STATUS="$LOG_DIR/run.status"
LOG="$LOG_DIR/qa.log"

mkdir -p "$LOG_DIR"

fail() {
  rc=$?
  printf 'failed rc=%s at %s\n' "$rc" "$(date --iso-8601=seconds)" > "$STATUS"
  exit "$rc"
}
trap fail ERR

printf 'running pid=%s at %s\n' "$$" "$(date --iso-8601=seconds)" > "$STATUS"
"$PYTHON" -u video_qa/run_infer.py \
  --model qwen2.5_vl_7b \
  --dataset streamingbench \
  --anno_path "$ANNO" \
  --num_chunks 1 \
  --devices 0 \
  --sample_fps 0.5 \
  --kv_size 2000 \
  --compress_mode hermes \
  --kv_head_budget_scores "$LU_SCORE" \
  --kv_head_budget_scheme sparsemm \
  --kv_head_budget_sparsemm_ratio 0.2 \
  --kv_head_budget_sparsemm_window_size 32 \
  --kv_head_budget_union_cap_ratio 1.0 \
  --kv_head_ragged_prefill true \
  --output_tag lu-raw-k2000-resolved-all-blend05-clip075-150-native-51-150 \
  --debug false \
  > "$LOG" 2>&1

printf 'complete at %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
