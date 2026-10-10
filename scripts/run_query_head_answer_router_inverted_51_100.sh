#!/usr/bin/env bash
set -euo pipefail

cd /home/yujun/program/hermes

PYTHON=/home/yujun/.conda/envs/hermes-qwen/bin/python
LOG_DIR=logs/query_head_answer_router_51_100_20260916
STATUS="$LOG_DIR/inverted.status"
mkdir -p "$LOG_DIR"

export PYTHONPATH=/home/yujun/program/hermes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DECORD_EOF_RETRY_MAX=40960

printf 'starting query-head inverted control %s\n' "$(date -u +%FT%TZ)" > "$STATUS"
if "$PYTHON" -u video_qa/run_infer.py \
  --model qwen2.5_vl_7b \
  --dataset streamingbench \
  --anno_path results/myobservations/temporal_zone_eval51_100/filtered_anno_51_100.json \
  --num_chunks 1 \
  --devices 0 \
  --sample_fps 0.5 \
  --kv_size 1000 \
  --compress_mode hermes \
  --kv_head_budget_scores results/myobservations/lu_kv_offline_50v_20260913/lu_raw_blend05_clip075_150_runtime_score.csv \
  --kv_head_budget_scheme sparsemm \
  --kv_head_budget_sparsemm_ratio 0.2 \
  --kv_head_budget_sparsemm_window_size 32 \
  --kv_head_ragged_prefill true \
  --answer_head_router true \
  --answer_head_profile results/myobservations/query_head_answer_router_disjoint14_20260916/head_profile_scores.csv \
  --answer_head_router_mode inverted \
  --answer_head_router_strength 0.2 \
  --answer_head_min_gate 0.8 \
  --answer_head_max_gate 1.2 \
  --answer_head_start_layer 8 \
  --debug false \
  --output_tag lu-qhead-he-route-inverted-51-100-k1000 \
  > "$LOG_DIR/inverted.log" 2>&1; then
  printf 'complete query-head inverted control %s\n' "$(date -u +%FT%TZ)" >> "$STATUS"
else
  code=$?
  printf 'failed query-head inverted control exit_code=%s %s\n' "$code" "$(date -u +%FT%TZ)" >> "$STATUS"
  exit "$code"
fi
