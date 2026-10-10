#!/usr/bin/env bash
set -euo pipefail

cd /home/yujun/program/hermes

PYTHON=/home/yujun/.conda/envs/hermes-qwen/bin/python
ANNO=results/myobservations/answer_head_router_pilot12_20260916/anno_pilot12.json
PROFILE=results/myobservations/query_head_answer_router_disjoint14_20260916/head_profile_scores.csv
LU_SCORE=results/myobservations/lu_kv_offline_50v_20260913/lu_raw_blend05_clip075_150_runtime_score.csv
LOG_DIR=logs/query_head_answer_router_pilot12_20260916
STATUS="$LOG_DIR/run.status"

mkdir -p "$LOG_DIR"
export PYTHONPATH=/home/yujun/program/hermes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DECORD_EOF_RETRY_MAX=40960

COMMON=(
  --model qwen2.5_vl_7b
  --dataset streamingbench
  --anno_path "$ANNO"
  --num_chunks 1
  --devices 0
  --sample_fps 0.5
  --kv_size 1000
  --compress_mode hermes
  --kv_head_budget_scores "$LU_SCORE"
  --kv_head_budget_scheme sparsemm
  --kv_head_budget_sparsemm_ratio 0.2
  --kv_head_budget_sparsemm_window_size 32
  --kv_head_ragged_prefill true
  --answer_head_router true
  --answer_head_profile "$PROFILE"
  --answer_head_router_strength 0.2
  --answer_head_min_gate 0.8
  --answer_head_max_gate 1.2
  --answer_head_start_layer 8
  --debug false
)

printf 'starting query-head route %s\n' "$(date -u +%FT%TZ)" > "$STATUS"
if "$PYTHON" -u video_qa/run_infer.py "${COMMON[@]}" \
  --answer_head_router_mode question \
  --output_tag lu-qhead-he-route-question-pilot12-k1000 \
  > "$LOG_DIR/question.log" 2>&1; then
  printf 'complete query-head route %s\n' "$(date -u +%FT%TZ)" >> "$STATUS"
else
  code=$?
  printf 'failed query-head route exit_code=%s %s\n' "$code" "$(date -u +%FT%TZ)" >> "$STATUS"
  exit "$code"
fi

printf 'starting query-head inverted control %s\n' "$(date -u +%FT%TZ)" >> "$STATUS"
if "$PYTHON" -u video_qa/run_infer.py "${COMMON[@]}" \
  --answer_head_router_mode inverted \
  --output_tag lu-qhead-he-route-inverted-pilot12-k1000 \
  > "$LOG_DIR/inverted.log" 2>&1; then
  printf 'complete query-head inverted control %s\n' "$(date -u +%FT%TZ)" >> "$STATUS"
else
  code=$?
  printf 'failed query-head inverted control exit_code=%s %s\n' "$code" "$(date -u +%FT%TZ)" >> "$STATUS"
  exit "$code"
fi
