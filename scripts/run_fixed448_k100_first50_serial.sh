#!/usr/bin/env bash
set -euo pipefail

cd /home/yujun/program/hermes
export PYTHONPATH=/home/yujun/program/hermes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DECORD_EOF_RETRY_MAX=40960

PYTHON=/home/yujun/.conda/envs/hermes-qwen/bin/python
ANNO=results/myobservations/temporal_zone_eval20/filtered_anno_first50_userpaths.json
LU_SCORE=results/myobservations/lu_kv_offline_50v_20260913/lu_raw_k100_w16_r02_scaled_exact_blend05_clip075_150_runtime_score.csv
LOG_DIR=logs/fixed448_k100_w16_r02_first50
STATUS="$LOG_DIR/serial.status"
mkdir -p "$LOG_DIR"

COMMON=(
  --model qwen2.5_vl_7b
  --dataset streamingbench
  --anno_path "$ANNO"
  --num_chunks 1
  --devices 0
  --sample_fps 0.5
  --kv_size 100
  --compress_mode hermes
  --kv_head_ragged_prefill true
  --kv_head_budget_scheme sparsemm
  --kv_head_budget_sparsemm_ratio 0.2
  --kv_head_budget_sparsemm_window_size 16
  --kv_head_budget_union_cap_ratio 1.0
  --fixed_frame_canvas_size 448
  --fixed_frame_pad_mode letterbox
  --debug false
)

fail() {
  rc=$?
  printf 'failed rc=%s at %s\n' "$rc" "$(date --iso-8601=seconds)" > "$STATUS"
  exit "$rc"
}
trap fail ERR

printf 'running lu pid=%s at %s\n' "$$" "$(date --iso-8601=seconds)" > "$STATUS"
"$PYTHON" -u video_qa/run_infer.py "${COMMON[@]}" \
  --kv_head_budget_scores "$LU_SCORE" \
  --output_tag lu-k100-w16-r02-scaled-fixed448-first50 \
  > "$LOG_DIR/lu.log" 2>&1

printf 'running hermes pid=%s at %s\n' "$$" "$(date --iso-8601=seconds)" > "$STATUS"
"$PYTHON" -u video_qa/run_infer.py "${COMMON[@]}" \
  --kv_head_budget_scores results/observations/memory_core_budget_scores/uniform.csv \
  --output_tag hermes-uniform-fixed448-k100-w16-r02-first50 \
  > "$LOG_DIR/hermes.log" 2>&1

printf 'complete at %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
