#!/usr/bin/env bash
set -euo pipefail

cd /home/yujun/program/hermes
export PYTHONPATH=/home/yujun/program/hermes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DECORD_EOF_RETRY_MAX=40960

PYTHON=/home/yujun/.conda/envs/hermes-qwen/bin/python
ROOT=results/myobservations/lu_head_budget_defer_pilot_k1000_20260921
ANNO=results/myobservations/lu_failure_trace_first150_k1000_20260921/filtered_anno.json
LU_SCORE=results/myobservations/lu_kv_offline_50v_20260913/lu_raw_blend05_clip075_150_runtime_score.csv
LOG_DIR=logs/lu_head_budget_defer_pilot_k1000_20260921
STATUS="$LOG_DIR/run.status"

mkdir -p "$ROOT" "$LOG_DIR"

fail() {
  rc=$?
  printf 'failed rc=%s at %s\n' "$rc" "$(date --iso-8601=seconds)" > "$STATUS"
  exit "$rc"
}
trap fail ERR

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
  --kv_head_budget_union_cap_ratio 1.0
  --kv_head_ragged_prefill true
  --defer_last_frame_compress true
  --defer_last_frame_budget_mode strict
  --defer_last_frame_head_budget_threshold 1000
  --debug false
)

for ratio in 0.1 0.2 0.3; do
  label=${ratio/./}
  printf 'running ratio=%s pid=%s at %s\n' "$ratio" "$$" \
    "$(date --iso-8601=seconds)" > "$STATUS"
  "$PYTHON" -u video_qa/run_infer.py "${COMMON[@]}" \
    --defer_last_frame_keep_ratio "$ratio" \
    --ragged_frame_trace_path "$ROOT/cache_trace_r${label}.jsonl" \
    --output_tag "lu-head-budget-defer-r${label}-native-k1000" \
    > "$LOG_DIR/r${label}.log" 2>&1
done

"$PYTHON" scripts/analyze_lu_head_budget_defer_pilot.py > "$ROOT/analysis_stdout.json"
printf 'complete at %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
