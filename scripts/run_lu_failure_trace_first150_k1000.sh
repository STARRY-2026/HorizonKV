#!/usr/bin/env bash
set -euo pipefail

cd /home/yujun/program/hermes
export PYTHONPATH=/home/yujun/program/hermes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DECORD_EOF_RETRY_MAX=40960

PYTHON=/home/yujun/.conda/envs/hermes-qwen/bin/python
ROOT=results/myobservations/lu_failure_trace_first150_k1000_20260921
ANNO="$ROOT/filtered_anno.json"
LOG_DIR=logs/lu_failure_trace_first150_k1000_20260921
STATUS="$LOG_DIR/run.status"

mkdir -p "$LOG_DIR"

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
  --kv_head_budget_scheme sparsemm
  --kv_head_budget_sparsemm_ratio 0.2
  --kv_head_budget_sparsemm_window_size 32
  --kv_head_budget_union_cap_ratio 1.0
  --kv_head_ragged_prefill true
  --ragged_readout_trace_last_n 1
  --debug false
)

printf 'running hermes pid=%s at %s\n' "$$" "$(date --iso-8601=seconds)" > "$STATUS"
"$PYTHON" -u video_qa/run_infer.py "${COMMON[@]}" \
  --kv_head_budget_scores results/observations/memory_core_budget_scores/uniform.csv \
  --ragged_frame_trace_path "$ROOT/hermes_cache_trace.jsonl" \
  --ragged_readout_trace_path "$ROOT/hermes_readout_trace.jsonl" \
  --output_tag lu-failure-trace-hermes-native-k1000 \
  > "$LOG_DIR/hermes.log" 2>&1

printf 'running lu pid=%s at %s\n' "$$" "$(date --iso-8601=seconds)" > "$STATUS"
"$PYTHON" -u video_qa/run_infer.py "${COMMON[@]}" \
  --kv_head_budget_scores results/myobservations/lu_kv_offline_50v_20260913/lu_raw_blend05_clip075_150_runtime_score.csv \
  --ragged_frame_trace_path "$ROOT/lu_cache_trace.jsonl" \
  --ragged_readout_trace_path "$ROOT/lu_readout_trace.jsonl" \
  --output_tag lu-failure-trace-lu-native-k1000 \
  > "$LOG_DIR/lu.log" 2>&1

printf 'complete at %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
