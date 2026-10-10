#!/usr/bin/env bash
set -uo pipefail

cd /home/yujun/program/hermes

export PYTHONPATH=/home/yujun/program/hermes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DECORD_EOF_RETRY_MAX=40960

PYTHON=/home/yujun/.conda/envs/hermes-qwen/bin/python
ANNO=data/streamingbench/timestamp_v8_diagnostic3_userpaths.json
TRACE_DIR=results/cvpr/timestamp_v8_cache_trace
STATUS=logs/timestamp_v8_trace_diagnostic3.status
TIMESTAMP_LOG=logs/timestamp_v8_trace_diagnostic3.log
HERMES_LOG=logs/hermes_trace_diagnostic3.log
CURRENT_PIDS=logs/half_k1000_fixed448_2gpu.worker_pids

mkdir -p "$TRACE_DIR" logs

if [[ "${RUN_TRACE_NOW:-0}" != "1" && -f "$CURRENT_PIDS" ]]; then
  # shellcheck disable=SC1090
  source "$CURRENT_PIDS"
  echo "queued behind pids ${timestamp_v8_pid:-unknown},${hermes_pid:-unknown} $(date --iso-8601=seconds)" > "$STATUS"
  while kill -0 "${timestamp_v8_pid:-0}" 2>/dev/null || kill -0 "${hermes_pid:-0}" 2>/dev/null; do
    sleep 300
  done
fi

COMMON_ARGS=(
  --model qwen2.5_vl_7b
  --dataset streamingbench
  --anno_path "$ANNO"
  --num_chunks 1
  --sample_fps 0.5
  --kv_size 1000
  --compress_mode hermes
  --kv_head_ragged_prefill true
  --fixed_frame_canvas_size 448
  --fixed_frame_pad_mode letterbox
  --debug false
)

run_timestamp_v8() {
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" video_qa/run_infer.py \
    "${COMMON_ARGS[@]}" \
    --devices 0 \
    --kv_head_budget_scores results/myobservations/relocatev3/v8_blend3_scores.csv \
    --kv_head_budget_scheme sparsemm \
    --kv_head_budget_sparsemm_ratio 0.2 \
    --kv_head_budget_sparsemm_window_size 32 \
    --kv_head_budget_union_cap_ratio 1.0 \
    --timestamp_v8_policy results/myobservations/relocatev3/v8_budget.csv \
    --timestamp_v8_grace_frames 1 \
    --ragged_frame_trace_path "$TRACE_DIR/timestamp_v8_trace.jsonl" \
    --output_tag timestamp-v8-trace3-k1000-canvas448 \
    > "$TIMESTAMP_LOG" 2>&1
}

run_hermes() {
  CUDA_VISIBLE_DEVICES=1 "$PYTHON" video_qa/run_infer.py \
    "${COMMON_ARGS[@]}" \
    --devices 1 \
    --ragged_frame_trace_path "$TRACE_DIR/hermes_trace.jsonl" \
    --output_tag hermes-trace3-k1000-canvas448 \
    > "$HERMES_LOG" 2>&1
}

echo "running $(date --iso-8601=seconds)" > "$STATUS"
run_timestamp_v8 &
timestamp_trace_pid=$!
run_hermes &
hermes_trace_pid=$!
printf 'timestamp_trace_pid=%s\nhermes_trace_pid=%s\n' "$timestamp_trace_pid" "$hermes_trace_pid" \
  > logs/timestamp_v8_trace_diagnostic3.worker_pids

wait "$timestamp_trace_pid"
timestamp_rc=$?
wait "$hermes_trace_pid"
hermes_rc=$?

if (( timestamp_rc != 0 || hermes_rc != 0 )); then
  echo "failed timestamp_rc=$timestamp_rc hermes_rc=$hermes_rc $(date --iso-8601=seconds)" > "$STATUS"
  exit 1
fi

"$PYTHON" scripts/analyze_timestamp_v8_cache_trace.py \
  --timestamp_trace "$TRACE_DIR/timestamp_v8_trace.jsonl" \
  --hermes_trace "$TRACE_DIR/hermes_trace.jsonl" \
  --output "$TRACE_DIR/analysis.md"

echo "done $(date --iso-8601=seconds)" > "$STATUS"
