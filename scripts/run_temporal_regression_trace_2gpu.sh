#!/usr/bin/env bash
set -euo pipefail

cd /home/yujun/program/hermes
export PYTHONPATH=/home/yujun/program/hermes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DECORD_EOF_RETRY_MAX=40960

PYTHON=/home/yujun/.conda/envs/hermes-qwen/bin/python
ROOT=results/myobservations/temporal_regression_diagnostic
ANNO="$ROOT/filtered_anno.json"

COMMON=(
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
  --ragged_readout_trace_last_n 1
  --debug false
)

"$PYTHON" video_qa/run_infer.py "${COMMON[@]}" \
  --devices 0 \
  --ragged_frame_trace_path "$ROOT/hermes_cache_trace.jsonl" \
  --ragged_readout_trace_path "$ROOT/hermes_readout_trace.jsonl" \
  --output_tag temporal-regression-trace-hermes \
  > logs/temporal_regression_trace_hermes.log 2>&1 &
HERMES_PID=$!

"$PYTHON" video_qa/run_infer.py "${COMMON[@]}" \
  --devices 1 \
  --timestamp_v8_policy results/myobservations/temporal_zone_eval50/matched_profiles.csv \
  --timestamp_v8_grace_frames 0 \
  --ragged_frame_trace_path "$ROOT/matched_cache_trace.jsonl" \
  --ragged_readout_trace_path "$ROOT/matched_readout_trace.jsonl" \
  --output_tag temporal-regression-trace-matched \
  > logs/temporal_regression_trace_matched.log 2>&1 &
MATCHED_PID=$!

wait "$HERMES_PID"
wait "$MATCHED_PID"
