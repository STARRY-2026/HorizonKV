#!/usr/bin/env bash
set -euo pipefail

# usage: run_content_sim_trace.sh <anno_relpath> <trace_relpath> <output_tag> <log_name>
ANNO="$1"
TRACE="$2"
TAG="$3"
LOG_NAME="$4"

cd /home/yujun/program/hermes
export PYTHONPATH=/home/yujun/program/hermes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DECORD_EOF_RETRY_MAX=40960

PYTHON=/home/yujun/.conda/envs/hermes-qwen/bin/python
LOG_DIR=logs/content_sim
mkdir -p "$LOG_DIR"

"$PYTHON" video_qa/run_infer.py \
  --model qwen2.5_vl_7b \
  --dataset streamingbench \
  --anno_path "$ANNO" \
  --num_chunks 1 \
  --devices 0 \
  --sample_fps 0.5 \
  --kv_size 1000000 \
  --compress_mode hermes \
  --kv_head_ragged_prefill true \
  --prospective_lifecycle_trace_path "$TRACE" \
  --prospective_lifecycle_trace_last_n 1 \
  --fixed_frame_canvas_size 448 \
  --fixed_frame_pad_mode letterbox \
  --debug false \
  --output_tag "$TAG" \
  > "$LOG_DIR/$LOG_NAME.log" 2>&1
