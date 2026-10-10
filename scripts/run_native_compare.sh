#!/usr/bin/env bash
set -euo pipefail

# usage: run_native_compare.sh <anno_relpath> <output_tag> <log_name> <kv_size> <budget_scores_or_uniform>
ANNO="$1"
TAG="$2"
LOG_NAME="$3"
KV_SIZE="$4"
SCORES="$5"

cd /home/yujun/program/hermes
export PYTHONPATH=/home/yujun/program/hermes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DECORD_EOF_RETRY_MAX=40960

PYTHON=/home/yujun/.conda/envs/hermes-qwen/bin/python
LOG_DIR=logs/native_compare
mkdir -p "$LOG_DIR"

ARGS=(
  --model qwen2.5_vl_7b
  --dataset streamingbench
  --anno_path "$ANNO"
  --num_chunks 1
  --devices 0
  --sample_fps 0.5
  --kv_size "$KV_SIZE"
  --compress_mode hermes
  --kv_head_ragged_prefill true
  --debug false
  --output_tag "$TAG"
)

if [ "$SCORES" != "uniform" ]; then
  ARGS+=(--kv_head_budget_scores "$SCORES" --kv_head_budget_scheme relative)
fi

"$PYTHON" video_qa/run_infer.py "${ARGS[@]}" \
  > "$LOG_DIR/$LOG_NAME.log" 2>&1
