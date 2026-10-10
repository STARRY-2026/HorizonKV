#!/usr/bin/env bash
set -euo pipefail

# usage: run_native_k1000_budget.sh <anno_relpath> <budget_scores_path> <output_tag> <log_name>
ANNO="$1"
SCORES="$2"
TAG="$3"
LOG_NAME="$4"

cd /home/yujun/program/hermes
export PYTHONPATH=/home/yujun/program/hermes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DECORD_EOF_RETRY_MAX=40960

PYTHON=/home/yujun/.conda/envs/hermes-qwen/bin/python
LOG_DIR=logs/native_k1000_budget
mkdir -p "$LOG_DIR"

"$PYTHON" video_qa/run_infer.py \
  --model qwen2.5_vl_7b \
  --dataset streamingbench \
  --anno_path "$ANNO" \
  --num_chunks 1 \
  --devices 0 \
  --sample_fps 0.5 \
  --kv_size 1000 \
  --compress_mode hermes \
  --kv_head_budget_scores "$SCORES" \
  --kv_head_budget_scheme sparsemm \
  --kv_head_budget_sparsemm_ratio 0.2 \
  --kv_head_budget_sparsemm_window_size 32 \
  --kv_head_ragged_prefill true \
  --debug false \
  --output_tag "$TAG" \
  > "$LOG_DIR/$LOG_NAME.log" 2>&1
