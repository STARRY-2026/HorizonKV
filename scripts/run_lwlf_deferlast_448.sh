#!/usr/bin/env bash
set -euo pipefail

# usage: run_lwlf_deferlast_448.sh <anno_relpath> <kv_size> <output_tag> <log_name>
ANNO="$1"
KV_SIZE="$2"
TAG="$3"
LOG_NAME="$4"

cd /home/yujun/program/hermes
export PYTHONPATH=/home/yujun/program/hermes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DECORD_EOF_RETRY_MAX=40960

PYTHON=/home/yujun/.conda/envs/hermes-qwen/bin/python
LOG_DIR=logs/lwlf_deferlast_448
mkdir -p "$LOG_DIR"

"$PYTHON" video_qa/run_infer.py \
  --model qwen2.5_vl_7b \
  --dataset streamingbench \
  --anno_path "$ANNO" \
  --num_chunks 1 \
  --devices 0 \
  --sample_fps 0.5 \
  --kv_size "$KV_SIZE" \
  --compress_mode hermes \
  --kv_head_budget_scores results/observations/memory_core_budget_scores/uniform.csv \
  --kv_head_budget_scheme sparsemm \
  --kv_head_budget_sparsemm_ratio 0.2 \
  --kv_head_budget_sparsemm_window_size 32 \
  --kv_head_layerwise_last_frame true \
  --kv_head_layerwise_shallow_end 10 \
  --kv_head_layerwise_mid_end 20 \
  --kv_head_layerwise_shallow_ratio 1.0 \
  --kv_head_layerwise_mid_ratio 0.0 \
  --kv_head_layerwise_deep_ratio 0.0 \
  --defer_last_frame_compress true \
  --defer_last_frame_budget_mode strict \
  --kv_head_ragged_prefill true \
  --fixed_frame_canvas_size 448 \
  --fixed_frame_pad_mode letterbox \
  --debug false \
  --output_tag "$TAG" \
  > "$LOG_DIR/$LOG_NAME.log" 2>&1
