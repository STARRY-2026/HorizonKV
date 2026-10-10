#!/usr/bin/env bash
set -euo pipefail

cd /home/yujun/program/hermes
export PYTHONPATH=/home/yujun/program/hermes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DECORD_EOF_RETRY_MAX=40960

PYTHON=/home/yujun/.conda/envs/hermes-qwen/bin/python
LOG_DIR=logs/he_head_budget
mkdir -p "$LOG_DIR"

"$PYTHON" video_qa/run_infer.py \
  --model qwen2.5_vl_7b \
  --dataset streamingbench \
  --anno_path results/myobservations/temporal_zone_eval20/filtered_anno_first20.json \
  --num_chunks 1 \
  --devices 0 \
  --sample_fps 0.5 \
  --kv_size 2000 \
  --compress_mode hermes \
  --kv_head_budget_scores results/myobservations/he_head_classification/specialist_budget_score.csv \
  --kv_head_budget_scheme relative \
  --kv_head_ragged_prefill true \
  --fixed_frame_canvas_size 448 \
  --fixed_frame_pad_mode letterbox \
  --debug false \
  --output_tag specialist-budget-history-recent-first20-k2000 \
  > "$LOG_DIR/specialist_first20_k2000.log" 2>&1
