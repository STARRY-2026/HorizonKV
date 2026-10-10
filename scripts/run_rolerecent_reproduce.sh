#!/usr/bin/env bash
set -euo pipefail

cd /home/yujun/program/hermes
export PYTHONPATH=/home/yujun/program/hermes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DECORD_EOF_RETRY_MAX=40960

PYTHON=/home/yujun/.conda/envs/hermes-qwen/bin/python
LOG_DIR=logs/reproduce
mkdir -p "$LOG_DIR"

"$PYTHON" video_qa/run_infer.py \
  --model qwen2.5_vl_7b \
  --dataset streamingbench \
  --anno_path data/streamingbench/streamingbench_realtime_half_userpaths.json \
  --num_chunks 1 \
  --devices 0 \
  --sample_fps 0.5 \
  --kv_size 6000 \
  --compress_mode hermes \
  --kv_head_budget_scores results/observations/memory_core_budget_scores/uniform.csv \
  --kv_head_budget_scheme sparsemm \
  --kv_head_budget_sparsemm_ratio 0.2 \
  --kv_head_budget_sparsemm_window_size 32 \
  --kv_head_role_recent_scores results/observations/memory_core_budget_scores/ours_top100.csv \
  --kv_head_role_recent_memory_quantile 0.75 \
  --kv_head_role_recent_current_quantile 0.25 \
  --kv_head_role_recent_memory_count 16 \
  --kv_head_role_recent_current_count 16 \
  --kv_head_role_recent_memory_tokens 0 \
  --kv_head_role_recent_mixed_tokens 2000 \
  --kv_head_role_recent_current_tokens 4000 \
  --kv_head_role_recent_window_mode override \
  --kv_head_role_recent_uniform_budget true \
  --kv_head_ragged_prefill true \
  --defer_last_frame_compress true \
  --defer_last_frame_budget_mode strict \
  --debug false \
  --output_tag rolerecent-reproduce \
  > "$LOG_DIR/rolerecent_reproduce.log" 2>&1
