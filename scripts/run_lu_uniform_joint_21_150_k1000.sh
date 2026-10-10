#!/usr/bin/env bash
set -euo pipefail

cd /home/yujun/program/hermes
log_dir=logs/lu_uniform_joint_21_150_k1000
mkdir -p "$log_dir"
printf 'running pid=%s %s\n' "$$" "$(date -u +%FT%TZ)" > "$log_dir/run.status"

export PYTHONPATH=/home/yujun/program/hermes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DECORD_EOF_RETRY_MAX=40960

if /home/yujun/.conda/envs/hermes-qwen/bin/python -u video_qa/run_infer.py \
  --model qwen2.5_vl_7b \
  --dataset streamingbench \
  --anno_path results/myobservations/uniform_joint_online_21_150_20260920/anno_21_150.json \
  --num_chunks 1 \
  --devices 0 \
  --sample_fps 0.5 \
  --kv_size 1000 \
  --compress_mode hermes \
  --kv_head_budget_scores results/myobservations/lu_kv_offline_50v_20260913/lu_raw_blend05_clip075_150_runtime_score.csv \
  --kv_head_budget_scheme sparsemm \
  --kv_head_budget_sparsemm_ratio 0.2 \
  --kv_head_budget_sparsemm_window_size 32 \
  --kv_head_selector_policy uniform_joint \
  --kv_head_ragged_prefill true \
  --debug false \
  --output_tag lu-uniform-joint-native-21-150-k1000 \
  > "$log_dir/run.log" 2>&1; then
  printf 'complete %s\n' "$(date -u +%FT%TZ)" >> "$log_dir/run.status"
else
  code=$?
  printf 'failed exit_code=%s %s\n' "$code" "$(date -u +%FT%TZ)" >> "$log_dir/run.status"
  exit "$code"
fi
