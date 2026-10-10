#!/usr/bin/env bash
set -euo pipefail

cd /home/yujun/program/hermes
export PYTHONPATH=/home/yujun/program/hermes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DECORD_EOF_RETRY_MAX=40960

EXPERIMENT_ANNO="${EXPERIMENT_ANNO:-results/myobservations/k250_cachecomp_score_pilot_20260923/anno_10v.json}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-cachecomp-k250-a01-kd2-10v-tmux-20260923}"
EXPERIMENT_BUDGET_SCORES="${EXPERIMENT_BUDGET_SCORES:-results/observations/memory_core_budget_scores/uniform.csv}"

exec /home/yujun/.conda/envs/hermes-qwen/bin/python video_qa/run_infer.py \
  --model qwen2.5_vl_7b \
  --dataset streamingbench \
  --anno_path "$EXPERIMENT_ANNO" \
  --num_chunks 1 \
  --devices 0 \
  --sample_fps 0.5 \
  --kv_size 250 \
  --compress_mode hermes \
  --kv_head_budget_scores "$EXPERIMENT_BUDGET_SCORES" \
  --kv_head_budget_scheme sparsemm \
  --kv_head_budget_sparsemm_ratio 0.2 \
  --kv_head_budget_sparsemm_window_size 32 \
  --kv_head_budget_union_cap_ratio 1.0 \
  --kv_head_ragged_prefill true \
  --kv_head_recency_scores results/myobservations/k250_cachecomp_score_pilot_20260923/recent_ratio_profile.csv \
  --kv_head_recency_metric recent_ratio \
  --kv_head_recency_delta 0.1 \
  --kv_head_recency_tau 1.0 \
  --kv_head_recency_alpha_min 0.0 \
  --kv_head_recency_alpha_max 1.0 \
  --kv_head_recency_k_delta 2.0 \
  --kv_head_recency_k_min 0.0 \
  --kv_head_recency_k_max 20.0 \
  --output_tag "$EXPERIMENT_TAG" \
  --debug false
