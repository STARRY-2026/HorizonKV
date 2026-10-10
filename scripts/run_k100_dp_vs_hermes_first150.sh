#!/usr/bin/env bash
set -euo pipefail

cd /home/yujun/program/hermes
export PYTHONPATH=/home/yujun/program/hermes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DECORD_EOF_RETRY_MAX=40960

PYTHON=/home/yujun/.conda/envs/hermes-qwen/bin/python
PROFILE=results/myobservations/two_bank_grid_dp_k100_floor75_20260929
LOG_DIR=logs/k100_dp_vs_hermes_first150_20260929
STATUS="$LOG_DIR/run.status"
RESULT_ROOT=results/qwen2.5_vl_7b/streamingbench
BASELINE_TAG=hermes-uniform-native-k100-first150-20260929
OUR_TAG=dp-cachecomp-alpha-k100-native-first150-20260929
BASELINE_DIR="$RESULT_ROOT/fps0.5-kv100-hermes-kvheadbudget-uniform-sparsemm-union1-r0.2-w32-raggedprefill-$BASELINE_TAG"
OUR_DIR="$RESULT_ROOT/fps0.5-kv100-hermes-kvheadbudget-dp_k100_exact_budget_scores-sparsemm-union1-r0.2-w32-raggedprefill-headalpha-recent_ratio_profile-d0.1-t1-a0-1-kd2-$OUR_TAG"
CURRENT=setup

mkdir -p "$LOG_DIR"
on_error() {
  rc=$?
  printf 'failed run=%s rc=%s at %s\n' "$CURRENT" "$rc" "$(date -u +'%F %T UTC')" > "$STATUS"
  exit "$rc"
}
trap on_error ERR

COMMON=(
  --model qwen2.5_vl_7b
  --dataset streamingbench
  --anno_path results/myobservations/native_k500_first150/anno_first150.json
  --num_chunks 1
  --devices 0
  --sample_fps 0.5
  --kv_size 100
  --compress_mode hermes
  --kv_head_ragged_prefill true
  --kv_head_budget_scheme sparsemm
  --kv_head_budget_sparsemm_ratio 0.2
  --kv_head_budget_sparsemm_window_size 32
  --kv_head_budget_union_cap_ratio 1.0
  --debug false
)

CURRENT=hermes_uniform
printf 'running %s at %s\n' "$CURRENT" "$(date -u +'%F %T UTC')" > "$STATUS"
"$PYTHON" video_qa/run_infer.py "${COMMON[@]}" \
  --kv_head_budget_scores results/observations/memory_core_budget_scores/uniform.csv \
  --output_tag "$BASELINE_TAG" \
  > "$LOG_DIR/hermes_uniform.log" 2>&1
test -s "$BASELINE_DIR/results.csv"
rg -q '^#Samples: 754$' "$BASELINE_DIR/eval_results.txt"

CURRENT=dp_alpha_k
printf 'running %s at %s\n' "$CURRENT" "$(date -u +'%F %T UTC')" > "$STATUS"
"$PYTHON" video_qa/run_infer.py "${COMMON[@]}" \
  --kv_head_budget_scores "$PROFILE/dp_k100_exact_budget_scores.csv" \
  --kv_head_recency_scores "$PROFILE/recent_ratio_profile.csv" \
  --kv_head_recency_metric recent_ratio \
  --kv_head_recency_delta 0.1 \
  --kv_head_recency_tau 1.0 \
  --kv_head_recency_alpha_min 0.0 \
  --kv_head_recency_alpha_max 1.0 \
  --kv_head_recency_k_delta 2.0 \
  --kv_head_recency_k_min 0.0 \
  --kv_head_recency_k_max 20.0 \
  --output_tag "$OUR_TAG" \
  > "$LOG_DIR/dp_alpha_k.log" 2>&1
test -s "$OUR_DIR/results.csv"
rg -q '^#Samples: 754$' "$OUR_DIR/eval_results.txt"

CURRENT=complete
printf 'complete at %s\n' "$(date -u +'%F %T UTC')" > "$STATUS"
