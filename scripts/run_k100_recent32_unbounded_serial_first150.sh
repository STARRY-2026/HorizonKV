#!/usr/bin/env bash
set -euo pipefail

cd /home/yujun/program/hermes
export PYTHONPATH=/home/yujun/program/hermes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DECORD_EOF_RETRY_MAX=40960

PYTHON=/home/yujun/.conda/envs/hermes-qwen/bin/python
PROFILE=results/myobservations/two_bank_k100_recent32_only_20260930
LOG_DIR=logs/k100_recent32_unbounded_first150_20260930
STATUS="$LOG_DIR/run.status"
ROOT=results/qwen2.5_vl_7b/streamingbench
PLAIN_TAG=dp-recent32-unbounded-noak-k100-first150-20260930
AK_TAG=dp-recent32-unbounded-ak-k100-first150-20260930
COMMON_DIR="$ROOT/fps0.5-kv100-hermes-kvheadbudget-exact_budget_scores-sparsemm-union1-r0-w32-raggedprefill"
PLAIN_DIR="$COMMON_DIR-$PLAIN_TAG"
AK_DIR="$COMMON_DIR-headalpha-recent_ratio_profile-d0.1-t1-a0-1-kd2-$AK_TAG"
CURRENT=setup

mkdir -p "$LOG_DIR"
if [[ -e "$PLAIN_DIR" || -e "$AK_DIR" ]]; then
  printf 'Refusing to overwrite existing result directory\n' >&2
  exit 1
fi
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
  --kv_head_budget_sparsemm_ratio 0.0
  --kv_head_budget_sparsemm_window_size 32
  --kv_head_budget_union_cap_ratio 1.0
  --kv_head_budget_scores "$PROFILE/exact_budget_scores.csv"
  --debug false
)

CURRENT=budget_only
printf 'running %s at %s\n' "$CURRENT" "$(date -u +'%F %T UTC')" > "$STATUS"
"$PYTHON" video_qa/run_infer.py "${COMMON[@]}" \
  --output_tag "$PLAIN_TAG" \
  > "$LOG_DIR/budget_only.log" 2>&1
test -s "$PLAIN_DIR/results.csv"
rg -q '^#Samples: 754$' "$PLAIN_DIR/eval_results.txt"

CURRENT=budget_plus_alpha_k
printf 'running %s at %s\n' "$CURRENT" "$(date -u +'%F %T UTC')" > "$STATUS"
"$PYTHON" video_qa/run_infer.py "${COMMON[@]}" \
  --kv_head_recency_scores "$PROFILE/recent_ratio_profile.csv" \
  --kv_head_recency_metric recent_ratio \
  --kv_head_recency_delta 0.1 \
  --kv_head_recency_tau 1.0 \
  --kv_head_recency_alpha_min 0.0 \
  --kv_head_recency_alpha_max 1.0 \
  --kv_head_recency_k_delta 2.0 \
  --kv_head_recency_k_min 0.0 \
  --kv_head_recency_k_max 20.0 \
  --output_tag "$AK_TAG" \
  > "$LOG_DIR/budget_plus_alpha_k.log" 2>&1
test -s "$AK_DIR/results.csv"
rg -q '^#Samples: 754$' "$AK_DIR/eval_results.txt"

CURRENT=complete
printf 'complete at %s\n' "$(date -u +'%F %T UTC')" > "$STATUS"
