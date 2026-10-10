#!/usr/bin/env bash
set -euo pipefail

cd /home/yujun/program/hermes
export PYTHONPATH=/home/yujun/program/hermes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DECORD_EOF_RETRY_MAX=40960

PYTHON=/home/yujun/.conda/envs/hermes-qwen/bin/python
ANNO=results/myobservations/native_k500_first150/anno_first150.json
K500_STATUS=logs/lu_k500_first150/run.status
LOG_DIR=logs/native_k250_first150
STATUS="$LOG_DIR/serial.status"

mkdir -p "$LOG_DIR"

fail() {
  rc=$?
  printf 'failed rc=%s at %s\n' "$rc" "$(date --iso-8601=seconds)" > "$STATUS"
  exit "$rc"
}
trap fail ERR

printf 'queued behind K500 at %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
while true; do
  k500_state=$(awk 'NR==1 {print $1}' "$K500_STATUS" 2>/dev/null || true)
  if [[ "$k500_state" == "complete" ]]; then
    break
  fi
  if [[ "$k500_state" == "failed" ]]; then
    printf 'blocked because K500 failed at %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
    exit 1
  fi
  sleep 30
done

COMMON=(
  --model qwen2.5_vl_7b
  --dataset streamingbench
  --anno_path "$ANNO"
  --num_chunks 1
  --devices 0
  --sample_fps 0.5
  --kv_size 250
  --compress_mode hermes
  --kv_head_ragged_prefill true
  --kv_head_budget_scheme sparsemm
  --kv_head_budget_sparsemm_ratio 0.2
  --kv_head_budget_sparsemm_window_size 32
  --kv_head_budget_union_cap_ratio 1.0
  --debug false
)

run_one() {
  name=$1
  score=$2
  tag=$3
  printf 'running %s at %s\n' "$name" "$(date --iso-8601=seconds)" > "$STATUS"
  "$PYTHON" video_qa/run_infer.py "${COMMON[@]}" \
    --kv_head_budget_scores "$score" \
    --output_tag "$tag" \
    > "$LOG_DIR/${name}.log" 2>&1
}

run_one \
  hermes_uniform \
  results/observations/memory_core_budget_scores/uniform.csv \
  hermes-uniform-native-k250-first150

run_one \
  lu_old \
  results/myobservations/lu_kv_offline_50v_20260913/lu_raw_blend05_clip075_150_runtime_score.csv \
  lu-raw-blend05-clip075-150-native-first150-k250

printf 'complete at %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
