#!/usr/bin/env bash
set -euo pipefail

cd /home/yujun/program/hermes
export PYTHONPATH=/home/yujun/program/hermes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DECORD_EOF_RETRY_MAX=40960

PYTHON=/home/yujun/.conda/envs/hermes-qwen/bin/python
ROOT=results/myobservations/protected_regression_diagnostic_101_200
ANNO="$ROOT/filtered_anno.json"
PROFILE=results/myobservations/temporal_survival_profiles_disjoint
STATUS=logs/temporal_survival_protected_regression_2gpu.status

COMMON=(
  --model qwen2.5_vl_7b
  --dataset streamingbench
  --anno_path "$ANNO"
  --num_chunks 1
  --sample_fps 0.5
  --kv_size 1000
  --compress_mode hermes
  --kv_head_ragged_prefill true
  --fixed_frame_canvas_size 448
  --fixed_frame_pad_mode letterbox
  --timestamp_v8_grace_frames 0
  --debug false
)

run_one() {
  local name="$1" gpu="$2" policy="$3"
  echo "running gpu=$gpu $(date --iso-8601=seconds)" > "logs/temporal_survival_${name}.status"
  "$PYTHON" video_qa/run_infer.py "${COMMON[@]}" \
    --devices "$gpu" \
    --timestamp_v8_policy "$policy" \
    --output_tag "temporal-survival-${name}-protected-regression" \
    > "logs/temporal_survival_${name}.log" 2>&1
  echo "done gpu=$gpu $(date --iso-8601=seconds)" > "logs/temporal_survival_${name}.status"
}

echo "running $(date --iso-8601=seconds)" > "$STATUS"
run_one head 0 "$PROFILE/survival_head.csv" & head_pid=$!
run_one content 1 "$PROFILE/survival_content.csv" & content_pid=$!
printf 'head=%s\ncontent=%s\n' "$head_pid" "$content_pid" \
  > logs/temporal_survival_protected_regression_2gpu.worker_pids
wait "$head_pid"
wait "$content_pid"
echo "done $(date --iso-8601=seconds)" > "$STATUS"
