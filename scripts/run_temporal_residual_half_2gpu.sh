#!/usr/bin/env bash
set -uo pipefail

cd /home/yujun/program/hermes
export PYTHONPATH=/home/yujun/program/hermes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DECORD_EOF_RETRY_MAX=40960

PYTHON=/home/yujun/.conda/envs/hermes-qwen/bin/python
ANNO=data/streamingbench/streamingbench_realtime_half_userpaths.json
PROFILE_DIR=results/myobservations/temporal_utility_profiles_v1
STATUS=logs/temporal_residual_half_2gpu.status

run_one() {
  local name="$1" gpu="$2" profile="$3"
  local status="logs/temporal_residual_${name}_half.status"
  echo "running gpu=$gpu $(date --iso-8601=seconds)" > "$status"
  "$PYTHON" video_qa/run_infer.py \
    --model qwen2.5_vl_7b \
    --dataset streamingbench \
    --anno_path "$ANNO" \
    --num_chunks 1 \
    --devices "$gpu" \
    --sample_fps 0.5 \
    --kv_size 1000 \
    --compress_mode hermes \
    --timestamp_v8_policy "$PROFILE_DIR/$profile.csv" \
    --timestamp_v8_grace_frames 0 \
    --kv_head_ragged_prefill true \
    --fixed_frame_canvas_size 448 \
    --fixed_frame_pad_mode letterbox \
    --debug false \
    --output_tag "temporal-residual-${name}-d01-uniform-half-k1000-canvas448" \
    > "logs/temporal_residual_${name}_half.log" 2>&1
  local rc=$?
  if (( rc == 0 )); then
    echo "done gpu=$gpu $(date --iso-8601=seconds)" > "$status"
  else
    echo "failed rc=$rc gpu=$gpu $(date --iso-8601=seconds)" > "$status"
  fi
  return "$rc"
}

echo "running $(date --iso-8601=seconds)" > "$STATUS"
run_one head 0 residual_head & head_pid=$!
run_one shuffled 1 residual_shuffled & shuffled_pid=$!
printf 'head=%s\nshuffled=%s\n' "$head_pid" "$shuffled_pid" \
  > logs/temporal_residual_half_2gpu.worker_pids
wait "$head_pid"; head_rc=$?
wait "$shuffled_pid"; shuffled_rc=$?
if (( head_rc == 0 && shuffled_rc == 0 )); then
  echo "done $(date --iso-8601=seconds)" > "$STATUS"
  exit 0
fi
echo "failed head=$head_rc shuffled=$shuffled_rc $(date --iso-8601=seconds)" > "$STATUS"
exit 1
