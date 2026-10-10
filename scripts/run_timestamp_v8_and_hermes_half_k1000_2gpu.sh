#!/usr/bin/env bash
set -uo pipefail

cd /home/yujun/program/hermes

export PYTHONPATH=/home/yujun/program/hermes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DECORD_EOF_RETRY_MAX=40960

PYTHON=/home/yujun/.conda/envs/hermes-qwen/bin/python
ANNO=data/streamingbench/streamingbench_realtime_half_userpaths.json
STATUS=logs/half_k1000_fixed448_2gpu.status
TIMESTAMP_STATUS=logs/timestamp_v8_half_k1000_fixed448.status
HERMES_STATUS=logs/hermes_half_k1000_fixed448.status
TIMESTAMP_LOG=logs/timestamp_v8_half_k1000_fixed448.log
HERMES_LOG=logs/hermes_half_k1000_fixed448.log

COMMON_ARGS=(
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
  --debug false
)

mkdir -p logs

run_timestamp_v8() {
  echo "running gpu=0 $(date --iso-8601=seconds)" > "$TIMESTAMP_STATUS"
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" video_qa/run_infer.py \
    "${COMMON_ARGS[@]}" \
    --devices 0 \
    --kv_head_budget_scores results/myobservations/relocatev3/v8_blend3_scores.csv \
    --kv_head_budget_scheme sparsemm \
    --kv_head_budget_sparsemm_ratio 0.2 \
    --kv_head_budget_sparsemm_window_size 32 \
    --kv_head_budget_union_cap_ratio 1.0 \
    --timestamp_v8_policy results/myobservations/relocatev3/v8_budget.csv \
    --timestamp_v8_grace_frames 1 \
    --output_tag timestamp-v8-half-k1000-canvas448 \
    > "$TIMESTAMP_LOG" 2>&1
  rc=$?
  if (( rc == 0 )); then
    echo "done gpu=0 $(date --iso-8601=seconds)" > "$TIMESTAMP_STATUS"
  else
    echo "failed rc=$rc gpu=0 $(date --iso-8601=seconds)" > "$TIMESTAMP_STATUS"
  fi
  return "$rc"
}

run_hermes() {
  echo "running gpu=1 $(date --iso-8601=seconds)" > "$HERMES_STATUS"
  CUDA_VISIBLE_DEVICES=1 "$PYTHON" video_qa/run_infer.py \
    "${COMMON_ARGS[@]}" \
    --devices 1 \
    --output_tag hermes-half-k1000-canvas448-uniform \
    > "$HERMES_LOG" 2>&1
  rc=$?
  if (( rc == 0 )); then
    echo "done gpu=1 $(date --iso-8601=seconds)" > "$HERMES_STATUS"
  else
    echo "failed rc=$rc gpu=1 $(date --iso-8601=seconds)" > "$HERMES_STATUS"
  fi
  return "$rc"
}

echo "both running $(date --iso-8601=seconds)" > "$STATUS"
run_timestamp_v8 &
timestamp_pid=$!
run_hermes &
hermes_pid=$!
printf 'timestamp_v8_pid=%s\nhermes_pid=%s\n' "$timestamp_pid" "$hermes_pid" \
  > logs/half_k1000_fixed448_2gpu.worker_pids

wait "$timestamp_pid"
timestamp_rc=$?
wait "$hermes_pid"
hermes_rc=$?

if (( timestamp_rc == 0 && hermes_rc == 0 )); then
  echo "both done $(date --iso-8601=seconds)" > "$STATUS"
  exit 0
fi

echo "failed timestamp_v8_rc=$timestamp_rc hermes_rc=$hermes_rc $(date --iso-8601=seconds)" > "$STATUS"
exit 1
