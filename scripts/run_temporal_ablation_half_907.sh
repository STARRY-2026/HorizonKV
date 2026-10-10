#!/usr/bin/env bash
set -uo pipefail

cd /home/yujun/program/hermes
export PYTHONPATH=/home/yujun/program/hermes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DECORD_EOF_RETRY_MAX=40960

PYTHON=/home/yujun/.conda/envs/hermes-qwen/bin/python
ANNO=data/streamingbench/streamingbench_realtime_half_userpaths.json
PROFILE=results/myobservations/temporal_half_ablation_profiles
LOG_DIR=logs/907

mkdir -p "$LOG_DIR"

run_one() {
  local name="$1" gpu="$2" profile_name="$3"
  local status="$LOG_DIR/${name}.status"
  local log="$LOG_DIR/${name}.log"
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
    --timestamp_v8_policy "$PROFILE/${profile_name}.csv" \
    --timestamp_v8_grace_frames 0 \
    --kv_head_ragged_prefill true \
    --fixed_frame_canvas_size 448 \
    --fixed_frame_pad_mode letterbox \
    --debug false \
    --output_tag "907-${name}-half-k1000" \
    > "$log" 2>&1
  local rc=$?
  if (( rc == 0 )); then
    echo "done gpu=$gpu $(date --iso-8601=seconds)" > "$status"
  else
    echo "failed rc=$rc gpu=$gpu $(date --iso-8601=seconds)" > "$status"
  fi
  return "$rc"
}

echo "running $(date --iso-8601=seconds)" > "$LOG_DIR/launcher.status"
run_one content 0 survival_content & p0=$!
run_one relative_head 0 relative_head & p1=$!
run_one global 1 survival_global & p2=$!
run_one relative_shuffled 1 relative_shuffled & p3=$!
printf 'content=%s\nrelative_head=%s\nglobal=%s\nrelative_shuffled=%s\n' \
  "$p0" "$p1" "$p2" "$p3" > "$LOG_DIR/worker_pids"

rc=0
for pid in "$p0" "$p1" "$p2" "$p3"; do
  wait "$pid" || rc=1
done
if (( rc == 0 )); then
  echo "done $(date --iso-8601=seconds)" > "$LOG_DIR/launcher.status"
else
  echo "failed $(date --iso-8601=seconds)" > "$LOG_DIR/launcher.status"
fi
exit "$rc"
