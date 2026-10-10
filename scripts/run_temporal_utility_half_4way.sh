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
STATUS=logs/temporal_utility_half_4way.status
WORKERS=logs/temporal_utility_half_4way.worker_pids

mkdir -p logs

run_one() {
  local gpu="$1" experiment="$2"
  local profile="$PROFILE_DIR/$experiment.csv"
  local log="logs/temporal_utility_${experiment}_half.log"
  local job_status="logs/temporal_utility_${experiment}_half.status"
  echo "running gpu=$gpu $(date --iso-8601=seconds)" > "$job_status"
  "$PYTHON" video_qa/run_infer.py \
    --model qwen2.5_vl_7b \
    --dataset streamingbench \
    --anno_path "$ANNO" \
    --num_chunks 1 \
    --devices "$gpu" \
    --sample_fps 0.5 \
    --kv_size 1000 \
    --compress_mode hermes \
    --kv_head_budget_scores results/myobservations/relocatev3/v8_blend3_scores.csv \
    --kv_head_budget_scheme sparsemm \
    --kv_head_budget_sparsemm_ratio 0.2 \
    --kv_head_budget_sparsemm_window_size 32 \
    --kv_head_budget_union_cap_ratio 1.0 \
    --timestamp_v8_policy "$profile" \
    --timestamp_v8_grace_frames 0 \
    --kv_head_ragged_prefill true \
    --fixed_frame_canvas_size 448 \
    --fixed_frame_pad_mode letterbox \
    --debug false \
    --output_tag "temporal-utility-${experiment}-tieavg-v1-half-k1000-canvas448" \
    > "$log" 2>&1
  local rc=$?
  if (( rc == 0 )); then
    echo "done gpu=$gpu $(date --iso-8601=seconds)" > "$job_status"
  else
    echo "failed rc=$rc gpu=$gpu $(date --iso-8601=seconds)" > "$job_status"
  fi
  return "$rc"
}

echo "running $(date --iso-8601=seconds)" > "$STATUS"
run_one 0 e1_global & p1=$!
run_one 1 e2_head_temporal & p2=$!
run_one 0 e3_full & p3=$!
run_one 1 e4_shuffled & p4=$!
printf 'e1_global=%s\ne2_head_temporal=%s\ne3_full=%s\ne4_shuffled=%s\n' \
  "$p1" "$p2" "$p3" "$p4" > "$WORKERS"

rc=0
for pid in "$p1" "$p2" "$p3" "$p4"; do
  wait "$pid" || rc=1
done
if (( rc == 0 )); then
  echo "done $(date --iso-8601=seconds)" > "$STATUS"
else
  echo "failed $(date --iso-8601=seconds)" > "$STATUS"
fi
exit "$rc"
