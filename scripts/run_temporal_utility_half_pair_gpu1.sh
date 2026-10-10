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

run_one() {
  local experiment="$1"
  local log="logs/temporal_utility_${experiment}_half.log"
  local status="logs/temporal_utility_${experiment}_half.status"
  echo "running gpu=1 retry $(date --iso-8601=seconds)" > "$status"
  "$PYTHON" video_qa/run_infer.py \
    --model qwen2.5_vl_7b \
    --dataset streamingbench \
    --anno_path "$ANNO" \
    --num_chunks 1 \
    --devices 1 \
    --sample_fps 0.5 \
    --kv_size 1000 \
    --compress_mode hermes \
    --kv_head_budget_scores results/myobservations/relocatev3/v8_blend3_scores.csv \
    --kv_head_budget_scheme sparsemm \
    --kv_head_budget_sparsemm_ratio 0.2 \
    --kv_head_budget_sparsemm_window_size 32 \
    --kv_head_budget_union_cap_ratio 1.0 \
    --timestamp_v8_policy "$PROFILE_DIR/$experiment.csv" \
    --timestamp_v8_grace_frames 0 \
    --kv_head_ragged_prefill true \
    --fixed_frame_canvas_size 448 \
    --fixed_frame_pad_mode letterbox \
    --debug false \
    --output_tag "temporal-utility-${experiment}-tieavg-v1-half-k1000-canvas448" \
    > "$log" 2>&1
  local rc=$?
  if (( rc == 0 )); then
    echo "done gpu=1 $(date --iso-8601=seconds)" > "$status"
  else
    echo "failed rc=$rc gpu=1 $(date --iso-8601=seconds)" > "$status"
  fi
  return "$rc"
}

run_one e1_global & p1=$!
run_one e3_full & p3=$!
printf 'e1_global=%s\ne3_full=%s\n' "$p1" "$p3" \
  > logs/temporal_utility_half_pair_gpu1.worker_pids
wait "$p1"; rc1=$?
wait "$p3"; rc3=$?
if (( rc1 == 0 && rc3 == 0 )); then
  echo "done $(date --iso-8601=seconds)" > logs/temporal_utility_half_pair_gpu1.status
  exit 0
fi
echo "failed e1=$rc1 e3=$rc3 $(date --iso-8601=seconds)" \
  > logs/temporal_utility_half_pair_gpu1.status
exit 1
