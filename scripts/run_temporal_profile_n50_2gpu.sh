#!/usr/bin/env bash
set -euo pipefail

cd /home/yujun/program/hermes
export PYTHONPATH=/home/yujun/program/hermes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PYTHON=/home/yujun/.conda/envs/hermes-qwen/bin/python
ROOT=results/myobservations/temporal_topk_profile_50v_fixed448

run_shard() {
  local gpu="$1"
  "$PYTHON" head_analysis/profile_horizonkv_pilot.py \
    --anno_path "$ROOT/new40_gpu${gpu}.json" \
    --device "$gpu" \
    --num_videos 20 \
    --max_last_end 100000 \
    --sample_fps 0.5 \
    --fixed_frame_canvas_size 448 \
    --query_pool last_n \
    --last_n 4 \
    --topk_values 1,4,16,64,100 \
    --seed 0 \
    --save_dir "$ROOT/shard_gpu${gpu}" \
    > "logs/temporal_profile_n50_gpu${gpu}.log" 2>&1
}

run_shard 0 & P0=$!
run_shard 1 & P1=$!
printf 'gpu0_pid=%s\ngpu1_pid=%s\n' "$P0" "$P1" > logs/temporal_profile_n50.worker_pids
wait "$P0"
wait "$P1"

"$PYTHON" scripts/merge_temporal_profile_n50.py \
  > logs/temporal_profile_n50_merge.log 2>&1
"$PYTHON" head_analysis/analyze_temporal_topk_profile.py \
  --raw_csv "$ROOT/raw_age_bin_attention.csv" \
  --save_dir "$ROOT/analysis" \
  > logs/temporal_profile_n50_analysis.log 2>&1
"$PYTHON" head_analysis/analyze_temporal_vector_stability.py \
  --raw_csv "$ROOT/raw_age_bin_attention.csv" \
  --save_dir "$ROOT/vector_stability" \
  > logs/temporal_profile_n50_vector_stability.log 2>&1
