#!/usr/bin/env bash
set -euo pipefail

cd /home/yujun/program/hermes
export PYTHONPATH=/home/yujun/program/hermes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PYTHON=/home/yujun/.conda/envs/hermes-llava/bin/python
ROOT=results/myobservations/temporal_profile_llavaov7b_same50

"$PYTHON" scripts/build_llava_same50_shards.py

run_shard() {
  local gpu="$1"
  local name="$2"
  "$PYTHON" head_analysis/profile_horizonkv_pilot.py \
    --model llava_ov_7b \
    --anno_path "$ROOT/input_split_${name}.json" \
    --num_videos 25 \
    --max_last_end 100000 \
    --sample_fps 0.5 \
    --encode_chunk_size 16 \
    --query_pool last_n \
    --last_n 4 \
    --topk_values 1,4,16,64,100 \
    --device "$gpu" \
    --save_dir "$ROOT/shard_${name}" \
    > "$ROOT/shard_${name}.log" 2>&1
}

run_shard 0 even & pid_even=$!
run_shard 1 odd & pid_odd=$!
printf 'even_pid=%s\nodd_pid=%s\n' "$pid_even" "$pid_odd" > "$ROOT/worker_pids"

set +e
wait "$pid_even"; rc_even=$?
wait "$pid_odd"; rc_odd=$?
set -e
printf 'even_rc=%s\nodd_rc=%s\n' "$rc_even" "$rc_odd" > "$ROOT/worker_status"
if [[ "$rc_even" -ne 0 || "$rc_odd" -ne 0 ]]; then
  exit 1
fi

"$PYTHON" scripts/merge_llava_same50_profile.py
mkdir -p "$ROOT/analysis" "$ROOT/vector_stability"
"$PYTHON" head_analysis/analyze_temporal_topk_profile.py \
  --raw_csv "$ROOT/raw_age_bin_attention.csv" \
  --save_dir "$ROOT/analysis" \
  --topk_values 1,4,16,64,100
"$PYTHON" head_analysis/analyze_temporal_vector_stability.py \
  --raw_csv "$ROOT/raw_age_bin_attention.csv" \
  --save_dir "$ROOT/vector_stability"
