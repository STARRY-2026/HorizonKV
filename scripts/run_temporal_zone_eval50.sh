#!/usr/bin/env bash
set -euo pipefail

cd /home/yujun/program/hermes

export PYTHONPATH=/home/yujun/program/hermes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DECORD_EOF_RETRY_MAX=40960

PYTHON=/home/yujun/.conda/envs/hermes-qwen/bin/python
ANNO=results/myobservations/temporal_zone_eval50/filtered_anno_first50.json
MODE=${1:-matched}
DEVICE=${2:-0}
NUM_CHUNKS=${3:-1}
GRACE_FRAMES=0

case "$MODE" in
  matched)
    POLICY=results/myobservations/temporal_zone_eval50/matched_profiles.csv
    ;;
  global)
    POLICY=results/myobservations/temporal_zone_eval50/global_profiles.csv
    ;;
  shuffled)
    POLICY=results/myobservations/temporal_zone_eval50/shuffled_profiles_seed2024.csv
    ;;
  selective)
    POLICY=results/myobservations/temporal_zone_eval50/matched_selective_protect_top1of8.csv
    GRACE_FRAMES=1
    ;;
  selective_top100)
    POLICY=results/myobservations/temporal_zone_eval50/matched_selective_protect_top1of8_top100current.csv
    GRACE_FRAMES=1
    ;;
  *)
    echo "Usage: $0 [matched|global|shuffled|selective|selective_top100] [devices] [num_chunks]" >&2
    exit 2
    ;;
esac

mkdir -p logs
"$PYTHON" video_qa/run_infer.py \
  --model qwen2.5_vl_7b \
  --dataset streamingbench \
  --anno_path "$ANNO" \
  --num_chunks "$NUM_CHUNKS" \
  --devices "$DEVICE" \
  --sample_fps 0.5 \
  --kv_size 1000 \
  --compress_mode hermes \
  --kv_head_ragged_prefill true \
  --timestamp_v8_policy "$POLICY" \
  --timestamp_v8_grace_frames "$GRACE_FRAMES" \
  --fixed_frame_canvas_size 448 \
  --fixed_frame_pad_mode letterbox \
  --debug false \
  --output_tag "temporal-zone-${MODE}-first50-k1000" \
  > "logs/temporal_zone_${MODE}_first50_k1000.log" 2>&1
