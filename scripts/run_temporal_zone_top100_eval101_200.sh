#!/usr/bin/env bash
set -euo pipefail

cd /home/yujun/program/hermes
export PYTHONPATH=/home/yujun/program/hermes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DECORD_EOF_RETRY_MAX=40960

PYTHON=/home/yujun/.conda/envs/hermes-qwen/bin/python
ANNO=results/myobservations/temporal_zone_eval101_200/filtered_anno_101_200.json
POLICY=results/myobservations/temporal_zone_eval50/matched_selective_protect_top1of8_top100current.csv
DEVICES=${1:-0,1}
NUM_CHUNKS=${2:-2}

"$PYTHON" video_qa/run_infer.py \
  --model qwen2.5_vl_7b \
  --dataset streamingbench \
  --anno_path "$ANNO" \
  --num_chunks "$NUM_CHUNKS" \
  --devices "$DEVICES" \
  --sample_fps 0.5 \
  --kv_size 1000 \
  --compress_mode hermes \
  --kv_head_ragged_prefill true \
  --timestamp_v8_policy "$POLICY" \
  --timestamp_v8_grace_frames 1 \
  --fixed_frame_canvas_size 448 \
  --fixed_frame_pad_mode letterbox \
  --debug false \
  --output_tag "temporal-zone-top100-videos101-200-k1000" \
  > logs/temporal_zone_top100_eval101_200.log 2>&1
