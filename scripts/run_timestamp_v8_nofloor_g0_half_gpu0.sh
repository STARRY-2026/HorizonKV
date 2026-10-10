#!/usr/bin/env bash
set -uo pipefail

cd /home/yujun/program/hermes
export PYTHONPATH=/home/yujun/program/hermes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DECORD_EOF_RETRY_MAX=40960

PYTHON=/home/yujun/.conda/envs/hermes-qwen/bin/python
STATUS=logs/timestamp_v8_nofloor_g0_half_gpu0.status

echo "running $(date --iso-8601=seconds)" > "$STATUS"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" video_qa/run_infer.py \
  --model qwen2.5_vl_7b \
  --dataset streamingbench \
  --anno_path data/streamingbench/streamingbench_realtime_half_userpaths.json \
  --num_chunks 1 \
  --devices 0 \
  --sample_fps 0.5 \
  --kv_size 1000 \
  --compress_mode hermes \
  --kv_head_budget_scores results/myobservations/relocatev3/v8_blend3_scores.csv \
  --kv_head_budget_scheme sparsemm \
  --kv_head_budget_sparsemm_ratio 0.2 \
  --kv_head_budget_sparsemm_window_size 32 \
  --kv_head_budget_floor_ratio 0 \
  --kv_head_budget_union_cap_ratio 1.0 \
  --timestamp_v8_policy results/myobservations/relocatev3/v8_budget.csv \
  --timestamp_v8_grace_frames 0 \
  --kv_head_ragged_prefill true \
  --fixed_frame_canvas_size 448 \
  --fixed_frame_pad_mode letterbox \
  --debug false \
  --output_tag timestamp-v8-nofloor-g0-half-k1000-canvas448 \
  > logs/timestamp_v8_nofloor_g0_half_gpu0.log 2>&1
rc=$?

if (( rc == 0 )); then
  echo "done $(date --iso-8601=seconds)" > "$STATUS"
else
  echo "failed rc=$rc $(date --iso-8601=seconds)" > "$STATUS"
fi
exit "$rc"
