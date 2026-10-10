#!/usr/bin/env bash
set -euo pipefail

cd /home/yujun/program/hermes

export PYTHONPATH=/home/yujun/program/hermes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DECORD_EOF_RETRY_MAX=40960

PYTHON=/home/yujun/.conda/envs/hermes-qwen/bin/python
ANNO=data/streamingbench/streamingbench_realtime_half.json
STATUS=logs/half_k2000_fixed448.status
TIMESTAMP_LOG=logs/timestamp_v8_half_k2000_fixed448.log
HERMES_LOG=logs/hermes_half_k2000_fixed448.log

COMMON_ARGS=(
  --model qwen2.5_vl_7b
  --dataset streamingbench
  --anno_path "$ANNO"
  --num_chunks 1
  --sample_fps 0.5
  --kv_size 2000
  --compress_mode hermes
  --kv_head_ragged_prefill true
  --fixed_frame_canvas_size 448
  --fixed_frame_pad_mode letterbox
  --debug false
)

mkdir -p logs
trap 'rc=$?; echo "failed rc=${rc} $(date --iso-8601=seconds)" > "$STATUS"; exit "$rc"' ERR

echo "timestamp_v8 running $(date --iso-8601=seconds)" > "$STATUS"
"$PYTHON" video_qa/run_infer.py \
  "${COMMON_ARGS[@]}" \
  --kv_head_budget_scores results/myobservations/relocatev3/v8_blend3_scores.csv \
  --kv_head_budget_scheme sparsemm \
  --kv_head_budget_sparsemm_ratio 0.2 \
  --kv_head_budget_sparsemm_window_size 32 \
  --kv_head_budget_union_cap_ratio 1.0 \
  --timestamp_v8_policy results/myobservations/relocatev3/v8_budget.csv \
  --timestamp_v8_grace_frames 1 \
  --output_tag timestamp-v8-half-k2000-canvas448 \
  > "$TIMESTAMP_LOG" 2>&1

echo "timestamp_v8 done; hermes running $(date --iso-8601=seconds)" > "$STATUS"
"$PYTHON" video_qa/run_infer.py \
  "${COMMON_ARGS[@]}" \
  --output_tag hermes-half-k2000-canvas448-uniform \
  > "$HERMES_LOG" 2>&1

echo "timestamp_v8 done; hermes done $(date --iso-8601=seconds)" > "$STATUS"
