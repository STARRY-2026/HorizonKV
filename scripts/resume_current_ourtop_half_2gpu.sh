#!/usr/bin/env bash
set -uo pipefail

cd /home/yujun/program/hermes
export PYTHONPATH=/home/yujun/program/hermes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DECORD_EOF_RETRY_MAX=40960

PYTHON=/home/yujun/.conda/envs/hermes-qwen/bin/python
ROOT=results/myobservations/current_ourtop_k1000_fixed448_g0
STATUS=logs/current_ourtop_half_2gpu_resume.status
LOG=logs/current_ourtop_half_2gpu_resume.log
SAVE_DIR=results/qwen2.5_vl_7b/streamingbench/fps0.5-kv1000-hermes-kvheadbudget-current_ourtop_scores-sparsemm-union1-r0.2-w32-raggedprefill-timestampv8-current_ourtop_v8_policy-g0-ffc448-letterbox-current-ourtop-timestamp-v8-g0-half-k1000-canvas448

if [ -f "$SAVE_DIR/results.csv" ]; then
  echo "already_done $(date --iso-8601=seconds)" > "$STATUS"
  exit 0
fi

echo "resuming $(date --iso-8601=seconds)" > "$STATUS"
"$PYTHON" video_qa/run_infer.py \
  --model qwen2.5_vl_7b \
  --dataset streamingbench \
  --anno_path data/streamingbench/streamingbench_realtime_half_userpaths.json \
  --num_chunks 2 \
  --devices 0,1 \
  --sample_fps 0.5 \
  --kv_size 1000 \
  --compress_mode hermes \
  --kv_head_budget_scores "$ROOT/current_ourtop_scores.csv" \
  --kv_head_budget_scheme sparsemm \
  --kv_head_budget_sparsemm_ratio 0.2 \
  --kv_head_budget_sparsemm_window_size 32 \
  --kv_head_budget_floor_ratio 0 \
  --kv_head_budget_union_cap_ratio 1.0 \
  --timestamp_v8_policy "$ROOT/current_ourtop_v8_policy.csv" \
  --timestamp_v8_grace_frames 0 \
  --kv_head_ragged_prefill true \
  --fixed_frame_canvas_size 448 \
  --fixed_frame_pad_mode letterbox \
  --debug false \
  --output_tag current-ourtop-timestamp-v8-g0-half-k1000-canvas448 \
  > "$LOG" 2>&1
rc=$?

if (( rc == 0 )); then
  echo "done $(date --iso-8601=seconds)" > "$STATUS"
else
  echo "failed rc=$rc $(date --iso-8601=seconds)" > "$STATUS"
fi
exit "$rc"
