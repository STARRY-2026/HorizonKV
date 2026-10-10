#!/usr/bin/env bash
set -euo pipefail

cd /home/yujun/program/hermes

export PYTHONPATH=/home/yujun/program/hermes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DECORD_EOF_RETRY_MAX=40960

PYTHON=/home/yujun/.conda/envs/hermes-qwen/bin/python
ANNO=results/cvpr/timestamp_v8_eval50_subset/filtered_anno.json
MODE=${1:-both}

COMMON_ARGS=(
  --model qwen2.5_vl_7b
  --dataset streamingbench
  --anno_path "$ANNO"
  --num_chunks 1
  --sample_fps 0.5
  --kv_size 4000
  --compress_mode hermes
  --kv_head_ragged_prefill true
  --fixed_frame_canvas_size 448
  --fixed_frame_pad_mode letterbox
  --debug false
)

run_timestamp_v8() {
  "$PYTHON" video_qa/run_infer.py \
    "${COMMON_ARGS[@]}" \
    --kv_head_budget_scores results/myobservations/relocatev3/v8_blend3_scores.csv \
    --kv_head_budget_scheme sparsemm \
    --kv_head_budget_sparsemm_ratio 0.2 \
    --kv_head_budget_sparsemm_window_size 32 \
    --kv_head_budget_union_cap_ratio 1.0 \
    --timestamp_v8_policy results/myobservations/relocatev3/v8_budget.csv \
    --timestamp_v8_grace_frames 1 \
    --output_tag timestamp-v8-eval50-canvas448 \
    > logs/timestamp_v8_eval50_k4000_canvas448.log 2>&1
}

run_hermes() {
  "$PYTHON" video_qa/run_infer.py \
    "${COMMON_ARGS[@]}" \
    --output_tag hermes-eval50-canvas448-uniform4000 \
    > logs/hermes_eval50_k4000_canvas448.log 2>&1
}

case "$MODE" in
  timestamp_v8)
    run_timestamp_v8
    ;;
  hermes)
    run_hermes
    ;;
  both)
    run_timestamp_v8
    run_hermes
    ;;
  *)
    echo "Usage: $0 [timestamp_v8|hermes|both]" >&2
    exit 2
    ;;
esac
