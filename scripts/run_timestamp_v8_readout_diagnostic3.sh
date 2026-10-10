#!/usr/bin/env bash
set -uo pipefail

cd /home/yujun/program/hermes
export PYTHONPATH=/home/yujun/program/hermes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DECORD_EOF_RETRY_MAX=40960

PYTHON=/home/yujun/.conda/envs/hermes-qwen/bin/python
ANNO=data/streamingbench/timestamp_v8_diagnostic3_userpaths.json
OUT=results/cvpr/timestamp_v8_readout_diagnostic3
STATUS=logs/timestamp_v8_readout_diagnostic3.status
mkdir -p "$OUT" logs

COMMON=(
  --model qwen2.5_vl_7b
  --dataset streamingbench
  --anno_path "$ANNO"
  --num_chunks 1
  --sample_fps 0.5
  --kv_size 1000
  --compress_mode hermes
  --kv_head_ragged_prefill true
  --fixed_frame_canvas_size 448
  --fixed_frame_pad_mode letterbox
  --debug false
)

run_timestamp() {
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" video_qa/run_infer.py \
    "${COMMON[@]}" --devices 0 \
    --kv_head_budget_scores results/myobservations/relocatev3/v8_blend3_scores.csv \
    --kv_head_budget_scheme sparsemm \
    --kv_head_budget_sparsemm_ratio 0.2 \
    --kv_head_budget_sparsemm_window_size 32 \
    --kv_head_budget_union_cap_ratio 1.0 \
    --timestamp_v8_policy results/myobservations/relocatev3/v8_budget.csv \
    --timestamp_v8_grace_frames 1 \
    --ragged_readout_trace_path "$OUT/timestamp_v8_readout.jsonl" \
    --ragged_readout_trace_last_n 1 \
    --output_tag timestamp-v8-readout3-k1000-canvas448 \
    > logs/timestamp_v8_readout_diagnostic3.log 2>&1
}

run_hermes() {
  CUDA_VISIBLE_DEVICES=1 "$PYTHON" video_qa/run_infer.py \
    "${COMMON[@]}" --devices 1 \
    --ragged_readout_trace_path "$OUT/hermes_readout.jsonl" \
    --ragged_readout_trace_last_n 1 \
    --output_tag hermes-readout3-k1000-canvas448 \
    > logs/hermes_readout_diagnostic3.log 2>&1
}

echo "running $(date --iso-8601=seconds)" > "$STATUS"
run_timestamp &
timestamp_pid=$!
run_hermes &
hermes_pid=$!
printf 'timestamp_pid=%s\nhermes_pid=%s\n' "$timestamp_pid" "$hermes_pid" \
  > logs/timestamp_v8_readout_diagnostic3.worker_pids

wait "$timestamp_pid"
timestamp_rc=$?
wait "$hermes_pid"
hermes_rc=$?
if (( timestamp_rc != 0 || hermes_rc != 0 )); then
  echo "failed timestamp_rc=$timestamp_rc hermes_rc=$hermes_rc $(date --iso-8601=seconds)" > "$STATUS"
  exit 1
fi

TIMESTAMP_RESULTS=results/qwen2.5_vl_7b/streamingbench/fps0.5-kv1000-hermes-kvheadbudget-v8_blend3_scores-sparsemm-union1-r0.2-w32-raggedprefill-timestampv8-v8_budget-g1-ffc448-letterbox-timestamp-v8-readout3-k1000-canvas448/results.csv
HERMES_RESULTS=results/qwen2.5_vl_7b/streamingbench/fps0.5-kv1000-hermes-ffc448-letterbox-hermes-readout3-k1000-canvas448/results.csv

"$PYTHON" scripts/analyze_timestamp_v8_readout_trace.py \
  --timestamp_trace "$OUT/timestamp_v8_readout.jsonl" \
  --hermes_trace "$OUT/hermes_readout.jsonl" \
  --timestamp_results "$TIMESTAMP_RESULTS" \
  --hermes_results "$HERMES_RESULTS" \
  --output "$OUT/analysis.md"

echo "done $(date --iso-8601=seconds)" > "$STATUS"
