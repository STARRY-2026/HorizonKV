#!/usr/bin/env bash
set -uo pipefail

cd /home/yujun/program/hermes
export PYTHONPATH=/home/yujun/program/hermes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DECORD_EOF_RETRY_MAX=40960

PYTHON=/home/yujun/.conda/envs/hermes-qwen/bin/python
ROOT=results/myobservations/temporal_utility_regression_diagnostic
ANNO="$ROOT/filtered_anno.json"
PROFILE=results/myobservations/temporal_utility_profiles_v1
STATUS=logs/temporal_utility_regression_trace_4way.status

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
  --ragged_readout_trace_last_n 1
  --debug false
)

run_one() {
  local name="$1" gpu="$2" policy="$3"
  local args=("${COMMON[@]}")
  if [[ -n "$policy" ]]; then
    args+=(
      --kv_head_budget_scores results/myobservations/relocatev3/v8_blend3_scores.csv
      --kv_head_budget_scheme sparsemm
      --kv_head_budget_sparsemm_ratio 0.2
      --kv_head_budget_sparsemm_window_size 32
      --kv_head_budget_union_cap_ratio 1.0
      --timestamp_v8_policy "$policy"
      --timestamp_v8_grace_frames 0
    )
  fi
  echo "running gpu=$gpu $(date --iso-8601=seconds)" > "logs/temporal_utility_diag_${name}.status"
  "$PYTHON" video_qa/run_infer.py "${args[@]}" \
    --devices "$gpu" \
    --ragged_frame_trace_path "$ROOT/${name}_cache_trace.jsonl" \
    --ragged_readout_trace_path "$ROOT/${name}_readout_trace.jsonl" \
    --output_tag "temporal-utility-diagnostic-${name}" \
    > "logs/temporal_utility_diag_${name}.log" 2>&1
  local rc=$?
  if (( rc == 0 )); then
    echo "done gpu=$gpu $(date --iso-8601=seconds)" > "logs/temporal_utility_diag_${name}.status"
  else
    echo "failed rc=$rc gpu=$gpu $(date --iso-8601=seconds)" > "logs/temporal_utility_diag_${name}.status"
  fi
  return "$rc"
}

echo "running $(date --iso-8601=seconds)" > "$STATUS"
run_one hermes 0 "" & p0=$!
run_one full 0 "$PROFILE/e3_full.csv" & p1=$!
run_one content 1 "$PROFILE/e0_content.csv" & p2=$!
run_one shuffled 1 "$PROFILE/e4_shuffled.csv" & p3=$!
printf 'hermes=%s\nfull=%s\ncontent=%s\nshuffled=%s\n' "$p0" "$p1" "$p2" "$p3" \
  > logs/temporal_utility_regression_trace_4way.worker_pids
rc=0
for pid in "$p0" "$p1" "$p2" "$p3"; do
  wait "$pid" || rc=1
done
if (( rc == 0 )); then
  echo "done $(date --iso-8601=seconds)" > "$STATUS"
else
  echo "failed $(date --iso-8601=seconds)" > "$STATUS"
fi
exit "$rc"
