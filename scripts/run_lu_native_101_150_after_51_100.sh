#!/usr/bin/env bash
set -euo pipefail

cd /home/yujun/program/hermes
PREVIOUS_PID=2615045
PREVIOUS_REPORT=results/qwen2.5_vl_7b/streamingbench/fps0.5-kv1000-hermes-kvheadbudget-lu_raw_blend05_clip075_150_runtime_score-sparsemm-union1-r0.2-w32-raggedprefill-lu-raw-blend05-clip075-150-native-51-100-k1000/eval_results.txt
STATUS=logs/lu_kv_native_101_150/queue.status

printf 'queued_after_pid=%s\n' "$PREVIOUS_PID" > "$STATUS"
while kill -0 "$PREVIOUS_PID" 2>/dev/null; do
  sleep 30
done
if [[ ! -f "$PREVIOUS_REPORT" ]]; then
  printf 'blocked: previous experiment ended without %s\n' "$PREVIOUS_REPORT" >> "$STATUS"
  exit 1
fi

printf 'starting\n' >> "$STATUS"
export PYTHONPATH=/home/yujun/program/hermes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DECORD_EOF_RETRY_MAX=40960
/home/yujun/.conda/envs/hermes-qwen/bin/python -u video_qa/run_infer.py \
  --model qwen2.5_vl_7b --dataset streamingbench \
  --anno_path results/myobservations/lu_kv_native_101_150/anno_101_150.json \
  --num_chunks 1 --devices 0 --sample_fps 0.5 --kv_size 1000 \
  --compress_mode hermes \
  --kv_head_budget_scores results/myobservations/lu_kv_offline_50v_20260913/lu_raw_blend05_clip075_150_runtime_score.csv \
  --kv_head_budget_scheme sparsemm --kv_head_budget_sparsemm_ratio 0.2 \
  --kv_head_budget_sparsemm_window_size 32 --kv_head_ragged_prefill true \
  --debug false --output_tag lu-raw-blend05-clip075-150-native-101-150-k1000 \
  > logs/lu_kv_native_101_150/raw_lu_native_101_150.log 2>&1
printf 'complete\n' >> "$STATUS"
