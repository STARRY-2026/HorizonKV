#!/usr/bin/env bash
set -euo pipefail

cd /home/yujun/program/hermes

ANNO=results/myobservations/lu_kv_native_151_250/anno_151_250.json
LU_STATUS=logs/lu_kv_native_151_250/run.status
LU_REPORT=results/qwen2.5_vl_7b/streamingbench/fps0.5-kv1000-hermes-kvheadbudget-lu_raw_blend05_clip075_150_runtime_score-sparsemm-union1-r0.2-w32-raggedprefill-lu-raw-blend05-clip075-150-native-151-250-k1000/eval_results.txt
QUEUE_DIR=logs/hermes_uniform_native_151_250
QUEUE_STATUS="$QUEUE_DIR/queue.status"
UNIFORM_REPORT=results/qwen2.5_vl_7b/streamingbench/fps0.5-kv1000-hermes-kvheadbudget-uniform-sparsemm-union1-r0.2-w32-raggedprefill-hermes-uniform-native-k1000-151-250/eval_results.txt

mkdir -p "$QUEUE_DIR"
printf 'queued %s\n' "$(date -u +%FT%TZ)" > "$QUEUE_STATUS"

while [[ ! -f "$LU_REPORT" ]]; do
  if [[ -f "$LU_STATUS" ]] && grep -q '^failed ' "$LU_STATUS"; then
    printf 'blocked: Raw-LU failed %s\n' "$(date -u +%FT%TZ)" >> "$QUEUE_STATUS"
    exit 1
  fi
  sleep 30
done

printf 'starting %s\n' "$(date -u +%FT%TZ)" >> "$QUEUE_STATUS"
if bash scripts/run_native_k1000_budget.sh \
  "$ANNO" \
  results/observations/memory_core_budget_scores/uniform.csv \
  hermes-uniform-native-k1000-151-250 \
  hermes_uniform_native_k1000_151_250; then
  if [[ ! -f "$UNIFORM_REPORT" ]]; then
    printf 'failed: inference returned without report %s\n' "$(date -u +%FT%TZ)" >> "$QUEUE_STATUS"
    exit 1
  fi
  printf 'complete %s\n' "$(date -u +%FT%TZ)" >> "$QUEUE_STATUS"
else
  exit_code=$?
  printf 'failed exit_code=%s %s\n' "$exit_code" "$(date -u +%FT%TZ)" >> "$QUEUE_STATUS"
  exit "$exit_code"
fi
