#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:?usage: run_native_k800_first150_one.sh hermes|lu}"
cd /home/yujun/program/hermes
export PYTHONPATH=/home/yujun/program/hermes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DECORD_EOF_RETRY_MAX=40960

PYTHON=/home/yujun/.conda/envs/hermes-qwen/bin/python
ANNO=results/myobservations/native_k500_first150/anno_first150.json
LOG_DIR=logs/native_k800_first150
mkdir -p "$LOG_DIR"

case "$METHOD" in
  hermes)
    SCORE=results/observations/memory_core_budget_scores/uniform.csv
    TAG=hermes-uniform-native-k800-first150
    ;;
  hermes_rerun)
    SCORE=results/observations/memory_core_budget_scores/uniform.csv
    TAG=hermes-uniform-native-k800-first150-rerun
    ;;
  lu)
    SCORE=results/myobservations/lu_kv_offline_50v_20260913/lu_raw_k800_scaled_exact_blend05_clip075_150_runtime_score.csv
    TAG=lu-raw-blend05-clip075-150-native-first150-k800
    ;;
  *)
    printf 'unknown method: %s\n' "$METHOD" >&2
    exit 2
    ;;
esac

STATUS="$LOG_DIR/$METHOD.status"
LOG="$LOG_DIR/$METHOD.log"

fail() {
  rc=$?
  printf 'failed rc=%s at %s\n' "$rc" "$(date --iso-8601=seconds)" > "$STATUS"
  exit "$rc"
}
trap fail ERR

printf 'running pid=%s at %s\n' "$$" "$(date --iso-8601=seconds)" > "$STATUS"
"$PYTHON" -u video_qa/run_infer.py \
  --model qwen2.5_vl_7b \
  --dataset streamingbench \
  --anno_path "$ANNO" \
  --num_chunks 1 \
  --devices 0 \
  --sample_fps 0.5 \
  --kv_size 800 \
  --compress_mode hermes \
  --kv_head_budget_scores "$SCORE" \
  --kv_head_budget_scheme sparsemm \
  --kv_head_budget_sparsemm_ratio 0.2 \
  --kv_head_budget_sparsemm_window_size 32 \
  --kv_head_budget_union_cap_ratio 1.0 \
  --kv_head_ragged_prefill true \
  --output_tag "$TAG" \
  --debug false \
  > "$LOG" 2>&1

printf 'complete at %s\n' "$(date --iso-8601=seconds)" > "$STATUS"
