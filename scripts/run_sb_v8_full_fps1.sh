#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

PYTHON_BIN="${PYTHON_BIN:-/home/yujun/.conda/envs/hermes-qwen/bin/python}"
DEVICES="${DEVICES:-0}"
ANNOTATION="$PROJECT_DIR/data/streamingbench/streamingbench_realtime.json"
BUDGET_SCORES="$PROJECT_DIR/results/myobservations/relocatev3/v8_blend3_scores.csv"

for required_file in "$PYTHON_BIN" "$ANNOTATION" "$BUDGET_SCORES"; do
    if [ ! -f "$required_file" ]; then
        echo "Required file not found: $required_file" >&2
        exit 1
    fi
done

cd "$PROJECT_DIR"
exec "$PYTHON_BIN" video_qa/run_infer.py \
    --model qwen2.5_vl_7b \
    --dataset streamingbench \
    --anno_path "$ANNOTATION" \
    --num_chunks 1 \
    --devices "$DEVICES" \
    --sample_fps 1.0 \
    --kv_size 6000 \
    --compress_mode hermes \
    --kv_head_budget_scores "$BUDGET_SCORES" \
    --kv_head_budget_scheme sparsemm \
    --kv_head_budget_sparsemm_ratio 0.2 \
    --kv_head_budget_sparsemm_window_size 32 \
    --kv_head_budget_union_cap_ratio 1.0 \
    --kv_head_ragged_prefill true \
    --output_tag "v8-full-fps1" \
    --debug false
