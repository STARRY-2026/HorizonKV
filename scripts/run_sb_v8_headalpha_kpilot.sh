#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

PYTHON_BIN="${PYTHON_BIN:-/home/yujun/.conda/envs/hermes-qwen/bin/python}"
DEVICES="${DEVICES:-0}"
K_DELTA="${K_DELTA:?Set K_DELTA to 4 or 8}"
PROFILE="$PROJECT_DIR/results/observations/obs_prev_current_chunk_attention_eager_gpu2_n16_o500_paper_kv/kv_head_profile_scores.csv"
BUDGET_SCORES="$PROJECT_DIR/results/myobservations/relocatev3/v8_blend3_scores.csv"
ANNOTATION="$PROJECT_DIR/data/streamingbench/headalpha_failure12_fullvideos.json"

for required_file in "$PYTHON_BIN" "$PROFILE" "$BUDGET_SCORES" "$ANNOTATION"; do
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
    --sample_fps 0.5 \
    --kv_size 6000 \
    --compress_mode hermes \
    --kv_head_budget_scores "$BUDGET_SCORES" \
    --kv_head_budget_scheme sparsemm \
    --kv_head_budget_sparsemm_ratio 0.2 \
    --kv_head_budget_sparsemm_window_size 32 \
    --kv_head_budget_union_cap_ratio 1.0 \
    --kv_head_ragged_prefill true \
    --kv_head_recency_scores "$PROFILE" \
    --kv_head_recency_metric b_log_per_token_ratio \
    --kv_head_recency_delta 0.2 \
    --kv_head_recency_tau 1.0 \
    --kv_head_recency_alpha_min 0.0 \
    --kv_head_recency_alpha_max 1.0 \
    --kv_head_recency_k_delta "$K_DELTA" \
    --kv_head_recency_k_min 0.0 \
    --kv_head_recency_k_max 20.0 \
    --output_tag "failure12-kd${K_DELTA}" \
    --debug false
