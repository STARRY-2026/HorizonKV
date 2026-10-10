#!/usr/bin/env bash
set -euo pipefail

cd /home/yujun/program/hermes
export PYTHONPATH=/home/yujun/program/hermes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DECORD_EOF_RETRY_MAX=40960

PY=/home/yujun/.conda/envs/hermes-qwen/bin/python
ANNO=results/myobservations/temporal_zone_eval50/filtered_anno_first50.json
ROOT=results/myobservations/lu_kv_offline_50v_20260913
LOG=logs/lu_kv_constrained_first50
mkdir -p "$LOG"

run_one() {
  local scores="$1"
  local tag="$2"
  "$PY" -u video_qa/run_infer.py \
    --model qwen2.5_vl_7b --dataset streamingbench --anno_path "$ANNO" \
    --num_chunks 1 --devices 0 --sample_fps 0.5 --kv_size 1000 \
    --compress_mode hermes --kv_head_budget_scores "$scores" \
    --kv_head_budget_scheme sparsemm --kv_head_budget_sparsemm_ratio 0.2 \
    --kv_head_budget_sparsemm_window_size 32 --kv_head_ragged_prefill true \
    --fixed_frame_canvas_size 448 --fixed_frame_pad_mode letterbox \
    --debug false --output_tag "$tag" > "$LOG/$tag.log" 2>&1
}

run_one results/observations/memory_core_budget_scores/uniform.csv \
  ragged-uniform-first50-k1000-canvas448
run_one "$ROOT/lu_raw_blend05_clip075_150_runtime_score.csv" \
  lu-raw-blend05-clip075-150-first50-k1000-canvas448
run_one "$ROOT/lu_correct_layernorm_blend05_clip075_150_runtime_score.csv" \
  lu-correct-layernorm-blend05-clip075-150-first50-k1000-canvas448
