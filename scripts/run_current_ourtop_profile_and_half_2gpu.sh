#!/usr/bin/env bash
set -uo pipefail

cd /home/yujun/program/hermes
export PYTHONPATH=/home/yujun/program/hermes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DECORD_EOF_RETRY_MAX=40960

PYTHON=/home/yujun/.conda/envs/hermes-qwen/bin/python
ROOT=results/myobservations/current_ourtop_k1000_fixed448_g0
STATUS=logs/current_ourtop_profile_and_half_2gpu.status
PROFILE_LOG=logs/current_ourtop_profile_2gpu.log
EVAL_LOG=logs/current_ourtop_half_2gpu.log
FULL=data/streamingbench/streamingbench_realtime.json
EVAL_ANNO=data/streamingbench/streamingbench_realtime_half_userpaths.json
CAL0="$ROOT/calibration_complement_2_gpu0.json"
CAL1="$ROOT/calibration_complement_2_gpu1.json"
TRACE0="$ROOT/readout_gpu0.jsonl"
TRACE1="$ROOT/readout_gpu1.jsonl"
OLD_SCORES=results/observations/memory_core_budget_scores/ours_top100.csv
OLD_POLICY=results/myobservations/relocatev3/v8_budget.csv
BOOTSTRAP_SCORES=results/myobservations/relocatev3/v8_blend3_scores.csv
NEW_SCORES="$ROOT/current_ourtop_scores.csv"
NEW_POLICY="$ROOT/current_ourtop_v8_policy.csv"

mkdir -p "$ROOT" logs
echo "profiling $(date --iso-8601=seconds)" > "$STATUS"

"$PYTHON" - "$FULL" "$EVAL_ANNO" "$CAL0" "$CAL1" <<'PY'
import json, sys
from pathlib import Path
full_path, eval_path, out0, out1 = sys.argv[1:]
full = json.load(open(full_path, encoding="utf-8"))
evaluation = json.load(open(eval_path, encoding="utf-8"))
eval_ids = {int(row["video_id"]) for row in evaluation}
video_root = Path("/home/yujun/program/streaming_data/streamingbench/videos")
calibration = []
for source in full:
    if int(source["video_id"]) in eval_ids:
        continue
    row = dict(source)
    row["video_path"] = str(video_root / Path(row["video_path"]).name)
    if Path(row["video_path"]).is_file():
        calibration.append(row)
    if len(calibration) == 4:
        break
if len(calibration) != 4:
    raise SystemExit(f"Expected 4 disjoint calibration videos, got {len(calibration)}")
for path, rows in ((out0, calibration[:2]), (out1, calibration[2:])):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle)
print("calibration video IDs:", [row["video_id"] for row in calibration])
PY

run_profile() {
  local gpu="$1" anno="$2" trace="$3" tag="$4"
  "$PYTHON" video_qa/run_infer.py \
    --model qwen2.5_vl_7b \
    --dataset streamingbench \
    --anno_path "$anno" \
    --num_chunks 1 \
    --devices "$gpu" \
    --sample_fps 0.5 \
    --kv_size 1000 \
    --compress_mode hermes \
    --kv_head_budget_scores "$BOOTSTRAP_SCORES" \
    --kv_head_budget_scheme sparsemm \
    --kv_head_budget_sparsemm_ratio 0.2 \
    --kv_head_budget_sparsemm_window_size 32 \
    --kv_head_budget_floor_ratio 0 \
    --kv_head_budget_union_cap_ratio 1.0 \
    --timestamp_v8_policy "$OLD_POLICY" \
    --timestamp_v8_grace_frames 0 \
    --kv_head_ragged_prefill true \
    --ragged_readout_trace_path "$trace" \
    --ragged_readout_trace_last_n 0 \
    --fixed_frame_canvas_size 448 \
    --fixed_frame_pad_mode letterbox \
    --debug false \
    --output_tag "$tag"
}

run_profile 0 "$CAL0" "$TRACE0" current-ourtop-profile-gpu0 > "$PROFILE_LOG.gpu0" 2>&1 &
p0=$!
run_profile 1 "$CAL1" "$TRACE1" current-ourtop-profile-gpu1 > "$PROFILE_LOG.gpu1" 2>&1 &
p1=$!
wait "$p0"; rc0=$?
wait "$p1"; rc1=$?
if (( rc0 != 0 || rc1 != 0 )); then
  echo "profile_failed gpu0=$rc0 gpu1=$rc1 $(date --iso-8601=seconds)" > "$STATUS"
  exit 1
fi

"$PYTHON" head_analysis/build_current_ourtop_scores.py \
  --trace "$TRACE0" \
  --trace "$TRACE1" \
  --old_scores "$OLD_SCORES" \
  --old_policy "$OLD_POLICY" \
  --output_scores "$NEW_SCORES" \
  --output_policy "$NEW_POLICY" \
  --summary "$ROOT/summary.json" \
  > "$PROFILE_LOG" 2>&1
if (( $? != 0 )); then
  echo "aggregate_failed $(date --iso-8601=seconds)" > "$STATUS"
  exit 1
fi

echo "evaluating $(date --iso-8601=seconds)" > "$STATUS"
"$PYTHON" video_qa/run_infer.py \
  --model qwen2.5_vl_7b \
  --dataset streamingbench \
  --anno_path "$EVAL_ANNO" \
  --num_chunks 2 \
  --devices 0,1 \
  --sample_fps 0.5 \
  --kv_size 1000 \
  --compress_mode hermes \
  --kv_head_budget_scores "$NEW_SCORES" \
  --kv_head_budget_scheme sparsemm \
  --kv_head_budget_sparsemm_ratio 0.2 \
  --kv_head_budget_sparsemm_window_size 32 \
  --kv_head_budget_floor_ratio 0 \
  --kv_head_budget_union_cap_ratio 1.0 \
  --timestamp_v8_policy "$NEW_POLICY" \
  --timestamp_v8_grace_frames 0 \
  --kv_head_ragged_prefill true \
  --fixed_frame_canvas_size 448 \
  --fixed_frame_pad_mode letterbox \
  --debug false \
  --output_tag current-ourtop-timestamp-v8-g0-half-k1000-canvas448 \
  > "$EVAL_LOG" 2>&1
rc=$?

if (( rc == 0 )); then
  echo "done $(date --iso-8601=seconds)" > "$STATUS"
else
  echo "eval_failed rc=$rc $(date --iso-8601=seconds)" > "$STATUS"
fi
exit "$rc"
