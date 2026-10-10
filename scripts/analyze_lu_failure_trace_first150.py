#!/usr/bin/env python3
"""Compare HERMES and LU cache/readout traces on the K=1000 failure subset."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path("results/myobservations/lu_failure_trace_first150_k1000_20260921")
RESULT_ROOT = Path("results/qwen2.5_vl_7b/streamingbench")


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def keyed(rows):
    return {(str(row["video_id"]), int(row["question_index"])): row for row in rows}


def load_results(pattern):
    paths = list(RESULT_ROOT.glob(pattern + "/results.csv"))
    if len(paths) != 1:
        raise RuntimeError((pattern, paths))
    grouped = defaultdict(list)
    with paths[0].open() as handle:
        for row in csv.DictReader(handle):
            grouped[str(row["video_id"])].append(row)
    return {(video_id, index): row for video_id, rows in grouped.items()
            for index, row in enumerate(rows)}


def frame_distribution(record, field):
    values = defaultdict(float)
    for head in record["heads"]:
        for frame, value in head[field].items():
            values[int(frame)] += float(value)
    total = sum(values.values())
    return values, total


def distribution_metrics(values, total, current_frame, fps):
    if total <= 0:
        return {"mean_age_sec": None, "recent10_mass": None,
                "recent30_mass": None, "num_frames": 0, "top_frames_sec": []}
    normalized = {frame: value / total for frame, value in values.items()}
    ages = {frame: (current_frame - frame) / fps for frame in values}
    top = sorted(normalized, key=normalized.get, reverse=True)[:5]
    return {
        "mean_age_sec": sum(normalized[f] * ages[f] for f in normalized),
        "recent10_mass": sum(normalized[f] for f in normalized if ages[f] <= 10),
        "recent30_mass": sum(normalized[f] for f in normalized if ages[f] <= 30),
        "num_frames": len(normalized),
        "top_frames_sec": [round(frame / fps, 1) for frame in top],
    }


def overlap(left, left_total, right, right_total):
    if left_total <= 0 or right_total <= 0:
        return None
    keys = set(left) | set(right)
    return sum(min(left.get(k, 0) / left_total, right.get(k, 0) / right_total)
               for k in keys)


manifest = json.loads((ROOT / "manifest.json").read_text())
targets = {(str(row["video_id"]), int(row["question_index"])): row
           for row in manifest["targets"]}

h_cache = keyed(load_jsonl(ROOT / "hermes_cache_trace.jsonl"))
l_cache = keyed(load_jsonl(ROOT / "lu_cache_trace.jsonl"))
h_read = keyed(load_jsonl(ROOT / "hermes_readout_trace.jsonl"))
l_read = keyed(load_jsonl(ROOT / "lu_readout_trace.jsonl"))
h_result = load_results("*lu-failure-trace-hermes-native-k1000")
l_result = load_results("*lu-failure-trace-lu-native-k1000")

rows = []
for key, target in targets.items():
    hc, lc, hr, lr = h_cache[key], l_cache[key], h_read[key], l_read[key]
    fps = float(hc["sample_fps"])
    current = int(hc["current_frame_id"])
    hcv, hct = frame_distribution(hc, "after_frame_tokens")
    lcv, lct = frame_distribution(lc, "after_frame_tokens")
    hrv, hrt = frame_distribution(hr, "frame_mass")
    lrv, lrt = frame_distribution(lr, "frame_mass")
    hcm = distribution_metrics(hcv, hct, current, fps)
    lcm = distribution_metrics(lcv, lct, current, fps)
    hrm = distribution_metrics(hrv, hrt, current, fps)
    lrm = distribution_metrics(lrv, lrt, current, fps)
    h_pred = h_result[key]["pred_choice"]
    l_pred = l_result[key]["pred_choice"]
    rows.append({
        "video_id": key[0], "question_index": key[1],
        "task": target["task"], "ability": target["ability"],
        "end_time": target["end_time"], "question": target["question"],
        "correct_choice": target["correct_choice"],
        "expected_hermes": target["hermes_pred_choice"],
        "expected_lu": target["matched_pred_choice"],
        "trace_hermes": h_pred, "trace_lu": l_pred,
        "target_reproduced": h_pred == target["hermes_pred_choice"]
                             and l_pred == target["matched_pred_choice"],
        "cache_overlap": overlap(hcv, hct, lcv, lct),
        "hermes_cache_mean_age": hcm["mean_age_sec"],
        "lu_cache_mean_age": lcm["mean_age_sec"],
        "hermes_cache_recent10": hcm["recent10_mass"],
        "lu_cache_recent10": lcm["recent10_mass"],
        "hermes_cache_frames": hcm["num_frames"],
        "lu_cache_frames": lcm["num_frames"],
        "readout_overlap": overlap(hrv, hrt, lrv, lrt),
        "hermes_readout_mean_age": hrm["mean_age_sec"],
        "lu_readout_mean_age": lrm["mean_age_sec"],
        "hermes_readout_recent10": hrm["recent10_mass"],
        "lu_readout_recent10": lrm["recent10_mass"],
        "hermes_top_read_frames_sec": json.dumps(hrm["top_frames_sec"]),
        "lu_top_read_frames_sec": json.dumps(lrm["top_frames_sec"]),
    })

with (ROOT / "target_trace_summary.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

numeric = lambda field: [float(row[field]) for row in rows if row[field] is not None]
summary = {
    "targets": len(rows),
    "fully_reproduced": sum(bool(row["target_reproduced"]) for row in rows),
    "hermes_target_correct": sum(row["trace_hermes"] == row["correct_choice"] for row in rows),
    "lu_target_correct": sum(row["trace_lu"] == row["correct_choice"] for row in rows),
    "mean_cache_distribution_overlap": float(np.mean(numeric("cache_overlap"))),
    "mean_readout_distribution_overlap": float(np.mean(numeric("readout_overlap"))),
    "mean_cache_age_delta_lu_minus_hermes_sec": float(np.mean([
        row["lu_cache_mean_age"] - row["hermes_cache_mean_age"] for row in rows])),
    "mean_readout_age_delta_lu_minus_hermes_sec": float(np.mean([
        row["lu_readout_mean_age"] - row["hermes_readout_mean_age"] for row in rows])),
    "per_target": rows,
}
(ROOT / "target_trace_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({key: value for key, value in summary.items() if key != "per_target"},
                 indent=2, ensure_ascii=False))
