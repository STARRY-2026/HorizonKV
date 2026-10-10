#!/usr/bin/env python3
"""Compare strict high-budget-head latest-frame banks with the LU baseline."""

import csv
import glob
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path("results/myobservations/lu_head_budget_defer_pilot_k1000_20260921")
SOURCE = Path("results/myobservations/lu_failure_trace_first150_k1000_20260921")
RESULT_ROOT = Path("results/qwen2.5_vl_7b/streamingbench")


def load_results(tag):
    paths = list(RESULT_ROOT.glob(f"*{tag}/results.csv"))
    if len(paths) != 1:
        raise RuntimeError((tag, paths))
    grouped = defaultdict(list)
    with paths[0].open() as handle:
        for row in csv.DictReader(handle):
            grouped[str(row["video_id"])].append(row)
    return {(video_id, index): row for video_id, rows in grouped.items()
            for index, row in enumerate(rows)}


manifest = json.loads((SOURCE / "manifest.json").read_text())
target_keys = {
    (str(row["video_id"]), int(row["question_index"]))
    for row in manifest["targets"]
}
baseline = load_results("exact-token-lu-native-k1000")
variants = {
    ratio: load_results(f"lu-head-budget-defer-r{label}-native-k1000")
    for ratio, label in ((0.1, "01"), (0.2, "02"), (0.3, "03"))
}

summary = {
    "baseline": {
        "correct": sum(r["pred_choice"] == r["correct_choice"]
                       for r in baseline.values()),
        "questions": len(baseline),
        "target_correct": sum(
            baseline[k]["pred_choice"] == baseline[k]["correct_choice"]
            for k in target_keys),
    },
    "variants": {},
}
for ratio, rows in variants.items():
    if rows.keys() != baseline.keys():
        raise RuntimeError(f"result-key mismatch for ratio={ratio}")
    gains = [k for k in rows
             if baseline[k]["pred_choice"] != baseline[k]["correct_choice"]
             and rows[k]["pred_choice"] == rows[k]["correct_choice"]]
    losses = [k for k in rows
              if baseline[k]["pred_choice"] == baseline[k]["correct_choice"]
              and rows[k]["pred_choice"] != rows[k]["correct_choice"]]
    changed = [k for k in rows
               if baseline[k]["pred_choice"] != rows[k]["pred_choice"]]
    target_gains = [k for k in gains if k in target_keys]
    control_losses = [k for k in losses if k not in target_keys]
    summary["variants"][str(ratio)] = {
        "correct": sum(r["pred_choice"] == r["correct_choice"]
                       for r in rows.values()),
        "questions": len(rows),
        "net_vs_baseline": len(gains) - len(losses),
        "gains": len(gains),
        "losses": len(losses),
        "changed_answers": len(changed),
        "target_correct": sum(
            rows[k]["pred_choice"] == rows[k]["correct_choice"]
            for k in target_keys),
        "target_gains": [list(k) for k in target_gains],
        "control_losses": [list(k) for k in control_losses],
        "changes": [
            {
                "video_id": k[0],
                "question_index": k[1],
                "is_target": k in target_keys,
                "question": rows[k]["question"],
                "correct": rows[k]["correct_choice"],
                "baseline": baseline[k]["pred_choice"],
                "variant": rows[k]["pred_choice"],
            }
            for k in changed
        ],
    }

(ROOT / "analysis.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(summary, indent=2, ensure_ascii=False))
