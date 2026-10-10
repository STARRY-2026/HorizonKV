#!/usr/bin/env python3
"""Paired comparison of uniform-joint and deployed LU-KV on first 20 videos."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("results/myobservations/uniform_joint_online_first20_20260920")
NEW = Path(
    "results/qwen2.5_vl_7b/streamingbench/"
    "fps0.5-kv1000-hermes-kvheadbudget-lu_raw_blend05_clip075_150_runtime_score-"
    "sparsemm-union1-r0.2-w32-raggedprefill-lu-uniform-joint-native-first20-k1000/results.csv"
)
BASE = Path(
    "results/qwen2.5_vl_7b/streamingbench/"
    "fps0.5-kv1000-hermes-kvheadbudget-lu_raw_blend05_clip075_150_runtime_score-"
    "sparsemm-union1-r0.2-w32-raggedprefill-lu-raw-blend05-clip075-150-native-first50-k1000/results.csv"
)


def exact_two_sided_binomial(b, c):
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


anno = json.loads((ROOT / "anno_first20.json").read_text())
video_ids = {int(video["video_id"]) for video in anno}
new = pd.read_csv(NEW)
base = pd.read_csv(BASE)
base = base[base["video_id"].astype(int).isin(video_ids)].copy()
# Some StreamingBench videos repeat identical question text at different
# timestamps. Results preserve annotation order, so pair by video and the
# within-video row index rather than by question text alone.
base["question_index"] = base.groupby("video_id", sort=False).cumcount()
new["question_index"] = new.groupby("video_id", sort=False).cumcount()
keys = ["video_id", "question_index"]
paired = base.merge(new, on=keys, suffixes=("_deployed", "_joint"), validate="one_to_one")
if not (paired["question_deployed"] == paired["question_joint"]).all():
    raise RuntimeError("Question order mismatch between deployed and joint results")
if len(paired) != len(new):
    raise RuntimeError(f"Pairing mismatch: deployed={len(base)} joint={len(new)} paired={len(paired)}")

deployed_correct = paired["qa_acc_deployed"].to_numpy() > 0
joint_correct = paired["qa_acc_joint"].to_numpy() > 0
gain = (~deployed_correct) & joint_correct
loss = deployed_correct & (~joint_correct)

video_delta = []
for _, frame in paired.groupby("video_id"):
    video_delta.append(
        (frame["qa_acc_joint"].to_numpy() > 0).mean()
        - (frame["qa_acc_deployed"].to_numpy() > 0).mean()
    )
video_delta = np.asarray(video_delta)
rng = np.random.default_rng(20260920)
boot = np.asarray([
    video_delta[rng.integers(0, len(video_delta), len(video_delta))].mean()
    for _ in range(20000)
])

tasks = []
for task, frame in paired.groupby("task_joint"):
    d = (frame["qa_acc_deployed"].to_numpy() > 0)
    j = (frame["qa_acc_joint"].to_numpy() > 0)
    tasks.append({
        "task": task,
        "n": int(len(frame)),
        "deployed_accuracy": float(d.mean()),
        "joint_accuracy": float(j.mean()),
        "delta": float(j.mean() - d.mean()),
    })

report = {
    "videos": len(video_ids),
    "questions": int(len(paired)),
    "deployed_accuracy": float(deployed_correct.mean()),
    "joint_accuracy": float(joint_correct.mean()),
    "delta": float(joint_correct.mean() - deployed_correct.mean()),
    "deployed_wrong_joint_right": int(gain.sum()),
    "deployed_right_joint_wrong": int(loss.sum()),
    "mcnemar_exact_p": exact_two_sided_binomial(int(gain.sum()), int(loss.sum())),
    "video_bootstrap_delta_95": np.quantile(boot, [.025, .975]).tolist(),
    "positive_videos": int((video_delta > 0).sum()),
    "negative_videos": int((video_delta < 0).sum()),
    "tied_videos": int((video_delta == 0).sum()),
    "tasks": sorted(tasks, key=lambda row: row["task"]),
    "new_results": str(NEW),
    "deployed_results": str(BASE),
}
(ROOT / "paired_report.json").write_text(json.dumps(report, indent=2) + "\n")

lines = [
    "# LU-KV Uniform-Joint Online Pilot",
    "",
    f"- Videos: {report['videos']}",
    f"- Questions: {report['questions']}",
    f"- Deployed selector: {100*report['deployed_accuracy']:.2f}%",
    f"- Uniform-joint selector: {100*report['joint_accuracy']:.2f}%",
    f"- Delta: {100*report['delta']:+.2f} percentage points",
    f"- Paired flips: +{report['deployed_wrong_joint_right']} / -{report['deployed_right_joint_wrong']}",
    f"- Exact McNemar p: {report['mcnemar_exact_p']:.4f}",
    "- Video-bootstrap 95% CI: "
    f"[{100*report['video_bootstrap_delta_95'][0]:+.2f}, "
    f"{100*report['video_bootstrap_delta_95'][1]:+.2f}] percentage points",
    "",
    "| Task | N | Deployed | Uniform joint | Delta |",
    "|---|---:|---:|---:|---:|",
]
for row in report["tasks"]:
    lines.append(
        f"| {row['task']} | {row['n']} | {100*row['deployed_accuracy']:.2f}% | "
        f"{100*row['joint_accuracy']:.2f}% | {100*row['delta']:+.2f} |"
    )
(ROOT / "paired_report.md").write_text("\n".join(lines) + "\n")
print(json.dumps(report, indent=2))
