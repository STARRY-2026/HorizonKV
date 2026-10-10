#!/usr/bin/env python3
"""Paired deployed-vs-joint reports for videos 21–150 and all first 150."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


BASE_ROOT = Path("results/qwen2.5_vl_7b/streamingbench")
OBS_ROOT = Path("results/myobservations/uniform_joint_online_21_150_20260920")
BASE_PATHS = [
    BASE_ROOT / ("fps0.5-kv1000-hermes-kvheadbudget-lu_raw_blend05_clip075_150_runtime_score-"
                 "sparsemm-union1-r0.2-w32-raggedprefill-lu-raw-blend05-clip075-150-native-first50-k1000/results.csv"),
    BASE_ROOT / ("fps0.5-kv1000-hermes-kvheadbudget-lu_raw_blend05_clip075_150_runtime_score-"
                 "sparsemm-union1-r0.2-w32-raggedprefill-lu-raw-blend05-clip075-150-native-51-100-k1000/results.csv"),
    BASE_ROOT / ("fps0.5-kv1000-hermes-kvheadbudget-lu_raw_blend05_clip075_150_runtime_score-"
                 "sparsemm-union1-r0.2-w32-raggedprefill-lu-raw-blend05-clip075-150-native-101-150-k1000/results.csv"),
]
NEW_21_150 = BASE_ROOT / ("fps0.5-kv1000-hermes-kvheadbudget-lu_raw_blend05_clip075_150_runtime_score-"
                           "sparsemm-union1-r0.2-w32-raggedprefill-lu-uniform-joint-native-21-150-k1000/results.csv")
NEW_1_20 = BASE_ROOT / ("fps0.5-kv1000-hermes-kvheadbudget-lu_raw_blend05_clip075_150_runtime_score-"
                         "sparsemm-union1-r0.2-w32-raggedprefill-lu-uniform-joint-native-first20-k1000/results.csv")


def exact_two_sided_binomial(gain, loss):
    total = gain + loss
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, k) for k in range(min(gain, loss) + 1)) / 2 ** total
    return min(1.0, 2 * tail)


def pair_frames(base, joint):
    base = base.copy()
    joint = joint.copy()
    base["question_index"] = base.groupby("video_id", sort=False).cumcount()
    joint["question_index"] = joint.groupby("video_id", sort=False).cumcount()
    paired = base.merge(joint, on=["video_id", "question_index"],
                        suffixes=("_deployed", "_joint"), validate="one_to_one")
    if len(paired) != len(base) or len(paired) != len(joint):
        raise RuntimeError((len(base), len(joint), len(paired)))
    if not (paired["question_deployed"] == paired["question_joint"]).all():
        raise RuntimeError("Question order mismatch")
    return paired


def summarize(paired, seed=20260920):
    deployed = paired["qa_acc_deployed"].to_numpy() > 0
    joint = paired["qa_acc_joint"].to_numpy() > 0
    gain = (~deployed) & joint
    loss = deployed & (~joint)
    deltas = []
    for _, frame in paired.groupby("video_id"):
        deltas.append((frame["qa_acc_joint"].to_numpy() > 0).mean()
                      - (frame["qa_acc_deployed"].to_numpy() > 0).mean())
    deltas = np.asarray(deltas)
    rng = np.random.default_rng(seed)
    boot = np.asarray([deltas[rng.integers(0, len(deltas), len(deltas))].mean()
                       for _ in range(20000)])
    tasks = []
    for task, frame in paired.groupby("task_joint"):
        d = frame["qa_acc_deployed"].to_numpy() > 0
        j = frame["qa_acc_joint"].to_numpy() > 0
        tasks.append({"task": task, "n": int(len(frame)),
                      "deployed_accuracy": float(d.mean()),
                      "joint_accuracy": float(j.mean()),
                      "delta": float(j.mean() - d.mean())})
    return {
        "videos": int(paired["video_id"].nunique()), "questions": int(len(paired)),
        "deployed_accuracy": float(deployed.mean()), "joint_accuracy": float(joint.mean()),
        "delta": float(joint.mean() - deployed.mean()),
        "deployed_wrong_joint_right": int(gain.sum()),
        "deployed_right_joint_wrong": int(loss.sum()),
        "mcnemar_exact_p": exact_two_sided_binomial(int(gain.sum()), int(loss.sum())),
        "video_bootstrap_delta_95": np.quantile(boot, [.025, .975]).tolist(),
        "positive_videos": int((deltas > 0).sum()),
        "negative_videos": int((deltas < 0).sum()),
        "tied_videos": int((deltas == 0).sum()),
        "tasks": sorted(tasks, key=lambda row: row["task"]),
    }


def markdown(title, report):
    lines = [f"# {title}", "",
             f"- Videos/questions: {report['videos']} / {report['questions']}",
             f"- Deployed: {100*report['deployed_accuracy']:.2f}%",
             f"- Uniform joint: {100*report['joint_accuracy']:.2f}%",
             f"- Delta: {100*report['delta']:+.2f} percentage points",
             f"- Paired flips: +{report['deployed_wrong_joint_right']} / -{report['deployed_right_joint_wrong']}",
             f"- Exact McNemar p: {report['mcnemar_exact_p']:.4f}",
             f"- Video-bootstrap 95% CI: [{100*report['video_bootstrap_delta_95'][0]:+.2f}, {100*report['video_bootstrap_delta_95'][1]:+.2f}] points",
             "", "| Task | N | Deployed | Joint | Delta |", "|---|---:|---:|---:|---:|"]
    for row in report["tasks"]:
        lines.append(f"| {row['task']} | {row['n']} | {100*row['deployed_accuracy']:.2f}% | "
                     f"{100*row['joint_accuracy']:.2f}% | {100*row['delta']:+.2f} |")
    return "\n".join(lines) + "\n"


base = pd.concat([pd.read_csv(path) for path in BASE_PATHS], ignore_index=True)
joint_21_150 = pd.read_csv(NEW_21_150)
ids_21_150 = set(joint_21_150["video_id"].astype(int))
base_21_150 = base[base["video_id"].astype(int).isin(ids_21_150)].copy()
report_21_150 = summarize(pair_frames(base_21_150, joint_21_150))

joint_1_150 = pd.concat([pd.read_csv(NEW_1_20), joint_21_150], ignore_index=True)
report_1_150 = summarize(pair_frames(base, joint_1_150), seed=20260921)
reports = {"videos_21_150": report_21_150, "videos_1_150": report_1_150}
(OBS_ROOT / "paired_reports.json").write_text(json.dumps(reports, indent=2) + "\n")
(OBS_ROOT / "paired_report_21_150.md").write_text(markdown("Uniform Joint: Videos 21–150", report_21_150))
(OBS_ROOT / "paired_report_1_150.md").write_text(markdown("Uniform Joint: Videos 1–150", report_1_150))
print(json.dumps(reports, indent=2))
