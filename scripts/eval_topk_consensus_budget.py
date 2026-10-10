#!/usr/bin/env python3
"""Evaluate top-k-consensus budget maps on disjoint LU utility traces."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path("results/myobservations")
PROFILE = ROOT / "local_global_topk_50v_20260922/local_global_topk_head_ranks.csv"
PROFILE_MANIFEST = ROOT / "lu_kv_offline_50v_20260913/manifest.json"
EVAL = ROOT / "lu_kv_balanced100v_20260915/traces"
OLD = ROOT / "lu_kv_offline_50v_20260913"
OUTPUT = ROOT / "topk_consensus_budget_heldout86_20260922"
RECENT = 32
K_VALUES = (800, 1000)
BLEND = 0.5


def budget_from_score(score, k):
    """Proportional allocation with the same 0.75–1.5 K bounds as old LU."""
    score = np.asarray(score, dtype=np.float64)
    target = k * (1 + BLEND * (score / score.mean() - 1))
    lower, upper = int(.75 * k), int(1.5 * k)
    lo, hi = -2 * k, 2 * k
    for _ in range(100):
        mid = (lo + hi) / 2
        if np.clip(target + mid, lower, upper).sum() < 112 * k:
            lo = mid
        else:
            hi = mid
    projected = np.clip(target + (lo + hi) / 2, lower, upper)
    result = np.floor(projected).astype(np.int64)
    missing = 112 * k - int(result.sum())
    candidates = np.where(result < upper)[0]
    order = candidates[np.argsort(-(projected[candidates] - result[candidates]), kind="stable")]
    result[order[:missing]] += 1
    assert result.sum() == 112 * k
    assert lower <= result.min() and result.max() <= upper
    return result


def read_budget(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = np.zeros(112, dtype=np.int64)
    for row in rows:
        result[4 * int(row["layer"]) + int(row["kv_head"])] = int(row["budget"])
    return result


def retention(utility, ranking, budget):
    heads, n = utility.shape
    history = n - RECENT
    recent_mass = utility[:, history:].sum(axis=1, dtype=np.float64)
    ordered = np.take_along_axis(utility[:, :history], ranking, axis=1)
    prefix = np.cumsum(ordered, axis=1, dtype=np.float64)
    count = np.clip(budget - RECENT, 0, history)
    selected = np.zeros(heads, dtype=np.float64)
    positive = count > 0
    selected[positive] = prefix[np.arange(heads)[positive], count[positive] - 1]
    return float((recent_mass + selected).sum() / utility.sum(dtype=np.float64))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with PROFILE.open(newline="") as handle:
        profile = list(csv.DictReader(handle))
    assert len(profile) == 112
    calibration_ids = {str(v["video_id"]) for v in json.loads(PROFILE_MANIFEST.read_text())["videos"]}
    eval_videos = json.loads((EVAL / "manifest.json").read_text())["videos"]
    heldout = [v for v in eval_videos if str(v["video_id"]) not in calibration_ids]
    assert len(heldout) == 86

    budgets = {}
    for k in K_VALUES:
        old_path = (OLD / "lu_raw_k800_resolved_all_blend05_clip075_150_budget.csv"
                    if k == 800 else OLD / "lu_raw_blend05_clip075_150_budget.csv")
        policies = {"uniform": np.full(112, k, dtype=np.int64),
                    "old_lu50_constrained": read_budget(old_path)}
        for metric in ("overlap_auc", f"overlap_k{k}", "overlap_k250"):
            score = np.array([min(float(row[f"uniform_local_{metric}"]),
                                  float(row[f"uniform_global_{metric}"]))
                              for row in profile])
            name = f"topk_consensus_{metric}_blend05"
            policies[name] = budget_from_score(score, k)
            with (OUTPUT / f"{name}_K{k}_budget.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["layer", "kv_head", "budget"])
                writer.writerows((h // 4, h % 4, int(value)) for h, value in enumerate(policies[name]))
        budgets[k] = policies

    video_values = {k: defaultdict(lambda: defaultdict(list)) for k in K_VALUES}
    eligible_counts = {k: 0 for k in K_VALUES}
    for video in heldout:
        video_id = str(video["video_id"])
        for question in video["conversations"]:
            stem = f"v{video_id}_q{question['source_question_index']}"
            with np.load(EVAL / f"{stem}.npz") as z:
                utility = z["utility"].astype(np.float64)
                ranking = z["ranking"].astype(np.int64)
            for k in K_VALUES:
                if utility.shape[1] < k:
                    continue
                eligible_counts[k] += 1
                for name, budget in budgets[k].items():
                    video_values[k][video_id][name].append(retention(utility, ranking, budget))

    report = {"calibration_videos": len(calibration_ids),
              "evaluation_videos": len(heldout), "overlap_videos": 0,
              "selector": "deployed HERMES ranking from original LU traces",
              "metric": "video-balanced retained raw projected future-answer utility",
              "budget_mapping": "K*(1+0.5*(consensus_score/mean_score-1)); clip 0.75K..1.5K; exact sum",
              "results": {}}
    rng = np.random.default_rng(20260922)
    for k in K_VALUES:
        video_ids = sorted(video_values[k])
        names = list(budgets[k])
        matrix = np.array([[np.mean(video_values[k][vid][name]) for name in names]
                           for vid in video_ids])
        means = matrix.mean(axis=0)
        baseline = matrix[:, names.index("old_lu50_constrained")]
        entries = {}
        for j, name in enumerate(names):
            diff = matrix[:, j] - baseline
            sample = rng.choice(diff, size=(10000, len(diff)), replace=True).mean(axis=1)
            entries[name] = {
                "video_mean_retention": float(means[j]),
                "vs_old_lu50_pp": float(100 * diff.mean()),
                "vs_old_lu50_bootstrap95_pp": (100 * np.percentile(sample, [2.5, 97.5])).tolist(),
                "budget_min": int(budgets[k][name].min()),
                "budget_max": int(budgets[k][name].max()),
            }
        report["results"][str(k)] = {"eligible_videos": len(video_ids),
                                      "eligible_observations": eligible_counts[k],
                                      "policies": entries}
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
