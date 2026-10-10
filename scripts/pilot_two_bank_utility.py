#!/usr/bin/env python3
"""Small offline pilot for a fixed recent/history KV-cache split.

This is a selector-utility proxy test, not compressed-cache QA.  Candidate
regions are fixed by an eight-second boundary before any capacity is chosen.
Recent keeps the newest r tokens; history keeps the q highest HERMES scores.
"""

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path("results/myobservations/lu_kv_offline_50v_20260913")
OUT = Path("results/myobservations/two_bank_pilot_20260923")
RECENT_SECONDS = 8.0
TOTAL_PER_HEAD = 1000
RECENT_CHOICES = (32, 64, 128, 256, 512)
HEADS = 112


def load_observations():
    observations = []
    for path in sorted(ROOT.glob("v*_q*.npz")):
        with np.load(path) as z:
            u = z["utility"].astype(np.float64)
            score = z["hermes_score"].astype(np.float64)
            ranking = z["ranking"].astype(np.int64)
            times = z["source_times"].astype(np.float64)
        n = u.shape[1]
        if not np.all(np.diff(times) >= 0):
            raise ValueError(f"Source timestamps are not ordered: {path}")
        recent_start = int(np.searchsorted(times, times[-1] - RECENT_SECONDS, side="right"))
        recent_count = n - recent_start
        if n < TOTAL_PER_HEAD or recent_count < max(RECENT_CHOICES):
            continue
        if recent_start < TOTAL_PER_HEAD - min(RECENT_CHOICES):
            continue
        rows = np.arange(HEADS)[:, None]
        total_utility = float(u.sum())
        baseline = float(u[:, n-32:].sum() + u[rows, ranking[:, :TOTAL_PER_HEAD-32]].sum())
        # Need only the largest historical capacity. Sort the short candidate
        # prefix after partition to get nested top-q sets for all r choices.
        max_q = TOTAL_PER_HEAD - min(RECENT_CHOICES)
        hist_score = score[:, :recent_start]
        top = np.argpartition(-hist_score, max_q-1, axis=1)[:, :max_q]
        top_score = np.take_along_axis(hist_score, top, axis=1)
        top = np.take_along_axis(top, np.argsort(-top_score, axis=1, kind="stable"), axis=1)
        ordered_utility = u[rows, top]
        hist_prefix = np.cumsum(ordered_utility, axis=1)
        choices = np.empty((HEADS, len(RECENT_CHOICES)), dtype=np.float64)
        soft_choices = np.empty_like(choices)
        for j, r in enumerate(RECENT_CHOICES):
            q = TOTAL_PER_HEAD - r
            choices[:, j] = u[:, n-r:].sum(axis=1) + hist_prefix[:, q-1]
            if r == 32:
                soft_choices[:, j] = u[:, n-r:].sum(axis=1) + u[rows, ranking[:, :q]].sum(axis=1)
            else:
                older_score = score[:, :n-r]
                older_top = np.argpartition(-older_score, q-1, axis=1)[:, :q]
                soft_choices[:, j] = u[:, n-r:].sum(axis=1) + u[rows, older_top].sum(axis=1)
        if not np.allclose(soft_choices[:, 0].sum(), baseline, rtol=0, atol=1e-7):
            raise AssertionError(f"Recent-32 soft control differs from baseline: {path}")
        if np.any(choices > u.sum(axis=1)[:, None] + 1e-8):
            raise AssertionError(f"Hard split retained more than full-cache utility: {path}")
        vid = path.stem.split("_")[0]
        observations.append({
            "video": vid, "observation": path.stem,
            "total": total_utility, "baseline": baseline,
            "choice_utility": choices, "soft_choice_utility": soft_choices,
            "recent_candidates": recent_count,
        })
    return observations


def video_means(observations, choice_key):
    videos = sorted({o["video"] for o in observations})
    return videos, {
        vid: np.mean([o[choice_key] / o["total"] for o in observations
                      if o["video"] == vid], axis=0)
        for vid in videos
    }


def evaluate(observations, choice_key):
    videos, profiles = video_means(observations, choice_key)
    if len(videos) < 3:
        raise ValueError("Need at least three videos for held-out calibration")
    rows = []
    choices_by_fold = []
    for heldout in videos:
        train = np.mean([profiles[v] for v in videos if v != heldout], axis=0)
        head_choice = train.argmax(axis=1)
        shared_choice = int(train.sum(axis=0).argmax())
        choices_by_fold.append(head_choice)
        test = [o for o in observations if o["video"] == heldout]
        result = {"video": heldout, "observations": len(test),
                  "head_recent_choice_count": {str(r): int(np.sum(head_choice == j))
                                               for j, r in enumerate(RECENT_CHOICES)},
                  "shared_recent_choice": int(RECENT_CHOICES[shared_choice])}
        for name in ("actual_hermes", "two_bank_r32", "shared_recent", "head_recent"):
            vals = []
            for o in test:
                if name == "actual_hermes":
                    retained = o["baseline"]
                elif name == "two_bank_r32":
                    retained = float(o[choice_key][:, 0].sum())
                elif name == "shared_recent":
                    retained = float(o[choice_key][:, shared_choice].sum())
                else:
                    retained = float(o[choice_key][np.arange(HEADS), head_choice].sum())
                vals.append(retained / o["total"])
            result[name] = float(np.mean(vals))
        rows.append(result)
    metrics = {name: float(np.mean([r[name] for r in rows]))
               for name in ("actual_hermes", "two_bank_r32", "shared_recent", "head_recent")}
    rng = np.random.default_rng(20260923)
    def paired(a, b):
        diff = np.array([r[a] - r[b] for r in rows])
        boot = rng.choice(diff, size=(10000, len(diff)), replace=True).mean(axis=1)
        return {"mean_pp": float(diff.mean()*100),
                "video_bootstrap_95_pp": [float(x*100) for x in np.quantile(boot, [.025, .975])],
                "positive_videos": int(np.sum(diff > 1e-12)),
                "negative_videos": int(np.sum(diff < -1e-12))}
    counts = np.bincount(np.concatenate(choices_by_fold), minlength=len(RECENT_CHOICES))
    return {
        "protocol": {"recent_seconds": RECENT_SECONDS, "total_per_head": TOTAL_PER_HEAD,
                     "recent_choices": RECENT_CHOICES, "model_metric": "full-cache raw projected answer utility",
                     "selector": "hard: fixed last-8-second candidate bank; newest r recent tokens; top-q older HERMES scores. soft: newest r protected, top-q all unprotected HERMES scores",
                     "calibration": "leave-one-video-out; each training video equal weight",
                     "not_measured": "compressed-cache QA accuracy or sequential eviction survival"},
        "videos": len(videos), "observations": len(observations),
        "recent_candidates": sorted({o["recent_candidates"] for o in observations}),
        "video_mean_retention": metrics,
        "head_vs_r32": paired("head_recent", "two_bank_r32"),
        "head_vs_shared": paired("head_recent", "shared_recent"),
        "head_vs_actual_hermes": paired("head_recent", "actual_hermes"),
        "r32_vs_actual_hermes": paired("two_bank_r32", "actual_hermes"),
        "loov_head_choice_counts": {str(r): int(counts[j]) for j, r in enumerate(RECENT_CHOICES)},
        "per_video": rows,
    }


def main():
    observations = load_observations()
    report = evaluate(observations, "choice_utility")
    soft = evaluate(observations, "soft_choice_utility")
    report["soft_reservoir"] = {k: v for k, v in soft.items() if k != "per_video"}
    report["soft_reservoir_per_video"] = soft["per_video"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    with (OUT / "per_video.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "observations", "actual_hermes",
                                             "two_bank_r32", "shared_recent", "head_recent",
                                             "shared_recent_choice"])
        writer.writeheader()
        writer.writerows({k: row[k] for k in writer.fieldnames} for row in report["per_video"])
    with (OUT / "soft_reservoir_per_video.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "observations", "actual_hermes",
                                             "two_bank_r32", "shared_recent", "head_recent",
                                             "shared_recent_choice"])
        writer.writeheader()
        writer.writerows({k: row[k] for k in writer.fieldnames}
                         for row in report["soft_reservoir_per_video"])
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("per_video", "soft_reservoir_per_video")}, indent=2))


if __name__ == "__main__":
    main()
