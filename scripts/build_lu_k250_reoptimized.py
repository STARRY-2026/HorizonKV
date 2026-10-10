#!/usr/bin/env python3
"""Re-solve deployable LU budgets at K=250 from the frozen 50-video traces."""

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from head_analysis.hermes_head_budget import load_head_scores
from head_analysis.hermes_kv_head_budget import build_kv_head_budget_table
from head_analysis.lu_kv_offline import make_curves, solve_curves


ROOT = Path("results/myobservations/lu_kv_offline_50v_20260913")
CURRENT_SCORE = ROOT / "lu_raw_blend05_clip075_150_runtime_score.csv"
K = 250
RECENT = 32
SELECTOR_FLOOR = int((K - RECENT) * 0.2) + RECENT
LOWER, UPPER, BLEND = 188, 375, 0.5


def exact_bounded_round(raw):
    raw = np.asarray(raw, dtype=np.float64)
    lo, hi = -10000.0, 10000.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if np.clip(raw + mid, LOWER, UPPER).sum() < 112 * K:
            lo = mid
        else:
            hi = mid
    projected = np.clip(raw + (lo + hi) / 2, LOWER, UPPER)
    out = np.floor(projected).astype(np.int64)
    missing = 112 * K - int(out.sum())
    candidates = np.where(out < UPPER)[0]
    order = candidates[np.argsort(-(projected[candidates] - out[candidates]), kind="stable")]
    out[order[:missing]] += 1
    assert out.sum() == 112 * K
    assert out.min() >= LOWER and out.max() <= UPPER
    return out


def write_table(path, field, values):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["layer", "kv_head", field])
        for index, value in enumerate(values):
            writer.writerow([index // 4, index % 4, int(value)])


manifest = json.loads((ROOT / "manifest.json").read_text())["videos"]
observations = []
per_video_all = defaultdict(list)
per_video_matched = defaultdict(list)
task_counts = Counter()

for video in manifest:
    for question in video["conversations"]:
        stem = f"v{video['video_id']}_q{question['source_question_index']}"
        meta = json.loads((ROOT / f"{stem}.json").read_text())
        num_tokens = int(meta["visual_tokens"])
        if num_tokens < K:
            continue
        with np.load(ROOT / f"{stem}.npz") as data:
            curves = make_curves(data["ranking"], data["utility"], RECENT)
        solved, _ = solve_curves(curves, 112 * K, SELECTOR_FLOOR)
        video_id = str(video["video_id"])
        per_video_all[video_id].append(solved)
        if num_tokens >= 1000:
            per_video_matched[video_id].append(solved)
        observations.append({
            "stem": stem,
            "video_id": video_id,
            "task": question["task"],
            "num_tokens": num_tokens,
        })
        task_counts[question["task"]] += 1


def aggregate(per_video):
    video_means = [np.mean(parts, axis=0) for parts in per_video.values()]
    raw = np.mean(video_means, axis=0)
    return exact_bounded_round(K + BLEND * (raw - K))


resolved_all = aggregate(per_video_all)
resolved_matched = aggregate(per_video_matched)
current_scores = load_head_scores(str(CURRENT_SCORE), 28, 28)
current_scaled = build_kv_head_budget_table(
    current_scores,
    K,
    scheme="sparsemm",
    sparsemm_ratio=0.2,
    sparsemm_window_size=RECENT,
).reshape(-1).astype(np.int64)

policies = {
    "uniform": np.full(112, K, dtype=np.int64),
    "current_scaled_from_k1000": current_scaled,
    "resolved_k250_matched79": resolved_matched,
    "resolved_k250_all": resolved_all,
}

retention = {"all_k250_eligible": defaultdict(list), "matched_k1000_eligible": defaultdict(list)}
for observation in observations:
    with np.load(ROOT / f"{observation['stem']}.npz") as data:
        curves = make_curves(data["ranking"], data["utility"], RECENT)
    total = float(curves[:, -1].sum())
    groups = ["all_k250_eligible"]
    if observation["num_tokens"] >= 1000:
        groups.append("matched_k1000_eligible")
    for group in groups:
        for name, budget in policies.items():
            effective = np.minimum(budget, curves.shape[1] - 1)
            kept = float(curves[np.arange(112), effective].sum())
            retention[group][name].append(kept / total)


def comparison(reference, candidate):
    delta = candidate - reference
    return {
        "pearson": float(np.corrcoef(reference, candidate)[0, 1]),
        "mean_absolute_difference": float(np.abs(delta).mean()),
        "max_absolute_difference": int(np.abs(delta).max()),
        "heads_changed": int(np.count_nonzero(delta)),
        "heads_crossing_uniform_250": int(np.sum((reference - K) * (candidate - K) < 0)),
        "delta_range": [int(delta.min()), int(delta.max())],
        "largest_shifts": [
            {
                "layer": int(index // 4),
                "kv_head": int(index % 4),
                "current": int(reference[index]),
                "candidate": int(candidate[index]),
                "delta": int(delta[index]),
            }
            for index in np.argsort(-np.abs(delta), kind="stable")[:20]
        ],
    }


for name, budget in (
    ("lu_raw_k250_resolved_matched79_blend05_clip075_150", resolved_matched),
    ("lu_raw_k250_resolved_all_blend05_clip075_150", resolved_all),
):
    write_table(ROOT / f"{name}_budget.csv", "budget", budget)
    write_table(ROOT / f"{name}_runtime_score.csv", "score", budget - SELECTOR_FLOOR)

report = {
    "K": K,
    "recent": RECENT,
    "selector_floor": SELECTOR_FLOOR,
    "blend_with_uniform": BLEND,
    "bounds": [LOWER, UPPER],
    "num_all_k250_observations": len(observations),
    "num_all_k250_videos": len(per_video_all),
    "num_matched_k1000_observations": sum(len(x) for x in per_video_matched.values()),
    "num_matched_k1000_videos": len(per_video_matched),
    "all_k250_task_counts": dict(sorted(task_counts.items())),
    "budgets": {
        name: {
            "sum": int(values.sum()),
            "range": [int(values.min()), int(values.max())],
            "std": float(values.std()),
        }
        for name, values in policies.items()
    },
    "comparison_to_current": {
        "resolved_k250_matched79": comparison(current_scaled, resolved_matched),
        "resolved_k250_all": comparison(current_scaled, resolved_all),
    },
    "offline_retention": {
        group: {
            name: float(np.mean(values))
            for name, values in group_values.items()
        }
        for group, group_values in retention.items()
    },
}

(ROOT / "lu_k250_reoptimized_report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
