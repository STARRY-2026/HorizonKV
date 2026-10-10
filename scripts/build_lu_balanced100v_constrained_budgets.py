#!/usr/bin/env python3
"""Build deployable K=1000 budgets from the balanced 100-video LU traces."""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from head_analysis.lu_kv_offline import make_curves, solve_curves


ROOT = Path("results/myobservations/lu_kv_balanced100v_20260915/traces")
K = 1000
RECENT = 32
SELECTOR_FLOOR = 225
LOWER, UPPER, BLEND = 750, 1500, 0.5
NAME = "lu_raw_balanced100v_blend05_clip075_150"


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


manifest = json.loads((ROOT / "manifest.json").read_text())["videos"]
per_video = defaultdict(list)
used = []
for video in manifest:
    for question in video["conversations"]:
        stem = f"v{video['video_id']}_q{question['source_question_index']}"
        with np.load(ROOT / f"{stem}.npz") as data:
            utility = data["utility"].astype(np.float64)
            ranking = data["ranking"]
        if utility.shape[1] < K:
            continue
        curves = make_curves(ranking, utility, RECENT)
        budget, _ = solve_curves(curves, 112 * K, SELECTOR_FLOOR)
        per_video[str(video["video_id"])].append(budget)
        used.append(stem)

video_means = [np.mean(parts, axis=0) for parts in per_video.values()]
raw_lu = np.mean(video_means, axis=0)
budget = exact_bounded_round(K + BLEND * (raw_lu - K))

budget_path = ROOT / f"{NAME}_budget.csv"
score_path = ROOT / f"{NAME}_runtime_score.csv"
with budget_path.open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["layer", "kv_head", "budget"])
    for i, value in enumerate(budget):
        writer.writerow([i // 4, i % 4, int(value)])
with score_path.open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["layer", "kv_head", "score"])
    for i, value in enumerate(budget):
        writer.writerow([i // 4, i % 4, int(value - SELECTOR_FLOOR)])

report = {
    "name": NAME,
    "source": str(ROOT),
    "K": K,
    "blend_with_uniform": BLEND,
    "bounds": [LOWER, UPPER],
    "selector_floor": SELECTOR_FLOOR,
    "num_videos": len(per_video),
    "num_observations": len(used),
    "budget_sum": int(budget.sum()),
    "budget_range": [int(budget.min()), int(budget.max())],
    "budget_path": str(budget_path),
    "runtime_score_path": str(score_path),
    "videos": sorted(per_video),
    "observations": used,
}
(ROOT / f"{NAME}_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k not in ("videos", "observations")}, indent=2))
