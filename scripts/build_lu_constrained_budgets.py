#!/usr/bin/env python3
"""Build conservative K=1000 LU budgets from the frozen 50-video traces."""

import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from head_analysis.lu_kv_offline import make_curves, solve_curves


ROOT = Path("results/myobservations/lu_kv_offline_50v_20260913")
K = 1000
SELECTOR_FLOOR = 225
LOWER, UPPER, BLEND = 750, 1500, 0.5


def exact_bounded_round(raw):
    raw = np.asarray(raw, dtype=np.float64)
    # Project onto the bounded fixed-sum simplex by bisection.
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
    assert out.sum() == 112 * K and out.min() >= LOWER and out.max() <= UPPER
    return out


def build(correct_only, layer_normalized):
    manifest = json.loads((ROOT / "manifest.json").read_text())["videos"]
    per_video = defaultdict(list)
    used = []
    for video in manifest:
        for question in video["conversations"]:
            stem = f"v{video['video_id']}_q{question['source_question_index']}"
            meta = json.loads((ROOT / f"{stem}.json").read_text())
            if correct_only and not meta["full_cache_correct"]:
                continue
            with np.load(ROOT / f"{stem}.npz") as data:
                utility = data["utility"].astype(np.float64)
                ranking = data["ranking"]
            if utility.shape[1] < K:
                continue
            if layer_normalized:
                shaped = utility.reshape(28, 4, -1)
                scale = shaped.mean(axis=(1, 2), keepdims=True)
                utility = (shaped / np.maximum(scale, 1e-12)).reshape(112, -1)
            curves = make_curves(ranking, utility, 32)
            budget, _ = solve_curves(curves, 112 * K, SELECTOR_FLOOR)
            per_video[str(video["video_id"])].append(budget)
            used.append(stem)
    video_means = [np.mean(parts, axis=0) for parts in per_video.values()]
    lu = np.mean(video_means, axis=0)
    constrained = exact_bounded_round(K + BLEND * (lu - K))
    return constrained, used, sorted(per_video)


def write_variant(name, correct_only, layer_normalized):
    budget, used, videos = build(correct_only, layer_normalized)
    rows = [(l, h, int(budget[l * 4 + h]), int(budget[l * 4 + h] - SELECTOR_FLOOR))
            for l in range(28) for h in range(4)]
    with (ROOT / f"{name}_budget.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["layer", "kv_head", "budget"])
        w.writerows((l, h, b) for l, h, b, _ in rows)
    with (ROOT / f"{name}_runtime_score.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["layer", "kv_head", "score"])
        w.writerows((l, h, s) for l, h, _, s in rows)
    report = {"name": name, "K": K, "blend_with_uniform": BLEND,
              "bounds": [LOWER, UPPER], "selector_floor": SELECTOR_FLOOR,
              "correct_only": correct_only, "layer_normalized": layer_normalized,
              "num_videos": len(videos), "num_observations": len(used),
              "budget_sum": int(budget.sum()), "budget_range": [int(budget.min()), int(budget.max())],
              "videos": videos, "observations": used}
    (ROOT / f"{name}_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in report if k not in ("videos", "observations")}, indent=2))


write_variant("lu_raw_blend05_clip075_150", False, False)
write_variant("lu_correct_layernorm_blend05_clip075_150", True, True)
