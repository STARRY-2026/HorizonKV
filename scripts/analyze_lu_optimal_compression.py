#!/usr/bin/env python3
"""Find the curve-level LU oracle gain at the plotted compression points."""

import glob
import heapq
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from head_analysis.lu_kv_offline import concave_blocks


ROOT = Path("results/myobservations/lu_kv_offline_50v_20260913")
COMPRESSION = np.asarray(
    [0, .1, .2, .3, .4, .5, .6, .7, .8, .85, .9, .925, .95, .975, .99],
    dtype=np.float64,
)
RECENT = 32


def load_curves(path):
    with np.load(path) as data:
        utility = data["utility"].astype(np.float64)
        ranking = data["ranking"].astype(np.int64)
    heads, tokens = utility.shape
    recent = min(RECENT, tokens)
    history = utility[:, :tokens - recent]
    ordered = np.take_along_axis(history, ranking[:, :history.shape[1]], axis=1)
    curves = np.zeros((heads, tokens + 1), dtype=np.float64)
    curves[:, recent] = utility[:, tokens - recent:].sum(axis=1)
    curves[:, recent + 1:] = curves[:, recent, None] + np.cumsum(ordered, axis=1)
    return curves


def solve_many(curves, per_head_targets):
    """Reuse one concave-envelope allocation path for all requested totals."""
    heads, size = curves.shape
    lower = min(RECENT, size - 1)
    blocks = [concave_blocks(np.diff(row[lower:])) for row in curves]
    heap = []
    for head, sequence in enumerate(blocks):
        if sequence:
            gain, count = sequence[0]
            heapq.heappush(heap, (-gain / count, head, 0, count))
    budget = np.full(heads, lower, dtype=np.int64)
    current = heads * lower
    result = {}
    for target_per_head in sorted(set(per_head_targets)):
        target = heads * target_per_head
        while current < target:
            negative_gain, head, block_index, left = heapq.heappop(heap)
            take = min(left, target - current)
            budget[head] += take
            current += take
            left -= take
            if left:
                heapq.heappush(heap, (negative_gain, head, block_index, left))
            elif block_index + 1 < len(blocks[head]):
                gain, count = blocks[head][block_index + 1]
                heapq.heappush(
                    heap, (-gain / count, head, block_index + 1, count)
                )
        result[target_per_head] = float(
            curves[np.arange(heads), budget].sum()
        )
    return result


by_video = defaultdict(list)
budget_rows = []
for npz_path in sorted(glob.glob(str(ROOT / "v*_q*.npz"))):
    stem = Path(npz_path).stem
    meta = json.loads((ROOT / f"{stem}.json").read_text())
    curves = load_curves(npz_path)
    tokens = curves.shape[1] - 1
    budgets = [
        max(RECENT, min(tokens, int(round((1.0 - ratio) * tokens))))
        for ratio in COMPRESSION
    ]
    optimized = solve_many(curves, budgets)
    total = float(curves[:, -1].sum())
    values = []
    for ratio, budget in zip(COMPRESSION, budgets):
        uniform = float(curves[:, budget].sum()) / total
        oracle = optimized[budget] / total
        values.append((uniform, oracle, oracle - uniform))
        budget_rows.append((float(ratio), int(budget), int(tokens)))
    by_video[str(meta["video_id"])].append(np.asarray(values))

video_values = [np.mean(np.stack(rows), axis=0) for rows in by_video.values()]
mean = np.mean(np.stack(video_values), axis=0)
rows = []
for index, ratio in enumerate(COMPRESSION):
    budgets = np.asarray([b for c, b, _ in budget_rows if c == ratio])
    tokens = np.asarray([n for c, _, n in budget_rows if c == ratio])
    rows.append({
        "compression": float(ratio),
        "uniform_coverage": float(mean[index, 0]),
        "oracle_coverage": float(mean[index, 1]),
        "oracle_gain": float(mean[index, 2]),
        "budget_mean": float(budgets.mean()),
        "budget_median": float(np.median(budgets)),
        "budget_min": int(budgets.min()),
        "budget_max": int(budgets.max()),
        "mean_retained_ratio": float((budgets / tokens).mean()),
    })

best = max(rows, key=lambda row: row["oracle_gain"])
print(json.dumps({
    "definition": "per-observation concave-envelope allocation oracle minus uniform",
    "videos": len(by_video),
    "observations": sum(len(rows_) for rows_ in by_video.values()),
    "best": best,
    "rows": rows,
}, indent=2))
