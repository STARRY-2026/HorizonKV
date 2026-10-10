#!/usr/bin/env python3
"""Scale the original K=1000 LU profile to K=1200 with an exact global total."""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from head_analysis.hermes_head_budget import load_head_scores
from head_analysis.hermes_kv_head_budget import build_kv_head_budget_table


ROOT = Path("results/myobservations/lu_kv_offline_50v_20260913")
SOURCE = ROOT / "lu_raw_blend05_clip075_150_runtime_score.csv"
K = 1200
RECENT = 32
FLOOR = int((K - RECENT) * 0.2) + RECENT
TARGET = 112 * K

scores = load_head_scores(str(SOURCE), 28, 28)
budget = build_kv_head_budget_table(
    scores,
    K,
    scheme="sparsemm",
    sparsemm_ratio=0.2,
    sparsemm_window_size=RECENT,
).reshape(-1).astype(np.int64)

excess = int(budget.sum() - TARGET)
if excess > 0:
    order = np.argsort(-budget, kind="stable")
    budget[order[:excess]] -= 1
elif excess < 0:
    order = np.argsort(budget, kind="stable")
    budget[order[:-excess]] += 1

assert int(budget.sum()) == TARGET
assert int(budget.min()) >= FLOOR

for suffix, field, values in (
    ("budget", "budget", budget),
    ("runtime_score", "score", budget - FLOOR),
):
    path = ROOT / f"lu_raw_k1200_scaled_exact_blend05_clip075_150_{suffix}.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["layer", "kv_head", field])
        for index, value in enumerate(values):
            writer.writerow([index // 4, index % 4, int(value)])

print({
    "sum": int(budget.sum()),
    "mean": float(budget.mean()),
    "range": [int(budget.min()), int(budget.max())],
    "selector_floor": FLOOR,
    "corrected_tokens": abs(excess),
})
