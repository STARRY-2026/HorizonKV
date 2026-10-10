#!/usr/bin/env python3
"""Derive K=500 head budgets and recency profile from cached two-bank curves."""

import csv
import json
import sys
from pathlib import Path

import numpy as np

from allocate_two_bank_grid_dp_floor750 import exact_floor_allocation


CURVES = Path("results/myobservations/two_bank_grid_dp_20260923/head_n_v_curves.npz")
OUT = Path("results/myobservations/two_bank_grid_dp_k500_floor375_20260925")
HEADS = 112
AVERAGE = 500
FLOOR = 375
CAP = 750
WINDOW = 32
SPARSEMM_RATIO = 0.2


def write_csv(path, rows, columns):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    with np.load(CURVES) as data:
        curves = data["values"]
        endpoints = data["endpoint_recent"]
    allocation, objective = exact_floor_allocation(
        curves, progress=True, total=HEADS * AVERAGE, floor=FLOOR, cap=CAP
    )
    recent = endpoints[np.arange(HEADS), allocation]
    history = allocation - recent
    assert int(allocation.sum()) == HEADS * AVERAGE
    assert np.all(recent >= 0) and np.all(history >= 0)
    assert np.isclose(objective, curves[np.arange(HEADS), allocation].sum())

    base = AVERAGE - WINDOW
    minimum = int(base * SPARSEMM_RATIO) + WINDOW
    budget_rows = []
    profile_rows = []
    allocation_rows = []
    for head in range(HEADS):
        layer, kv_head = divmod(head, 4)
        capacity = int(allocation[head])
        r = int(recent[head])
        h = int(history[head])
        allocation_rows.append({
            "head_id": head, "layer": layer, "kv_head": kv_head,
            "capacity_n": capacity, "recent_r": r, "history_q": h,
            "history_share": h / capacity,
            "selected_proxy": float(curves[head, capacity]),
        })
        budget_rows.append({
            "layer": layer, "kv_head": kv_head, "score": capacity - minimum
        })
        profile_rows.append({
            "layer": layer, "kv_head": kv_head, "recent_ratio": r / capacity,
            "capacity_n": capacity, "recent_r": r, "history_q": h,
        })

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "per_head_allocation.csv", allocation_rows,
              list(allocation_rows[0]))
    write_csv(OUT / "dp_k500_exact_budget_scores.csv", budget_rows,
              ["layer", "kv_head", "score"])
    write_csv(OUT / "recent_ratio_profile.csv", profile_rows,
              list(profile_rows[0]))

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from head_analysis.hermes_head_budget import load_head_scores
    from head_analysis.hermes_kv_head_budget import build_kv_head_budget_table

    scores = load_head_scores(str(OUT / "dp_k500_exact_budget_scores.csv"), 28, 28)
    runtime = build_kv_head_budget_table(
        scores, AVERAGE, num_layers=28, num_query_heads=28,
        num_kv_heads=4, scheme="sparsemm", sparsemm_ratio=SPARSEMM_RATIO,
        sparsemm_window_size=WINDOW,
    )
    if not np.array_equal(runtime.reshape(-1), allocation):
        raise AssertionError("Runtime budget mapping differs from DP allocation")

    report = {
        "protocol": {
            "average_slots_per_head": AVERAGE,
            "total_visual_slots": int(allocation.sum()),
            "head_floor": FLOOR,
            "head_cap": CAP,
            "sparsemm_window": WINDOW,
            "sparsemm_ratio": SPARSEMM_RATIO,
            "source_curves": str(CURVES),
            "runtime_budget_mapping": "exact per head",
        },
        "allocation": {
            "recent_total": int(recent.sum()),
            "history_total": int(history.sum()),
            "recent_share": float(recent.sum() / allocation.sum()),
            "head_capacity_min": int(allocation.min()),
            "head_capacity_max": int(allocation.max()),
            "heads_at_floor": int(np.sum(allocation == FLOOR)),
            "heads_at_cap": int(np.sum(allocation == CAP)),
            "objective": objective,
        },
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
