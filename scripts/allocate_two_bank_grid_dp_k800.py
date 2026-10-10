#!/usr/bin/env python3
"""Exact global two-bank DP allocation for mean K=800 with a 75% head floor."""

import csv
import json
from pathlib import Path

import numpy as np

from allocate_two_bank_grid_dp_floor750 import exact_floor_allocation


ROOT = Path("results/myobservations/two_bank_grid_dp_20260923")
OUT = Path("results/myobservations/two_bank_grid_dp_k800_floor600_20260928")
HEADS = 112
MEAN = 800
FLOOR = 600
CAP = 1200


def main():
    with np.load(ROOT / "head_n_v_curves.npz") as data:
        values = data["values"]
        endpoints = data["endpoint_recent"]
    allocation, objective = exact_floor_allocation(
        values, progress=True, total=HEADS * MEAN, floor=FLOOR, cap=CAP
    )
    recent = endpoints[np.arange(HEADS), allocation]
    history = allocation - recent
    assert int(allocation.sum()) == HEADS * MEAN
    assert np.isclose(objective, values[np.arange(HEADS), allocation].sum())
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [{
        "head_id": head, "layer": head // 4, "kv_head": head % 4,
        "capacity_n": int(allocation[head]), "recent_r": int(recent[head]),
        "history_q": int(history[head]),
        "selected_proxy": float(values[head, allocation[head]]),
    } for head in range(HEADS)]
    with (OUT / "per_head_allocation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "mean_budget": MEAN, "total_budget": HEADS * MEAN,
        "per_head_floor": FLOOR, "per_head_cap": CAP,
        "recent_total": int(recent.sum()), "history_total": int(history.sum()),
        "objective": objective,
        "source": str(ROOT / "head_n_v_curves.npz"),
        "caveat": "In-sample full-cache answer-utility proxy, not compressed QA",
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
