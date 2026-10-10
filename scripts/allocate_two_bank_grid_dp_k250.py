#!/usr/bin/env python3
"""Allocate K=250/head using cached exact 1-token two-bank n-v curves."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from allocate_two_bank_grid_dp_floor750 import exact_floor_allocation


ROOT = Path("results/myobservations/two_bank_grid_dp_20260923")
OUT = Path("results/myobservations/two_bank_grid_dp_k250_floor188_20260923")
HEADS = 112
AVERAGE = 250
TOTAL = HEADS * AVERAGE
FLOOR = 188  # nearest integer to 0.75 * 250 = 187.5
CAP = 375    # 1.5 * 250


def main():
    with np.load(ROOT/"head_n_v_curves.npz") as z:
        curves = z["values"]
        endpoints = z["endpoint_recent"]
    alloc, objective = exact_floor_allocation(
        curves, progress=True, total=TOTAL, floor=FLOOR, cap=CAP)
    heads = np.arange(HEADS)
    recent = endpoints[heads, alloc]
    history = alloc - recent
    if (recent < 0).any() or (history < 0).any() or recent.sum()+history.sum() != TOTAL:
        raise AssertionError("Invalid bank allocation")
    if not np.isclose(objective, curves[heads,alloc].sum(), atol=1e-9):
        raise AssertionError("Objective differs from selected curve values")

    OUT.mkdir(parents=True, exist_ok=True)
    rows = [{"head_id": int(h), "layer": int(h//4), "kv_head": int(h%4),
             "capacity_n": int(alloc[h]), "recent_r": int(recent[h]),
             "history_q": int(history[h]),
             "history_share": float(history[h]/alloc[h]),
             "selected_proxy": float(curves[h,alloc[h]])} for h in heads]
    with (OUT/"per_head_allocation.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(15.2, 5.3), layout="constrained")
    ax.bar(heads, history, width=.88, color="#355C7D", label="History cache")
    ax.bar(heads, recent, bottom=history, width=.88, color="#E5A54B", label="Recent cache")
    ax.axhline(FLOOR, color="#B15A40", linestyle="--", linewidth=.9,
               label="Floor: 188")
    ax.axhline(AVERAGE, color="#555555", linestyle=":", linewidth=.9,
               label="Mean: 250")
    ax.set(xlim=(-1, HEADS), ylim=(0, 400),
           xlabel="Physical KV head (layer × 4 + head)",
           ylabel="Visual KV slots",
           title="Two-bank exact-DP allocation at K=250 (28,000 total slots)")
    ax.set_xticks(np.arange(0, HEADS, 8))
    ax.legend(ncol=4, frameon=False, loc="upper left")
    fig.savefig(OUT/"per_head_allocation.png", dpi=270)
    fig.savefig(OUT/"per_head_allocation.pdf")
    plt.close(fig)

    report = {
        "protocol": {"videos": 48, "observations": 68,
                     "average_slots_per_head": AVERAGE, "total_visual_slots": TOTAL,
                     "head_capacity_range": [FLOOR,CAP], "grid_step": 1,
                     "optimizer": "exact global DP over previously computed two-bank n-v curves",
                     "recent_candidates": "newest 512 visual tokens in last two Qwen visual time units",
                     "caveat": "In-sample full-cache answer-utility proxy, not compressed-cache QA"},
        "allocation": {"recent_total": int(recent.sum()),
                       "history_total": int(history.sum()),
                       "recent_share": float(recent.sum()/TOTAL),
                       "history_share": float(history.sum()/TOTAL),
                       "head_capacity_quantiles": np.quantile(alloc,[0,.1,.25,.5,.75,.9,1]).tolist(),
                       "head_recent_quantiles": np.quantile(recent,[0,.1,.25,.5,.75,.9,1]).tolist(),
                       "heads_at_floor": int(np.sum(alloc==FLOOR)),
                       "heads_at_cap": int(np.sum(alloc==CAP)),
                       "heads_zero_recent": int(np.sum(recent==0)),
                       "heads_recent_below_32": int(np.sum(recent<32)),
                       "objective": objective}}
    (OUT/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__ == "__main__":
    main()
