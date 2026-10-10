#!/usr/bin/env python3
"""Matched 1-token comparison of unified-ranking and two-bank allocations."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from allocate_two_bank_grid_dp_floor750 import exact_floor_allocation


ROOT = Path("results/myobservations/lu_kv_offline_50v_20260913")
TWO = Path("results/myobservations/two_bank_grid_dp_floor750_20260923")
GRID = Path("results/myobservations/two_bank_grid_dp_20260923")
OUT = Path("results/myobservations/single_vs_two_bank_floor750_20260923")
HEADS = 112
CAP = 1500
RECENT_PROTECTED = 32
ROWS = np.arange(HEADS)[:, None]


def load_unified_curves():
    """Old HERMES policy: protect Recent-32, rank all remaining tokens together."""
    by_video = defaultdict(list)
    for path in sorted(ROOT.glob("v*_q*.npz")):
        with np.load(path) as z:
            u = z["utility"].astype(np.float64)
            ranking = z["ranking"].astype(np.int64)
            times = z["source_times"].astype(np.float64)
        unique = np.unique(times)
        if len(unique) < 2:
            continue
        start = int(np.searchsorted(times, unique[-2], side="left"))
        if len(times) - start < 512 or start < 1472:
            continue
        if u.shape[0] != HEADS or ranking.shape[0] != HEADS or ranking.shape[1] < CAP-RECENT_PROTECTED:
            raise AssertionError(f"Bad trace dimensions: {path}")
        full = float(u.sum())
        recent32 = u[:, -RECENT_PROTECTED:].sum(axis=1)
        ranked = u[ROWS, ranking[:, :CAP-RECENT_PROTECTED]]
        prefix = np.column_stack((np.zeros(HEADS), np.cumsum(ranked, axis=1)))
        curve = np.zeros((HEADS, CAP+1))
        curve[:, RECENT_PROTECTED:] = (recent32[:, None] + prefix) / full
        by_video[path.stem.split("_")[0]].append(curve)
    return {video: np.mean(observations, axis=0) for video, observations in by_video.items()}, \
           {video: len(observations) for video, observations in by_video.items()}


def main():
    per_video_curves, observation_counts = load_unified_curves()
    videos = sorted(per_video_curves)
    one_curve = np.mean([per_video_curves[v] for v in videos], axis=0)
    with np.load(GRID/"head_n_v_curves.npz") as z:
        two_curve = z["values"]
        endpoints = z["endpoint_recent"]
    two_rows = list(csv.DictReader((TWO/"per_head_allocation.csv").open()))
    two_alloc = np.array([int(row["floor750_n"]) for row in two_rows])
    if len(videos) != 48 or sum(observation_counts.values()) != 68:
        raise AssertionError("One-dimensional comparison did not recover the same cohort")
    one_alloc, one_objective = exact_floor_allocation(one_curve, progress=True)
    ix = np.arange(HEADS)
    two_objective = float(two_curve[ix, two_alloc].sum())
    one_on_two_alloc = float(one_curve[ix, two_alloc].sum())
    two_on_one_alloc = float(two_curve[ix, one_alloc].sum())
    one_recent32 = np.full(HEADS, RECENT_PROTECTED)
    two_recent = endpoints[ix, two_alloc]
    assert one_alloc.sum() == two_alloc.sum() == 112000
    per_video = []
    for video in videos:
        per_video.append({"video": video, "observations": observation_counts[video],
                          "one_dim_proxy": float(per_video_curves[video][ix, one_alloc].sum())})
    per_head = []
    for h in range(HEADS):
        per_head.append({"head_id": h, "layer": h//4, "kv_head": h%4,
                         "one_dim_n": int(one_alloc[h]), "two_bank_n": int(two_alloc[h]),
                         "delta_n_two_minus_one": int(two_alloc[h]-one_alloc[h]),
                         "two_recent_r": int(two_recent[h]),
                         "two_history_q": int(two_alloc[h]-two_recent[h]),
                         "one_policy_recent_floor": RECENT_PROTECTED,
                         "one_selected_proxy": float(one_curve[h,one_alloc[h]]),
                         "two_selected_proxy": float(two_curve[h,two_alloc[h]])})
    OUT.mkdir(parents=True, exist_ok=True)
    for name, entries in (("per_head_allocation.csv", per_head), ("per_video.csv", per_video)):
        with (OUT/name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(entries[0]))
            writer.writeheader()
            writer.writerows(entries)
    np.savez_compressed(OUT/"one_dim_n_v_curves.npz", values=one_curve,
                        capacity=np.arange(CAP+1))

    heads = np.arange(HEADS)
    fig, ax = plt.subplots(figsize=(14.7, 5.2), layout="constrained")
    ax.plot(heads, one_alloc, color="#466D94", marker=".", markersize=3,
            linewidth=1, label="One-dimensional HERMES ranking")
    ax.plot(heads, two_alloc, color="#D75D42", marker=".", markersize=3,
            linewidth=1, label="Two-bank n-v")
    ax.axhline(750, color="#777777", linestyle="--", linewidth=.8)
    ax.axhline(1500, color="#777777", linestyle="--", linewidth=.8)
    ax.set(xlim=(-1,HEADS), ylim=(700,1550), xlabel="Physical KV head (layer × 4 + head)",
           ylabel="Allocated visual KV slots",
           title="Matched one-dimensional vs two-bank allocation (floor 750, cap 1500)")
    ax.set_xticks(np.arange(0,HEADS,8))
    ax.legend(frameon=False)
    fig.savefig(OUT/"allocation_comparison.png", dpi=250)
    fig.savefig(OUT/"allocation_comparison.pdf")
    plt.close(fig)

    report = {"protocol": {"videos": len(videos), "observations": sum(observation_counts.values()),
                           "total_slots": 112000, "head_bounds": [750,1500], "grid_step": 1,
                           "optimizer": "same exact global DP",
                           "one_dim_selector": "protected Recent-32 + HERMES ranking over all older candidates",
                           "two_bank_selector": "newest r from last two visual units + HERMES top-q from strictly older history",
                           "caveat": "In-sample full-cache answer-utility proxy, not compressed-cache QA"},
              "one_dim": {"proxy": one_objective, "heads_at_floor": int(np.sum(one_alloc==750)),
                          "heads_at_cap": int(np.sum(one_alloc==CAP)),
                          "budget_quantiles": np.quantile(one_alloc,[0,.1,.25,.5,.75,.9,1]).tolist()},
              "two_bank": {"proxy": two_objective, "heads_at_floor": int(np.sum(two_alloc==750)),
                           "heads_at_cap": int(np.sum(two_alloc==CAP)),
                           "recent_total": int(two_recent.sum())},
              "comparison": {"two_minus_one_proxy_pp": 100*(two_objective-one_objective),
                             "mean_abs_budget_delta": float(np.abs(two_alloc-one_alloc).mean()),
                             "token_reassigned": int(np.abs(two_alloc-one_alloc).sum()//2),
                             "same_budget_heads": int(np.sum(two_alloc==one_alloc)),
                             "one_dim_score_on_two_bank_alloc": one_on_two_alloc,
                             "two_bank_score_on_one_dim_alloc": two_on_one_alloc}}
    (OUT/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__ == "__main__":
    main()
