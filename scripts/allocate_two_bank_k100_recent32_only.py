#!/usr/bin/env python3
"""Exact K=100 two-bank allocation with only a Recent-32 lower bound.

Uses the same 68 observations / 48 videos as the original two-bank curves,
extending their shared measurable domain to 512 recent + 1536 history tokens.
The resulting 2048-token per-head domain is a data-support limit, not a
chosen budget regularizer. This is an offline full-cache utility proxy.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from allocate_two_bank_grid_dp_floor750 import exact_floor_allocation


ROOT = Path("results/myobservations/lu_kv_offline_50v_20260913")
OUT = Path("results/myobservations/two_bank_k100_recent32_only_20260930")
HEADS = 112
TOTAL = 11200
MIN_RECENT = 32
MAX_RECENT = 512
MAX_HISTORY = 1536
MAX_CAPACITY = MAX_RECENT + MAX_HISTORY


def load_mean_profiles():
    by_video = defaultdict(list)
    rows = np.arange(HEADS)[:, None]
    for path in sorted(ROOT.glob("v*_q*.npz")):
        with np.load(path) as data:
            utility = data["utility"].astype(np.float64)
            score = data["hermes_score"].astype(np.float64)
            times = data["source_times"].astype(np.float64)
        unique = np.unique(times)
        if len(unique) < 2:
            continue
        start = int(np.searchsorted(times, unique[-2], side="left"))
        if start < MAX_HISTORY or len(times) - start < MAX_RECENT:
            continue
        full = float(utility.sum())
        recent = np.concatenate((np.zeros((HEADS, 1)),
                                 np.cumsum(utility[:, start:][:, ::-1], axis=1)[:, :MAX_RECENT]),
                                axis=1) / full
        history_score = score[:, :start]
        top = np.argpartition(-history_score, MAX_HISTORY - 1, axis=1)[:, :MAX_HISTORY]
        top_score = np.take_along_axis(history_score, top, axis=1)
        top = np.take_along_axis(top, np.argsort(-top_score, axis=1, kind="stable"), axis=1)
        history = np.concatenate((np.zeros((HEADS, 1)),
                                  np.cumsum(utility[rows, top], axis=1)), axis=1) / full
        by_video[path.stem.split("_")[0]].append((recent, history))
    video_means = [(np.mean([x[0] for x in obs], axis=0),
                    np.mean([x[1] for x in obs], axis=0))
                   for obs in by_video.values()]
    recent = np.mean([x[0] for x in video_means], axis=0)
    history = np.mean([x[1] for x in video_means], axis=0)
    return recent, history, len(by_video), sum(map(len, by_video.values()))


def constrained_curves(recent, history):
    values = np.full((HEADS, MAX_CAPACITY + 1), -np.inf)
    endpoints = np.full((HEADS, MAX_CAPACITY + 1), -1, dtype=np.int16)
    for n in range(MIN_RECENT, MAX_CAPACITY + 1):
        rr = np.arange(max(MIN_RECENT, n - MAX_HISTORY), min(MAX_RECENT, n) + 1)
        qq = n - rr
        candidate = recent[:, rr] + history[:, qq]
        best = np.argmax(candidate, axis=1)
        values[:, n] = candidate[np.arange(HEADS), best]
        endpoints[:, n] = rr[best]
    return values, endpoints


def write_csv(path, records):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    recent_profile, history_profile, videos, observations = load_mean_profiles()
    if (videos, observations) != (48, 68):
        raise AssertionError(f"Unexpected calibration cohort: {videos} videos, {observations} observations")
    values, endpoints = constrained_curves(recent_profile, history_profile)
    allocation, objective = exact_floor_allocation(
        values, progress=True, total=TOTAL, floor=MIN_RECENT, cap=MAX_CAPACITY)
    r = endpoints[np.arange(HEADS), allocation]
    q = allocation - r
    if not np.all(r >= MIN_RECENT) or int(allocation.sum()) != TOTAL:
        raise AssertionError("Allocation violates the Recent-32 or total-budget constraint")

    OUT.mkdir(parents=True, exist_ok=True)
    rows = [{"head_id": h, "layer": h // 4, "kv_head": h % 4,
             "capacity_n": int(allocation[h]), "recent_r": int(r[h]),
             "history_q": int(q[h]), "selected_proxy": float(values[h, allocation[h]])}
            for h in range(HEADS)]
    write_csv(OUT / "per_head_allocation.csv", rows)
    write_csv(OUT / "exact_budget_scores.csv", [
        {"layer": h // 4, "kv_head": h % 4,
         "score": int(allocation[h]) - MIN_RECENT}
        for h in range(HEADS)])
    write_csv(OUT / "recent_ratio_profile.csv", [
        {"layer": h // 4, "kv_head": h % 4,
         "recent_ratio": float(r[h] / allocation[h]),
         "capacity_n": int(allocation[h]), "recent_r": int(r[h]),
         "history_q": int(q[h])}
        for h in range(HEADS)])
    # SparseMM with ratio=0 maps nonnegative scores exactly to Recent-32
    # plus the globally normalized 7,616 additional visual slots.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from head_analysis.hermes_head_budget import load_head_scores
    from head_analysis.hermes_kv_head_budget import build_kv_head_budget_table
    runtime_scores = load_head_scores(str(OUT / "exact_budget_scores.csv"), 28, 28)
    mapped = build_kv_head_budget_table(
        runtime_scores, 100, num_layers=28, num_query_heads=28,
        num_kv_heads=4, scheme="sparsemm", sparsemm_ratio=0.0,
        sparsemm_window_size=32)
    if not np.array_equal(mapped.reshape(-1), allocation):
        raise AssertionError("Online budget mapping differs from Recent-32-only allocation")
    np.savez_compressed(OUT / "constrained_head_n_v_curves.npz",
                        values=values, endpoint_recent=endpoints)

    previous = list(csv.DictReader(Path(
        "results/myobservations/two_bank_grid_dp_k100_floor75_20260929/per_head_allocation.csv").open()))
    old_n = np.array([int(x["capacity_n"]) for x in previous])
    old_r = np.array([int(x["recent_r"]) for x in previous])
    old_q = old_n - old_r
    old_feasible = recent_profile[np.arange(HEADS), np.maximum(MIN_RECENT, old_r)] + \
                   history_profile[np.arange(HEADS), old_n - np.maximum(MIN_RECENT, old_r)]
    report = {
        "protocol": {"videos": videos, "observations": observations,
                     "total_visual_slots": TOTAL,
                     "only_policy_constraint": "at least 32 recent tokens per head",
                     "measured_domain": {"max_recent": MAX_RECENT,
                                         "max_history": MAX_HISTORY,
                                         "max_total": MAX_CAPACITY},
                     "metric": "in-sample full-cache projected answer utility, not compressed QA"},
        "allocation": {"objective": objective,
                       "recent_total": int(r.sum()), "history_total": int(q.sum()),
                       "min_head_capacity": int(allocation.min()),
                       "max_head_capacity": int(allocation.max()),
                       "heads_at_recent32": int(np.sum(r == MIN_RECENT)),
                       "heads_at_measured_max": int(np.sum(allocation == MAX_CAPACITY)),
                       "capacity_quantiles": np.quantile(allocation, [0, .1, .25, .5, .75, .9, 1]).tolist(),
                       "capacity_bins": {f"{lo}-{hi}": int(np.sum((allocation >= lo) & (allocation <= hi)))
                                         for lo, hi in ((32, 49), (50, 99), (100, 199),
                                                        (200, 499), (500, 999), (1000, 2048))}},
        "comparison_with_floor75_cap150": {
            "heads_increased": int(np.sum(allocation > old_n)),
            "heads_decreased": int(np.sum(allocation < old_n)),
            "tokens_reassigned": int(np.abs(allocation - old_n).sum() // 2),
            "old_allocation_with_recent32_proxy": float(old_feasible.sum()),
            "new_vs_old_feasible_relative_proxy_gain": float(objective / old_feasible.sum() - 1),
        },
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
