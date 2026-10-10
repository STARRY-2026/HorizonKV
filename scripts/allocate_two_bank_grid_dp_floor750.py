#!/usr/bin/env python3
"""Exact global allocation of cached two-bank head curves with a 750-token floor.

Reuses the 1-token per-head n-v curves from allocate_two_bank_grid_dp.py so
the only experimental change is the minimum per-head capacity.
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from allocate_token_gradient import load_profiles


ROOT = Path("results/myobservations/two_bank_grid_dp_20260923")
OUT = Path("results/myobservations/two_bank_grid_dp_floor750_20260923")
HEADS = 112
TOTAL = 112000
FLOOR = 750
CAP = 1500


def exact_floor_allocation(curves, progress=False, total=TOTAL, floor=FLOOR, cap=CAP):
    """Multiple-choice DP over surplus tokens after all heads receive a floor."""
    extra_total = total - HEADS * floor
    max_extra = cap - floor
    if curves.shape[0] != HEADS or curves.shape[1] < cap + 1:
        raise ValueError(f"Unexpected n-v shape: {curves.shape}")
    if not 0 <= extra_total <= HEADS * max_extra:
        raise ValueError("Infeasible total budget")
    prev = np.full(extra_total + 1, -np.inf)
    prev[0] = 0.0
    back = np.full((HEADS, extra_total + 1), -1, dtype=np.int16)
    prev_lo = prev_hi = 0
    for head in range(HEADS):
        processed = head + 1
        left = HEADS - processed
        lo = max(0, extra_total - left * max_extra)
        hi = min(extra_total, processed * max_extra)
        nxt = np.full_like(prev, -np.inf)
        chosen = back[head]
        for extra in range(max_extra + 1):
            dst_lo = max(lo, prev_lo + extra)
            dst_hi = min(hi, prev_hi + extra)
            if dst_lo > dst_hi:
                continue
            candidate = prev[dst_lo-extra:dst_hi-extra+1] + curves[head, floor+extra]
            current = nxt[dst_lo:dst_hi+1]
            better = candidate > current
            current[better] = candidate[better]
            chosen[dst_lo:dst_hi+1][better] = extra
        prev, prev_lo, prev_hi = nxt, lo, hi
        if progress and (processed % 16 == 0 or processed == HEADS):
            print(f"GLOBAL_DP {processed}/{HEADS} reachable={int(np.isfinite(prev).sum())}", flush=True)
    if not np.isfinite(prev[extra_total]):
        raise AssertionError("Final budget state is unreachable")
    alloc = np.empty(HEADS, dtype=np.int16)
    state = extra_total
    for head in range(HEADS-1, -1, -1):
        extra = int(back[head, state])
        if extra < 0:
            raise AssertionError(f"Broken traceback at head {head}, state {state}")
        alloc[head] = floor + extra
        state -= extra
    if state != 0 or alloc.sum() != total or alloc.min() < floor or alloc.max() > cap:
        raise AssertionError("Allocation violates bounds")
    return alloc, float(prev[extra_total])


def test_small():
    # Independent exhaustive check for a 3-head / 2-extra-token toy problem.
    toy = np.array([[1., 2., 5.], [2., 6., 7.], [3., 4., 9.]])
    dp = np.array([0., -np.inf, -np.inf])
    for h in range(3):
        nxt = np.full_like(dp, -np.inf)
        for k in range(3):
            nxt[k:] = np.maximum(nxt[k:], dp[:3-k] + toy[h, k])
        dp = nxt
    brute = max(toy[0,a] + toy[1,b] + toy[2,c]
                for a in range(3) for b in range(3) for c in range(3) if a+b+c == 2)
    assert np.isclose(dp[2], brute)


def video_value(profile, recent, history):
    idx = np.arange(HEADS)
    return float(np.sum(profile["recent"][idx, recent] + profile["history"][idx, history]))


def main():
    test_small()
    with np.load(ROOT / "head_n_v_curves.npz") as z:
        curves = z["values"]
        endpoints = z["endpoint_recent"]
    rows = list(csv.DictReader((ROOT / "per_head_allocation.csv").open()))
    old = np.array([int(row["capacity_n"]) for row in rows])
    old_r = np.array([int(row["recent_r"]) for row in rows])
    old_h = old - old_r
    alloc, objective = exact_floor_allocation(curves, progress=True)
    idx = np.arange(HEADS)
    recent = endpoints[idx, alloc]
    history = alloc - recent
    old_objective = float(curves[idx, old].sum())
    assert np.isclose(objective, curves[idx, alloc].sum(), atol=1e-9)
    assert objective <= old_objective + 1e-9

    profiles, _ = load_profiles()
    per_video = []
    for video, profile in sorted(profiles.items()):
        before = video_value(profile, old_r, old_h)
        after = video_value(profile, recent, history)
        per_video.append({"video": video, "observations": profile["observations"],
                          "unconstrained_proxy": before, "floor750_proxy": after,
                          "delta_pp": 100 * (after-before)})
    per_head = []
    for h in range(HEADS):
        per_head.append({"head_id": h, "layer": h//4, "kv_head": h%4,
                         "unconstrained_n": int(old[h]), "floor750_n": int(alloc[h]),
                         "delta_n": int(alloc[h]-old[h]), "recent_r": int(recent[h]),
                         "history_q": int(history[h]),
                         "history_share": float(history[h]/alloc[h]),
                         "selected_proxy": float(curves[h,alloc[h]])})
    OUT.mkdir(parents=True, exist_ok=True)
    for name, entries in (("per_head_allocation.csv", per_head), ("per_video.csv", per_video)):
        with (OUT/name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(entries[0]))
            writer.writeheader()
            writer.writerows(entries)

    heads = np.arange(HEADS)
    fig, axes = plt.subplots(2, 1, figsize=(15.2, 8), sharex=True, layout="constrained")
    for ax, cap, r, title in ((axes[0], old, old_r, "No per-head floor"),
                               (axes[1], alloc, recent, "750-token minimum per head")):
        ax.bar(heads, cap-r, width=.85, color="#355C7D", label="History")
        ax.bar(heads, r, bottom=cap-r, width=.85, color="#E5A54B", label="Recent")
        ax.axhline(750, color="#C95D42", linestyle="--", linewidth=.9)
        ax.axhline(1000, color="#555555", linestyle=":", linewidth=.9)
        ax.set(ylabel="Visual KV slots", title=title, xlim=(-1, HEADS), ylim=(0, 1550))
    axes[0].legend(ncol=2, frameon=False, loc="upper left")
    axes[1].set_xlabel("Physical KV head (layer × 4 + head)")
    axes[1].set_xticks(np.arange(0, HEADS, 8))
    fig.suptitle("Exact n-v allocation, same 112,000-token budget; dashed line: 750-token floor")
    fig.savefig(OUT/"allocation_comparison.png", dpi=250)
    fig.savefig(OUT/"allocation_comparison.pdf")
    plt.close(fig)

    diffs = np.array([row["delta_pp"] for row in per_video])
    report = {
        "protocol": {"videos": len(profiles), "observations": sum(p["observations"] for p in profiles.values()),
                     "total_visual_slots": TOTAL, "head_floor": FLOOR, "head_cap": CAP,
                     "curves": str(ROOT/"head_n_v_curves.npz"),
                     "caveat": "In-sample full-cache answer-utility proxy, not compressed-cache QA"},
        "unconstrained": {"objective": old_objective, "recent_total": int(old_r.sum()),
                          "history_total": int(old_h.sum()), "heads_zero": int(np.sum(old==0)),
                          "heads_at_cap": int(np.sum(old==CAP))},
        "floor750": {"objective": objective, "recent_total": int(recent.sum()),
                     "history_total": int(history.sum()), "history_share": float(history.sum()/TOTAL),
                     "heads_at_floor": int(np.sum(alloc==FLOOR)),
                     "heads_at_cap": int(np.sum(alloc==CAP)),
                     "head_capacity_quantiles": np.quantile(alloc, [0,.1,.25,.5,.75,.9,1]).tolist()},
        "comparison": {"objective_delta_pp": 100*(objective-old_objective),
                       "objective_relative_change": (objective/old_objective-1),
                       "capacity_mean_absolute_change": float(np.abs(alloc-old).mean()),
                       "token_reassigned": int(np.abs(alloc-old).sum()//2),
                       "per_video_delta_pp_mean": float(diffs.mean()),
                       "per_video_delta_pp_min": float(diffs.min()),
                       "per_video_delta_pp_max": float(diffs.max()),
                       "videos_improved": int(np.sum(diffs>1e-10)),
                       "videos_worsened": int(np.sum(diffs< -1e-10))}}
    (OUT/"report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
