#!/usr/bin/env python3
"""Allocate an exact global KV budget from the fixed two-bank proxy surface.

Compares the 2D allocation with a one-dimensional HERMES head-budget curve
under identical 32-token grid, per-head bounds, and leave-one-video-out folds.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_two_dimensional_utility_surface import (
    HEADS, HISTORY, RECENTS, ROOT, STEP, top_history_prefix,
)


OUT = Path("results/myobservations/dual_utility_allocation_20260923")
OLD_TABLE = ROOT / "lu_raw_blend05_clip075_150_budget.csv"
TOTAL = 112000
MIN_UNITS = 24  # 768 tokens; closest feasible grid value above old lower bound 750
MAX_UNITS = 46  # 1472 tokens; closest feasible grid value below old upper bound 1500
BUDGET_UNITS = np.arange(MIN_UNITS, MAX_UNITS + 1)
TARGET_UNITS = TOTAL // STEP
MIN_RECENT_UNITS = 1  # Match the existing HERMES Recent-32 protection floor.
ROWS = np.arange(HEADS)[:, None]


def load_profiles():
    old_table = np.genfromtxt(OLD_TABLE, delimiter=",", names=True)
    if len(old_table) != HEADS or int(old_table["budget"].sum()) != TOTAL:
        raise AssertionError("Old budget table is not 112 physical heads / exact budget")
    deployed = old_table["budget"].astype(int)
    by_video = defaultdict(list)
    for path in sorted(ROOT.glob("v*_q*.npz")):
        with np.load(path) as z:
            u = z["utility"].astype(np.float64)
            score = z["hermes_score"].astype(np.float64)
            times = z["source_times"].astype(np.float64)
            ranking = z["ranking"].astype(np.int64)
        unique = np.unique(times)
        if len(unique) < 2:
            continue
        start = int(np.searchsorted(times, unique[-2], side="left"))
        if len(times) - start < RECENTS[-1] or start < HISTORY[-1]:
            continue
        full = float(u.sum())
        if full <= 0:
            raise AssertionError(f"Invalid utility: {path}")
        recent_prefix = np.column_stack((np.zeros(HEADS), np.cumsum(u[:, start:][:, ::-1], axis=1)))
        history_prefix = top_history_prefix(score[:, :start], u[:, :start])
        two_dim = (recent_prefix[:, RECENTS, None] + history_prefix[:, None, HISTORY]) / full
        ranked_u = u[ROWS, ranking]
        ranked_prefix = np.column_stack((np.zeros(HEADS), np.cumsum(ranked_u, axis=1)))
        recent32 = u[:, -32:].sum(axis=1)
        old_grid = (recent32[:, None] + ranked_prefix[:, STEP * BUDGET_UNITS - 32]) / full
        old_fixed = float((recent32 + ranked_prefix[np.arange(HEADS), deployed - 32]).sum() / full)
        uniform = float((recent32 + ranked_prefix[:, 1000 - 32]).sum() / full)
        by_video[path.stem.split("_")[0]].append((two_dim, old_grid, old_fixed, uniform))
    profiles = {}
    for video, observations in by_video.items():
        profiles[video] = {
            "two_dim": np.mean([x[0] for x in observations], axis=0),
            "old_grid": np.mean([x[1] for x in observations], axis=0),
            "old_fixed": float(np.mean([x[2] for x in observations])),
            "uniform": float(np.mean([x[3] for x in observations])),
            "observations": len(observations),
        }
    return profiles, deployed


def reduce_surface(surface):
    """For every head and total budget, find the best recent/history split."""
    best_value = np.full((HEADS, len(BUDGET_UNITS)), -np.inf)
    best_recent = np.full((HEADS, len(BUDGET_UNITS)), -1, dtype=np.int16)
    for j, units in enumerate(BUDGET_UNITS):
        for ri, r in enumerate(RECENTS):
            if ri < MIN_RECENT_UNITS:
                continue
            hist = int(units * STEP - r)
            if hist < 0 or hist > HISTORY[-1] or hist % STEP:
                continue
            qi = hist // STEP
            value = surface[:, ri, qi]
            better = value > best_value[:, j]
            best_value[better, j] = value[better]
            best_recent[better, j] = ri
    if np.any(best_recent < 0):
        raise AssertionError("Unassigned head/budget option")
    return best_value, best_recent


def exact_dp(value):
    """Exact multiple-choice knapsack, with one budget choice per head."""
    dp = np.full(TARGET_UNITS + 1, -np.inf)
    dp[0] = 0
    back = np.full((HEADS, TARGET_UNITS + 1), -1, dtype=np.int8)
    for head in range(HEADS):
        next_dp = np.full_like(dp, -np.inf)
        for j, units in enumerate(BUDGET_UNITS):
            units = int(units)
            candidate = dp[:-units] + value[head, j]
            better = candidate > next_dp[units:]
            next_dp[units:][better] = candidate[better]
            back[head, units:][better] = j
        dp = next_dp
    if not np.isfinite(dp[TARGET_UNITS]):
        raise AssertionError("No exact global allocation")
    picked = np.empty(HEADS, dtype=np.int16)
    state = TARGET_UNITS
    for head in range(HEADS - 1, -1, -1):
        j = int(back[head, state])
        if j < 0:
            raise AssertionError("Broken DP traceback")
        picked[head] = j
        state -= int(BUDGET_UNITS[j])
    if state != 0:
        raise AssertionError("Wrong total after traceback")
    return picked


def greedy_gradient(surface):
    """Literal local largest-next-32-gradient baseline from minimum budgets."""
    _, best_recent = reduce_surface(surface)
    r = best_recent[:, 0].astype(int)
    q = MIN_UNITS - r
    remaining = TARGET_UNITS - HEADS * MIN_UNITS
    for _ in range(remaining):
        dr = np.full(HEADS, -np.inf)
        dh = np.full(HEADS, -np.inf)
        feasible = r + q < MAX_UNITS
        can_r = feasible & (r + 1 < len(RECENTS))
        can_h = feasible & (q + 1 < len(HISTORY))
        h = np.arange(HEADS)
        dr[can_r] = surface[h[can_r], r[can_r]+1, q[can_r]] - surface[h[can_r], r[can_r], q[can_r]]
        dh[can_h] = surface[h[can_h], r[can_h], q[can_h]+1] - surface[h[can_h], r[can_h], q[can_h]]
        best_r = int(np.argmax(dr))
        best_h = int(np.argmax(dh))
        if dr[best_r] >= dh[best_h]:
            r[best_r] += 1
        else:
            q[best_h] += 1
    if int((r+q).sum() * STEP) != TOTAL:
        raise AssertionError("Greedy exceeded or missed budget")
    return r, q


def allocate(surface):
    option, recent_by_total = reduce_surface(surface)
    picked = exact_dp(option)
    ri = recent_by_total[np.arange(HEADS), picked]
    qi = BUDGET_UNITS[picked] - ri
    if int((ri+qi).sum() * STEP) != TOTAL:
        raise AssertionError("2D allocation misses exact budget")
    return ri, qi


def selected_value(profile, r, q):
    return float(profile[np.arange(HEADS), r, q].sum())


def paired(rows, a, b):
    d = np.array([row[a] - row[b] for row in rows])
    rng = np.random.default_rng(20260923)
    boot = rng.choice(d, size=(10000, len(d)), replace=True).mean(axis=1)
    return {"mean_pp": float(100 * d.mean()),
            "bootstrap95_pp": (100 * np.quantile(boot, [.025, .975])).tolist(),
            "positive_videos": int(np.sum(d > 1e-12)),
            "negative_videos": int(np.sum(d < -1e-12))}


def plot_allocation(per_head):
    heads = np.arange(HEADS)
    recent = np.array([row["recent_r"] for row in per_head])
    history = np.array([row["history_q"] for row in per_head])
    old = np.array([row["old_single_curve_b"] for row in per_head])
    fig, ax = plt.subplots(figsize=(15.5, 5.5), layout="constrained")
    ax.bar(heads, history, width=.85, color="#355C7D", label="New: history q")
    ax.bar(heads, recent, bottom=history, width=.85, color="#E5A54B", label="New: recent r")
    ax.plot(heads, old, color="#C94343", linewidth=1.15, label="Old: unified head budget b")
    ax.axhline(1000, color="#454545", linewidth=.8, linestyle="--", label="Uniform b = 1000")
    ax.set(xlim=(-1, HEADS), ylim=(0, 1550), xlabel="Physical KV head (layer × 4 + head)",
           ylabel="Visual KV slots", title="Exact 112,000-slot allocation from the two-dimensional proxy")
    ax.set_xticks(np.arange(0, HEADS, 8))
    ax.legend(ncol=4, loc="upper center", frameon=False, fontsize=9)
    fig.savefig(OUT / "per_head_allocation.png", dpi=250)
    fig.savefig(OUT / "per_head_allocation.pdf")
    plt.close(fig)


def main():
    profiles, deployed = load_profiles()
    videos = sorted(profiles)
    train = np.mean([profiles[v]["two_dim"] for v in videos], axis=0)
    r, q = allocate(train)
    gr, gq = greedy_gradient(train)
    old_train = np.mean([profiles[v]["old_grid"] for v in videos], axis=0)
    old_choice = STEP * BUDGET_UNITS[exact_dp(old_train)]
    per_head = []
    for h in range(HEADS):
        per_head.append({"head_id": h, "layer": h // 4, "kv_head": h % 4,
                         "recent_r": int(STEP*r[h]), "history_q": int(STEP*q[h]),
                         "total_b": int(STEP*(r[h]+q[h])),
                         "greedy_r": int(STEP*gr[h]), "greedy_q": int(STEP*gq[h]),
                         "old_single_curve_b": int(old_choice[h]),
                         "old_deployed_b": int(deployed[h])})
    rows = []
    for held in videos:
        two_train = np.mean([profiles[v]["two_dim"] for v in videos if v != held], axis=0)
        old_fold_train = np.mean([profiles[v]["old_grid"] for v in videos if v != held], axis=0)
        fold_r, fold_q = allocate(two_train)
        fold_gr, fold_gq = greedy_gradient(two_train)
        fold_old = exact_dp(old_fold_train)
        test = profiles[held]
        rows.append({"video": held, "observations": test["observations"],
                     "two_dim_exact": selected_value(test["two_dim"], fold_r, fold_q),
                     "two_dim_greedy": selected_value(test["two_dim"], fold_gr, fold_gq),
                     "old_single_curve": float(test["old_grid"][np.arange(HEADS), fold_old].sum()),
                     "old_deployed_table": test["old_fixed"], "uniform_hermes": test["uniform"]})
    names = ("two_dim_exact", "two_dim_greedy", "old_single_curve", "old_deployed_table", "uniform_hermes")
    report = {
        "protocol": {
            "videos": len(videos), "observations": sum(profiles[v]["observations"] for v in videos),
            "total_visual_kv_slots": TOTAL, "step": STEP,
            "head_budget_bounds": [int(MIN_UNITS*STEP), int(MAX_UNITS*STEP)],
            "min_recent_tokens_per_head": int(MIN_RECENT_UNITS*STEP),
            "recent_candidate_set": "last two Qwen visual time units, 512 tokens",
            "two_dim": "newest r tokens, top-q HERMES scored strictly older tokens",
            "old_single_curve": "Recent-32 plus top-ranked other tokens; same LOOV cohort, grid and bounds",
            "old_deployed_table": "fixed pre-existing 750-1500 budget table; trained on overlapping 50-video corpus, so not a clean held-out comparator",
            "utility": "full-cache answer-readout proxy, not sequential cache survival or compressed QA accuracy",
        },
        "full_cohort_allocation": {
            "recent_total": int(STEP*r.sum()), "history_total": int(STEP*q.sum()),
            "head_total_quantiles": np.quantile(STEP*(r+q), [0,.1,.25,.5,.75,.9,1]).tolist(),
            "recent_quantiles": np.quantile(STEP*r, [0,.1,.25,.5,.75,.9,1]).tolist(),
            "history_quantiles": np.quantile(STEP*q, [0,.1,.25,.5,.75,.9,1]).tolist(),
            "recent_counts": {str(x): int(np.sum(STEP*r == x)) for x in np.unique(STEP*r)},
            "old_single_curve_quantiles": np.quantile(old_choice, [0,.1,.25,.5,.75,.9,1]).tolist(),
            "old_deployed_quantiles": np.quantile(deployed, [0,.1,.25,.5,.75,.9,1]).tolist(),
            "mean_abs_total_delta_vs_old_single_curve": float(np.mean(np.abs(STEP*(r+q)-old_choice))),
            "heads_same_total_as_old_single_curve": int(np.sum(STEP*(r+q)==old_choice)),
            "mean_abs_total_delta_vs_old_deployed": float(np.mean(np.abs(STEP*(r+q)-deployed))),
            "heads_new_b_larger_than_old_deployed": int(np.sum(STEP*(r+q)>deployed)),
            "heads_new_b_smaller_than_old_deployed": int(np.sum(STEP*(r+q)<deployed)),
            "train_exact_proxy": selected_value(train,r,q),
            "train_greedy_proxy": selected_value(train,gr,gq),
        },
        "loov_mean_video_retention": {n: float(np.mean([row[n] for row in rows])) for n in names},
        "loov_paired": {
            "two_dim_exact_minus_old_single_curve": paired(rows,"two_dim_exact","old_single_curve"),
            "two_dim_exact_minus_uniform_hermes": paired(rows,"two_dim_exact","uniform_hermes"),
            "two_dim_exact_minus_old_deployed_table": paired(rows,"two_dim_exact","old_deployed_table"),
            "two_dim_exact_minus_two_dim_greedy": paired(rows,"two_dim_exact","two_dim_greedy"),
        },
        "per_head": per_head, "per_video": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    plot_allocation(per_head)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    with (OUT / "per_head_allocation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_head[0]))
        writer.writeheader(); writer.writerows(per_head)
    with (OUT / "per_video.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({k:v for k,v in report.items() if k not in ("per_head","per_video")}, indent=2))


if __name__ == "__main__":
    main()
