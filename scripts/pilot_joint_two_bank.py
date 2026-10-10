#!/usr/bin/env python3
"""Leave-one-video-out proxy test of joint head and recent/history capacity.

At each Head, choose b in {750,1000,1250,1500} and protected recent r in
{32,64,128,256,512}, with exactly 112000 global KV slots. The recent pool is
the last eight seconds (512 visual tokens in these eligible observations).
"""

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path("results/myobservations/lu_kv_offline_50v_20260913")
OUT = Path("results/myobservations/two_bank_joint_pilot_20260923")
BUDGETS = np.array([750, 1000, 1250, 1500], dtype=np.int64)
RECENTS = np.array([32, 64, 128, 256, 512], dtype=np.int64)
TIME_WINDOW = 8.0
HEADS = 112
GLOBAL_SLOTS = 112000


def top_prefix_utility(score, utility, length):
    """Nested score-ranked prefix utility, ignoring order of exact score ties."""
    top = np.argpartition(-score, length - 1, axis=1)[:, :length]
    top_scores = np.take_along_axis(score, top, axis=1)
    top = np.take_along_axis(top, np.argsort(-top_scores, axis=1, kind="stable"), axis=1)
    rows = np.arange(HEADS)[:, None]
    return np.cumsum(utility[rows, top], axis=1)


def load(time_window=TIME_WINDOW, recents=RECENTS):
    result = []
    max_q = int(BUDGETS.max() - recents.min())
    for path in sorted(ROOT.glob("v*_q*.npz")):
        with np.load(path) as z:
            utility = z["utility"].astype(np.float64)
            score = z["hermes_score"].astype(np.float64)
            ranking = z["ranking"].astype(np.int64)
            times = z["source_times"].astype(np.float64)
        if not np.all(np.diff(times) >= 0):
            raise ValueError(f"Unordered source times: {path}")
        n = utility.shape[1]
        recent_start = int(np.searchsorted(times, times[-1] - time_window, side="right"))
        recent_count = n - recent_start
        if n < BUDGETS.max() or recent_count < recents.max() or recent_start < max_q:
            continue
        raw = np.empty((2, HEADS, len(BUDGETS), len(recents)), dtype=np.float64)
        hard_prefix = top_prefix_utility(score[:, :recent_start], utility[:, :recent_start], max_q)
        for ri, r in enumerate(recents):
            r = int(r)
            recent_value = utility[:, n-r:].sum(axis=1)
            if r == 32:
                # The exact original HERMES ranking is used here, so the
                # r=32 soft policy is identical to the deployed selector.
                rows = np.arange(HEADS)[:, None]
                soft_prefix = np.cumsum(utility[rows, ranking[:, :max_q]], axis=1)
            else:
                soft_prefix = top_prefix_utility(score[:, :n-r], utility[:, :n-r], max_q)
            for bi, b in enumerate(BUDGETS):
                q = int(b) - r
                raw[0, :, bi, ri] = recent_value + hard_prefix[:, q-1]
                raw[1, :, bi, ri] = recent_value + soft_prefix[:, q-1]
        if np.any(raw < -1e-8) or np.any(raw > utility.sum(axis=1)[None,:,None,None] + 1e-7):
            raise AssertionError(f"Invalid retained utility: {path}")
        full = float(utility.sum())
        result.append({"video": path.stem.split("_")[0], "observation": path.stem,
                       "full": full, "raw": raw, "recent_candidates": recent_count})
    return result


def solve(profile, optimize_recent):
    """Exact multiple-choice knapsack on the coarse budget grid."""
    value = profile.max(axis=2) if optimize_recent else profile[:, :, 0]
    recent_choice = profile.argmax(axis=2) if optimize_recent else np.zeros((HEADS, len(BUDGETS)), dtype=int)
    unit = (BUDGETS - int(BUDGETS.min())) // 250
    target = (GLOBAL_SLOTS - HEADS * int(BUDGETS.min())) // 250
    if target * 250 + HEADS * int(BUDGETS.min()) != GLOBAL_SLOTS:
        raise ValueError("Infeasible grid for exact global budget")
    dp = np.full(target + 1, -np.inf)
    dp[0] = 0.0
    back = np.full((HEADS, target + 1), -1, dtype=np.int8)
    for h in range(HEADS):
        nxt = np.full_like(dp, -np.inf)
        for bi, shift in enumerate(unit):
            shift = int(shift)
            candidates = dp[:target + 1 - shift] + value[h, bi]
            better = candidates > nxt[shift:]
            nxt[shift:][better] = candidates[better]
            back[h, shift:][better] = bi
        dp = nxt
    if not np.isfinite(dp[target]):
        raise AssertionError("No exact allocation")
    bidx = np.empty(HEADS, dtype=int)
    ridx = np.empty(HEADS, dtype=int)
    state = int(target)
    for h in range(HEADS - 1, -1, -1):
        bi = int(back[h, state])
        if bi < 0:
            raise AssertionError("Broken allocation traceback")
        bidx[h] = bi
        ridx[h] = int(recent_choice[h, bi])
        state -= int(unit[bi])
    if state != 0 or int(BUDGETS[bidx].sum()) != GLOBAL_SLOTS:
        raise AssertionError("Allocation does not satisfy exact budget")
    return bidx, ridx


def paired(rows, a, b, rng):
    diff = np.array([x[a] - x[b] for x in rows])
    boot = rng.choice(diff, size=(10000, len(diff)), replace=True).mean(axis=1)
    return {"mean_pp": float(diff.mean() * 100),
            "bootstrap95_pp": [float(x * 100) for x in np.quantile(boot, [.025, .975])],
            "positive_videos": int(np.sum(diff > 1e-12)),
            "negative_videos": int(np.sum(diff < -1e-12))}


def run(observations=None, time_window=TIME_WINDOW, recents=RECENTS):
    if observations is None:
        observations = load(time_window=time_window, recents=recents)
    videos = sorted({x["video"] for x in observations})
    if len(videos) < 3:
        raise ValueError("Insufficient videos")
    per_video = {vid: [x for x in observations if x["video"] == vid] for vid in videos}
    profiles = {vid: np.mean([x["raw"] / x["full"] for x in obs], axis=0)
                for vid, obs in per_video.items()}
    rows = []
    assignment_counts = {name: np.zeros((len(BUDGETS), len(recents)), dtype=int)
                         for name in ("hard_joint", "soft_joint")}
    head_ids = np.arange(HEADS)
    for heldout in videos:
        train = np.mean([profiles[vid] for vid in videos if vid != heldout], axis=0)
        schemes = {}
        for bank, prefix in ((0, "hard"), (1, "soft")):
            schemes[prefix + "_budget_only"] = (bank, *solve(train[bank], False))
            schemes[prefix + "_joint"] = (bank, *solve(train[bank], True))
        for name in ("hard_joint", "soft_joint"):
            _, bi, ri = schemes[name]
            np.add.at(assignment_counts[name], (bi, ri), 1)
        row = {"video": heldout, "observations": len(per_video[heldout])}
        for name in ("uniform_hermes", *schemes):
            ret = []
            for obs in per_video[heldout]:
                if name == "uniform_hermes":
                    amount = float(obs["raw"][1, :, 1, 0].sum())
                else:
                    bank, bi, ri = schemes[name]
                    amount = float(obs["raw"][bank, head_ids, bi, ri].sum())
                ret.append(amount / obs["full"])
            row[name] = float(np.mean(ret))
        rows.append(row)
    rng = np.random.default_rng(20260923)
    names = ("uniform_hermes", "hard_budget_only", "hard_joint", "soft_budget_only", "soft_joint")
    return {
        "protocol": {"bank_boundary_seconds": time_window, "budget_grid": BUDGETS.tolist(),
                     "recent_grid": recents.tolist(), "total_global_slots": GLOBAL_SLOTS,
                     "selection": "question-free HERMES score, full-cache answer utility",
                     "calibration": "leave-one-video-out, equal video weight",
                     "caveat": "proxy retention from static prefixes, not online cache survival or compressed QA"},
        "videos": len(videos), "observations": len(observations),
        "mean_video_retention": {n: float(np.mean([r[n] for r in rows])) for n in names},
        "paired": {"hard_joint_minus_hard_budget_only": paired(rows,"hard_joint","hard_budget_only",rng),
                   "hard_joint_minus_uniform": paired(rows,"hard_joint","uniform_hermes",rng),
                   "hard_joint_minus_soft_budget_only": paired(rows,"hard_joint","soft_budget_only",rng),
                   "soft_joint_minus_soft_budget_only": paired(rows,"soft_joint","soft_budget_only",rng),
                   "soft_joint_minus_uniform": paired(rows,"soft_joint","uniform_hermes",rng),
                   "soft_budget_only_minus_uniform": paired(rows,"soft_budget_only","uniform_hermes",rng)},
        "fold_assignment_counts": {k: v.tolist() for k,v in assignment_counts.items()},
        "per_video": rows,
    }


def main():
    report = run()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    with (OUT / "per_video.csv").open("w", newline="") as f:
        fields = list(report["per_video"][0])
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report["per_video"])
    print(json.dumps({k:v for k,v in report.items() if k != "per_video"}, indent=2))


if __name__ == "__main__":
    main()
