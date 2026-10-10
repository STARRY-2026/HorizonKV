#!/usr/bin/env python3
"""Literal one-token marginal-gain allocation over head × recent/history.

The fixed recent candidates are the last two Qwen visual time units. This
offline allocation uses full-cache projected answer utility and does not replay
compressed model states. Two versions are reported: pure gradient with only
Recent-32 and a 1500-token head cap, and fair-bounds gradient with each head
initialized at its training-optimal 750-token split before global greediness.
"""

import csv
import heapq
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_two_dimensional_utility_surface import HEADS, ROOT, top_history_prefix


OUT = Path("results/myobservations/token_gradient_allocation_20260923")
TOTAL = 112000
RECENT_MIN = 32
RECENT_MAX = 512
HISTORY_MAX = 1468
HEAD_MAX = 1500
HEAD_MIN = 750
ROWS = np.arange(HEADS)[:, None]
OLD_TABLE = ROOT / "lu_raw_blend05_clip075_150_budget.csv"


def load_profiles():
    deployed = np.genfromtxt(OLD_TABLE, delimiter=",", names=True)["budget"].astype(int)
    by_video = defaultdict(list)
    for path in sorted(ROOT.glob("v*_q*.npz")):
        with np.load(path) as z:
            u = z["utility"].astype(np.float64)
            score = z["hermes_score"].astype(np.float64)
            ranking = z["ranking"].astype(np.int64)
            times = z["source_times"].astype(np.float64)
        unique = np.unique(times)
        if len(unique) < 2:
            continue
        start = int(np.searchsorted(times, unique[-2], side="left"))
        if len(times) - start < RECENT_MAX or start < 1472:
            continue
        full = float(u.sum())
        recent = np.column_stack((np.zeros(HEADS), np.cumsum(u[:, start:][:, ::-1], axis=1)))
        hist = top_history_prefix(score[:, :start], u[:, :start])
        recent /= full
        hist /= full
        ranking_prefix = np.column_stack((np.zeros(HEADS), np.cumsum(u[ROWS, ranking], axis=1)))
        recent32 = u[:, -32:].sum(axis=1)
        baseline = float((recent32 + ranking_prefix[:, 968]).sum() / full)
        old_fixed = float((recent32 + ranking_prefix[np.arange(HEADS), deployed-32]).sum() / full)
        by_video[path.stem.split("_")[0]].append((recent[:, :RECENT_MAX+1], hist[:, :HISTORY_MAX+1], baseline, old_fixed))
    return {vid: {
        "recent": np.mean([o[0] for o in observations], axis=0),
        "history": np.mean([o[1] for o in observations], axis=0),
        "uniform": float(np.mean([o[2] for o in observations])),
        "old_fixed": float(np.mean([o[3] for o in observations])),
        "observations": len(observations),
    } for vid, observations in by_video.items()}, deployed


def fit_split_at_floor(recent, history):
    choices = np.arange(RECENT_MIN, RECENT_MAX+1)
    hchoices = HEAD_MIN - choices
    values = recent[:, choices] + history[:, hchoices]
    r = choices[np.argmax(values, axis=1)].astype(np.int16)
    q = (HEAD_MIN - r).astype(np.int16)
    return r, q


def gradient_greedy(recent, history, head_min):
    if head_min:
        r, q = fit_split_at_floor(recent, history)
    else:
        r = np.full(HEADS, RECENT_MIN, dtype=np.int16)
        q = np.zeros(HEADS, dtype=np.int16)
    versions = np.zeros(HEADS, dtype=np.int32)
    heap = []

    def push(head, bank):
        if int(r[head] + q[head]) >= HEAD_MAX:
            return
        if bank == 0:
            if r[head] >= RECENT_MAX:
                return
            gain = recent[head, r[head]+1] - recent[head, r[head]]
        else:
            if q[head] >= HISTORY_MAX:
                return
            gain = history[head, q[head]+1] - history[head, q[head]]
        heapq.heappush(heap, (-float(gain), head, bank, int(versions[head])))

    for head in range(HEADS):
        push(head, 0)
        push(head, 1)
    additions = TOTAL - int(r.sum() + q.sum())
    for _ in range(additions):
        while True:
            if not heap:
                raise AssertionError("No feasible marginal increment remains")
            _, head, bank, version = heapq.heappop(heap)
            if version == versions[head]:
                break
        if bank == 0:
            r[head] += 1
        else:
            q[head] += 1
        versions[head] += 1
        push(head, 0)
        push(head, 1)
    if int((r+q).sum()) != TOTAL or int((r+q).max()) > HEAD_MAX:
        raise AssertionError("Gradient allocation violates budget")
    if head_min and int((r+q).min()) < HEAD_MIN:
        raise AssertionError("Gradient allocation violates per-head floor")
    return r, q


def value(profile, r, q):
    return float((profile["recent"][np.arange(HEADS), r]
                  + profile["history"][np.arange(HEADS), q]).sum())


def paired(rows, a, b):
    d = np.array([x[a]-x[b] for x in rows])
    rng = np.random.default_rng(20260923)
    boot = rng.choice(d, size=(10000, len(d)), replace=True).mean(axis=1)
    return {"mean_pp": float(100*d.mean()),
            "bootstrap95_pp": (100*np.quantile(boot, [.025, .975])).tolist(),
            "positive_videos": int(sum(d > 1e-12)),
            "negative_videos": int(sum(d < -1e-12))}


def counts(values):
    unique, amount = np.unique(values, return_counts=True)
    return {str(int(x)): int(n) for x,n in zip(unique, amount)}


def plot_allocations(per_head):
    heads = np.arange(HEADS)
    old = np.array([x["old_deployed_total"] for x in per_head])
    fig, axes = plt.subplots(2, 1, figsize=(15.2, 8), sharex=True, layout="constrained")
    for ax, prefix, title in zip(axes, ("pure", "bounded"),
                                 ("Pure one-token gradient (no head floor)",
                                  "One-token gradient after 750-token/head initialization")):
        history = np.array([x[f"{prefix}_history_q"] for x in per_head])
        recent = np.array([x[f"{prefix}_recent_r"] for x in per_head])
        ax.bar(heads, history, width=.85, color="#355C7D", label="History q")
        ax.bar(heads, recent, bottom=history, width=.85, color="#E5A54B", label="Recent r")
        ax.plot(heads, old, color="#C94343", linewidth=1.05, label="Old deployed head budget")
        ax.axhline(1000, color="#555555", linestyle="--", linewidth=.75)
        ax.set(ylabel="Visual KV slots", title=title, ylim=(0,1550), xlim=(-1,HEADS))
    axes[0].legend(ncol=3, loc="upper center", frameon=False)
    axes[-1].set_xlabel("Physical KV head (layer × 4 + head)")
    axes[-1].set_xticks(np.arange(0, HEADS, 8))
    fig.suptitle("Global budget = 112,000; each increment selects the largest current marginal gain")
    fig.savefig(OUT / "one_token_gradient_allocations.png", dpi=230)
    fig.savefig(OUT / "one_token_gradient_allocations.pdf")
    plt.close(fig)


def main():
    profiles, deployed = load_profiles()
    videos = sorted(profiles)
    mean_recent = np.mean([profiles[v]["recent"] for v in videos], axis=0)
    mean_history = np.mean([profiles[v]["history"] for v in videos], axis=0)
    pure_r, pure_q = gradient_greedy(mean_recent, mean_history, head_min=False)
    fair_r, fair_q = gradient_greedy(mean_recent, mean_history, head_min=True)
    per_head = [{"head_id": h, "layer": h//4, "kv_head": h%4,
                 "pure_recent_r": int(pure_r[h]), "pure_history_q": int(pure_q[h]),
                 "pure_total": int(pure_r[h]+pure_q[h]),
                 "bounded_recent_r": int(fair_r[h]), "bounded_history_q": int(fair_q[h]),
                 "bounded_total": int(fair_r[h]+fair_q[h]),
                 "old_deployed_total": int(deployed[h])} for h in range(HEADS)]
    rows = []
    for held in videos:
        train_r = np.mean([profiles[v]["recent"] for v in videos if v != held], axis=0)
        train_h = np.mean([profiles[v]["history"] for v in videos if v != held], axis=0)
        r, q = gradient_greedy(train_r, train_h, head_min=True)
        pure_fold_r, pure_fold_q = gradient_greedy(train_r, train_h, head_min=False)
        test = profiles[held]
        rows.append({"video": held, "observations": test["observations"],
                     "bounded_gradient": value(test,r,q),
                     "pure_gradient": value(test,pure_fold_r,pure_fold_q),
                     "uniform_hermes": test["uniform"],
                     "old_deployed_table": test["old_fixed"]})
    report = {
        "protocol": {"videos": len(videos),
                     "observations": sum(profiles[v]["observations"] for v in videos),
                     "total_visual_slots": TOTAL, "gradient_step_tokens": 1,
                     "recent_candidate_window": "last two Qwen visual units",
                     "pure_bounds": "recent>=32, head_total<=1500, no head-total floor",
                     "bounded_bounds": "recent>=32, 750<=head_total<=1500; initialize at each head optimal 750-token split",
                     "caveat": "full-cache utility proxy; no compressed-cache question answering or online survivor replay"},
        "full_cohort_pure": {"recent_total": int(pure_r.sum()), "history_total": int(pure_q.sum()),
                             "total_quantiles": np.quantile(pure_r+pure_q,[0,.1,.25,.5,.75,.9,1]).tolist(),
                             "recent_quantiles": np.quantile(pure_r,[0,.1,.25,.5,.75,.9,1]).tolist(),
                             "head_total_at_cap": int(np.sum(pure_r+pure_q==HEAD_MAX)),
                             "train_proxy": value({"recent":mean_recent,"history":mean_history},pure_r,pure_q)},
        "full_cohort_bounded": {"recent_total": int(fair_r.sum()), "history_total": int(fair_q.sum()),
                                "total_quantiles": np.quantile(fair_r+fair_q,[0,.1,.25,.5,.75,.9,1]).tolist(),
                                "recent_quantiles": np.quantile(fair_r,[0,.1,.25,.5,.75,.9,1]).tolist(),
                                "head_total_at_floor": int(np.sum(fair_r+fair_q==HEAD_MIN)),
                                "head_total_at_cap": int(np.sum(fair_r+fair_q==HEAD_MAX)),
                                "train_proxy": value({"recent":mean_recent,"history":mean_history},fair_r,fair_q),
                                "mean_abs_total_delta_vs_old_deployed": float(np.mean(np.abs(fair_r+fair_q-deployed)))},
        "loov_mean_video_retention": {name: float(np.mean([x[name] for x in rows]))
                                       for name in ("pure_gradient","bounded_gradient","uniform_hermes","old_deployed_table")},
        "loov_paired": {"bounded_minus_uniform": paired(rows,"bounded_gradient","uniform_hermes"),
                        "bounded_minus_old_deployed": paired(rows,"bounded_gradient","old_deployed_table"),
                        "pure_minus_bounded": paired(rows,"pure_gradient","bounded_gradient"),
                        "pure_minus_old_deployed": paired(rows,"pure_gradient","old_deployed_table")},
        "per_head": per_head, "per_video": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    plot_allocations(per_head)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
    for name, data in (("per_head.csv",per_head),("per_video.csv",rows)):
        with (OUT/name).open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    print(json.dumps({k:v for k,v in report.items() if k not in ("per_head","per_video")},indent=2))


if __name__ == "__main__":
    main()
