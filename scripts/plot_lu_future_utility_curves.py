#!/usr/bin/env python3
"""Plot video-balanced head-wise future projected-utility retention curves."""

import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("results/myobservations/lu_kv_offline_50v_20260913")
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
RECENT = 32
COMPRESSION = np.asarray(
    [0, .1, .2, .3, .4, .5, .6, .7, .8, .85, .9, .925, .95, .975, .99],
    dtype=np.float64,
)


def observation_curves(npz_path):
    with np.load(npz_path) as data:
        utility = data["utility"].astype(np.float64)
        ranking = data["ranking"].astype(np.int64)
    heads, tokens = utility.shape
    recent = min(RECENT, tokens)
    recent_mass = utility[:, tokens - recent:].sum(axis=1)
    history_utility = utility[:, :tokens - recent]
    ordered = np.take_along_axis(history_utility, ranking[:, :history_utility.shape[1]], axis=1)
    prefix = np.concatenate([np.zeros((heads, 1)), np.cumsum(ordered, axis=1)], axis=1)
    total = utility.sum(axis=1)
    result = np.full((heads, len(COMPRESSION)), np.nan, dtype=np.float64)
    valid = total > 1e-12
    for i, compression in enumerate(COMPRESSION):
        budget = max(recent, min(tokens, int(round((1.0 - compression) * tokens))))
        history_budget = min(tokens - recent, max(0, budget - recent))
        retained = recent_mass + prefix[:, history_budget]
        result[valid, i] = retained[valid] / total[valid]
    return result


# Average observations within each video first, then give every video equal weight.
by_video = defaultdict(list)
for path in sorted(glob.glob(str(ROOT / "v*_q*.npz"))):
    stem = Path(path).stem
    meta = json.loads((ROOT / f"{stem}.json").read_text())
    by_video[str(meta["video_id"])].append(observation_curves(path))

video_curves = []
for video_id in sorted(by_video, key=lambda x: int(x)):
    stack = np.stack(by_video[video_id])
    video_curves.append(np.nanmean(stack, axis=0))
curves = np.nanmean(np.stack(video_curves), axis=0)
counts = np.sum(np.isfinite(np.stack(video_curves)), axis=0)
auc = np.trapezoid(np.nan_to_num(curves, nan=0.0), COMPRESSION, axis=1)

csv_path = OUT / "head_future_utility_curves.csv"
with csv_path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["layer", "kv_head", "auc"] +
                    [f"coverage_at_compression_{x:g}" for x in COMPRESSION])
    for head in range(112):
        writer.writerow([head // 4, head % 4, auc[head], *curves[head]])

top = np.argsort(auc)[-6:][::-1]
bottom = np.argsort(auc)[:6]
colors = plt.cm.viridis(np.linspace(0.05, 0.95, 28))
fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.4), gridspec_kw={"width_ratios": [1.15, 1]})

for head in range(112):
    axes[0].plot(COMPRESSION * 100, curves[head] * 100,
                 color=colors[head // 4], alpha=0.30, linewidth=0.85)
mean_curve = np.nanmean(curves, axis=0)
axes[0].plot(COMPRESSION * 100, mean_curve * 100, color="black", linewidth=2.6,
             label="Mean over 112 heads")
axes[0].set_title("All physical KV heads", weight="bold")
axes[0].legend(frameon=False, loc="lower left")

for head in top:
    axes[1].plot(COMPRESSION * 100, curves[head] * 100, linewidth=2.0,
                 label=f"High AUC: L{head//4}-H{head%4}")
for head in bottom:
    axes[1].plot(COMPRESSION * 100, curves[head] * 100, linewidth=1.7, linestyle="--",
                 label=f"Low AUC: L{head//4}-H{head%4}")
axes[1].set_title("Highest- and lowest-AUC heads", weight="bold")
axes[1].legend(fontsize=7.5, ncol=2, frameon=False, loc="lower left")

for ax in axes:
    ax.set_xlabel("Compression ratio (%)")
    ax.set_ylabel("Retained projected utility (%)")
    ax.set_xlim(0, 99)
    ax.set_ylim(0, 101)
    ax.grid(True, alpha=0.22)

fig.suptitle(
    "Head-wise Future-Utility Curves on StreamingBench Calibration Videos\n"
    "HERMES selector; question-conditioned full-cache projected utility",
    fontsize=14, weight="bold",
)
fig.text(
    0.5, 0.01,
    f"Video-balanced over {len(video_curves)} videos / "
    f"{sum(len(v) for v in by_video.values())} observations; recent {RECENT} tokens always retained. "
    "This is a surrogate, not causal evidence recall.",
    ha="center", fontsize=9,
)
fig.tight_layout(rect=[0, 0.055, 1, 0.90])
for suffix in ("png", "pdf"):
    fig.savefig(OUT / f"head_future_utility_curves.{suffix}", dpi=300, bbox_inches="tight")

summary = {
    "videos": len(video_curves),
    "observations": sum(len(v) for v in by_video.values()),
    "top_auc_heads": [{"layer": int(h // 4), "kv_head": int(h % 4), "auc": float(auc[h])}
                      for h in top],
    "bottom_auc_heads": [{"layer": int(h // 4), "kv_head": int(h % 4), "auc": float(auc[h])}
                         for h in bottom],
    "mean_coverage": {str(x): float(y) for x, y in zip(COMPRESSION, mean_curve)},
}
(OUT / "head_future_utility_curves_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
