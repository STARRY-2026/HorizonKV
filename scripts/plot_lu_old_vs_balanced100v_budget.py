#!/usr/bin/env python3
"""Plot old/new deployable LU budgets and their difference."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OLD = Path("results/myobservations/lu_kv_offline_50v_20260913/lu_raw_blend05_clip075_150_budget.csv")
NEW = Path("results/myobservations/lu_kv_balanced100v_20260915/traces/lu_raw_balanced100v_blend05_clip075_150_budget.csv")
OUT = Path("results/myobservations/lu_kv_balanced100v_20260915/figures")
OUT.mkdir(parents=True, exist_ok=True)


def load(path):
    result = np.zeros((28, 4), dtype=np.int64)
    with path.open() as handle:
        for row in csv.DictReader(handle):
            result[int(row["layer"]), int(row["kv_head"])] = int(row["budget"])
    return result


old, new = load(OLD), load(NEW)
delta = new - old
fig, axes = plt.subplots(1, 3, figsize=(12.8, 8.6), constrained_layout=True)
for ax, values, title in zip(
    axes[:2], (old, new), ("Old: 50-video calibration", "New: balanced 100-video calibration")
):
    image = ax.imshow(values, aspect="auto", cmap="viridis", vmin=750, vmax=1500)
    ax.set_title(title, weight="bold")
    ax.set_xlabel("Physical KV head")
    ax.set_ylabel("Layer")
    ax.set_xticks(range(4))
    fig.colorbar(image, ax=ax, shrink=0.78, label="Token budget")

limit = max(abs(delta.min()), abs(delta.max()))
image = axes[2].imshow(delta, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
axes[2].set_title("New − old", weight="bold")
axes[2].set_xlabel("Physical KV head")
axes[2].set_ylabel("Layer")
axes[2].set_xticks(range(4))
fig.colorbar(image, ax=axes[2], shrink=0.78, label="Budget change")
fig.suptitle("Deployable LU Head-Budget Change at K=1000\nBounds 750–1500; exact total 112,000", weight="bold")

for suffix in ("png", "pdf"):
    fig.savefig(OUT / f"lu_old_vs_balanced100v_budget_heatmap.{suffix}", dpi=300, bbox_inches="tight")

with (OUT / "lu_old_vs_balanced100v_budget_delta.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["layer", "kv_head", "old_budget", "new_budget", "delta"])
    for layer in range(28):
        for head in range(4):
            writer.writerow([layer, head, old[layer, head], new[layer, head], delta[layer, head]])
print({"min_delta": int(delta.min()), "max_delta": int(delta.max()),
       "mean_abs_delta": float(np.abs(delta).mean()),
       "pearson": float(np.corrcoef(old.ravel(), new.ravel())[0, 1])})
