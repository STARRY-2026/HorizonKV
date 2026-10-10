#!/usr/bin/env python3
"""Plot token-level and temporal similarity of Local/Global selections."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("results/myobservations/local_global_selected_time_50v_20260922")


def main():
    with (ROOT / "head_selected_time.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    report = json.loads((ROOT / "summary.json").read_text())
    high = {(layer, head) for layer, head in report["high_auc_heads"]}
    is_high = np.array([(int(row["layer"]), int(row["kv_head"])) in high for row in rows])
    layer = np.array([int(row["layer"]) for row in rows])

    def col(name):
        return np.array([float(row[name]) for row in rows])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, k in zip(axes[:2], (250, 1000)):
        x = col(f"k{k}_local_current25")
        y = col(f"k{k}_global_current25")
        ax.plot([0, 1], [0, 1], color="#888888", ls="--", lw=1)
        ax.scatter(x[~is_high], y[~is_high], c=layer[~is_high], cmap="viridis",
                   vmin=0, vmax=27, s=25, alpha=.75)
        ax.scatter(x[is_high], y[is_high], facecolors="none", edgecolors="#D55E00",
                   linewidths=1.6, s=95, label="Both top-10 AUC")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Local: share selected from last 25%",
               ylabel="Global: share selected from last 25%", title=f"K={k}: selected time")
        ax.grid(alpha=.2)
        ax.legend(loc="lower right", fontsize=8)
    x = col("consensus_topk_auc")
    y250 = col("k250_shared_fraction")
    y1000 = col("k1000_shared_fraction")
    axes[2].scatter(x, y250, c="#0072B2", s=29, alpha=.7, label="K=250")
    axes[2].scatter(x, y1000, c="#009E73", s=29, alpha=.7, label="K=1000")
    axes[2].set(xlabel="Local/Global consensus top-k AUC",
                ylabel="Same historical tokens selected / budget",
                title="Token-set overlap")
    axes[2].grid(alpha=.2)
    axes[2].legend(fontsize=8)
    fig.suptitle(f"Local vs Global retained-token timing: {report['videos']} videos, "
                 f"{report['observations']} observations", fontsize=13)
    fig.text(.5, .01, "Newest 32 visual tokens excluded from all comparisons; "
             "time normalized within each observed video prefix.",
             ha="center", color="#555555", fontsize=9)
    fig.tight_layout(rect=[0, .04, 1, .93])
    png = ROOT / "local_global_selected_time.png"
    pdf = ROOT / "local_global_selected_time.pdf"
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
