#!/usr/bin/env python3
"""Visualize Local/Global head-rank changes after binary top-k evaluation."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("results/myobservations/local_global_topk_50v_20260922")


def main():
    with (ROOT / "local_global_topk_head_ranks.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 112:
        raise ValueError(f"Expected 112 heads, got {len(rows)}")
    report = json.loads((ROOT / "local_global_topk_summary.json").read_text())
    layers = np.array([int(row["layer"]) for row in rows])

    def col(key):
        return np.array([float(row[key]) for row in rows])

    lu = col("uniform_local_utility_auc_rank")
    gu = col("uniform_global_utility_auc_rank")
    lt = col("uniform_local_overlap_auc_rank")
    gt = col("uniform_global_overlap_auc_rank")
    lk = col("uniform_local_overlap_k250_rank")
    gk = col("uniform_global_overlap_k250_rank")
    pairs = [
        (lu, gu, "Utility AUC: Local vs Global", "Local rank", "Global rank"),
        (lt, gt, "Top-k overlap AUC: Local vs Global", "Local rank", "Global rank"),
        (lk, gk, "K=250 top-k overlap: Local vs Global", "Local rank", "Global rank"),
        (lu, lt, "Local: utility AUC vs top-k AUC", "Utility AUC rank", "Top-k AUC rank"),
        (gu, gt, "Global: utility AUC vs top-k AUC", "Utility AUC rank", "Top-k AUC rank"),
        (lt, lk, "Local: top-k AUC vs K=250", "Top-k AUC rank", "K=250 overlap rank"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    for ax, (x, y, title, xlabel, ylabel) in zip(axes.flat, pairs):
        ax.plot([1, 112], [1, 112], color="#888888", ls="--", lw=1)
        sc = ax.scatter(x, y, c=layers, cmap="viridis", vmin=0, vmax=27,
                        s=26, alpha=.82, edgecolors="none")
        rho = np.corrcoef(x, y)[0, 1]
        ax.text(.04, .96, f"Spearman = {rho:.3f}", transform=ax.transAxes,
                va="top", fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 113)
        ax.set_ylim(113, 0)
        ax.grid(alpha=.15)
    fig.colorbar(sc, ax=axes, label="Layer", fraction=.022, pad=.025)
    fig.suptitle(
        f"Local / Global head ranks: {report['videos']} videos, "
        f"{report['observations']} observations", fontsize=14,
    )
    fig.subplots_adjust(top=.89, bottom=.07, left=.06, right=.9, wspace=.28, hspace=.3)
    png = ROOT / "local_global_topk_rank_changes.png"
    pdf = ROOT / "local_global_topk_rank_changes.pdf"
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
