#!/usr/bin/env python3
"""Overlay all physical KV-head n-v curves in raw and head-normalized scales."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np


ROOT = Path("results/myobservations/two_bank_grid_dp_20260923")


def main():
    with np.load(ROOT / "head_n_v_curves.npz") as data:
        n = data["capacity"]
        curves = data["values"]
        shares = data["full_head_share"]
    if curves.shape != (112, 1501) or shares.shape != (112,):
        raise ValueError(f"Unexpected curve shape: {curves.shape}, {shares.shape}")
    rows = list(csv.DictReader((ROOT / "per_head_allocation.csv").open()))
    if len(rows) != 112:
        raise ValueError("Expected 112 physical heads")
    budget = np.array([int(row["capacity_n"]) for row in rows])
    norm = Normalize(vmin=0, vmax=27)
    cmap = plt.get_cmap("turbo")

    fig, axes = plt.subplots(2, 1, figsize=(12.6, 9), sharex=True,
                             layout="constrained")
    for h in range(112):
        color = cmap(norm(h // 4))
        axes[0].plot(n, 100 * curves[h], color=color, alpha=.32, linewidth=.85)
        axes[1].plot(n, curves[h] / shares[h], color=color, alpha=.32, linewidth=.85)
    for ax in axes:
        ax.grid(alpha=.16)
        ax.set_xlim(0, 1500)
    axes[0].set_ylabel("Contribution to full-cache utility (pp)")
    axes[0].set_title("Original scale used by the global budget optimizer", loc="left")
    axes[1].set_ylabel("Fraction of this head's full utility")
    axes[1].set_xlabel("Per-head cache capacity n (visual tokens)")
    axes[1].set_title("Same 112 curves, normalized per head for shape comparison", loc="left")
    normalized_max = float(np.max(curves / shares[:, None]))
    axes[1].set_ylim(0, min(1.0, normalized_max * 1.05))
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes,
                       label="Transformer layer", fraction=.025, pad=.012)
    bar.set_ticks([0, 4, 8, 12, 16, 20, 24, 27])
    fig.suptitle("All 112 physical KV-head n–v curves (1-token grid)", fontsize=14)
    fig.savefig(ROOT / "all_head_n_v_curves.png", dpi=270)
    fig.savefig(ROOT / "all_head_n_v_curves.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12.8, 6.2), layout="constrained")
    order = np.argsort(-curves[:, -1])
    # Each row is one head, ordered by final raw projected utility. This is a
    # diagnostic complement to the literal overlay, not a replacement for it.
    image = ax.imshow(100 * curves[order], origin="lower", aspect="auto",
                      cmap="viridis", extent=(-.5, 1500.5, -.5, 111.5))
    ax.scatter(budget[order], np.arange(112), s=5, c="#e84a5f",
               edgecolors="none", label="Allocated capacity")
    ax.set(xlabel="Per-head cache capacity n (visual tokens)",
           ylabel="Head rank by utility at n=1500",
           title="Head-wise n–v map; red marks show the selected budget")
    fig.colorbar(image, ax=ax, label="Contribution to full-cache utility (pp)")
    ax.legend(loc="lower right", frameon=True, fontsize=8)
    fig.savefig(ROOT / "all_head_n_v_heatmap.png", dpi=270)
    fig.savefig(ROOT / "all_head_n_v_heatmap.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
