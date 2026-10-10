#!/usr/bin/env python3
"""Compare full-range AUC and high-compression retention for Joint attention."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("results/myobservations/uniform_selector_auc_50v_20260920")
COMPRESSION = np.array(
    [0, .1, .2, .3, .4, .5, .6, .7, .8, .85, .9, .925, .95, .975, .99]
)
LOW_INDEX = int(np.where(np.isclose(COMPRESSION, .975))[0][0])


def main():
    rows = {}
    with (ROOT / "uniform_selector_head_auc.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["policy"] == "uniform_joint":
                head = 4 * int(row["layer"]) + int(row["kv_head"])
                rows[head] = row
    if len(rows) != 112:
        raise ValueError(f"Expected 112 Joint heads, got {len(rows)}")

    curves = np.array([
        [float(rows[h][f"coverage_at_compression_{x:g}"]) for x in COMPRESSION]
        for h in range(112)
    ])
    auc = np.array([float(rows[h]["auc"]) for h in range(112)])
    low = curves[:, LOW_INDEX]
    top_auc = set(np.argsort(-auc, kind="stable")[:10])
    top_low = set(np.argsort(-low, kind="stable")[:10])
    groups = [
        ("Top AUC only", sorted(top_auc - top_low), "#D55E00"),
        ("Top 97.5% only", sorted(top_low - top_auc), "#0072B2"),
        ("Top in both", sorted(top_auc & top_low), "#009E73"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.3), sharey=True)
    for ax in axes:
        for curve in curves:
            ax.plot(COMPRESSION * 100, curve, color="#D0D2D4", lw=.75, alpha=.36)
        for name, heads, color in groups:
            ax.plot(
                COMPRESSION * 100, curves[heads].mean(axis=0),
                marker="o", ms=3, lw=2.5, color=color,
                label=f"{name} (n={len(heads)})",
            )
        ax.axvline(97.5, ls="--", lw=1.2, color="#333333", alpha=.8)
        ax.grid(alpha=.2)
        ax.set_xlabel("Visual-token compression (%)")
        ax.set_ylim(0, 1.03)
    axes[0].set_xlim(0, 100)
    axes[0].set_title("Full compression range")
    axes[0].set_ylabel("Retained future-answer utility / full utility")
    axes[0].legend(loc="lower left", fontsize=9, framealpha=.96)
    axes[1].set_xlim(80, 100)
    axes[1].set_title("Low-capacity region (97.5% target compression)")
    axes[1].annotate(
        "Low-budget comparison", xy=(97.5, .57), xytext=(89, .67),
        fontsize=9, arrowprops={"arrowstyle": "->", "color": "#333333"},
    )
    fig.suptitle(
        "Qwen2.5-VL-7B StreamingBench: Joint-attention head utility curves",
        fontsize=13,
    )
    fig.text(
        .5, .005,
        "50 calibration videos, 97 observations; at least newest 32 visual tokens retained. "
        "Gray: all 112 heads. Colored lines: group means.",
        ha="center", fontsize=9, color="#555555",
    )
    fig.tight_layout(rect=[0, .04, 1, .96])
    png = ROOT / "joint_auc_vs_low_budget_curves.png"
    pdf = ROOT / "joint_auc_vs_low_budget_curves.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    print(png)
    print(pdf)
    for name, heads, _ in groups:
        print(name, [(h // 4, h % 4) for h in heads],
              "mean AUC", round(float(auc[heads].mean()), 4),
              "mean retention at 97.5%", round(float(low[heads].mean()), 4))


if __name__ == "__main__":
    main()
