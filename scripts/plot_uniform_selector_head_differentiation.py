#!/usr/bin/env python3
"""Plot head differentiation from the uniform-selector AUC experiment."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("results/myobservations/uniform_selector_auc_50v_20260920")
CSV = ROOT / "uniform_selector_head_auc.csv"
COMPRESSION = np.asarray(
    [0, .1, .2, .3, .4, .5, .6, .7, .8, .85, .9, .925, .95, .975, .99]
)


def load_policy(name):
    rows = [row for row in csv.DictReader(CSV.open()) if row["policy"] == name]
    rows.sort(key=lambda row: (int(row["layer"]), int(row["kv_head"])))
    auc = np.asarray([float(row["auc"]) for row in rows])
    curves = np.asarray([
        [float(row[f"coverage_at_compression_{x:g}"]) for x in COMPRESSION]
        for row in rows
    ])
    return auc, curves


deployed_auc, _ = load_policy("deployed")
joint_auc, joint_curves = load_policy("uniform_joint")
local_auc, _ = load_policy("uniform_local")
global_auc, _ = load_policy("uniform_global")
recency_auc, _ = load_policy("uniform_recency")

top = np.argsort(joint_auc)[-6:][::-1]
bottom = np.argsort(joint_auc)[:6]
colors = plt.cm.viridis(np.linspace(0.05, 0.95, 28))

fig = plt.figure(figsize=(17.0, 11.5))
grid = fig.add_gridspec(2, 2, hspace=0.29, wspace=0.22)

# All curves: color encodes layer; transparency exposes density and outliers.
ax = fig.add_subplot(grid[0, 0])
for head in range(112):
    ax.plot(COMPRESSION * 100, joint_curves[head] * 100,
            color=colors[head // 4], alpha=0.35, linewidth=1.0)
ax.plot(COMPRESSION * 100, joint_curves.mean(axis=0) * 100,
        color="black", linewidth=2.8, label="Mean over 112 heads")
ax.fill_between(
    COMPRESSION * 100,
    np.quantile(joint_curves, .25, axis=0) * 100,
    np.quantile(joint_curves, .75, axis=0) * 100,
    color="black", alpha=.10, label="Head IQR",
)
ax.set_title("A. All heads under uniform joint ranking", weight="bold")
ax.set_xlabel("Compression ratio (%)")
ax.set_ylabel("Retained projected utility (%)")
ax.set_xlim(0, 99)
ax.set_ylim(0, 102)
ax.grid(alpha=.2)
ax.legend(frameon=False, loc="lower left")

# Extremes reveal that high/low heads are distributed across depth.
ax = fig.add_subplot(grid[0, 1])
for head in top:
    ax.plot(COMPRESSION * 100, joint_curves[head] * 100, linewidth=2.1,
            label=f"High L{head//4}-H{head%4} ({joint_auc[head]:.3f})")
for head in bottom:
    ax.plot(COMPRESSION * 100, joint_curves[head] * 100, linewidth=1.8,
            linestyle="--", label=f"Low L{head//4}-H{head%4} ({joint_auc[head]:.3f})")
ax.set_title("B. Highest- and lowest-AUC heads", weight="bold")
ax.set_xlabel("Compression ratio (%)")
ax.set_ylabel("Retained projected utility (%)")
ax.set_xlim(0, 99)
ax.set_ylim(0, 102)
ax.grid(alpha=.2)
ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower left")

# Side-by-side heatmaps expose both the old depth prior and the new head pattern.
sub = grid[1, 0].subgridspec(1, 3, wspace=.10)
heatmaps = [
    (deployed_auc.reshape(28, 4), "Deployed", .48, .82, "viridis"),
    (joint_auc.reshape(28, 4), "Uniform joint", .48, .82, "viridis"),
    ((joint_auc - deployed_auc).reshape(28, 4), "Joint − deployed", -.02, .24, "RdBu_r"),
]
heat_axes = []
for index, (values, title, vmin, vmax, cmap) in enumerate(heatmaps):
    axis = fig.add_subplot(sub[0, index])
    image = axis.imshow(values, aspect="auto", origin="lower", cmap=cmap,
                        vmin=vmin, vmax=vmax)
    axis.set_title(title, fontsize=10, weight="bold")
    axis.set_xlabel("KV head")
    axis.set_xticks(range(4))
    if index == 0:
        axis.set_ylabel("Layer")
        axis.set_yticks(range(0, 28, 3))
    else:
        axis.set_yticks([])
    heat_axes.append((axis, image))
fig.colorbar(heat_axes[1][1], ax=[x[0] for x in heat_axes[:2]], fraction=.025,
             pad=.02, label="AUC")
fig.colorbar(heat_axes[2][1], ax=heat_axes[2][0], fraction=.08, pad=.04,
             label="AUC delta")
heat_axes[0][0].text(-.46, 1.075, "C. AUC maps: routing artifact and residual head structure",
                     transform=heat_axes[0][0].transAxes, fontsize=12, weight="bold")

# Layer means show how attention removes most of the original depth separation.
ax = fig.add_subplot(grid[1, 1])
layers = np.arange(28)
for values, label, style, width in [
    (deployed_auc, "Deployed routing", "-", 2.5),
    (joint_auc, "Uniform joint", "-", 2.7),
    (local_auc, "Uniform local", "--", 1.5),
    (global_auc, "Uniform global", "--", 1.5),
    (recency_auc, "Uniform recency", ":", 2.0),
]:
    ax.plot(layers, values.reshape(28, 4).mean(axis=1), style,
            linewidth=width, label=label)
ax.axvspan(-.5, 7.5, color="#4C78A8", alpha=.06)
ax.axvspan(7.5, 18.5, color="#F2CF5B", alpha=.08)
ax.axvspan(18.5, 27.5, color="#59A14F", alpha=.06)
ax.set_title("D. Mean AUC by layer", weight="bold")
ax.set_xlabel("Layer")
ax.set_ylabel("AUC")
ax.set_xlim(-.5, 27.5)
ax.set_ylim(.42, .84)
ax.grid(alpha=.2)
ax.legend(frameon=False, fontsize=9, ncol=2)

fig.suptitle(
    "Physical KV-head differentiation after controlling the selector policy\n"
    "Qwen2.5-VL-7B · StreamingBench · 50 videos / 97 observations · Recent32 protected",
    fontsize=16, weight="bold", y=.98,
)
fig.text(
    .5, .012,
    "Uniform joint removes most shallow/deep separation, while substantial within-layer head variation remains. "
    "Metric is offline projected-utility retention, not QA accuracy.",
    ha="center", fontsize=10,
)

for suffix in ("png", "pdf"):
    fig.savefig(ROOT / f"uniform_selector_head_differentiation.{suffix}",
                dpi=250, bbox_inches="tight")
print(ROOT / "uniform_selector_head_differentiation.png")
