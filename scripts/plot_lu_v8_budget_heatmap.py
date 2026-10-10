#!/usr/bin/env python3
"""Plot physical KV-head budget comparison for K=1000."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from head_analysis.hermes_head_budget import load_head_scores
from head_analysis.hermes_kv_head_budget import build_kv_head_budget_table


ROOT = Path("results/myobservations/lu_kv_offline_50v_20260913")
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

v8_scores = load_head_scores(
    "results/myobservations/relocatev3/v8_blend3_scores.csv", 28, 28
)
v8 = build_kv_head_budget_table(
    v8_scores,
    1000,
    num_layers=28,
    num_query_heads=28,
    num_kv_heads=4,
    scheme="sparsemm",
    sparsemm_ratio=0.2,
    sparsemm_window_size=32,
)
lu = np.loadtxt(
    ROOT / "lu_raw_blend05_clip075_150_budget.csv",
    delimiter=",",
    skiprows=1,
    usecols=2,
    dtype=int,
).reshape(28, 4)

fig, axes = plt.subplots(
    1, 3, figsize=(11.2, 10.2),
    gridspec_kw={"width_ratios": [1, 1, 1.05]},
)
fig.subplots_adjust(left=0.055, right=0.94, bottom=0.085, top=0.91, wspace=0.38)
main_min = int(min(v8.min(), lu.min()))
main_max = int(max(v8.max(), lu.max()))
images = []
for ax, data, title in zip(
    axes[:2], [v8, lu], ["V8 budget", "Constrained Raw-LU budget"]
):
    im = ax.imshow(data, cmap="viridis", aspect="auto", vmin=main_min, vmax=main_max)
    images.append(im)
    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_xlabel("Physical KV head")
    ax.set_xticks(range(4))
    ax.set_yticks(range(28))
    ax.set_ylabel("Layer")
    for layer in range(28):
        for head in range(4):
            color = "white" if data[layer, head] < (main_min + main_max) / 2 else "black"
            ax.text(head, layer, str(int(data[layer, head])), ha="center", va="center",
                    fontsize=5.5, color=color)

delta = lu - v8
limit = int(np.abs(delta).max())
im_delta = axes[2].imshow(delta, cmap="RdBu_r", aspect="auto", vmin=-limit, vmax=limit)
axes[2].set_title("Difference: LU - V8", fontsize=13, weight="bold")
axes[2].set_xlabel("Physical KV head")
axes[2].set_ylabel("Layer")
axes[2].set_xticks(range(4))
axes[2].set_yticks(range(28))
for layer in range(28):
    for head in range(4):
        axes[2].text(head, layer, f"{int(delta[layer, head]):+d}", ha="center",
                     va="center", fontsize=5.5,
                     color="white" if abs(delta[layer, head]) > limit * 0.52 else "black")

fig.colorbar(images[0], ax=axes[:2], shrink=0.78, pad=0.025, label="KV pairs per physical head")
fig.colorbar(im_delta, ax=axes[2], shrink=0.78, pad=0.025, label="Budget difference")
fig.suptitle(
    "Qwen2.5-VL-7B head-wise KV allocation (K=1000, total target=112,000)",
    fontsize=14, weight="bold",
)
fig.text(
    0.5, 0.022,
    f"V8: min={v8.min()}, max={v8.max()}, sum={v8.sum()}  |  "
    f"LU: min={lu.min()}, max={lu.max()}, sum={lu.sum()}  |  "
    f"mean |difference|={np.abs(delta).mean():.1f}",
    ha="center", fontsize=10,
)

for suffix in ("png", "pdf"):
    fig.savefig(OUT / f"lu_vs_v8_budget_heatmap_k1000.{suffix}", dpi=300, bbox_inches="tight")
np.savetxt(OUT / "lu_minus_v8_budget_k1000.csv", delta, delimiter=",", fmt="%d")
print({"v8_sum": int(v8.sum()), "lu_sum": int(lu.sum()),
       "v8_range": [int(v8.min()), int(v8.max())],
       "lu_range": [int(lu.min()), int(lu.max())],
       "delta_range": [int(delta.min()), int(delta.max())],
       "mean_abs_delta": float(np.abs(delta).mean()), "output": str(OUT)})
