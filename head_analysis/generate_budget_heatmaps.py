"""
Generate actual budget allocation heatmaps for all HERMES experiments.

For each experiment, we:
  1. Load the score CSV
  2. Compute the actual budget_table[28, 4] using build_kv_head_budget_table()
     with the exact scheme & parameters from the experiment name
  3. Plot side-by-side: raw scores vs. actual budget allocation

Output: results/myobservations/budget_heatmaps/
"""

import argparse
import os
import sys
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make sure we can import project modules
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

from head_analysis.hermes_kv_head_budget import (
    build_kv_head_budget_table,
    _aggregate_query_scores_to_kv,
)


NUM_LAYERS = 28
NUM_KV_HEADS = 4
NUM_QUERY_HEADS = 28
KV_SIZE = 6000


# ═══════════════════════════════════════════════════════════════
# Experiment registry
# ═══════════════════════════════════════════════════════════════

def _load_standard_csv(path):
    """Load a score CSV with columns: layer, kv_head, score"""
    df = pd.read_csv(path)
    scores = np.zeros((NUM_LAYERS, NUM_KV_HEADS), dtype=np.float64)
    for _, row in df.iterrows():
        layer = int(row["layer"])
        kv_head = int(row["kv_head"])
        scores[layer, kv_head] = float(row.get("score", 0))
    return scores


def _load_effective_readout(path):
    """Load effective_readout_scores.csv — use internal_top100_mean_attention as score"""
    df = pd.read_csv(path)
    scores = np.zeros((NUM_LAYERS, NUM_KV_HEADS), dtype=np.float64)
    col = "internal_top100_mean_attention"
    for _, row in df.iterrows():
        layer = int(row["layer"])
        kv_head = int(row["kv_head"])
        scores[layer, kv_head] = float(row.get(col, 0))
    return scores


def _load_sparsemm_qwen25(path):
    """Load sparsemm_qwen25_kv_group_scores.csv — use score_sum as score"""
    df = pd.read_csv(path)
    scores = np.zeros((NUM_LAYERS, NUM_KV_HEADS), dtype=np.float64)
    for _, row in df.iterrows():
        layer = int(row["layer"])
        kv_head = int(row["kv_head"])
        scores[layer, kv_head] = float(row.get("score_sum", 0))
    return scores


# ── Define experiments with their exact configurations ──
EXPERIMENTS = OrderedDict()

# 1) ours_top100 + sparsemm r=0.2 w=32
EXPERIMENTS["ours_top100\n(sparsemm r=0.2)"] = {
    "score_path": "results/observations/memory_core_budget_scores/ours_top100.csv",
    "loader": _load_standard_csv,
    "scheme": "sparsemm",
    "sparsemm_ratio": 0.2,
    "sparsemm_window_size": 32,
    "num_keep": KV_SIZE,
}

# 2) ours_dual_top100 + sparsemm r=0.2 w=32
EXPERIMENTS["ours_dual_top100\n(sparsemm r=0.2)"] = {
    "score_path": "results/observations/memory_core_budget_scores_dual/ours_dual_top100.csv",
    "loader": _load_standard_csv,
    "scheme": "sparsemm",
    "sparsemm_ratio": 0.2,
    "sparsemm_window_size": 32,
    "num_keep": KV_SIZE,
}

# 3) effective_readout + sparsemm r=0.2 w=32
EXPERIMENTS["effective_readout\n(sparsemm r=0.2)"] = {
    "score_path": "results/observations/effective_memory_readout_core_top100_n4_o80/effective_readout_scores.csv",
    "loader": _load_effective_readout,
    "scheme": "sparsemm",
    "sparsemm_ratio": 0.2,
    "sparsemm_window_size": 32,
    "num_keep": KV_SIZE,
}

# 4) ours_top100 + sparsemm_per_layer_total r=0.2 w=32
EXPERIMENTS["ours_top100\n(per_layer_total r=0.2)"] = {
    "score_path": "results/observations/memory_core_budget_scores/ours_top100.csv",
    "loader": _load_standard_csv,
    "scheme": "sparsemm_per_layer_total",
    "sparsemm_ratio": 0.2,
    "sparsemm_window_size": 32,
    "num_keep": KV_SIZE,
}

# 5) score_v1 + sparsemm r=0.1 w=32
EXPERIMENTS["score_v1\n(sparsemm r=0.1)"] = {
    "score_path": "results/observations/score_v1/score_v1.csv",
    "loader": _load_standard_csv,
    "scheme": "sparsemm",
    "sparsemm_ratio": 0.1,
    "sparsemm_window_size": 32,
    "num_keep": KV_SIZE,
}

# 6) score_v1 + sparsemm r=0.2 w=32
EXPERIMENTS["score_v1\n(sparsemm r=0.2)"] = {
    "score_path": "results/observations/score_v1/score_v1.csv",
    "loader": _load_standard_csv,
    "scheme": "sparsemm",
    "sparsemm_ratio": 0.2,
    "sparsemm_window_size": 32,
    "num_keep": KV_SIZE,
}

# 7) sparsemm_qwen25 + sparsemm r=0.1 w=32
EXPERIMENTS["SparseMM-qwen25\n(sparsemm r=0.1)"] = {
    "score_path": "results/observations/sparsemm_qwen25_kv_group_scores.csv",
    "loader": _load_sparsemm_qwen25,
    "scheme": "sparsemm",
    "sparsemm_ratio": 0.1,
    "sparsemm_window_size": 32,
    "num_keep": KV_SIZE,
}

# 8) TAScore optimistic + sparsemm_per_layer_total r=0.2 w=32
EXPERIMENTS["TAScore-optimistic\n(per_layer_total r=0.2)"] = {
    "score_path": "results/observations/tascores/tascores_optimistic.csv",
    "loader": _load_standard_csv,
    "scheme": "sparsemm_per_layer_total",
    "sparsemm_ratio": 0.2,
    "sparsemm_window_size": 32,
    "num_keep": KV_SIZE,
}

# 9) TAScore balanced + sparsemm_per_layer_total r=0.2 w=32
EXPERIMENTS["TAScore-balanced\n(per_layer_total r=0.2)"] = {
    "score_path": "results/observations/tascores/tascores_balanced.csv",
    "loader": _load_standard_csv,
    "scheme": "sparsemm_per_layer_total",
    "sparsemm_ratio": 0.2,
    "sparsemm_window_size": 32,
    "num_keep": KV_SIZE,
}

# 10) uniform baseline (sparsemm — all heads get the same)
EXPERIMENTS["Uniform baseline\n(sparsemm r=0.2)"] = {
    "score_path": "results/observations/memory_core_budget_scores/uniform.csv",
    "loader": _load_standard_csv,
    "scheme": "sparsemm",
    "sparsemm_ratio": 0.2,
    "sparsemm_window_size": 32,
    "num_keep": KV_SIZE,
}

# 11) inverted control (sparsemm r=0.2)
EXPERIMENTS["Inverted control\n(sparsemm r=0.2)"] = {
    "score_path": "results/observations/memory_core_budget_scores/inverted_top100.csv",
    "loader": _load_standard_csv,
    "scheme": "sparsemm",
    "sparsemm_ratio": 0.2,
    "sparsemm_window_size": 32,
    "num_keep": KV_SIZE,
}


# ═══════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════

def _safe_normalize(arr):
    arr = np.asarray(arr, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def plot_budget_heatmap(budget_table, title, save_path, cmap="YlOrRd",
                        annotate=True, vmin=None, vmax=None):
    """Plot a single [L, H] budget heatmap with annotated cell values."""
    arr = _safe_normalize(budget_table)
    if vmin is None:
        vmin = arr.min()
    if vmax is None:
        vmax = arr.max()

    fig, ax = plt.subplots(figsize=(6, 8))
    im = ax.imshow(arr, cmap=cmap, aspect="auto", origin="lower",
                   vmin=vmin, vmax=vmax, interpolation="nearest")

    # Annotate each cell with the integer budget value
    if annotate:
        for layer in range(arr.shape[0]):
            for head in range(arr.shape[1]):
                val = int(arr[layer, head])
                # Use white text on dark cells, black on light
                frac = (arr[layer, head] - vmin) / max(vmax - vmin, 1e-6)
                color = "white" if frac > 0.55 else "black"
                ax.text(head, layer, str(val), ha="center", va="center",
                        fontsize=7, color=color, fontweight="bold")

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("KV Head", fontsize=10)
    ax.set_ylabel("Layer", fontsize=10)
    ax.set_xticks(range(arr.shape[1]))
    ax.set_xticklabels([str(h) for h in range(arr.shape[1])])
    ax.set_yticks(range(arr.shape[0]))
    ax.set_yticklabels([str(l) for l in range(arr.shape[0])], fontsize=6)

    # Mark layer-type boundaries
    short_term_threshold = int(NUM_LAYERS * 0.3)   # layer 8
    long_term_threshold = int(NUM_LAYERS * 0.7)     # layer 19
    for boundary, label, color in [
        (short_term_threshold - 0.5, "short→mid", "#2196F3"),
        (long_term_threshold - 0.5, "mid→long", "#F44336"),
    ]:
        ax.axhline(y=boundary, color=color, linestyle="--", linewidth=1.2, alpha=0.7)
        ax.text(3.6, boundary - 0.3, label, fontsize=7, color=color,
                ha="right", va="bottom", fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Budget (tokens)", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_score_vs_budget(scores, budget_table, title, save_path,
                         cmap_score="viridis", cmap_budget="YlOrRd"):
    """Plot raw scores (left) vs. actual budget (right) side-by-side."""
    scores_arr = _safe_normalize(scores)
    budget_arr = _safe_normalize(budget_table)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

    # Left: raw scores
    im1 = ax1.imshow(scores_arr, cmap=cmap_score, aspect="auto", origin="lower",
                     interpolation="nearest")
    ax1.set_title(f"{title}\nRaw Scores", fontsize=11, fontweight="bold")
    ax1.set_xlabel("KV Head")
    ax1.set_ylabel("Layer")
    ax1.set_xticks(range(scores_arr.shape[1]))
    ax1.set_xticklabels([str(h) for h in range(scores_arr.shape[1])])
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.85, pad=0.02)
    cbar1.set_label("Score", fontsize=8)

    # Right: budget allocation
    im2 = ax2.imshow(budget_arr, cmap=cmap_budget, aspect="auto", origin="lower",
                     interpolation="nearest")
    ax2.set_title("Actual Budget Allocation", fontsize=11, fontweight="bold")
    ax2.set_xlabel("KV Head")
    ax2.set_ylabel("Layer")
    ax2.set_xticks(range(budget_arr.shape[1]))
    ax2.set_xticklabels([str(h) for h in range(budget_arr.shape[1])])
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.85, pad=0.02)
    cbar2.set_label("Budget (tokens)", fontsize=8)

    # Layer boundaries
    for boundary, color in [
        (int(NUM_LAYERS * 0.3) - 0.5, "#2196F3"),
        (int(NUM_LAYERS * 0.7) - 0.5, "#F44336"),
    ]:
        for ax in (ax1, ax2):
            ax.axhline(y=boundary, color=color, linestyle="--",
                       linewidth=1.0, alpha=0.5)

    # Annotate budget cells
    for layer in range(budget_arr.shape[0]):
        for head in range(budget_arr.shape[1]):
            val = int(budget_arr[layer, head])
            frac = (budget_arr[layer, head] - budget_arr.min()) / max(
                budget_arr.max() - budget_arr.min(), 1e-6)
            color = "white" if frac > 0.55 else "black"
            ax2.text(head, layer, str(val), ha="center", va="center",
                     fontsize=5.5, color=color, fontweight="bold")

    plt.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_comparison_grid(all_budgets, all_names, save_path):
    """N×1 grid comparing budget heatmaps across all experiments."""
    n = len(all_budgets)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))

    # Use consistent vmin/vmax across all plots
    all_flat = np.concatenate([b.flatten() for b in all_budgets])
    vmin = all_flat.min()
    vmax = all_flat.max()

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 5.5),
                             squeeze=False)

    for idx, (budget, name) in enumerate(zip(all_budgets, all_names)):
        ax = axes[idx // ncols][idx % ncols]
        arr = _safe_normalize(budget)
        im = ax.imshow(arr, cmap="YlOrRd", aspect="auto", origin="lower",
                       vmin=vmin, vmax=vmax, interpolation="nearest")

        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.set_xlabel("KV Head", fontsize=7)
        ax.set_ylabel("Layer", fontsize=7)
        ax.set_xticks(range(arr.shape[1]))
        ax.set_xticklabels([str(h) for h in range(arr.shape[1])], fontsize=6)
        ax.set_yticks([0, 7, 14, 21, 27])
        ax.set_yticklabels(["0", "7", "14", "21", "27"], fontsize=6)

        # Layer type boundaries
        for boundary, color in [
            (int(NUM_LAYERS * 0.3) - 0.5, "#2196F3"),
            (int(NUM_LAYERS * 0.7) - 0.5, "#F44336"),
        ]:
            ax.axhline(y=boundary, color=color, linestyle="--",
                       linewidth=0.8, alpha=0.4)

        # Show budget value in each cell
        for layer in range(arr.shape[0]):
            for head in range(arr.shape[1]):
                val = int(arr[layer, head])
                frac = (arr[layer, head] - vmin) / max(vmax - vmin, 1e-6)
                color = "white" if frac > 0.6 else "black"
                ax.text(head, layer, str(val), ha="center", va="center",
                        fontsize=4.5, color=color, fontweight="bold")

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    # Shared colorbar
    fig.subplots_adjust(right=0.92, hspace=0.35, wspace=0.30)
    cbar_ax = fig.add_axes([0.94, 0.08, 0.015, 0.84])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Budget (tokens per KV-head)", fontsize=10)

    fig.suptitle("KV-Head Budget Allocation Comparison\n"
                 f"(num_keep={KV_SIZE}, 28 layers × 4 KV heads)",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_budget_statistics(all_budgets, all_names, save_path):
    """Bar chart: per-experiment budget range (min/mean/max per head)."""
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(all_names))
    width = 0.25

    mins = [b.min() for b in all_budgets]
    means = [b.mean() for b in all_budgets]
    maxs = [b.max() for b in all_budgets]

    ax.bar(x - width, mins, width, label="Min budget", color="#4CAF50", alpha=0.8)
    ax.bar(x, means, width, label="Mean budget", color="#2196F3", alpha=0.8)
    ax.bar(x + width, maxs, width, label="Max budget", color="#F44336", alpha=0.8)

    ax.axhline(y=KV_SIZE, color="gray", linestyle="--", linewidth=1,
               alpha=0.5, label=f"num_keep={KV_SIZE}")

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("\n", " ") for n in all_names],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Budget (tokens)", fontsize=11)
    ax.set_title("Budget Distribution Across Experiments\n"
                 "(min / mean / max per KV-head)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_layer_profile(all_budgets, all_names, save_path):
    """Layer-wise total budget (sum of 4 KV heads) per experiment."""
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_budgets)))
    for idx, (budget, name) in enumerate(zip(all_budgets, all_names)):
        layer_total = budget.sum(axis=1)  # sum over 4 KV heads
        short_name = name.replace("\n", " ")
        ax.plot(range(NUM_LAYERS), layer_total, color=colors[idx],
                linewidth=1.8, marker="o", markersize=3, label=short_name,
                alpha=0.85)

    ax.axhline(y=KV_SIZE * NUM_KV_HEADS, color="gray", linestyle="--",
               linewidth=1, alpha=0.5, label=f"Uniform ({KV_SIZE*NUM_KV_HEADS})")

    # Layer type boundaries
    for boundary, color, label in [
        (int(NUM_LAYERS * 0.3) - 0.5, "#2196F3", "short→mid"),
        (int(NUM_LAYERS * 0.7) - 0.5, "#F44336", "mid→long"),
    ]:
        ax.axvline(x=boundary, color=color, linestyle=":", linewidth=1.2, alpha=0.6)

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Total Budget (sum over 4 KV heads)", fontsize=11)
    ax.set_title("Layer-wise Total Budget Across Experiments", fontsize=13,
                 fontweight="bold")
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate budget allocation heatmaps for all experiments"
    )
    parser.add_argument("--save_dir",
                        default="results/myobservations/budget_heatmaps")
    parser.add_argument("--num_keep", type=int, default=KV_SIZE)
    args = parser.parse_args()

    save_dir = os.path.join(PROJ_ROOT, args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving outputs to: {save_dir}\n")

    all_budgets = []
    all_scores = []
    all_names = []

    for name, cfg in EXPERIMENTS.items():
        score_path = os.path.join(PROJ_ROOT, cfg["score_path"])
        print(f"{'─'*60}")
        print(f"Processing: {name.replace(chr(10), ' ')}")
        print(f"  Score file: {cfg['score_path']}")
        print(f"  Scheme: {cfg['scheme']}, r={cfg['sparsemm_ratio']}, "
              f"w={cfg['sparsemm_window_size']}")

        if not os.path.exists(score_path):
            print(f"  ⚠ SKIPPED: score file not found: {score_path}")
            continue

        # Load scores
        raw_scores = cfg["loader"](score_path)
        scores_2d = _safe_normalize(raw_scores)

        # Qwen2.5-VL-7B: GQA with 28 query heads → 4 KV heads
        # For build_kv_head_budget_table, we need query-head-level scores [28,28]
        # Since our score files are already at KV-head level [28,4],
        # we need to broadcast back to query heads or adapt the function.
        # The function internally does _aggregate_query_scores_to_kv,
        # so we pass KV-level scores directly and bypass the aggregation.

        # We'll compute the budget table directly using the scheme logic
        kv_scores = scores_2d  # already [L, H_kv]

        budget_table = build_kv_head_budget_table_from_kv(
            kv_scores,
            num_keep=cfg["num_keep"],
            num_layers=NUM_LAYERS,
            num_kv_heads=NUM_KV_HEADS,
            scheme=cfg["scheme"],
            sparsemm_ratio=cfg["sparsemm_ratio"],
            sparsemm_window_size=cfg["sparsemm_window_size"],
        )

        print(f"  Budget: min={budget_table.min()}, mean={budget_table.mean():.0f}, "
              f"max={budget_table.max()}, total={budget_table.sum()}")

        # --- Individual heatmaps ---
        safe_name = name.replace("\n", "_").replace(" ", "_").replace("(", "").replace(")", "")
        safe_name = safe_name.replace("=", "")

        # Score vs Budget side-by-side
        plot_score_vs_budget(
            scores_2d, budget_table, name,
            os.path.join(save_dir, f"score_vs_budget_{safe_name}.png"),
        )

        # Budget-only annotated heatmap
        plot_budget_heatmap(
            budget_table, f"Budget: {name}",
            os.path.join(save_dir, f"budget_{safe_name}.png"),
        )

        all_budgets.append(budget_table)
        all_scores.append(scores_2d)
        all_names.append(name)

    # --- Comparison plots ---
    if len(all_budgets) >= 2:
        print(f"\n{'─'*60}")
        print("Generating comparison plots...")

        plot_comparison_grid(
            all_budgets, all_names,
            os.path.join(save_dir, "comparison_grid.png"),
        )

        plot_budget_statistics(
            all_budgets, all_names,
            os.path.join(save_dir, "budget_statistics.png"),
        )

        plot_layer_profile(
            all_budgets, all_names,
            os.path.join(save_dir, "layer_profile.png"),
        )

    print(f"\n{'='*60}")
    print(f"Done! All outputs saved to: {save_dir}")
    print(f"Total experiments processed: {len(all_budgets)}")


# ═══════════════════════════════════════════════════════════════
# Adapted budget computation (works with KV-head-level scores)
# ═══════════════════════════════════════════════════════════════

def _finite_positive_weights(values):
    values = np.asarray(values, dtype=np.float64)
    return np.where(np.isfinite(values) & (values > 0), values, 1.0)


def _allocate_integer_budget(weights, total_budget, min_budget=0, max_budget=None):
    weights = _finite_positive_weights(weights)
    n_items = int(weights.size)
    total_budget = int(total_budget)
    if n_items == 0:
        return np.zeros(0, dtype=int)
    if total_budget <= 0:
        return np.zeros(n_items, dtype=int)

    min_budget = max(0, int(min_budget))
    if min_budget * n_items > total_budget:
        min_budget = total_budget // n_items

    if max_budget is None:
        max_budget = total_budget
    max_budget = max(int(max_budget), min_budget)

    raw = weights / weights.sum() * total_budget
    raw = np.clip(raw, min_budget, max_budget)
    if raw.sum() > 0:
        raw = raw * (total_budget / raw.sum())

    budgets = np.floor(raw).astype(int)
    budgets = np.clip(budgets, min_budget, max_budget)
    frac = raw - np.floor(raw)

    diff = total_budget - int(budgets.sum())
    while diff > 0:
        candidates = np.where(budgets < max_budget)[0]
        if candidates.size == 0:
            break
        idx = candidates[np.argmax(frac[candidates])]
        budgets[idx] += 1
        diff -= 1

    while diff < 0:
        candidates = np.where(budgets > min_budget)[0]
        if candidates.size == 0:
            break
        idx = candidates[np.argmin(frac[candidates])]
        budgets[idx] -= 1
        diff += 1

    return budgets.astype(int)


def build_kv_head_budget_table_from_kv(kv_scores, num_keep, num_layers=28,
                                        num_kv_heads=4,
                                        scheme="sparsemm",
                                        sparsemm_ratio=0.1,
                                        sparsemm_window_size=32):
    """Build [L, H] budget table from KV-head-level scores (bypasses query→KV aggregation)."""
    scheme = str(scheme or "relative").lower()

    if scheme == "sparsemm":
        base_capacity = max(1, int(num_keep) - int(sparsemm_window_size))
        min_cache = int(base_capacity * float(sparsemm_ratio))
        remain_capacity = (
            max(base_capacity - min_cache, 0)
            * int(num_layers)
            * int(num_kv_heads)
        )
        weights = np.nan_to_num(np.abs(kv_scores), nan=0.0, posinf=0.0, neginf=0.0)
        if weights.sum() <= 1e-12:
            weights = np.ones_like(weights, dtype=np.float64)
        weights = weights / weights.sum()
        budgets = np.round(weights * remain_capacity + min_cache).astype(int)
        budgets = budgets + int(sparsemm_window_size)
        return np.maximum(budgets, 1).astype(int)

    if scheme in ("sparsemm_layer_total", "sparsemm_layer", "sparsemm_total"):
        base_capacity = max(1, int(num_keep) - int(sparsemm_window_size))
        min_cache = int(base_capacity * float(sparsemm_ratio))
        min_budget = max(1, min_cache + int(sparsemm_window_size))
        target_total = int(num_keep) * int(num_layers)
        weights = np.nan_to_num(np.abs(kv_scores), nan=0.0, posinf=0.0, neginf=0.0)
        budgets_flat = _allocate_integer_budget(
            weights.reshape(-1),
            total_budget=target_total,
            min_budget=min_budget,
            max_budget=None,
        )
        return budgets_flat.reshape(int(num_layers), int(num_kv_heads)).astype(int)

    if scheme in ("sparsemm_per_layer_total", "sparsemm_layer_exact", "sparsemm_per_layer"):
        base_capacity = max(1, int(num_keep) - int(sparsemm_window_size))
        min_cache = int(base_capacity * float(sparsemm_ratio))
        min_budget = max(1, min_cache + int(sparsemm_window_size))
        weights = np.nan_to_num(np.abs(kv_scores), nan=0.0, posinf=0.0, neginf=0.0)
        budgets = np.zeros((int(num_layers), int(num_kv_heads)), dtype=int)
        for layer_idx in range(int(num_layers)):
            budgets[layer_idx] = _allocate_integer_budget(
                weights[layer_idx],
                total_budget=int(num_keep),
                min_budget=min_budget,
                max_budget=None,
            )
        return budgets.astype(int)

    # relative
    strength = 0.75
    min_ratio = 0.75
    max_ratio = 1.25
    if kv_scores.max() < 1e-8:
        weights = np.ones_like(kv_scores)
    else:
        global_mean = kv_scores.mean() + 1e-8
        weights = 1.0 + strength * (kv_scores / global_mean - 1.0)
        weights = np.clip(weights, min_ratio, max_ratio)

    budgets = np.zeros((num_layers, num_kv_heads), dtype=int)
    total_per_layer = int(num_keep) * int(num_kv_heads)
    min_h = max(1, int(round(num_keep * min_ratio)))
    max_h = max(min_h, int(round(num_keep * max_ratio)))
    for layer_idx in range(num_layers):
        budgets[layer_idx] = _allocate_integer_budget(
            weights[layer_idx],
            total_per_layer,
            min_budget=min_h,
            max_budget=max_h,
        )
    return budgets


if __name__ == "__main__":
    main()
