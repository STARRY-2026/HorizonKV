"""Generate per-head budget heatmaps for specific experiment directories.

Parses experiment directory names to extract scoring method, scheme, and parameters,
then computes the actual budget_table[28,4] and saves a heatmap inside each directory.
"""

import os, sys, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

NUM_LAYERS = 28
NUM_KV_HEADS = 4

# ── Score file registry ──
SCORE_REGISTRY = {
    "effective_readout_scores": {
        "path": "results/observations/effective_memory_readout_core_top100_n4_o80/effective_readout_scores.csv",
        "col": "internal_top100_mean_attention",
    },
    "ours_top100": {
        "path": "results/observations/memory_core_budget_scores/ours_top100.csv",
        "col": "score",
    },
    "ours_dual_top100": {
        "path": "results/observations/memory_core_budget_scores_dual/ours_dual_top100.csv",
        "col": "score",
    },
    "score_v1": {
        "path": "results/observations/score_v1/score_v1.csv",
        "col": "score",
    },
}


def load_scores(score_name):
    """Load [28,4] score matrix from a named score file."""
    info = SCORE_REGISTRY[score_name]
    path = os.path.join(PROJ_ROOT, info["path"])
    col = info["col"]
    df = pd.read_csv(path)
    scores = np.zeros((NUM_LAYERS, NUM_KV_HEADS), dtype=np.float64)
    for _, row in df.iterrows():
        l, h = int(row["layer"]), int(row["kv_head"])
        scores[l, h] = float(row.get(col, 0))
    return scores


# ── Budget computation (same logic as hermes_kv_head_budget.py) ──

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


def compute_sparsemm_budget(kv_scores, num_keep, r, w):
    """sparsemm scheme: global 112-head competition, total = avg num_keep per head."""
    base = max(1, num_keep - w)
    min_cache = int(base * r)
    remain = max(base - min_cache, 0) * NUM_LAYERS * NUM_KV_HEADS
    weights = np.nan_to_num(np.abs(kv_scores), nan=0.0, posinf=0.0, neginf=0.0)
    if weights.sum() <= 1e-12:
        weights = np.ones_like(weights)
    weights = weights / weights.sum()
    budgets = np.round(weights.reshape(-1) * remain + min_cache).astype(int) + w
    return np.maximum(budgets.reshape(NUM_LAYERS, NUM_KV_HEADS), 1).astype(int)


def compute_sparsemm_layer_total_budget(kv_scores, num_keep, r, w):
    """sparsemm_layer_total: global 112-head, total = num_keep * L."""
    base = max(1, num_keep - w)
    min_cache = int(base * r)
    min_budget = max(1, min_cache + w)
    target_total = num_keep * NUM_LAYERS
    weights = np.nan_to_num(np.abs(kv_scores), nan=0.0, posinf=0.0, neginf=0.0)
    return _allocate_integer_budget(
        weights.reshape(-1), total_budget=target_total, min_budget=min_budget
    ).reshape(NUM_LAYERS, NUM_KV_HEADS).astype(int)


def compute_sparsemm_per_layer_total_budget(kv_scores, num_keep, r, w):
    """sparsemm_per_layer_total: per-layer 4-head, each layer sums to num_keep."""
    base = max(1, num_keep - w)
    min_cache = int(base * r)
    min_budget = max(1, min_cache + w)
    weights = np.nan_to_num(np.abs(kv_scores), nan=0.0, posinf=0.0, neginf=0.0)
    budgets = np.zeros((NUM_LAYERS, NUM_KV_HEADS), dtype=int)
    for l in range(NUM_LAYERS):
        budgets[l] = _allocate_integer_budget(
            weights[l], total_budget=num_keep, min_budget=min_budget
        )
    return budgets.astype(int)


def compute_relative_budget(kv_scores, num_keep):
    """relative scheme: per-layer, heads clipped to [0.75*keep, 1.25*keep]."""
    strength, min_ratio, max_ratio = 0.75, 0.75, 1.25
    if kv_scores.max() < 1e-8:
        weights = np.ones_like(kv_scores)
    else:
        global_mean = kv_scores.mean() + 1e-8
        weights = 1.0 + strength * (kv_scores / global_mean - 1.0)
        weights = np.clip(weights, min_ratio, max_ratio)
    budgets = np.zeros((NUM_LAYERS, NUM_KV_HEADS), dtype=int)
    total_per_layer = num_keep * NUM_KV_HEADS
    min_h = max(1, int(round(num_keep * min_ratio)))
    max_h = max(min_h, int(round(num_keep * max_ratio)))
    for l in range(NUM_LAYERS):
        budgets[l] = _allocate_integer_budget(
            weights[l], total_per_layer, min_budget=min_h, max_budget=max_h
        )
    return budgets.astype(int)


# ── Heatmap plotting ──

def plot_budget_heatmap(budget_table, title, save_path, vmin=None, vmax=None):
    """Save a [L,H] annotated heatmap."""
    arr = np.asarray(budget_table, dtype=np.float64)
    if vmin is None:
        vmin = arr.min()
    if vmax is None:
        vmax = arr.max()

    fig, ax = plt.subplots(figsize=(5.5, 7.5))
    cmap = "YlOrRd" if vmax > vmin else "Blues"
    im = ax.imshow(arr, cmap=cmap, aspect="auto", origin="lower",
                   vmin=vmin, vmax=vmax, interpolation="nearest")

    for layer in range(arr.shape[0]):
        for head in range(arr.shape[1]):
            val = int(arr[layer, head])
            frac = (arr[layer, head] - vmin) / max(vmax - vmin, 1e-6)
            color = "white" if frac > 0.55 else "black"
            ax.text(head, layer, str(val), ha="center", va="center",
                    fontsize=6.5, color=color, fontweight="bold")

    # Layer type boundaries
    for boundary, color, label in [
        (int(NUM_LAYERS * 0.3) - 0.5, "#2196F3", "short→mid"),
        (int(NUM_LAYERS * 0.7) - 0.5, "#F44336", "mid→long"),
    ]:
        ax.axhline(y=boundary, color=color, linestyle="--", linewidth=1.2, alpha=0.7)
        ax.text(3.6, boundary - 0.3, label, fontsize=7, color=color,
                ha="right", va="bottom", fontweight="bold")

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("KV Head", fontsize=9)
    ax.set_ylabel("Layer", fontsize=9)
    ax.set_xticks(range(NUM_KV_HEADS))
    ax.set_xticklabels([str(h) for h in range(NUM_KV_HEADS)])
    ax.set_yticks(range(NUM_LAYERS))
    ax.set_yticklabels([str(l) for l in range(NUM_LAYERS)], fontsize=5.5)

    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Budget (tokens per KV-head)", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return arr.min(), arr.max()


# ── Experiment parser ──

def parse_experiment(dirname):
    """Parse experiment directory name to extract configuration.

    Returns dict with: score_name, scheme, r, w, is_baseline
    """
    # Baseline check
    if "kvheadbudget" not in dirname and "layerbudget" not in dirname:
        return {"is_baseline": True, "desc": "HERMES Baseline (uniform per-layer)"}

    # Extract score name
    m = re.search(r"kvheadbudget-([^-]+(?:[^-]+)?)-", dirname)
    if not m:
        return None
    score_tag = m.group(1)

    # Map to known scores
    score_map = {
        "effective_readout_scores": "effective_readout_scores",
        "ours_dual_top100": "ours_dual_top100",
        "ours_top100": "ours_top100",
        "score_v1": "score_v1",
        "sparsemm_qwen25": "sparsemm_qwen25",
        "qwen2.5-vl": "sparsemm_qwen25",
    }
    score_name = score_map.get(score_tag)
    if score_name is None:
        # Try partial match
        for key in score_map:
            if key in score_tag:
                score_name = score_map[key]
                break
    if score_name is None:
        return None

    # Extract scheme
    m2 = re.search(r"kvheadbudget-[^-]+-([a-z_]+)-", dirname)
    scheme = m2.group(1) if m2 else "relative"

    # Extract r and w
    r_match = re.search(r"-r(\d+\.?\d*)-", dirname)
    w_match = re.search(r"-w(\d+)", dirname)
    r_val = float(r_match.group(1)) if r_match else 0.1
    w_val = int(w_match.group(1)) if w_match else 32

    # Is ragged prefill?
    ragged = "raggedprefill" in dirname

    num_keep = 6000
    m_kv = re.search(r"kv(\d+)-", dirname)
    if m_kv:
        num_keep = int(m_kv.group(1))

    desc_parts = [score_name.replace("_", " ")]
    desc_parts.append(f"scheme={scheme}")
    desc_parts.append(f"r={r_val:.2g}")
    desc_parts.append(f"w={w_val}")
    if ragged:
        desc_parts.append("ragged")
    desc_parts.append(f"kv={num_keep}")

    return {
        "is_baseline": False,
        "score_name": score_name,
        "scheme": scheme,
        "r": r_val,
        "w": w_val,
        "num_keep": num_keep,
        "ragged": ragged,
        "desc": ", ".join(desc_parts),
    }


# ── Main ──

def main():
    base_dir = os.path.join(PROJ_ROOT, "results/qwen2.5_vl_7b/streamingbench")
    exp_dirs = [
        "fps0.5-kv6000-hermes-half",
        "fps0.5-kv6000-hermes-kvheadbudget-effective_readout_scores-sparsemm-union1-r0.2-w32-raggedprefill",
        "fps0.5-kv6000-hermes-kvheadbudget-ours_dual_top100-sparsemm-union1-r0.2-w32-raggedprefill",
        "fps0.5-kv6000-hermes-kvheadbudget-ours_top100-sparsemm-union1-r0.2-w32-raggedprefill",
        "fps0.5-kv6000-hermes-kvheadbudget-score_v1-sparsemm-union1-r0.1-w32-raggedprefill",
        "fps0.5-kv6000-hermes-kvheadbudget-score_v1-sparsemm-union1-r0.2-w32-raggedprefill",
    ]

    all_budgets = []
    all_labels = []

    for exp_name in exp_dirs:
        exp_path = os.path.join(base_dir, exp_name)
        print(f"\n{'='*60}")
        print(f"Experiment: {exp_name}")

        cfg = parse_experiment(exp_name)
        if cfg is None:
            print(f"  ⚠ Could not parse config, skipping")
            continue

        print(f"  Config: {cfg.get('desc', 'N/A')}")

        if cfg.get("is_baseline"):
            # Baseline HERMES: uniform allocation across all KV heads
            # Each layer gets num_keep=6000, shared by 4 KV heads via GQA
            budget = np.full((NUM_LAYERS, NUM_KV_HEADS), 6000, dtype=int)
            title = "HERMES Baseline (dense, no head budget)\nEach layer: 6000 tokens, shared by 4 KV heads"
            stats = "min=6000, mean=6000, max=6000, total=168000"
        else:
            score_name = cfg["score_name"]
            scheme = cfg["scheme"]
            r = cfg["r"]
            w = cfg["w"]
            num_keep = cfg["num_keep"]

            if score_name == "sparsemm_qwen25":
                scores = load_sparsemm_qwen25()
            else:
                scores = load_scores(score_name)

            # Compute budget
            if scheme == "sparsemm":
                budget = compute_sparsemm_budget(scores, num_keep, r, w)
            elif scheme in ("sparsemm_layer_total", "sparsemm_layer", "sparsemm_total"):
                budget = compute_sparsemm_layer_total_budget(scores, num_keep, r, w)
            elif scheme in ("sparsemm_per_layer_total", "sparsemm_per_layer_exact", "sparsemm_per_layer"):
                budget = compute_sparsemm_per_layer_total_budget(scores, num_keep, r, w)
            elif scheme == "relative":
                budget = compute_relative_budget(scores, num_keep)
            else:
                print(f"  ⚠ Unknown scheme: {scheme}, defaulting to sparsemm")
                budget = compute_sparsemm_budget(scores, num_keep, r, w)

            score_label = score_name.replace("_", " ")
            title = f"{score_label}\n{scheme}, r={r:.2g}, w={w}, kv={num_keep}"
            stats = (f"min={budget.min()}, mean={budget.mean():.0f}, "
                     f"max={budget.max()}, total={budget.sum()}")

        print(f"  Budget: {stats}")

        save_path = os.path.join(exp_path, "budget_heatmap.png")
        plot_budget_heatmap(budget, title, save_path)
        print(f"  ✓ Saved: {save_path}")

        all_budgets.append(budget)
        all_labels.append(exp_name.split("-kvheadbudget-")[-1].split("-union")[0]
                          if "kvheadbudget" in exp_name else "baseline-hermes")

    # ── Comparison grid ──
    if len(all_budgets) >= 2:
        print(f"\n{'='*60}")
        print("Generating comparison grid...")

        # Use shared color scale across all non-baseline experiments
        non_baseline = [b for b, l in zip(all_budgets, all_labels) if "baseline" not in l]
        if non_baseline:
            vmin = min(b.min() for b in non_baseline)
            vmax = max(b.max() for b in non_baseline)
        else:
            vmin, vmax = 6000, 6000

        n = len(all_budgets)
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 5.5),
                                 squeeze=False)

        # Short labels
        short_labels = []
        for exp_name in exp_dirs:
            if "kvheadbudget" in exp_name:
                m = re.search(r"kvheadbudget-([^-]+)-([^-]+)-", exp_name)
                if m:
                    short_labels.append(f"{m.group(1)}\n{m.group(2)}")
                else:
                    short_labels.append(exp_name[:60])
            else:
                short_labels.append("baseline\n(uniform)")

        for idx, (budget, label) in enumerate(zip(all_budgets, short_labels)):
            ax = axes[idx // ncols][idx % ncols]
            arr = np.asarray(budget, dtype=np.float64)
            cmap = "YlOrRd" if vmax > vmin else "Blues"
            im = ax.imshow(arr, cmap=cmap, aspect="auto", origin="lower",
                           vmin=vmin, vmax=vmax, interpolation="nearest")

            ax.set_title(label, fontsize=8, fontweight="bold")
            ax.set_xticks(range(NUM_KV_HEADS))
            ax.set_xticklabels([str(h) for h in range(NUM_KV_HEADS)], fontsize=6)
            ax.set_yticks([0, 7, 14, 21, 27])
            ax.set_yticklabels(["0", "7", "14", "21", "27"], fontsize=6)

            for boundary, color in [
                (int(NUM_LAYERS * 0.3) - 0.5, "#2196F3"),
                (int(NUM_LAYERS * 0.7) - 0.5, "#F44336"),
            ]:
                ax.axhline(y=boundary, color=color, linestyle="--",
                           linewidth=0.8, alpha=0.4)

            # Cell values
            for layer in range(arr.shape[0]):
                for head in range(arr.shape[1]):
                    val = int(arr[layer, head])
                    frac = (arr[layer, head] - vmin) / max(vmax - vmin, 1e-6)
                    color = "white" if frac > 0.55 else "black"
                    ax.text(head, layer, str(val), ha="center", va="center",
                            fontsize=4, color=color, fontweight="bold")

        for idx in range(n, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.subplots_adjust(right=0.92, hspace=0.35, wspace=0.30)
        cbar_ax = fig.add_axes([0.94, 0.08, 0.015, 0.84])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("Budget (tokens per KV-head)", fontsize=9)

        fig.suptitle("KV-Head Budget Allocation Comparison\n"
                     f"(num_keep=6000, sparsemm scheme, all ragged prefill)",
                     fontsize=12, fontweight="bold", y=0.98)

        comp_path = os.path.join(base_dir, "budget_heatmap_comparison.png")
        plt.savefig(comp_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Saved: {comp_path}")

    print(f"\n{'='*60}")
    print("Done.")


def load_sparsemm_qwen25():
    """Load SparseMM-qwen25 scores."""
    path = os.path.join(PROJ_ROOT, "results/observations/sparsemm_qwen25_kv_group_scores.csv")
    df = pd.read_csv(path)
    scores = np.zeros((NUM_LAYERS, NUM_KV_HEADS), dtype=np.float64)
    for _, row in df.iterrows():
        l, h = int(row["layer"]), int(row["kv_head"])
        scores[l, h] = float(row.get("score_sum", 0))
    return scores


if __name__ == "__main__":
    main()
