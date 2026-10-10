"""TAScore: Task-Adaptive Head Specialization Scoring.

Replaces score_v1's arbitrary weighted-sum + hard-max with:
  1. Per-task head profiling (preserves task-specific signal)
  2. Functional head clustering (data-driven, not hand-designed)
  3. Multiplicative scoring: utility^α × selectivity^β × boundary_penalty^γ
  4. Within-layer contrastive normalization
  5. Task-conditioned budget allocation

Usage:
  python build_task_adaptive_scores.py \
      --profile_csv results/observations/effective_memory_readout_core_top100_n4_o80/effective_readout_scores.csv \
      --raw_csv results/observations/effective_memory_readout_core_top100_n4_o80/raw_effective_readout.csv \
      --save_dir results/observations/tascores
"""

import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

EPS = 1e-12

# ── StreamingBench task definitions ──
MEMORY_HEAVY_TASKS = {"Causal Reasoning", "Clips Summarize", "Event Understanding",
                       "Prospective Reasoning", "Text-Rich Understanding"}
DETAIL_HEAVY_TASKS = {"Counting", "Object Recognition", "Action Recognition"}
BALANCED_TASKS = {"Attribute Recognition", "Spatial Understanding"}

ALL_TASKS = MEMORY_HEAVY_TASKS | DETAIL_HEAVY_TASKS | BALANCED_TASKS


# ═══════════════════════════════════════════════════════════════
# Phase 1: Per-Task Head Profiling
# ═══════════════════════════════════════════════════════════════

def load_raw_with_task(raw_csv_path):
    """Load raw observations with task labels."""
    df = pd.read_csv(raw_csv_path)
    required = {"layer", "kv_head", "task", "internal_readout_share",
                "internal_topk_concentration", "boundary_ratio",
                "current_readout_share"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Raw CSV missing columns: {sorted(missing)}")
    return df


def aggregate_per_task_head(raw_df, metric, num_layers=28, num_kv_heads=4):
    """Aggregate a metric per (task, layer, kv_head).

    Returns dict: task -> np.array[layer, kv_head]
    """
    profiles = {}
    for task in sorted(raw_df["task"].unique()):
        task_df = raw_df[raw_df["task"] == task]
        mat = np.full((num_layers, num_kv_heads), np.nan)
        counts = np.zeros((num_layers, num_kv_heads), dtype=int)
        for (layer, kv_head), group in task_df.groupby(["layer", "kv_head"]):
            vals = group[metric].dropna().values
            if len(vals) > 0:
                mat[layer, kv_head] = float(np.mean(vals))
                counts[layer, kv_head] = len(vals)
        profiles[task] = {"mean": mat, "count": counts}
    return profiles


def bayesian_shrinkage(task_profile, global_profile, min_obs=20):
    """Shrink per-task estimates toward global mean for low-N tasks.

    shrinkage = n_obs / (n_obs + min_obs)
    shrunk = shrinkage * task_mean + (1 - shrinkage) * global_mean
    """
    shrunk = {}
    for task in task_profile:
        n = task_profile[task]["count"]
        shrinkage = n / (n + min_obs)
        shrunk[task] = (
            shrinkage * np.nan_to_num(task_profile[task]["mean"], nan=0.0)
            + (1.0 - shrinkage) * np.nan_to_num(global_profile["mean"], nan=0.0)
        )
    return shrunk


def safe_positive(arr, floor=EPS):
    """Ensure all values are positive with a small floor."""
    arr = np.nan_to_num(np.asarray(arr, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(arr, floor)


# ═══════════════════════════════════════════════════════════════
# Phase 2: Functional Head Clustering
# ═══════════════════════════════════════════════════════════════

def build_head_features(task_profiles, num_layers=28, num_kv_heads=4):
    """Build (L×H, F) feature matrix for clustering.

    task_profiles structure: {metric: {task: {"mean": array[L,H], "count": array[L,H]}}}

    Features per head:
      - mean internal_readout_share across tasks
      - mean topk_concentration across tasks
      - mean boundary_ratio across tasks
      - mean current_readout_share across tasks
      - internal_readout for memory-heavy tasks
      - current_readout for detail-heavy tasks
    """
    features = []
    head_ids = []
    for layer in range(num_layers):
        for kv_head in range(num_kv_heads):
            feats = []
            # Global means across tasks for each metric
            for metric in ["internal_readout_share", "internal_topk_concentration",
                           "boundary_ratio", "current_readout_share"]:
                vals = []
                if metric in task_profiles:
                    for task, prof in task_profiles[metric].items():
                        v = prof["mean"][layer, kv_head]
                        if not np.isnan(v):
                            vals.append(v)
                feats.append(np.mean(vals) if vals else 0.0)

            # Task-grouped means
            internal_metric = task_profiles.get("internal_readout_share", {})
            current_metric = task_profiles.get("current_readout_share", {})

            mem_vals = [internal_metric[t]["mean"][layer, kv_head]
                       for t in MEMORY_HEAVY_TASKS if t in internal_metric
                       and not np.isnan(internal_metric[t]["mean"][layer, kv_head])]
            detail_vals = [current_metric[t]["mean"][layer, kv_head]
                          for t in DETAIL_HEAVY_TASKS if t in current_metric
                          and not np.isnan(current_metric[t]["mean"][layer, kv_head])]

            feats.append(np.nanmean(mem_vals) if mem_vals else 0.0)
            feats.append(np.nanmean(detail_vals) if detail_vals else 0.0)

            features.append(feats)
            head_ids.append((layer, kv_head))

    X = np.array(features, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    # Standardize
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + EPS)
    return X, head_ids


def cluster_heads(X, n_clusters=4, random_state=42):
    """Cluster heads into functional types."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    labels = kmeans.fit_predict(X)
    sil = silhouette_score(X, labels) if n_clusters >= 2 else 0.0
    return labels, kmeans.cluster_centers_, sil


def label_clusters(centroids, feature_names):
    """Auto-label clusters by centroid characteristics.

    Feature indices:
      0: internal_readout_share (global)
      1: internal_topk_concentration (global)
      2: boundary_ratio (global)
      3: current_readout_share (global)
      4: internal_readout (memory tasks)
      5: current_readout (detail tasks)
    """
    labels = []
    for c in centroids:
        internal = c[0]
        topk = c[1]
        boundary = c[2]
        current = c[3]
        mem_internal = c[4]
        detail_current = c[5]

        if internal > 0.3 and topk > 0.2 and boundary < 0.0:
            labels.append("retrieval")
        elif current > 0.3 and detail_current > 0.2 and topk > 0.1:
            labels.append("detail")
        elif boundary > 0.3:
            labels.append("boundary")
        else:
            labels.append("diffuse")
    return labels


# ═══════════════════════════════════════════════════════════════
# Phase 3: Task-Conditioned Product Scoring
# ═══════════════════════════════════════════════════════════════

def compute_utility(task, internal_readout, current_readout):
    """Task-conditioned utility: which memory region matters for this task?"""
    if task in MEMORY_HEAVY_TASKS:
        return internal_readout
    elif task in DETAIL_HEAVY_TASKS:
        return current_readout
    else:  # BALANCED
        return 0.5 * internal_readout + 0.5 * current_readout


def compute_task_scores(task_profiles, topk_conc, boundary_ratio_global,
                        alpha=1.0, beta=0.5, gamma=0.5,
                        num_layers=28, num_kv_heads=4):
    """Compute TAScore for each task.

    score[t, l, h] = utility[t,l,h]^α × selectivity[l,h]^β × boundary_penalty[l,h]^γ

    task_profiles structure: {metric: {task: {"mean": array[L,H], "count": array[L,H]}}}
    """
    internal_metric = task_profiles.get("internal_readout_share", {})
    current_metric = task_profiles.get("current_readout_share", {})

    # Global mean for Bayesian shrinkage
    all_internal_means = [internal_metric[t]["mean"] for t in internal_metric]
    global_internal_mean = np.nanmean(all_internal_means, axis=0) if all_internal_means else np.zeros((num_layers, num_kv_heads))
    all_current_means = [current_metric[t]["mean"] for t in current_metric]
    global_current_mean = np.nanmean(all_current_means, axis=0) if all_current_means else np.zeros((num_layers, num_kv_heads))

    internal_shrunk = bayesian_shrinkage(internal_metric, {"mean": global_internal_mean})
    current_shrunk = bayesian_shrinkage(current_metric, {"mean": global_current_mean})

    selectivity = safe_positive(topk_conc)
    boundary_penalty = safe_positive(1.0 - np.nan_to_num(boundary_ratio_global, nan=0.0))

    task_scores = {}
    all_tasks = sorted(set(internal_metric.keys()) | set(current_metric.keys()))
    for task in all_tasks:
        int_task = internal_shrunk.get(task, global_internal_mean)
        cur_task = current_shrunk.get(task, global_current_mean)
        utility = safe_positive(compute_utility(task, int_task, cur_task))

        score = (utility ** alpha) * (selectivity ** beta) * (boundary_penalty ** gamma)
        task_scores[task] = score

    return task_scores


# ═══════════════════════════════════════════════════════════════
# Phase 4: Within-Layer Contrastive Normalization
# ═══════════════════════════════════════════════════════════════

def contrastive_normalize(task_scores):
    """Normalize scores within each layer to mean=1.0."""
    normalized = {}
    for task, scores in task_scores.items():
        normed = np.zeros_like(scores)
        for layer in range(scores.shape[0]):
            layer_mean = scores[layer, :].mean()
            if layer_mean > EPS:
                normed[layer, :] = scores[layer, :] / layer_mean
            else:
                normed[layer, :] = 1.0
        normalized[task] = normed
    return normalized


# ═══════════════════════════════════════════════════════════════
# Phase 5: Budget Allocation
# ═══════════════════════════════════════════════════════════════

def proportional_allocate(weights, total_budget, min_budget, max_budget):
    """Proportional allocation with min/max constraints and largest remainder."""
    weights = safe_positive(weights)
    n = weights.size
    if n == 0:
        return np.zeros(0, dtype=int)

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


def build_budget_table(norm_scores, num_keep, num_layers=28, num_kv_heads=4,
                       mode="optimistic", task=None, sparsemm_window_size=32,
                       sparsemm_ratio=0.2):
    """Build per-KV-head budget table [L, H] from normalized scores.

    Args:
        norm_scores: dict {task: np.array[L,H]} or np.array[T,L,H]
        mode: 'task_conditioned' | 'optimistic' | 'balanced'
        task: task name (required if mode='task_conditioned')
    """
    if isinstance(norm_scores, dict):
        task_list = sorted(norm_scores.keys())
        score_stack = np.stack([norm_scores[t] for t in task_list])  # (T, L, H)
        task_index = {t: i for i, t in enumerate(task_list)}
    else:
        score_stack = np.asarray(norm_scores)

    if mode == "task_conditioned":
        if task is None or task not in task_index:
            raise ValueError(f"task={task} required for task_conditioned mode")
        weights = score_stack[task_index[task]]
    elif mode == "optimistic":
        weights = score_stack.max(axis=0)
    elif mode == "balanced":
        weights = score_stack.mean(axis=0)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    base_capacity = max(1, int(num_keep) - int(sparsemm_window_size))
    min_cache = int(base_capacity * float(sparsemm_ratio))
    min_budget = max(1, min_cache + int(sparsemm_window_size))
    target_total = int(num_keep) * int(num_layers)

    budgets_flat = proportional_allocate(
        weights.reshape(-1),
        total_budget=target_total,
        min_budget=min_budget,
        max_budget=int(num_keep * 1.5),
    )
    return budgets_flat.reshape(int(num_layers), int(num_kv_heads)).astype(int)


# ═══════════════════════════════════════════════════════════════
# Write outputs
# ═══════════════════════════════════════════════════════════════

def write_score_csv(df_template, scores, out_path, score_name="tascores"):
    """Write a score CSV compatible with hermes_kv_head_budget.py."""
    out = df_template[["layer", "kv_head"]].copy()
    out["score"] = np.asarray(scores.flatten(), dtype=np.float64)
    out[score_name] = out["score"]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"  wrote {out_path}")


def write_head_types(head_ids, cluster_labels, type_names, path):
    """Write head type classification."""
    records = []
    for (layer, kv_head), label, tname in zip(head_ids, cluster_labels, type_names):
        records.append({"layer": layer, "kv_head": kv_head,
                        "cluster_id": int(label), "head_type": tname})
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  wrote {path}")


# ═══════════════════════════════════════════════════════════════
# Control variants (for ablation)
# ═══════════════════════════════════════════════════════════════

def generate_controls(scores, df_template, save_dir, num_layers=28, num_kv_heads=4, seed=42):
    """Generate uniform, inverted, and random control score CSVs."""
    # Uniform
    uniform = np.ones((num_layers, num_kv_heads))
    write_score_csv(df_template, uniform, os.path.join(save_dir, "tascores_uniform.csv"), "uniform")

    # Inverted within each layer
    inv = np.zeros_like(scores)
    for l in range(num_layers):
        vals = scores[l, :]
        inv[l, :] = vals.max() + vals.min() - vals
    inv = safe_positive(inv)
    write_score_csv(df_template, inv, os.path.join(save_dir, "tascores_inverted.csv"), "inverted")

    # Random within each layer
    rng = np.random.default_rng(seed)
    rnd = np.zeros_like(scores)
    for l in range(num_layers):
        rnd[l, :] = rng.permutation(scores[l, :])
    write_score_csv(df_template, rnd, os.path.join(save_dir, f"tascores_random_seed{seed}.csv"), "random")


# ═══════════════════════════════════════════════════════════════
# Ablation: measure contribution of each factor
# ═══════════════════════════════════════════════════════════════

def ablation_scores(task_scores, task_profiles, topk_conc, boundary_ratio_global,
                    num_layers=28, num_kv_heads=4):
    """Generate ablated scores for factor importance measurement."""
    ablation = {}

    internal_metric = task_profiles.get("internal_readout_share", {})
    current_metric = task_profiles.get("current_readout_share", {})
    all_internal_means = [internal_metric[t]["mean"] for t in internal_metric]
    global_internal_mean = np.nanmean(all_internal_means, axis=0) if all_internal_means else np.zeros((num_layers, num_kv_heads))
    all_current_means = [current_metric[t]["mean"] for t in current_metric]
    global_current_mean = np.nanmean(all_current_means, axis=0) if all_current_means else np.zeros((num_layers, num_kv_heads))

    internal_shrunk = bayesian_shrinkage(internal_metric, {"mean": global_internal_mean})
    current_shrunk = bayesian_shrinkage(current_metric, {"mean": global_current_mean})

    # No utility (α=0)
    ablation["no_utility"] = {}
    for task in task_scores:
        selectivity = safe_positive(topk_conc)
        boundary_penalty = safe_positive(1.0 - np.nan_to_num(boundary_ratio_global, nan=0.0))
        ablation["no_utility"][task] = (selectivity ** 0.5) * (boundary_penalty ** 0.5)

    # No selectivity (β=0)
    ablation["no_selectivity"] = {}
    for task in task_scores:
        int_task = internal_shrunk.get(task, global_internal_mean)
        cur_task = current_shrunk.get(task, global_current_mean)
        utility = safe_positive(compute_utility(task, int_task, cur_task))
        boundary_penalty = safe_positive(1.0 - np.nan_to_num(boundary_ratio_global, nan=0.0))
        ablation["no_selectivity"][task] = utility * (boundary_penalty ** 0.5)

    # No boundary penalty (γ=0)
    ablation["no_boundary_penalty"] = {}
    for task in task_scores:
        ablation["no_boundary_penalty"][task] = task_scores[task] / safe_positive(
            1.0 - np.nan_to_num(boundary_ratio_global, nan=0.0)) ** 0.5

    # Utility only (β=0, γ=0)
    ablation["utility_only"] = {}
    for task in task_scores:
        int_task = internal_shrunk.get(task, global_internal_mean)
        cur_task = current_shrunk.get(task, global_current_mean)
        utility = safe_positive(compute_utility(task, int_task, cur_task))
        ablation["utility_only"][task] = utility

    return ablation


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="TAScore: Task-Adaptive Head Specialization Scoring")
    parser.add_argument("--raw_csv",
                        default="results/observations/effective_memory_readout_core_top100_n4_o80/raw_effective_readout.csv")
    parser.add_argument("--score_csv",
                        default="results/observations/effective_memory_readout_core_top100_n4_o80/effective_readout_scores.csv")
    parser.add_argument("--save_dir", default="results/observations/tascores")
    parser.add_argument("--num_layers", type=int, default=28)
    parser.add_argument("--num_kv_heads", type=int, default=4)
    parser.add_argument("--n_clusters", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Utility exponent (default 1.0)")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="Selectivity exponent (default 0.5)")
    parser.add_argument("--gamma", type=float, default=0.5,
                        help="Boundary penalty exponent (default 0.5)")
    parser.add_argument("--min_obs", type=int, default=20,
                        help="Minimum observations for Bayesian shrinkage")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    np.random.seed(args.seed)

    # ── Load data ──
    print("=" * 60)
    print("TAScore: Task-Adaptive Head Specialization Scoring")
    print("=" * 60)

    raw_df = load_raw_with_task(args.raw_csv)
    score_df = pd.read_csv(args.score_csv)
    df_template = score_df[["layer", "kv_head"]].drop_duplicates().sort_values(["layer", "kv_head"])
    print(f"Loaded {len(raw_df)} raw observations, {len(df_template)} heads")

    # ── Phase 1: Per-task profiling ──
    print("\n── Phase 1: Per-Task Head Profiling ──")
    task_profiles = {}
    task_counts = {}
    for metric in ["internal_readout_share", "internal_topk_concentration",
                   "boundary_ratio", "current_readout_share"]:
        task_profiles[metric] = aggregate_per_task_head(
            raw_df, metric, args.num_layers, args.num_kv_heads
        )
        if metric == "internal_readout_share":
            for task, prof in task_profiles[metric].items():
                task_counts[task] = int(prof["count"].sum())

    print(f"Tasks found: {sorted(task_profiles['internal_readout_share'].keys())}")
    for task, count in sorted(task_counts.items(), key=lambda x: -x[1]):
        print(f"  {task:30s}: {count:5d} observations")

    # ── Phase 2: Clustering ──
    print(f"\n── Phase 2: Functional Head Clustering (K={args.n_clusters}) ──")
    X, head_ids = build_head_features(task_profiles, args.num_layers, args.num_kv_heads)
    cluster_labels, centroids, sil_score = cluster_heads(X, args.n_clusters, args.seed)
    feature_names = ["internal_readout", "topk_conc", "boundary_ratio",
                     "current_readout", "mem_internal", "detail_current"]
    type_names = label_clusters(centroids, feature_names)
    print(f"Silhouette score: {sil_score:.4f}")
    for i, tname in enumerate(type_names):
        count = int((cluster_labels == i).sum())
        print(f"  Cluster {i} ({tname:12s}): {count:3d} heads")
    write_head_types(head_ids, cluster_labels, type_names,
                     os.path.join(args.save_dir, "head_types.csv"))

    # ── Phase 3: Product scoring ──
    print(f"\n── Phase 3: Task-Conditioned Product Scoring (α={args.alpha}, β={args.beta}, γ={args.gamma}) ──")
    # Global (task-agnostic) metrics
    topk_conc = np.zeros((args.num_layers, args.num_kv_heads))
    boundary_ratio_global = np.zeros((args.num_layers, args.num_kv_heads))
    for (layer, kv_head), group in raw_df.groupby(["layer", "kv_head"]):
        topk_conc[layer, kv_head] = group["internal_topk_concentration"].mean()
        boundary_ratio_global[layer, kv_head] = group["boundary_ratio"].mean()

    task_scores = compute_task_scores(
        task_profiles, topk_conc, boundary_ratio_global,
        alpha=args.alpha, beta=args.beta, gamma=args.gamma,
        num_layers=args.num_layers, num_kv_heads=args.num_kv_heads,
    )

    # ── Phase 4: Contrastive normalization ──
    print("\n── Phase 4: Within-Layer Contrastive Normalization ──")
    norm_scores = contrastive_normalize(task_scores)

    # ── Phase 5: Budget CSVs ──
    print("\n── Phase 5: Budget Allocation CSVs ──")

    # Task-conditioned
    task_cond_dir = os.path.join(args.save_dir, "task_conditioned")
    os.makedirs(task_cond_dir, exist_ok=True)
    for task in sorted(norm_scores.keys()):
        write_score_csv(df_template, norm_scores[task],
                        os.path.join(task_cond_dir, f"tascores_{task.replace(' ', '_')}.csv"),
                        f"tascores_{task.replace(' ', '_')}")

    # Optimistic ensemble
    optimistic = np.maximum.reduce([norm_scores[t] for t in sorted(norm_scores.keys())])
    write_score_csv(df_template, optimistic,
                    os.path.join(args.save_dir, "tascores_optimistic.csv"),
                    "tascores_optimistic")

    # Balanced ensemble
    balanced = np.mean([norm_scores[t] for t in sorted(norm_scores.keys())], axis=0)
    write_score_csv(df_template, balanced,
                    os.path.join(args.save_dir, "tascores_balanced.csv"),
                    "tascores_balanced")

    # ── Controls ──
    print("\n── Controls ──")
    generate_controls(balanced, df_template, args.save_dir, args.num_layers, args.num_kv_heads, args.seed)

    # ── Ablation ──
    print("\n── Ablation Scores ──")
    ablation = ablation_scores(task_scores, task_profiles, topk_conc, boundary_ratio_global,
                                args.num_layers, args.num_kv_heads)
    ablation_dir = os.path.join(args.save_dir, "ablation")
    os.makedirs(ablation_dir, exist_ok=True)
    for ab_name, ab_scores in ablation.items():
        ab_norm = contrastive_normalize(ab_scores)
        ab_ens = np.maximum.reduce([ab_norm[t] for t in sorted(ab_norm.keys())])
        write_score_csv(df_template, ab_ens,
                        os.path.join(ablation_dir, f"tascores_ablation_{ab_name}.csv"),
                        f"ablation_{ab_name}")

    # ── Summary ──
    summary = {
        "method": "TAScore",
        "formula": "score = utility^α × selectivity^β × boundary_penalty^γ",
        "alpha": args.alpha, "beta": args.beta, "gamma": args.gamma,
        "n_clusters": args.n_clusters, "silhouette_score": float(sil_score),
        "cluster_types": {int(i): name for i, name in enumerate(type_names)},
        "cluster_sizes": {int(i): int((cluster_labels == i).sum()) for i in range(args.n_clusters)},
        "num_tasks": len(task_profiles["internal_readout_share"]),
        "task_observation_counts": task_counts,
        "bayesian_shrinkage_min_obs": args.min_obs,
        "outputs": {
            "task_conditioned_dir": task_cond_dir,
            "optimistic": os.path.join(args.save_dir, "tascores_optimistic.csv"),
            "balanced": os.path.join(args.save_dir, "tascores_balanced.csv"),
            "head_types": os.path.join(args.save_dir, "head_types.csv"),
            "controls": {
                "uniform": os.path.join(args.save_dir, "tascores_uniform.csv"),
                "inverted": os.path.join(args.save_dir, "tascores_inverted.csv"),
                "random": os.path.join(args.save_dir, f"tascores_random_seed{args.seed}.csv"),
            },
            "ablation_dir": ablation_dir,
        },
    }
    summary_path = os.path.join(args.save_dir, "tascores_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Summary written to {summary_path}")
    print("✓ Done.")


if __name__ == "__main__":
    main()
