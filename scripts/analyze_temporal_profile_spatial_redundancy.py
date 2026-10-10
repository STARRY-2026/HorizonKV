"""HeteroCache-style intralayer redundancy audit for the 8-bin temporal profiles.

The eight-dim temporal profile of a physical KV head is the mean readout over
eight token-age bins (0-4s ... 256s+). HeteroCache's second insight is that
attention heads within the same layer are highly redundant (intralayer
similarity >> interlayer similarity). This script checks whether the same
"spatial" (intralayer) redundancy exists in the *temporal-profile* space: are
the four KV heads inside a layer near-duplicates in their temporal preferences?

Two representations are compared for each metric:
  * raw / L1 profile  : overall shape, dominated by the shared recency baseline
  * within-bin zscore : distinctive temporal preference beyond that baseline
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


AGE_BINS = [
    "0-4s",
    "4-8s",
    "8-16s",
    "16-32s",
    "32-64s",
    "64-128s",
    "128-256s",
    "256s+",
]
HEAD_KEYS = ["layer", "kv_head"]
METRICS = ["reuse_score", "top100_mean_attention"]
EPS = 1e-12


def load_profiles(raw_csv):
    columns = [
        "query_source",
        "bin_type",
        "bin_label",
        "token_count",
        "layer",
        "kv_head",
        *METRICS,
    ]
    frame = pd.read_csv(raw_csv, usecols=columns)
    frame = frame[
        (frame["query_source"] == "real")
        & (frame["bin_type"] == "age_sec")
        & (frame["token_count"] > 0)
        & frame["bin_label"].isin(AGE_BINS)
    ].copy()

    profiles = {}
    for metric in METRICS:
        matrix = (
            frame.groupby([*HEAD_KEYS, "bin_label"])[metric]
            .mean()
            .unstack()
            .reindex(columns=AGE_BINS)
        )
        profiles[metric] = {
            "index": matrix.index.tolist(),
            "raw": matrix.to_numpy(dtype=np.float64),
        }
    return profiles


def l1_normalize(values):
    values = np.nan_to_num(values, nan=0.0)
    return values / np.maximum(np.abs(values).sum(axis=1, keepdims=True), EPS)


def column_zscore(values):
    mean = np.nanmean(values, axis=0, keepdims=True)
    std = np.nanstd(values, axis=0, keepdims=True)
    return np.nan_to_num((values - mean) / np.maximum(std, EPS), nan=0.0)


def cosine_matrix(left, right=None):
    right = left if right is None else right
    left = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), EPS)
    right = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), EPS)
    return left @ right.T


def topk_overlap_matrix(values, k):
    order = np.argsort(-values, axis=1, kind="mergesort")[:, :k]
    n = values.shape[0]
    overlap = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        si = set(order[i].tolist())
        for j in range(i, n):
            sj = set(order[j].tolist())
            overlap[i, j] = overlap[j, i] = len(si & sj) / min(k, k)
    return overlap


def intra_inter_stats(sim, layers, heads):
    same_layer = layers[:, None] == layers[None, :]
    off_diag = ~np.eye(sim.shape[0], dtype=bool)
    intra = sim[same_layer & off_diag]
    inter = sim[~same_layer]

    # HeteroCache S_sim analog: per head, max similarity to another head in the
    # same layer (they take the median over steps; we only have one profile).
    s_sim = np.full(sim.shape[0], np.nan)
    for i in range(sim.shape[0]):
        cand = np.flatnonzero(same_layer[i] & off_diag[i])
        s_sim[i] = sim[i, cand].max() if len(cand) else np.nan

    return {
        "intra_mean": float(intra.mean()) if intra.size else float("nan"),
        "intra_median": float(np.median(intra)) if intra.size else float("nan"),
        "inter_mean": float(inter.mean()) if inter.size else float("nan"),
        "inter_median": float(np.median(inter)) if inter.size else float("nan"),
        "intra_minus_inter": float(intra.mean() - inter.mean()) if intra.size else float("nan"),
        "s_sim_mean": float(np.nanmean(s_sim)),
        "s_sim_median": float(np.nanmedian(s_sim)),
        "s_sim_above_0.8": float(np.nanmean(s_sim > 0.8)),
        "s_sim_above_0.9": float(np.nanmean(s_sim > 0.9)),
        "s_sim_above_0.95": float(np.nanmean(s_sim > 0.95)),
    }


def greedy_star_clusters(sim, layers, threshold):
    """Greedy star clustering per layer, as in HeteroCache.

    Returns total number of clusters (pivots) and the satellite fraction.
    """
    n = sim.shape[0]
    visited = np.zeros(n, dtype=bool)
    n_clusters = 0
    for i in range(n):
        if visited[i]:
            continue
        members = [i]
        visited[i] = True
        same_layer = np.flatnonzero(layers == layers[i])
        for j in same_layer:
            if j == i or visited[j]:
                continue
            if all(sim[j, m] >= threshold for m in members):
                members.append(j)
                visited[j] = True
        n_clusters += 1
    return {
        "threshold": threshold,
        "num_clusters": n_clusters,
        "num_heads": n,
        "satellite_fraction": float((n - n_clusters) / max(n, 1)),
    }


def summarize(profiles):
    index = profiles["reuse_score"]["index"]
    layers = np.asarray([item[0] for item in index])
    heads = np.asarray([item[1] for item in index])

    result = {"num_heads": len(index), "num_layers": int(np.max(layers) + 1)}
    result["per_layer"] = {}
    for metric in METRICS:
        raw = profiles[metric]["raw"]
        representations = {
            "raw": raw,
            "l1": l1_normalize(raw),
            "z": column_zscore(raw),
        }
        result[metric] = {}
        result["per_layer"][metric] = {}
        for name, values in representations.items():
            sim = cosine_matrix(values)
            result[metric][name] = intra_inter_stats(sim, layers, heads)
            result[metric][name]["clusters"] = {
                "tau_0.7": greedy_star_clusters(sim, layers, 0.7),
                "tau_0.8": greedy_star_clusters(sim, layers, 0.8),
                "tau_0.9": greedy_star_clusters(sim, layers, 0.9),
            }
            for k in (2, 3):
                overlap = topk_overlap_matrix(values, k)
                ov = intra_inter_stats(overlap, layers, heads)
                result[metric][name][f"top{k}_overlap_intra_mean"] = ov["intra_mean"]
                result[metric][name][f"top{k}_overlap_inter_mean"] = ov["inter_mean"]
            if name == "z":
                per_layer = {}
                for layer in np.unique(layers):
                    idx = np.flatnonzero(layers == layer)
                    block = sim[np.ix_(idx, idx)]
                    iu = np.triu_indices(len(idx), k=1)
                    pair_values = block[iu]
                    per_layer[int(layer)] = {
                        "n_heads": int(len(idx)),
                        "pair_cos_min": float(pair_values.min()),
                        "pair_cos_mean": float(pair_values.mean()),
                        "pair_cos_max": float(pair_values.max()),
                        "all_pairs_above_0.8": bool(np.all(pair_values > 0.8)),
                        "all_pairs_above_0.9": bool(np.all(pair_values > 0.9)),
                    }
                result["per_layer"][metric]["z"] = per_layer
        result["per_layer"][metric]["fully_redundant_layers_0.8"] = int(
            np.sum(
                [
                    v["all_pairs_above_0.8"]
                    for v in result["per_layer"][metric]["z"].values()
                ]
            )
        )
        result["per_layer"][metric]["fully_redundant_layers_0.9"] = int(
            np.sum(
                [
                    v["all_pairs_above_0.9"]
                    for v in result["per_layer"][metric]["z"].values()
                ]
            )
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen_root", default="results/myobservations/temporal_topk_profile_50v_fixed448")
    parser.add_argument("--llava_root", default="results/myobservations/temporal_profile_llavaov7b_same50")
    parser.add_argument("--output_dir", default="results/myobservations/temporal_profile_spatial_redundancy")
    args = parser.parse_args()

    roots = {
        "Qwen2.5-VL-7B": Path(args.qwen_root),
        "LLaVA-OV-7B": Path(args.llava_root),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_result = {}
    for model, root in roots.items():
        profiles = load_profiles(root / "raw_age_bin_attention.csv")
        all_result[model] = summarize(profiles)

    (output_dir / "spatial_redundancy_summary.json").write_text(
        json.dumps(all_result, indent=2) + "\n"
    )
    print(json.dumps(all_result, indent=2))


if __name__ == "__main__":
    main()
