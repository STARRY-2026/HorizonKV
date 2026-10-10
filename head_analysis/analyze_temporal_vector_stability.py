"""Test whether multi-bin temporal profiles are stable and head-specific."""

import argparse
import json
import os

import matplotlib.pyplot as plt
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
EPS = 1e-12


def build_matrix(frame, source, metric, parity):
    subset = frame[
        (frame["query_source"] == source)
        & (frame["video_idx"] % 2 == parity)
        & (frame["bin_label"].isin(AGE_BINS))
        & (frame["token_count"] > 0)
    ]
    matrix = subset.groupby([*HEAD_KEYS, "bin_label"])[metric].mean().unstack()
    return matrix.reindex(columns=AGE_BINS)


def row_l1(matrix):
    values = matrix.to_numpy(dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0)
    denom = np.abs(values).sum(axis=1, keepdims=True)
    return values / np.maximum(denom, EPS)


def column_zscore(matrix):
    values = matrix.to_numpy(dtype=np.float64)
    means = np.nanmean(values, axis=0, keepdims=True)
    stds = np.nanstd(values, axis=0, keepdims=True)
    return np.nan_to_num((values - means) / np.maximum(stds, EPS), nan=0.0)


def cosine_matrix(left, right):
    left_norm = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), EPS)
    right_norm = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), EPS)
    return left_norm @ right_norm.T


def spearman_values(left, right):
    left_rank = pd.Series(left).rank().to_numpy()
    right_rank = pd.Series(right).rank().to_numpy()
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def permutation_pvalue(similarity, trials=10000, seed=0):
    rng = np.random.default_rng(seed)
    observed = float(np.diag(similarity).mean())
    exceed = 0
    for _ in range(trials):
        permuted = rng.permutation(similarity.shape[0])
        value = float(similarity[np.arange(similarity.shape[0]), permuted].mean())
        exceed += value >= observed
    return (exceed + 1) / (trials + 1)


def summarize(left, right, index):
    similarity = cosine_matrix(left, right)
    same = np.diag(similarity)
    off_diagonal = similarity[~np.eye(similarity.shape[0], dtype=bool)]
    layers = np.asarray([item[0] for item in index])
    kv_heads = np.asarray([item[1] for item in index])
    same_layer_other_mask = (layers[:, None] == layers[None, :]) & (
        kv_heads[:, None] != kv_heads[None, :]
    )
    same_layer_other = similarity[same_layer_other_mask]
    nearest = similarity.argmax(axis=1)
    nearest_within_layer = []
    for row_idx, layer in enumerate(layers):
        candidates = np.flatnonzero(layers == layer)
        nearest_within_layer.append(candidates[np.argmax(similarity[row_idx, candidates])])
    pair_left = cosine_matrix(left, left)
    pair_right = cosine_matrix(right, right)
    upper = np.triu_indices(left.shape[0], k=1)
    return {
        "same_head_cosine_mean": float(same.mean()),
        "same_head_cosine_median": float(np.median(same)),
        "different_head_cosine_mean": float(off_diagonal.mean()),
        "different_head_cosine_median": float(np.median(off_diagonal)),
        "same_minus_different_mean": float(same.mean() - off_diagonal.mean()),
        "same_layer_other_cosine_mean": float(same_layer_other.mean()),
        "same_layer_other_cosine_median": float(np.median(same_layer_other)),
        "same_minus_same_layer_other_mean": float(
            same.mean() - same_layer_other.mean()
        ),
        "same_head_above_different_95pct_fraction": float(
            np.mean(same > np.percentile(off_diagonal, 95))
        ),
        "nearest_head_identification_accuracy": float(
            np.mean(nearest == np.arange(len(nearest)))
        ),
        "nearest_within_layer_identification_accuracy": float(
            np.mean(np.asarray(nearest_within_layer) == np.arange(len(nearest)))
        ),
        "pairwise_geometry_spearman": spearman_values(
            pair_left[upper], pair_right[upper]
        ),
        "permutation_pvalue": float(permutation_pvalue(similarity)),
    }


def plot_split_heatmaps(left, right, index, output_path, title):
    average = (left + right) / 2.0
    order = np.argsort(np.argmax(average, axis=1) + 0.01 * average[:, -1])
    limit = float(np.percentile(np.abs(np.concatenate([left.ravel(), right.ravel()])), 98))
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 8.0), sharey=True)
    for ax, values, name in zip(axes, [left, right], ["Video split A", "Video split B"]):
        image = ax.imshow(values[order], cmap="RdBu_r", aspect="auto", vmin=-limit, vmax=limit)
        ax.set_title(name)
        ax.set_xlabel("Token age")
        ax.set_xticks(range(len(AGE_BINS)))
        ax.set_xticklabels(AGE_BINS, rotation=45, ha="right")
    axes[0].set_ylabel("Physical KV heads (same order)")
    labels = [f"L{index[i][0]}H{index[i][1]}" for i in order]
    ticks = np.arange(0, len(labels), 10)
    axes[0].set_yticks(ticks)
    axes[0].set_yticklabels([labels[i] for i in ticks], fontsize=7)
    fig.colorbar(image, ax=axes, pad=0.02, label="Within-bin standardized readout")
    fig.suptitle(title)
    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.12, top=0.92, wspace=0.08)
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_csv", required=True)
    parser.add_argument("--save_dir", required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.raw_csv)
    frame = frame[frame["bin_type"] == "age_sec"]
    os.makedirs(args.save_dir, exist_ok=True)
    summary = {"age_bins": AGE_BINS, "num_heads": 112, "sources": {}}

    for source in ["pseudo", "real"]:
        source_results = {}
        for metric in ["reuse_score", "top100_mean_attention"]:
            left_frame = build_matrix(frame, source, metric, parity=0)
            right_frame = build_matrix(frame, source, metric, parity=1)
            common = left_frame.index.intersection(right_frame.index)
            left_frame = left_frame.loc[common]
            right_frame = right_frame.loc[common]

            l1_result = summarize(
                row_l1(left_frame), row_l1(right_frame), common.tolist()
            )
            left_z = column_zscore(left_frame)
            right_z = column_zscore(right_frame)
            z_result = summarize(left_z, right_z, common.tolist())
            source_results[metric] = {
                "l1_normalized_profile": l1_result,
                "within_bin_head_zscore_profile": z_result,
            }
            plot_split_heatmaps(
                left_z,
                right_z,
                common.tolist(),
                os.path.join(args.save_dir, f"{source}_{metric}_split_heatmaps.png"),
                f"{source.capitalize()} queries: {metric}",
            )
        summary["sources"][source] = source_results

    with open(os.path.join(args.save_dir, "vector_stability_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
