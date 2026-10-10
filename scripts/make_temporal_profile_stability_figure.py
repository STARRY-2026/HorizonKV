"""Create the paper-style figure for stable head-specific temporal profiles."""

import argparse
import os

import matplotlib.pyplot as plt
from matplotlib import patches
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
]
AGE_LABELS = ["0-4", "4-8", "8-16", "16-32", "32-64", "64-128", "128-256"]
AGE_CENTERS = np.asarray([2, 6, 12, 24, 48, 96, 192], dtype=np.float64)
HEAD_KEYS = ["layer", "kv_head"]
EPS = 1e-12


def profile_matrix(frame, parity=None):
    subset = frame[
        (frame["query_source"] == "pseudo")
        & (frame["bin_type"] == "age_sec")
        & (frame["token_count"] > 0)
        & (frame["bin_label"].isin(AGE_BINS))
    ]
    if parity is not None:
        subset = subset[subset["video_idx"] % 2 == parity]
    matrix = subset.groupby([*HEAD_KEYS, "bin_label"])["reuse_score"].mean().unstack()
    return matrix.reindex(columns=AGE_BINS)


def l1_normalize(values):
    values = np.nan_to_num(values.astype(np.float64), nan=0.0)
    return values / np.maximum(values.sum(axis=1, keepdims=True), EPS)


def column_zscore(values):
    mean = np.nanmean(values, axis=0, keepdims=True)
    std = np.nanstd(values, axis=0, keepdims=True)
    return np.nan_to_num((values - mean) / np.maximum(std, EPS), nan=0.0)


def cosine_matrix(left, right):
    left = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), EPS)
    right = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), EPS)
    return left @ right.T


def choose_representatives(profile, index):
    log_age = np.log2(AGE_CENTERS)
    centroid = profile @ log_age
    order = np.argsort(centroid)
    groups = {
        "Recent-dominant": order[: max(8, len(order) // 5)],
        "Sustained": order[len(order) * 2 // 5 : len(order) * 3 // 5],
        "Long-range": order[-max(8, len(order) // 5) :],
    }
    selected = {}
    for label, candidates in groups.items():
        center = profile[candidates].mean(axis=0)
        distances = np.linalg.norm(profile[candidates] - center, axis=1)
        selected[label] = int(candidates[np.argmin(distances)])
    return selected


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3, width=0.8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.raw_csv)
    split_a = profile_matrix(frame, parity=0)
    split_b = profile_matrix(frame, parity=1)
    combined = profile_matrix(frame)
    common = split_a.index.intersection(split_b.index).intersection(combined.index)
    split_a = split_a.loc[common]
    split_b = split_b.loc[common]
    combined = combined.loc[common]

    a_raw = split_a.to_numpy(dtype=np.float64)
    b_raw = split_b.to_numpy(dtype=np.float64)
    combined_profile = l1_normalize(combined.to_numpy(dtype=np.float64))
    a_profile = l1_normalize(a_raw)
    b_profile = l1_normalize(b_raw)
    a_z = column_zscore(a_raw)
    b_z = column_zscore(b_raw)

    # Ordering is learned on split A only and then frozen for split B.
    order_score = a_profile @ np.log2(AGE_CENTERS)
    order = np.argsort(order_score)
    representatives = choose_representatives(combined_profile, common.tolist())

    similarity = cosine_matrix(a_z, b_z)
    same = np.diag(similarity)
    layers = np.asarray([idx[0] for idx in common])
    heads = np.asarray([idx[1] for idx in common])
    same_layer_other_mask = (layers[:, None] == layers[None, :]) & (
        heads[:, None] != heads[None, :]
    )
    same_layer_other = similarity[same_layer_other_mask]
    nearest_within_layer = []
    for row_idx, layer in enumerate(layers):
        candidates = np.flatnonzero(layers == layer)
        nearest_within_layer.append(candidates[np.argmax(similarity[row_idx, candidates])])
    identification = np.mean(np.asarray(nearest_within_layer) == np.arange(len(common)))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(13.2, 9.2), facecolor="white")
    outer = fig.add_gridspec(
        2,
        3,
        width_ratios=[1, 1, 0.82],
        height_ratios=[1.55, 1],
        left=0.07,
        right=0.96,
        bottom=0.08,
        top=0.92,
        wspace=0.22,
        hspace=0.38,
    )

    limit = float(np.nanpercentile(np.abs(np.concatenate([a_z.ravel(), b_z.ravel()])), 98))
    heat_axes = [fig.add_subplot(outer[0, 0]), fig.add_subplot(outer[0, 1])]
    image = None
    row_position = {original_idx: pos for pos, original_idx in enumerate(order)}
    representative_colors = {
        "Recent-dominant": "#168AAD",
        "Sustained": "#52796F",
        "Long-range": "#D1495B",
    }
    for panel, (ax, values, title) in enumerate(
        zip(heat_axes, [a_z, b_z], ["Video split A", "Video split B"])
    ):
        image = ax.imshow(
            values[order],
            aspect="auto",
            interpolation="nearest",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        ax.set_title(title, pad=7, weight="bold")
        ax.set_xlabel("Token age (seconds)")
        ax.set_xticks(range(len(AGE_LABELS)))
        ax.set_xticklabels(AGE_LABELS, rotation=35, ha="right")
        ax.set_yticks([])
        if panel == 0:
            ax.set_ylabel("112 physical KV heads\n(same order in both splits)")
        for label, original_idx in representatives.items():
            y = row_position[original_idx]
            ax.add_patch(
                patches.Rectangle(
                    (-0.49, y - 0.49),
                    len(AGE_BINS) - 0.02,
                    0.98,
                    fill=False,
                    linewidth=1.2,
                    edgecolor=representative_colors[label],
                    clip_on=False,
                )
            )
            if panel == 0:
                layer, head = common[original_idx]
                ax.text(
                    len(AGE_BINS) - 0.62,
                    y,
                    f"L{layer}H{head}",
                    ha="right",
                    va="center",
                    fontsize=7,
                    color=representative_colors[label],
                    weight="bold",
                    bbox={
                        "boxstyle": "square,pad=0.12",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.82,
                    },
                )

    colorbar = fig.colorbar(image, ax=heat_axes, fraction=0.025, pad=0.025)
    colorbar.ax.set_title("Relative\nreadout\n(z-score)", fontsize=7, pad=5)
    heat_axes[0].text(-0.18, 1.04, "a", transform=heat_axes[0].transAxes, fontsize=14, weight="bold")

    curve_grid = outer[1, :2].subgridspec(1, 3, wspace=0.32)
    curve_axes = []
    for axis_idx, (label, original_idx) in enumerate(representatives.items()):
        ax = fig.add_subplot(curve_grid[0, axis_idx])
        curve_axes.append(ax)
        color = representative_colors[label]
        layer, head = common[original_idx]
        ax.plot(
            range(len(AGE_BINS)),
            a_profile[original_idx],
            color=color,
            marker="o",
            markersize=3.5,
            linewidth=1.8,
            label="Split A",
        )
        ax.plot(
            range(len(AGE_BINS)),
            b_profile[original_idx],
            color=color,
            marker="o",
            markersize=3,
            linewidth=1.5,
            linestyle="--",
            alpha=0.8,
            label="Split B",
        )
        ax.set_title(f"{label}\nL{layer}H{head}", color=color, weight="bold")
        ax.set_xticks(range(len(AGE_LABELS)))
        ax.set_xticklabels(AGE_LABELS, rotation=45, ha="right")
        ax.set_xlabel("Token age (s)")
        if axis_idx == 0:
            ax.set_ylabel("Normalized readout")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#D9DEE3", linewidth=0.6, alpha=0.8)
        style_axis(ax)
    curve_axes[0].text(-0.35, 1.13, "b", transform=curve_axes[0].transAxes, fontsize=14, weight="bold")
    curve_axes[-1].legend(frameon=False, loc="upper right")

    stats_grid = outer[:, 2].subgridspec(2, 1, height_ratios=[1.25, 0.75], hspace=0.42)
    ax_similarity = fig.add_subplot(stats_grid[0, 0])
    violin = ax_similarity.violinplot(
        [same_layer_other, same],
        positions=[0, 1],
        widths=0.72,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    for body, color in zip(violin["bodies"], ["#AAB4BE", "#2F6690"]):
        body.set_facecolor(color)
        body.set_edgecolor("white")
        body.set_alpha(0.9)
    violin["cmedians"].set_color("#20262E")
    violin["cmedians"].set_linewidth(1.5)
    rng = np.random.default_rng(4)
    sample_other = rng.choice(same_layer_other, size=min(180, len(same_layer_other)), replace=False)
    ax_similarity.scatter(
        rng.normal(0, 0.045, len(sample_other)),
        sample_other,
        s=6,
        color="#65727E",
        alpha=0.25,
        linewidth=0,
    )
    ax_similarity.scatter(
        rng.normal(1, 0.045, len(same)),
        same,
        s=7,
        color="#184E77",
        alpha=0.32,
        linewidth=0,
    )
    ax_similarity.axhline(0, color="#9AA4AE", linewidth=0.7)
    ax_similarity.set_xticks([0, 1])
    ax_similarity.set_xticklabels(["Other heads\nin same layer", "Same physical\nhead"])
    ax_similarity.set_ylabel("Cross-split profile similarity")
    ax_similarity.set_title("Same head, stable fingerprint", weight="bold")
    ax_similarity.text(
        0.98,
        0.04,
        f"mean: {same.mean():.3f} vs. {same_layer_other.mean():.3f}\npermutation $p < 10^{{-4}}$",
        transform=ax_similarity.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    style_axis(ax_similarity)
    ax_similarity.text(-0.2, 1.06, "c", transform=ax_similarity.transAxes, fontsize=14, weight="bold")

    ax_id = fig.add_subplot(stats_grid[1, 0])
    bars = ax_id.bar(
        [0, 1],
        [0.25, identification],
        color=["#C4CBD2", "#2F6690"],
        width=0.62,
    )
    ax_id.set_ylim(0, 1)
    ax_id.set_ylabel("Identification accuracy")
    ax_id.set_xticks([0, 1])
    ax_id.set_xticklabels(["Random", "Temporal\nprofile"])
    ax_id.set_title("Identify a head among 4\nheads in the same layer", weight="bold")
    for bar, value in zip(bars, [0.25, identification]):
        ax_id.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.035,
            f"{value * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )
    ax_id.grid(axis="y", color="#D9DEE3", linewidth=0.6, alpha=0.8)
    style_axis(ax_id)

    fig.suptitle(
        "Physical KV heads exhibit stable and distinct temporal profiles",
        x=0.5,
        y=0.975,
        fontsize=15,
        weight="bold",
    )
    os.makedirs(args.output_dir, exist_ok=True)
    png_path = os.path.join(args.output_dir, "stable_head_temporal_profiles.png")
    pdf_path = os.path.join(args.output_dir, "stable_head_temporal_profiles.pdf")
    fig.savefig(png_path, dpi=300, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
    print(png_path)
    print(pdf_path)
    print(
        {
            "same_head_mean": float(same.mean()),
            "same_layer_other_mean": float(same_layer_other.mean()),
            "identification_accuracy": float(identification),
            "representatives": {
                label: f"L{common[idx][0]}H{common[idx][1]}"
                for label, idx in representatives.items()
            },
        }
    )


if __name__ == "__main__":
    main()
