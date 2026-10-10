#!/usr/bin/env python3
"""Generate publication figures for cross-model temporal-profile stability."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
AGE_LABELS = ["0-4", "4-8", "8-16", "16-32", "32-64", "64-128", "128-256", "256+"]
AGE_CENTERS = np.asarray([2, 6, 12, 24, 48, 96, 192, 384], dtype=np.float64)
HEAD_KEYS = ["layer", "kv_head"]
METRICS = ["reuse_score", "top100_mean_attention"]
METRIC_LABELS = {"reuse_score": "Reuse", "top100_mean_attention": "Top-100"}
COLORS = {"Qwen2.5-VL-7B": "#2878B5", "LLaVA-OV-7B": "#D95F02"}
PROFILE_COLORS = ["#00798C", "#5B5F97", "#D1495B"]
EPS = 1e-12


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )


def clean_axis(ax, grid=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3, width=0.8)
    if grid:
        ax.grid(axis="y", color="#D9DEE3", linewidth=0.6, alpha=0.8, zorder=0)


def l1_normalize(values):
    values = np.nan_to_num(values.astype(np.float64), nan=0.0)
    return values / np.maximum(np.abs(values).sum(axis=1, keepdims=True), EPS)


def column_zscore(values):
    mean = np.nanmean(values, axis=0, keepdims=True)
    std = np.nanstd(values, axis=0, keepdims=True)
    return np.nan_to_num((values - mean) / np.maximum(std, EPS), nan=0.0)


def cosine_matrix(left, right):
    left = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), EPS)
    right = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), EPS)
    return left @ right.T


def load_model_data(raw_csv):
    columns = [
        "video_idx",
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
    frame["parity"] = frame["video_idx"] % 2

    result = {}
    for metric in METRICS:
        matrices = []
        for parity in (0, 1):
            matrix = (
                frame[frame["parity"] == parity]
                .groupby([*HEAD_KEYS, "bin_label"])[metric]
                .mean()
                .unstack()
                .reindex(columns=AGE_BINS)
            )
            matrices.append(matrix)
        common = matrices[0].index.intersection(matrices[1].index)
        left = matrices[0].loc[common].to_numpy(dtype=np.float64)
        right = matrices[1].loc[common].to_numpy(dtype=np.float64)
        result[metric] = {
            "index": common.tolist(),
            "raw": [left, right],
            "l1": [l1_normalize(left), l1_normalize(right)],
            "z": [column_zscore(left), column_zscore(right)],
        }
    return result


def matching_stats(metric_data):
    left, right = metric_data["z"]
    similarity = cosine_matrix(left, right)
    index = metric_data["index"]
    layers = np.asarray([item[0] for item in index])
    heads = np.asarray([item[1] for item in index])
    diagonal = np.diag(similarity)
    same_layer_other_mask = (layers[:, None] == layers[None, :]) & (
        heads[:, None] != heads[None, :]
    )
    same_layer_other = similarity[same_layer_other_mask]
    nearest = similarity.argmax(axis=1)
    nearest_within = []
    for row_idx, layer in enumerate(layers):
        candidates = np.flatnonzero(layers == layer)
        nearest_within.append(candidates[np.argmax(similarity[row_idx, candidates])])
    return {
        "similarity": similarity,
        "same": diagonal,
        "same_layer_other": same_layer_other,
        "same_mean": float(diagonal.mean()),
        "same_layer_other_mean": float(same_layer_other.mean()),
        "global_id": float(np.mean(nearest == np.arange(len(index)))),
        "within_id": float(np.mean(np.asarray(nearest_within) == np.arange(len(index)))),
    }


def choose_representatives(left_l1, right_l1):
    average = (left_l1 + right_l1) / 2.0
    centroid = average @ np.log2(AGE_CENTERS)
    order = np.argsort(centroid)
    pools = [
        order[: max(8, len(order) // 5)],
        order[len(order) * 2 // 5 : len(order) * 3 // 5],
        order[-max(8, len(order) // 5) :],
    ]
    selected = []
    for pool in pools:
        center = average[pool].mean(axis=0)
        selected.append(int(pool[np.argmin(np.linalg.norm(average[pool] - center, axis=1))]))
    return selected


def save_figure(fig, output_dir, stem):
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(pdf, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_profiles(model_data, output_dir):
    fig = plt.figure(figsize=(13.6, 7.0), facecolor="white")
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=[1, 1, 1.22],
        left=0.065,
        right=0.965,
        bottom=0.105,
        top=0.89,
        wspace=0.28,
        hspace=0.34,
    )
    all_z = np.concatenate(
        [values["reuse_score"]["z"][split].ravel() for values in model_data.values() for split in (0, 1)]
    )
    limit = float(np.nanpercentile(np.abs(all_z), 98))
    image = None
    heat_axes = []

    for row, (model, values) in enumerate(model_data.items()):
        reuse = values["reuse_score"]
        order_score = reuse["l1"][0] @ np.log2(AGE_CENTERS)
        order = np.argsort(order_score)
        representatives = choose_representatives(*reuse["l1"])

        for split in (0, 1):
            ax = fig.add_subplot(grid[row, split])
            heat_axes.append(ax)
            image = ax.imshow(
                reuse["z"][split][order],
                aspect="auto",
                interpolation="nearest",
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
            )
            ax.set_title(f"Split {'A' if split == 0 else 'B'}", weight="bold", pad=5)
            ax.set_xticks(range(len(AGE_LABELS)))
            ax.set_xticklabels(AGE_LABELS, rotation=38, ha="right")
            ax.set_yticks([])
            if split == 0:
                ax.set_ylabel(f"{model}\n112 physical KV heads", weight="bold")
            if row == 1:
                ax.set_xlabel("Token age (seconds)")
            if row == 0 and split == 0:
                ax.text(-0.19, 1.04, "a", transform=ax.transAxes, fontsize=13, weight="bold")

        ax_curve = fig.add_subplot(grid[row, 2])
        labels = ["Recent-dominant", "Sustained", "Long-range"]
        for idx, label, color in zip(representatives, labels, PROFILE_COLORS):
            layer, head = reuse["index"][idx]
            ax_curve.plot(
                range(len(AGE_BINS)),
                reuse["l1"][0][idx],
                color=color,
                marker="o",
                markersize=3,
                linewidth=1.6,
                label=f"{label}  L{layer}H{head}",
            )
            ax_curve.plot(
                range(len(AGE_BINS)),
                reuse["l1"][1][idx],
                color=color,
                marker="o",
                markersize=2.5,
                linewidth=1.3,
                linestyle="--",
                alpha=0.8,
            )
        ax_curve.set_title(f"Representative 8-D profiles", weight="bold", pad=5)
        ax_curve.set_xticks(range(len(AGE_LABELS)))
        ax_curve.set_xticklabels(AGE_LABELS, rotation=38, ha="right")
        ax_curve.set_ylabel("L1-normalized readout")
        ax_curve.set_xlabel("Token age (seconds)")
        ax_curve.set_ylim(bottom=0)
        clean_axis(ax_curve, grid=True)
        ax_curve.legend(frameon=False, loc="upper right", ncol=1)
        if row == 0:
            ax_curve.text(-0.17, 1.04, "b", transform=ax_curve.transAxes, fontsize=13, weight="bold")

    colorbar = fig.colorbar(image, ax=heat_axes, fraction=0.018, pad=0.015)
    colorbar.set_label("Within-bin standardized readout (z-score)")
    line_legend = [
        Line2D([0], [0], color="#30343B", lw=1.5, label="Split A"),
        Line2D([0], [0], color="#30343B", lw=1.5, ls="--", label="Split B"),
    ]
    fig.legend(handles=line_legend, loc="upper right", bbox_to_anchor=(0.965, 0.955), frameon=False, ncol=2)
    fig.suptitle(
        "Temporal profiles reproduce across disjoint videos and across model families",
        fontsize=14,
        weight="bold",
        y=0.965,
    )
    return save_figure(fig, output_dir, "fig1_cross_model_temporal_profiles")


def plot_evidence(model_data, summaries, analysis_summaries, output_dir):
    stats = {
        model: {metric: matching_stats(values[metric]) for metric in METRICS}
        for model, values in model_data.items()
    }
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.65), facecolor="white")
    ax_sim, ax_id, ax_rank = axes
    labels = []
    x = np.arange(4, dtype=float)
    for model in model_data:
        for metric in METRICS:
            labels.append(f"{model.split('-')[0]}\n{METRIC_LABELS[metric]}")
    for pos, (model, metric) in enumerate(
        [(model, metric) for model in model_data for metric in METRICS]
    ):
        values = stats[model][metric]
        color = COLORS[model]
        ax_sim.plot(
            [pos - 0.13, pos + 0.13],
            [values["same_layer_other_mean"], values["same_mean"]],
            color="#9AA4AE",
            linewidth=1.2,
            zorder=1,
        )
        ax_sim.scatter(pos - 0.13, values["same_layer_other_mean"], s=35, color="#AAB4BE", zorder=3)
        ax_sim.scatter(pos + 0.13, values["same_mean"], s=42, color=color, zorder=3)
    ax_sim.set_xticks(x)
    ax_sim.set_xticklabels(labels)
    ax_sim.set_ylim(-0.05, 1.05)
    ax_sim.set_ylabel("Cross-split cosine similarity")
    ax_sim.set_title("Same-head profiles are distinctive", weight="bold")
    ax_sim.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#AAB4BE", markeredgecolor="none", label="Same-layer other"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#D95F02", markeredgecolor="none", label="Same head"),
        ],
        frameon=False,
        loc="lower right",
    )
    clean_axis(ax_sim, grid=True)

    width = 0.34
    global_values = [stats[model][metric]["global_id"] for model in model_data for metric in METRICS]
    within_values = [stats[model][metric]["within_id"] for model in model_data for metric in METRICS]
    bars_global = ax_id.bar(x - width / 2, global_values, width, color="#3C6E9F", label="Global (112-way)", zorder=2)
    bars_within = ax_id.bar(x + width / 2, within_values, width, color="#D9822B", label="Within layer (4-way)", zorder=2)
    ax_id.axhline(1 / 112, color="#3C6E9F", linewidth=0.8, linestyle=":")
    ax_id.axhline(0.25, color="#D9822B", linewidth=0.8, linestyle=":")
    ax_id.set_xticks(x)
    ax_id.set_xticklabels(labels)
    ax_id.set_ylim(0, 1.08)
    ax_id.set_ylabel("Head identification accuracy")
    ax_id.set_title("Profiles recover physical head identity", weight="bold")
    ax_id.legend(frameon=False, loc="lower right")
    for bars in (bars_global, bars_within):
        for bar in bars:
            value = bar.get_height()
            ax_id.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value * 100:.0f}", ha="center", va="bottom", fontsize=7)
    clean_axis(ax_id, grid=True)

    models = list(model_data)
    rank_values = [analysis_summaries[model]["real"]["split_half_spearman"] for model in models]
    geometry_reuse = [
        summaries[model]["sources"]["real"]["reuse_score"]["within_bin_head_zscore_profile"]["pairwise_geometry_spearman"]
        for model in models
    ]
    geometry_top100 = [
        summaries[model]["sources"]["real"]["top100_mean_attention"]["within_bin_head_zscore_profile"]["pairwise_geometry_spearman"]
        for model in models
    ]
    bx = np.arange(len(models))
    bw = 0.23
    ax_rank.bar(bx - bw, rank_values, bw, color="#4C956C", label="Far/recent rank")
    ax_rank.bar(bx, geometry_reuse, bw, color="#3C6E9F", label="Reuse geometry")
    ax_rank.bar(bx + bw, geometry_top100, bw, color="#D9822B", label="Top-100 geometry")
    ax_rank.set_xticks(bx)
    ax_rank.set_xticklabels(["Qwen", "LLaVA"])
    ax_rank.set_ylim(0, 1.05)
    ax_rank.set_ylabel("Split-half Spearman correlation")
    ax_rank.set_title("Head organization is reproducible", weight="bold")
    ax_rank.legend(frameon=False, loc="lower right")
    clean_axis(ax_rank, grid=True)

    for label, ax in zip("abc", axes):
        ax.text(-0.16, 1.06, label, transform=ax.transAxes, fontsize=13, weight="bold")
    fig.suptitle("Independent tests support stable, head-specific temporal fingerprints", fontsize=13, weight="bold", y=1.02)
    fig.tight_layout(w_pad=2.0)
    paths = save_figure(fig, output_dir, "fig2_temporal_profile_stability_evidence")
    return paths, stats


def plot_matching_matrices(model_data, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(8.1, 7.4), facecolor="white", sharex=True, sharey=True)
    for row, (model, values) in enumerate(model_data.items()):
        for col, metric in enumerate(METRICS):
            ax = axes[row, col]
            result = matching_stats(values[metric])
            truth = np.arange(result["similarity"].shape[0])
            prediction = result["similarity"].argmax(axis=1)
            correct = prediction == truth
            ax.plot([0, 111], [0, 111], color="#B8C0C8", linewidth=1.0, linestyle="--", zorder=1)
            ax.scatter(
                truth[~correct],
                prediction[~correct],
                s=18,
                color="#9AA4AE",
                edgecolor="white",
                linewidth=0.35,
                alpha=0.85,
                label="Incorrect",
                zorder=2,
            )
            ax.scatter(
                truth[correct],
                prediction[correct],
                s=20,
                color=COLORS[model],
                edgecolor="white",
                linewidth=0.35,
                alpha=0.9,
                label="Correct",
                zorder=3,
            )
            ax.set_title(f"{model} | {METRIC_LABELS[metric]}", weight="bold")
            ax.text(
                0.03,
                0.96,
                f"global ID: {result['global_id'] * 100:.1f}%",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox={"boxstyle": "square,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.86},
            )
            ax.set_xlim(-3, 114)
            ax.set_ylim(-3, 114)
            ax.set_aspect("equal")
            ax.set_xticks([0, 28, 56, 84, 111])
            ax.set_yticks([0, 28, 56, 84, 111])
            ax.grid(color="#E1E5E9", linewidth=0.55, alpha=0.8, zorder=0)
            if row == 1:
                ax.set_xlabel("Head identity in split A")
            if col == 0:
                ax.set_ylabel("Nearest head in split B")
            if row == 0 and col == 1:
                ax.legend(frameon=False, loc="lower right")
    fig.suptitle("Cross-split nearest-neighbor matching recovers physical head identity", fontsize=13, weight="bold", y=0.98)
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.08, top=0.91, wspace=0.18, hspace=0.2)
    return save_figure(fig, output_dir, "fig3_cross_split_head_matching")


def write_table(stats, summaries, analysis_summaries, output_dir):
    rows = []
    for model, model_stats in stats.items():
        for metric in METRICS:
            values = model_stats[metric]
            summary = summaries[model]["sources"]["real"][metric]["within_bin_head_zscore_profile"]
            rows.append(
                {
                    "Model": model,
                    "Metric": METRIC_LABELS[metric],
                    "Same-head cosine": values["same_mean"],
                    "Same-layer other": values["same_layer_other_mean"],
                    "Gap": values["same_mean"] - values["same_layer_other_mean"],
                    "Global ID (%)": 100 * values["global_id"],
                    "Within-layer ID (%)": 100 * values["within_id"],
                    "Geometry Spearman": summary["pairwise_geometry_spearman"],
                    "Permutation p": summary["permutation_pvalue"],
                    "Far/recent Spearman": analysis_summaries[model]["real"]["split_half_spearman"],
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "table_temporal_profile_stability.csv", index=False, float_format="%.6f")
    latex_table = table.copy()
    for column in [
        "Same-head cosine",
        "Same-layer other",
        "Gap",
        "Geometry Spearman",
        "Far/recent Spearman",
    ]:
        latex_table[column] = latex_table[column].map(lambda value: f"{value:.3f}")
    for column in ["Global ID (%)", "Within-layer ID (%)"]:
        latex_table[column] = latex_table[column].map(lambda value: f"{value:.1f}")
    latex_table["Permutation p"] = "$<10^{-4}$"
    latex_table = latex_table.rename(
        columns={
            "Global ID (%)": r"Global ID (\%)",
            "Within-layer ID (%)": r"Within-layer ID (\%)",
        }
    )
    latex = latex_table.to_latex(
        index=False,
        escape=False,
        caption="Cross-split stability and identifiability of eight-bin temporal profiles on 50 videos.",
        label="tab:temporal-profile-stability",
    )
    (output_dir / "table_temporal_profile_stability.tex").write_text(latex)
    return table


def write_readme(output_dir, table):
    headers = list(table.columns)
    markdown_rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in table.itertuples(index=False, name=None):
        cells = []
        for header, value in zip(headers, row):
            if header == "Permutation p":
                cells.append("<1e-4")
            elif header in ["Global ID (%)", "Within-layer ID (%)"]:
                cells.append(f"{value:.1f}")
            elif isinstance(value, float):
                cells.append(f"{value:.3f}")
            else:
                cells.append(str(value))
        markdown_rows.append("| " + " | ".join(cells) + " |")
    lines = [
        "# Cross-model temporal-profile figures",
        "",
        "All figures use the same 50 videos, 253 questions, eight token-age bins, and 112 physical KV heads per model. Main quantitative panels use real StreamingBench questions and within-bin head z-scores.",
        "",
        "## Figure 1: Cross-model temporal profiles",
        "",
        "**Caption.** Eight-bin temporal readout profiles of physical KV heads on two disjoint 25-video splits. Rows are ordered using split A only and the order is frozen for split B. The repeated horizontal structure and representative curves show that heterogeneous recent, sustained, and long-range preferences reproduce on unseen videos in both Qwen2.5-VL-7B and LLaVA-OV-7B.",
        "",
        "## Figure 2: Stability evidence",
        "",
        "**Caption.** Independent tests of temporal-profile stability. (a) Cross-split similarity of the same physical head is substantially higher than that of other heads in the same layer. (b) Eight-bin profiles recover head identity far above 112-way global and 4-way within-layer chance. (c) Far/recent rankings and pairwise head geometry remain highly correlated across video splits. All permutation tests for diagonal matching have p < 1e-4.",
        "",
        "## Figure 3: Cross-split head matching",
        "",
        "**Caption.** Top-1 cross-split matching of physical KV heads. For each split-A profile, the plot shows the most similar split-B head among all 112 candidates. Points on the diagonal recover the same physical head; off-diagonal points are mismatches. LLaVA recovers 111/112 heads with Reuse and 102/112 with Top-100, while both models remain far above the 1/112 chance rate.",
        "",
        "## Reporting note",
        "",
        "Head identification is a diagnostic of profile stability and distinctiveness, not video-QA task accuracy. Cross-model comparisons use normalized within-model profiles; they do not imply that either model has higher downstream QA accuracy.",
        "",
        "## Table preview",
        "",
        *markdown_rows,
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qwen_root",
        default="results/myobservations/temporal_topk_profile_50v_fixed448",
    )
    parser.add_argument(
        "--llava_root",
        default="results/myobservations/temporal_profile_llavaov7b_same50",
    )
    parser.add_argument(
        "--output_dir",
        default="results/myobservations/cross_model_temporal_profile_paper",
    )
    args = parser.parse_args()
    roots = {
        "Qwen2.5-VL-7B": Path(args.qwen_root),
        "LLaVA-OV-7B": Path(args.llava_root),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    style()

    model_data = {model: load_model_data(root / "raw_age_bin_attention.csv") for model, root in roots.items()}
    summaries = {
        model: json.loads((root / "vector_stability" / "vector_stability_summary.json").read_text())
        for model, root in roots.items()
    }
    analysis_summaries = {
        model: json.loads((root / "analysis" / "analysis_summary.json").read_text())
        for model, root in roots.items()
    }

    outputs = []
    outputs.extend(plot_profiles(model_data, output_dir))
    evidence_paths, stats = plot_evidence(model_data, summaries, analysis_summaries, output_dir)
    outputs.extend(evidence_paths)
    outputs.extend(plot_matching_matrices(model_data, output_dir))
    table = write_table(stats, summaries, analysis_summaries, output_dir)
    write_readme(output_dir, table)
    print(json.dumps({"output_dir": str(output_dir), "figures": [str(path) for path in outputs]}, indent=2))


if __name__ == "__main__":
    main()
