"""Analyze matched recent-vs-history strong reads from temporal profiling CSVs."""

import argparse
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RECENT_BINS = ["0-4s", "4-8s", "8-16s"]
FAR_BINS = ["64-128s", "128-256s"]
HEAD_KEYS = ["layer", "kv_head"]
OBS_KEYS = ["video_idx", "question_idx", "query_source", "query_name", *HEAD_KEYS]
EPS = 1e-12


def spearman(left, right):
    common = left.index.intersection(right.index)
    if len(common) < 3:
        return None
    value = left.loc[common].rank().corr(right.loc[common].rank())
    return float(value) if math.isfinite(value) else None


def matched_scores(frame, source, k):
    subset = frame[frame["query_source"] == source]
    wide = subset.pivot_table(
        index=OBS_KEYS,
        columns="bin_label",
        values=f"top{k}_mean_attention",
        aggfunc="mean",
    )
    recent = wide[[b for b in RECENT_BINS if b in wide]].mean(axis=1)
    far = wide[[b for b in FAR_BINS if b in wide]].mean(axis=1)
    paired = pd.DataFrame({"recent": recent, "far": far}).dropna().reset_index()
    paired["log_far_recent"] = np.log(
        (paired["far"] + EPS) / (paired["recent"] + EPS)
    )
    return paired


def summarize_source(frame, source, topk_values):
    score_table = None
    paired_by_k = {}
    for k in topk_values:
        paired = matched_scores(frame, source, k)
        paired_by_k[k] = paired
        grouped = paired.groupby(HEAD_KEYS)["log_far_recent"].agg(["mean", "count"])
        grouped = grouped.rename(
            columns={"mean": f"top{k}_log_far_recent", "count": f"top{k}_num_pairs"}
        )
        grouped[f"top{k}_geomean_far_recent"] = np.exp(
            grouped[f"top{k}_log_far_recent"]
        )
        score_table = grouped if score_table is None else score_table.join(grouped)

    rank_table = score_table[
        [f"top{k}_log_far_recent" for k in topk_values]
    ].rank()
    topk_corr = rank_table.corr().to_dict()
    primary_k = max(topk_values)
    primary = paired_by_k[primary_k]
    even = primary[primary["video_idx"] % 2 == 0].groupby(HEAD_KEYS)[
        "log_far_recent"
    ].mean()
    odd = primary[primary["video_idx"] % 2 == 1].groupby(HEAD_KEYS)[
        "log_far_recent"
    ].mean()

    summary = {
        "source": source,
        "num_matched_query_probes": int(
            primary[["video_idx", "question_idx", "query_name"]]
            .drop_duplicates()
            .shape[0]
        ),
        "primary_k": primary_k,
        "split_half_spearman": spearman(even, odd),
        "topk_rank_correlation": topk_corr,
    }
    return score_table.reset_index(), summary


def plot_heatmap(scores, value_column, output_path):
    matrix = scores.pivot(index="layer", columns="kv_head", values=value_column)
    fig, ax = plt.subplots(figsize=(5.2, 7.2))
    limit = float(np.nanpercentile(np.abs(matrix.to_numpy()), 95))
    image = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-limit, vmax=limit)
    ax.set_xlabel("Physical KV head")
    ax.set_ylabel("Layer")
    ax.set_xticks(matrix.columns)
    ax.set_yticks(np.arange(0, len(matrix.index), 2))
    ax.set_yticklabels(matrix.index[::2])
    colorbar = fig.colorbar(image, ax=ax, pad=0.03)
    colorbar.set_label("log(top-K far / recent attention)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_csv", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--topk_values", default="1,4,16,64,100")
    args = parser.parse_args()
    topk_values = sorted({int(v) for v in args.topk_values.split(",") if int(v) > 0})

    frame = pd.read_csv(args.raw_csv)
    frame = frame[(frame["bin_type"] == "age_sec") & (frame["token_count"] > 0)]
    os.makedirs(args.save_dir, exist_ok=True)

    outputs = {}
    scores_by_source = {}
    for source in ("pseudo", "real"):
        scores, source_summary = summarize_source(frame, source, topk_values)
        scores_by_source[source] = scores
        scores.to_csv(os.path.join(args.save_dir, f"head_scores_{source}.csv"), index=False)
        outputs[source] = source_summary

    primary_k = max(topk_values)
    column = f"top{primary_k}_log_far_recent"
    pseudo = scores_by_source["pseudo"].set_index(HEAD_KEYS)[column]
    real = scores_by_source["real"].set_index(HEAD_KEYS)[column]
    outputs["real_pseudo_spearman"] = spearman(pseudo, real)
    outputs["age_bins"] = {
        "recent": RECENT_BINS,
        "far": FAR_BINS,
        "unobserved": ["256s+"]
        if not (frame["bin_label"] == "256s+").any()
        else [],
    }
    outputs["num_physical_kv_heads"] = int(frame[HEAD_KEYS].drop_duplicates().shape[0])

    with open(os.path.join(args.save_dir, "analysis_summary.json"), "w") as handle:
        json.dump(outputs, handle, indent=2)
    plot_heatmap(
        scores_by_source["pseudo"],
        column,
        os.path.join(args.save_dir, f"pseudo_top{primary_k}_far_recent_heatmap.png"),
    )
    plot_heatmap(
        scores_by_source["pseudo"],
        column,
        os.path.join(args.save_dir, f"pseudo_top{primary_k}_far_recent_heatmap.pdf"),
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
