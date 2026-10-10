"""Validate historical-evidence Head stability from an existing raw profile."""

import argparse
import csv
import json
import math
import os

import numpy as np
import pandas as pd


HEAD_KEYS = ["layer", "kv_head"]


def finite_spearman(left, right):
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3:
        return None
    left_rank = pd.Series(left[valid]).rank(method="average").to_numpy(float)
    right_rank = pd.Series(right[valid]).rank(method="average").to_numpy(float)
    value = np.corrcoef(left_rank, right_rank)[0, 1]
    return float(value) if math.isfinite(float(value)) else None


def topk_set(frame, metric, k):
    ranked = frame.dropna(subset=[metric]).nlargest(k, metric)
    return set(zip(ranked["layer"].astype(int), ranked["kv_head"].astype(int)))


def compare_profiles(left, right, metric, top_ks):
    merged = left.merge(right, on=HEAD_KEYS, suffixes=("_left", "_right"))
    result = {
        "num_heads": int(len(merged)),
        "spearman": finite_spearman(
            merged[f"{metric}_left"].to_numpy(float),
            merged[f"{metric}_right"].to_numpy(float),
        ),
    }
    for k in top_ks:
        a = topk_set(left, metric, k)
        b = topk_set(right, metric, k)
        result[f"top{k}_overlap"] = int(len(a & b))
        result[f"top{k}_jaccard"] = float(len(a & b) / max(len(a | b), 1))
    return result


def head_profile(frame, metric):
    return frame.groupby(HEAD_KEYS, as_index=False)[metric].mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_csv", required=True)
    parser.add_argument("--metric", default="internal_top100_mean_attention")
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--top_ks", type=int, nargs="+", default=[10, 16, 23])
    parser.add_argument("--save_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    df = pd.read_csv(args.raw_csv)
    needed = set(HEAD_KEYS + ["video_idx", args.metric, "memory_readout_share", "transition_boundary_ratio"])
    missing = sorted(needed - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in raw profile: {missing}")

    df = df.replace([np.inf, -np.inf], np.nan)
    videos = np.asarray(sorted(df["video_idx"].dropna().astype(int).unique()))
    if len(videos) < 4:
        raise ValueError("At least four videos are required for split validation.")

    midpoint = len(videos) // 2
    calibration_videos = videos[:midpoint]
    evaluation_videos = videos[midpoint:]
    calibration = head_profile(df[df.video_idx.isin(calibration_videos)], args.metric)
    evaluation = head_profile(df[df.video_idx.isin(evaluation_videos)], args.metric)
    disjoint = compare_profiles(calibration, evaluation, args.metric, args.top_ks)
    disjoint.update(
        {
            "calibration_videos": calibration_videos.tolist(),
            "evaluation_videos": evaluation_videos.tolist(),
        }
    )

    rng = np.random.default_rng(args.seed)
    repeated = []
    for repeat in range(args.repeats):
        shuffled = rng.permutation(videos)
        left_videos = shuffled[:midpoint]
        right_videos = shuffled[midpoint:]
        left = head_profile(df[df.video_idx.isin(left_videos)], args.metric)
        right = head_profile(df[df.video_idx.isin(right_videos)], args.metric)
        row = {"repeat": repeat}
        row.update(compare_profiles(left, right, args.metric, args.top_ks))
        repeated.append(row)

    repeated_df = pd.DataFrame(repeated)
    repeated_path = os.path.join(args.save_dir, "split_half_repeats.csv")
    repeated_df.to_csv(repeated_path, index=False, quoting=csv.QUOTE_MINIMAL)

    full_primary = head_profile(df, args.metric)
    full_memory = head_profile(df, "memory_readout_share")
    boundary_comparison = compare_profiles(
        full_primary.rename(columns={args.metric: "comparison_metric"}),
        full_memory.rename(columns={"memory_readout_share": "comparison_metric"}),
        "comparison_metric",
        args.top_ks,
    )
    head_boundary = head_profile(df, "transition_boundary_ratio")
    merged_boundary = full_primary.merge(head_boundary, on=HEAD_KEYS)
    boundary_comparison["primary_vs_boundary_ratio_spearman"] = finite_spearman(
        merged_boundary[args.metric].to_numpy(float),
        merged_boundary["transition_boundary_ratio"].to_numpy(float),
    )

    repeat_summary = {}
    for column in repeated_df.columns:
        if column in {"repeat", "num_heads"}:
            continue
        values = repeated_df[column].dropna().to_numpy(float)
        repeat_summary[column] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        }

    summary = {
        "source": os.path.abspath(args.raw_csv),
        "metric": args.metric,
        "num_rows": int(len(df)),
        "num_videos": int(len(videos)),
        "num_heads": int(len(full_primary)),
        "video_disjoint_fixed_split": disjoint,
        "repeated_random_split_half": {
            "repeats": int(args.repeats),
            "statistics": repeat_summary,
            "raw_csv": repeated_path,
        },
        "boundary_sensitivity": boundary_comparison,
        "global_observation_means": {
            "memory_readout_share": float(df["memory_readout_share"].mean()),
            "transition_boundary_ratio": float(df["transition_boundary_ratio"].mean()),
            "internal_top100_mean_attention": float(df[args.metric].mean()),
        },
    }
    summary_path = os.path.join(args.save_dir, "historical_evidence_validation.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
