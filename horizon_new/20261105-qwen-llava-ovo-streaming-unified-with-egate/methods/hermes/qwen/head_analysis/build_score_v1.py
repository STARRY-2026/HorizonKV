"""Build score_v1.csv by merging memory-readout and prev/current KV-head profiles.

This script targets detail-sensitive tasks such as Counting and Object Recognition.
It combines three branches:
1. current_detail: preserve heads that read recent/local visual details
2. sharp_selective: reward sharp and selective readout heads
3. long_memory: keep strong long-term memory heads as a fallback

The final score is:
    score_v1 = max(
        0.60 * current_detail + 0.40 * sharp_selective,
        0.70 * long_memory + 0.30 * sharp_selective,
    ) + 0.01
"""

import argparse
import os

import numpy as np
import pandas as pd


EPS = 1e-12


def safe_array(series):
    values = np.asarray(series, dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return values


def normalize_clip_p95(values, eps=EPS):
    """Clamp to the 95th percentile, then normalize to [0, 1]."""
    values = safe_array(values)
    values = np.maximum(values, 0.0)
    p95 = np.percentile(values, 95.0)
    if p95 <= eps:
        vmax = values.max()
        if vmax <= eps:
            return np.zeros_like(values, dtype=np.float64)
        return np.clip(values / vmax, 0.0, 1.0)
    return np.clip(values, 0.0, p95) / p95


def normalize_log1p_clip_p95(values):
    values = np.maximum(safe_array(values), 0.0)
    return normalize_clip_p95(np.log1p(values))


def ensure_columns(df, required, path):
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")


def load_and_merge(memory_csv, kv_csv, num_layers, num_kv_heads):
    memory_df = pd.read_csv(memory_csv)
    kv_df = pd.read_csv(kv_csv)

    ensure_columns(
        memory_df,
        {
            "layer",
            "kv_head",
            "current_core_readout_share",
            "current_readout_share",
            "spiky_readout_score",
            "internal_topk_concentration",
            "boundary_suppression",
            "internal_top100_mean_attention",
            "clean_internal_readout_share",
        },
        memory_csv,
    )
    ensure_columns(
        kv_df,
        {"layer", "kv_head", "s_current_share", "r_per_token_ratio"},
        kv_csv,
    )

    merged = pd.merge(
        memory_df,
        kv_df[["layer", "kv_head", "s_current_share", "r_per_token_ratio"]],
        on=["layer", "kv_head"],
        how="inner",
        validate="one_to_one",
    )
    merged = merged.sort_values(["layer", "kv_head"]).reset_index(drop=True)

    expected = int(num_layers) * int(num_kv_heads)
    if len(merged) != expected:
        raise ValueError(
            f"Expected {expected} merged rows ({num_layers} layers x {num_kv_heads} kv heads), "
            f"got {len(merged)}"
        )
    return merged


def compute_score_v1(df):
    current_core = normalize_clip_p95(df["current_core_readout_share"])
    current_readout = normalize_clip_p95(df["current_readout_share"])
    s_current = normalize_clip_p95(df["s_current_share"])
    r_density = normalize_log1p_clip_p95(df["r_per_token_ratio"])

    spiky = normalize_clip_p95(df["spiky_readout_score"])
    topk = normalize_clip_p95(df["internal_topk_concentration"])
    boundary = normalize_clip_p95(df["boundary_suppression"])

    long_top100 = normalize_clip_p95(df["internal_top100_mean_attention"])
    clean_internal = normalize_clip_p95(df["clean_internal_readout_share"])

    current_detail = (
        0.40 * current_core
        + 0.25 * current_readout
        + 0.20 * s_current
        + 0.15 * r_density
    )
    sharp_selective = 0.45 * spiky + 0.35 * topk + 0.20 * boundary
    long_memory = 0.70 * long_top100 + 0.30 * clean_internal

    branch_detail = 0.60 * current_detail + 0.40 * sharp_selective
    branch_memory = 0.70 * long_memory + 0.30 * sharp_selective
    score_v1 = np.maximum(branch_detail, branch_memory) + 0.01

    out = df[["layer", "kv_head"]].copy()
    out["score"] = score_v1
    out["score_v1"] = score_v1
    out["current_detail"] = current_detail
    out["sharp_selective"] = sharp_selective
    out["long_memory"] = long_memory
    out["branch_detail"] = branch_detail
    out["branch_memory"] = branch_memory
    out["norm_current_core_readout_share"] = current_core
    out["norm_current_readout_share"] = current_readout
    out["norm_s_current_share"] = s_current
    out["norm_log1p_r_per_token_ratio"] = r_density
    out["norm_spiky_readout_score"] = spiky
    out["norm_internal_topk_concentration"] = topk
    out["norm_boundary_suppression"] = boundary
    out["norm_internal_top100_mean_attention"] = long_top100
    out["norm_clean_internal_readout_share"] = clean_internal
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--memory_csv",
        default="results/observations/effective_memory_readout_core_top100_n4_o80/effective_readout_scores.csv",
        help="CSV from effective memory readout profiling.",
    )
    parser.add_argument(
        "--kv_csv",
        default="results/observations/obs_prev_current_chunk_attention_eager_gpu2_n16_o500_paper_kv/kv_head_profile_scores.csv",
        help="CSV from prev/current KV-head profiling.",
    )
    parser.add_argument(
        "--save_dir",
        default="results/observations/score_v1",
        help="Directory to save score_v1.csv and debug output.",
    )
    parser.add_argument("--num_layers", type=int, default=28)
    parser.add_argument("--num_kv_heads", type=int, default=4)
    args = parser.parse_args()

    merged = load_and_merge(
        args.memory_csv,
        args.kv_csv,
        num_layers=args.num_layers,
        num_kv_heads=args.num_kv_heads,
    )
    result = compute_score_v1(merged)

    os.makedirs(args.save_dir, exist_ok=True)
    score_path = os.path.join(args.save_dir, "score_v1.csv")
    debug_path = os.path.join(args.save_dir, "score_v1_debug.csv")

    result[["layer", "kv_head", "score", "score_v1"]].to_csv(score_path, index=False)
    result.to_csv(debug_path, index=False)

    print(f"wrote {score_path}")
    print(f"wrote {debug_path}")
    print(
        "[score_v1] summary:",
        f"min={result['score'].min():.4f}",
        f"max={result['score'].max():.4f}",
        f"mean={result['score'].mean():.4f}",
    )


if __name__ == "__main__":
    main()
