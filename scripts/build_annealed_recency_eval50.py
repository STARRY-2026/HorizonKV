#!/usr/bin/env python3
"""Build the first head-specific annealed-recency policy."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


def percentile_ranks(values):
    ranks = pd.Series(values).rank(method="average").to_numpy(dtype=np.float64) - 1.0
    return ranks / max(len(ranks) - 1, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_csv",
        default="results/myobservations/temporal_topk_profile_10v_fixed448/"
        "raw_pseudo_query_age_bins.csv",
    )
    parser.add_argument(
        "--output_dir",
        default="results/myobservations/annealed_recency_eval50",
    )
    parser.add_argument("--alpha_min", type=float, default=0.15)
    parser.add_argument("--alpha_max", type=float, default=0.75)
    parser.add_argument("--alpha_floor", type=float, default=0.05)
    parser.add_argument("--alpha_tau_sec", type=float, default=4.0)
    parser.add_argument("--protect_fraction", type=float, default=0.125)
    args = parser.parse_args()

    raw = pd.read_csv(args.raw_csv)
    current = raw[
        (raw["query_source"] == "pseudo")
        & (raw["bin_type"] == "age_sec")
        & (raw["bin_label"] == "0-4s")
        & (raw["token_count"] > 0)
        & raw["top100_mean_attention"].notna()
    ]
    keys = pd.MultiIndex.from_product(
        [range(28), range(4)], names=["layer", "kv_head"]
    )
    strength = (
        current.groupby(["layer", "kv_head"])["top100_mean_attention"]
        .mean()
        .reindex(keys)
    )
    if strength.isna().any():
        raise ValueError("Current Top-100 readout does not cover all 112 KV heads")

    ranks = percentile_ranks(strength.to_numpy(dtype=np.float64))
    alpha_initial = args.alpha_min + (args.alpha_max - args.alpha_min) * ranks
    protect_count = int(np.floor(len(keys) * args.protect_fraction))
    protect = np.zeros(len(keys), dtype=np.int64)
    top = np.argsort(-strength.to_numpy(dtype=np.float64), kind="stable")[:protect_count]
    protect[top] = 1

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    policy_path = output / "annealed_head_top100_protect.csv"
    with policy_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "layer", "kv_head", "profile_mode", "alpha_initial",
            "alpha_floor", "alpha_tau_sec", "protect_latest",
            "current_top100_attention", "current_rank",
        ])
        for idx, (layer, head) in enumerate(keys):
            writer.writerow([
                layer,
                head,
                "annealed_head_top100_protect",
                f"{alpha_initial[idx]:.10g}",
                f"{args.alpha_floor:.10g}",
                f"{args.alpha_tau_sec:.10g}",
                int(protect[idx]),
                f"{strength.iloc[idx]:.10g}",
                f"{ranks[idx]:.10g}",
            ])

    manifest = {
        "source": args.raw_csv,
        "current_metric": "mean pseudo-query top100_mean_attention in 0-4s",
        "alpha_initial_formula": "alpha_min + (alpha_max-alpha_min)*global_percentile_rank",
        "alpha_min": args.alpha_min,
        "alpha_max": args.alpha_max,
        "alpha_floor": args.alpha_floor,
        "alpha_tau_sec": args.alpha_tau_sec,
        "protect_fraction": args.protect_fraction,
        "protected_heads": protect_count,
        "policy": str(policy_path),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
