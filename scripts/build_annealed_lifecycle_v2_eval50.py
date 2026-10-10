#!/usr/bin/env python3
"""Build the baseline-anchored Head-specific lifecycle policy."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


def percentile_ranks(values):
    ranks = pd.Series(values).rank(method="average").to_numpy(dtype=np.float64) - 1.0
    return ranks / max(len(ranks) - 1, 1)


def hermes_layer_parameters(layer):
    if layer < 8:
        return 1.0, 20.0
    if layer >= 19:
        return 0.0, 0.0
    progress = (layer - 8) / float(19 - 8)
    return 0.75 - 0.6 * progress, 20.0 - 12.0 * progress


def gated_top_fraction(ranks, fraction):
    threshold = 1.0 - float(fraction)
    return np.clip((ranks - threshold) / max(float(fraction), 1e-12), 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_csv",
        default="results/myobservations/temporal_topk_profile_10v_fixed448/"
        "raw_pseudo_query_age_bins.csv",
    )
    parser.add_argument(
        "--output_dir",
        default="results/myobservations/annealed_lifecycle_v2_eval50",
    )
    parser.add_argument("--specialist_fraction", type=float, default=0.125)
    parser.add_argument("--alpha_grace_sec", type=float, default=2.0)
    parser.add_argument("--alpha_tau_sec", type=float, default=8.0)
    args = parser.parse_args()

    raw = pd.read_csv(args.raw_csv)
    valid = raw[
        (raw["query_source"] == "pseudo")
        & (raw["bin_type"] == "age_sec")
        & (raw["token_count"] > 0)
        & raw["top100_mean_attention"].notna()
    ]
    keys = pd.MultiIndex.from_product(
        [range(28), range(4)], names=["layer", "kv_head"]
    )

    def strength_for(bins):
        values = (
            valid[valid["bin_label"].isin(bins)]
            .groupby(["layer", "kv_head"])["top100_mean_attention"]
            .mean()
            .reindex(keys)
        )
        if values.isna().any():
            raise ValueError(f"Top-100 readout does not cover all heads for {bins}")
        return values.to_numpy(dtype=np.float64)

    current_strength = strength_for(["0-4s"])
    memory_strength = strength_for(["64-128s", "128-256s"])
    current_rank = percentile_ranks(current_strength)
    memory_rank = percentile_ranks(memory_strength)
    current_gate = gated_top_fraction(current_rank, args.specialist_fraction)
    memory_gate = gated_top_fraction(memory_rank, args.specialist_fraction)

    base_alpha = np.empty(len(keys), dtype=np.float64)
    base_k = np.empty(len(keys), dtype=np.float64)
    for idx, (layer, _) in enumerate(keys):
        base_alpha[idx], base_k[idx] = hermes_layer_parameters(layer)
    alpha_start = base_alpha + (1.0 - base_alpha) * current_gate
    alpha_end = base_alpha * (1.0 - memory_gate)
    recency_k = base_k + (20.0 - base_k) * current_gate
    protect = (current_gate > 0.0).astype(np.int64)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    policy_path = output / "annealed_lifecycle_v2.csv"
    with policy_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "layer", "kv_head", "profile_mode", "alpha_initial",
            "alpha_floor", "alpha_tau_sec", "protect_latest",
            "alpha_grace_sec", "recency_k_decay", "base_alpha", "base_k_decay",
            "current_top100_attention", "memory_top100_attention",
            "current_rank", "memory_rank", "current_gate", "memory_gate",
        ])
        for idx, (layer, head) in enumerate(keys):
            writer.writerow([
                layer, head, "annealed_lifecycle_v2",
                f"{alpha_start[idx]:.10g}", f"{alpha_end[idx]:.10g}",
                f"{args.alpha_tau_sec:.10g}", int(protect[idx]),
                f"{args.alpha_grace_sec:.10g}", f"{recency_k[idx]:.10g}",
                f"{base_alpha[idx]:.10g}", f"{base_k[idx]:.10g}",
                f"{current_strength[idx]:.10g}", f"{memory_strength[idx]:.10g}",
                f"{current_rank[idx]:.10g}", f"{memory_rank[idx]:.10g}",
                f"{current_gate[idx]:.10g}", f"{memory_gate[idx]:.10g}",
            ])

    manifest = {
        "source": args.raw_csv,
        "current_metric": "mean pseudo-query top100_mean_attention in 0-4s",
        "memory_metric": "mean pseudo-query top100_mean_attention in 64-256s",
        "specialist_fraction": args.specialist_fraction,
        "current_specialists": int((current_gate > 0.0).sum()),
        "memory_specialists": int((memory_gate > 0.0).sum()),
        "overlap_specialists": int(((current_gate > 0.0) & (memory_gate > 0.0)).sum()),
        "alpha_grace_sec": args.alpha_grace_sec,
        "alpha_tau_sec": args.alpha_tau_sec,
        "policy": str(policy_path),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
