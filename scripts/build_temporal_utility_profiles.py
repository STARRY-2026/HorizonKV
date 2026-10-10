#!/usr/bin/env python3
"""Build frozen temporal-utility policies from disjoint pseudo-query traces."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


AGE_BINS = [
    "0-4s", "4-8s", "8-16s", "16-32s", "32-64s", "64-128s", "128-256s"
]
OUTPUT_COLUMNS = [
    "time_0_4", "time_4_8", "time_8_16", "time_16_32",
    "time_32_64", "time_64_128", "time_128_plus",
]
OBS_KEYS = ["video_idx", "question_idx", "query_name", "layer", "kv_head"]


def percentile_rank(values):
    return pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=np.float64)


def interpolate_rows(values):
    frame = pd.DataFrame(values, columns=AGE_BINS)
    frame = frame.interpolate(axis=1, limit_direction="both")
    if frame.isna().any().any():
        raise ValueError("Temporal profiles contain an entirely unobserved age bin")
    array = frame.to_numpy(dtype=np.float64)
    maxima = np.maximum(array.max(axis=1, keepdims=True), 1e-12)
    return np.clip(array / maxima, 0.0, 1.0)


def write_policy(path, curves, affinities, mode, temporal_weight, novelty_weight, tau):
    fieldnames = [
        "layer", "kv_head", "profile_mode", *OUTPUT_COLUMNS,
        "current_affinity", "temporal_weight", "novelty_weight", "novelty_tau_sec",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for flat_idx in range(curves.shape[0]):
            row = {
                "layer": flat_idx // 4,
                "kv_head": flat_idx % 4,
                "profile_mode": mode,
                "current_affinity": f"{affinities[flat_idx]:.10g}",
                "temporal_weight": f"{temporal_weight:.10g}",
                "novelty_weight": f"{novelty_weight:.10g}",
                "novelty_tau_sec": f"{tau:.10g}",
            }
            row.update({name: f"{value:.10g}" for name, value in zip(OUTPUT_COLUMNS, curves[flat_idx])})
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--temporal_weight", type=float, default=0.2)
    parser.add_argument("--novelty_weight", type=float, default=0.2)
    parser.add_argument("--novelty_tau_sec", type=float, default=4.0)
    parser.add_argument("--residual_weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument(
        "--exclude_video_indices", default="",
        help="Comma-separated raw-profile video_idx values excluded from calibration.",
    )
    args = parser.parse_args()

    raw = pd.read_csv(args.raw_csv)
    excluded_video_indices = {
        int(value)
        for value in args.exclude_video_indices.split(",")
        if value.strip()
    }
    if excluded_video_indices:
        raw = raw[
            ~raw["video_idx"].astype(int).isin(excluded_video_indices)
        ].copy()
    frame = raw[
        (raw["query_source"] == "pseudo")
        & (raw["bin_type"] == "age_sec")
        & raw["bin_label"].isin(AGE_BINS)
        & (raw["token_count"] > 0)
        & raw["top100_mean_attention"].notna()
        & (raw["top100_mean_attention"] > 0)
    ].copy()

    # A valid observation must offer a far-history alternative. Otherwise a
    # recent-only prefix would be mistaken for evidence of a recent preference.
    eligible = frame[frame["bin_label"] == "64-128s"][OBS_KEYS].drop_duplicates()
    frame = frame.merge(eligible, on=OBS_KEYS, how="inner")
    if frame.empty:
        raise ValueError("No pseudo-query observations contain 64-128s candidates")

    wide = frame.pivot_table(
        index=OBS_KEYS,
        columns="bin_label",
        values="top100_mean_attention",
        aggfunc="mean",
    )
    available_bins = [name for name in AGE_BINS if name in wide]
    relative = wide[available_bins].div(wide[available_bins].max(axis=1), axis=0)
    profile = relative.groupby(["layer", "kv_head"]).median()
    profile = profile.reindex(
        pd.MultiIndex.from_product([range(28), range(4)], names=["layer", "kv_head"])
    )
    curves = interpolate_rows(profile.reindex(columns=AGE_BINS).to_numpy())

    current = frame[frame["bin_label"] == "0-4s"].groupby(
        ["layer", "kv_head"]
    )["top100_mean_attention"].apply(lambda x: float(np.exp(np.log(x).mean())))
    current = current.reindex(profile.index)
    if current.isna().any():
        raise ValueError("Current affinity does not cover all physical KV heads")
    affinities = percentile_rank(current.to_numpy(dtype=np.float64))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    global_curve = np.repeat(curves.mean(axis=0, keepdims=True), curves.shape[0], axis=0)
    global_affinity = np.full_like(affinities, affinities.mean())

    rng = np.random.default_rng(args.seed)
    shuffled_curves = curves.copy()
    shuffled_affinities = affinities.copy()
    permutations = {}
    for layer in range(28):
        start = layer * 4
        permutation = rng.permutation(4)
        shuffled_curves[start:start + 4] = curves[start + permutation]
        shuffled_affinities[start:start + 4] = affinities[start + permutation]
        permutations[str(layer)] = permutation.tolist()

    policies = {
        "e0_content": (curves, affinities, 0.0, 0.0),
        "e1_global": (global_curve, global_affinity, args.temporal_weight, args.novelty_weight),
        "e2_head_temporal": (curves, affinities, args.temporal_weight, 0.0),
        "e3_full": (curves, affinities, args.temporal_weight, args.novelty_weight),
        "e4_shuffled": (shuffled_curves, shuffled_affinities, args.temporal_weight, args.novelty_weight),
        "residual_zero": (curves, affinities, 0.0, 0.0),
        "residual_head": (curves, affinities, args.residual_weight, 0.0),
        "residual_shuffled": (shuffled_curves, shuffled_affinities, args.residual_weight, 0.0),
    }
    survival = np.flip(np.cumsum(np.flip(curves, axis=1), axis=1), axis=1)
    survival /= np.maximum(survival[:, :1], 1e-12)
    global_survival = np.repeat(
        survival.mean(axis=0, keepdims=True), survival.shape[0], axis=0
    )
    shuffled_survival = survival.copy()
    for layer in range(28):
        start = layer * 4
        permutation = np.asarray(permutations[str(layer)], dtype=np.int64)
        shuffled_survival[start:start + 4] = survival[start + permutation]
    policies.update({
        "survival_content": (survival, affinities, 0.0, 0.0),
        "survival_global": (global_survival, global_affinity, 0.2, 0.0),
        "survival_head025": (survival, affinities, 0.25, 0.0),
        "survival_head050": (survival, affinities, 0.5, 0.0),
        "survival_head": (survival, affinities, 1.0, 0.0),
        "survival_shuffled": (shuffled_survival, shuffled_affinities, 1.0, 0.0),
        "relative_head": (survival, affinities, 0.2, 0.0),
        "relative_shuffled": (shuffled_survival, shuffled_affinities, 0.2, 0.0),
    })
    for mode, (mode_curves, mode_affinities, temporal_weight, novelty_weight) in policies.items():
        write_policy(
            output_dir / f"{mode}.csv",
            mode_curves,
            mode_affinities,
            mode,
            temporal_weight,
            novelty_weight,
            args.novelty_tau_sec,
        )

    summary = {
        "source": str(Path(args.raw_csv)),
        "profile_source": "pseudo queries only",
        "eligibility": "observation contains nonempty 64-128s bin",
        "num_eligible_observations": int(wide.reset_index()[OBS_KEYS[:3]].drop_duplicates().shape[0]),
        "num_heads": int(curves.shape[0]),
        "age_bins": AGE_BINS,
        "temporal_weight": args.temporal_weight,
        "novelty_weight": args.novelty_weight,
        "novelty_tau_sec": args.novelty_tau_sec,
        "residual_weight": args.residual_weight,
        "shuffle_seed": args.seed,
        "within_layer_permutations": permutations,
        "half_labels_used": False,
        "excluded_video_indices": sorted(excluded_video_indices),
    }
    (output_dir / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
