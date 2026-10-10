#!/usr/bin/env python3
"""Build temporal-zone policies and the first-50 StreamingBench-half subset."""

import argparse
import csv
import json
import os

import numpy as np
import pandas as pd


AGE_BINS = [
    "0-4s", "4-8s", "8-16s", "16-32s", "32-64s", "64-128s", "128-256s"
]
ZONE_BINS = {
    "recent_ratio": AGE_BINS[:3],
    "middle_ratio": AGE_BINS[3:5],
    "far_ratio": AGE_BINS[5:7],
}


def build_profiles(raw_csv, shrinkage=0.5, minimum=0.15, seed=2024):
    frame = pd.read_csv(raw_csv)
    frame = frame[
        (frame["query_source"] == "pseudo")
        & (frame["bin_type"] == "age_sec")
        & (frame["bin_label"].isin(AGE_BINS))
        & (frame["token_count"] > 0)
    ]
    matrix = (
        frame.groupby(["layer", "kv_head", "bin_label"])["reuse_score"]
        .mean()
        .unstack()
        .reindex(columns=AGE_BINS)
        .sort_index()
    )
    if matrix.shape != (112, 7) or matrix.isna().any().any():
        raise ValueError(f"Expected a complete 112x7 profile matrix, got {matrix.shape}")

    zone_values = np.column_stack(
        [matrix[bins].sum(axis=1).to_numpy() for bins in ZONE_BINS.values()]
    )
    profiles = zone_values / zone_values.sum(axis=1, keepdims=True)
    profiles = (1.0 - shrinkage) * profiles + shrinkage / 3.0
    profiles = minimum + (1.0 - 3.0 * minimum) * profiles
    profiles = profiles / profiles.sum(axis=1, keepdims=True)

    keys = list(matrix.index)
    global_profile = profiles.mean(axis=0, keepdims=True)
    global_profiles = np.repeat(global_profile, len(keys), axis=0)
    rng = np.random.default_rng(seed)
    shuffled_profiles = profiles[rng.permutation(len(keys))]
    normalized = matrix.to_numpy(dtype=np.float64)
    normalized = normalized / normalized.sum(axis=1, keepdims=True)
    current_affinity = normalized[:, 0] - normalized[:, 3:].mean(axis=1)
    protect_latest = np.zeros(len(keys), dtype=np.int64)
    top_count = len(keys) // 8
    top_indices = np.argsort(-current_affinity, kind="stable")[:top_count]
    protect_latest[top_indices] = 1
    current_rows = frame[frame["bin_label"] == "0-4s"]
    top100_current = (
        current_rows.groupby(["layer", "kv_head"])["top100_mean_attention"]
        .mean()
        .reindex(keys)
        .to_numpy(dtype=np.float64)
    )
    top100_current = np.nan_to_num(top100_current, nan=0.0)
    protect_latest_top100 = np.zeros(len(keys), dtype=np.int64)
    top100_indices = np.argsort(-top100_current, kind="stable")[:top_count]
    protect_latest_top100[top100_indices] = 1
    return (
        keys, profiles, global_profiles, shuffled_profiles,
        protect_latest, current_affinity,
        protect_latest_top100, top100_current,
    )


def write_policy(path, keys, profiles, protect_latest=None, current_affinity=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        extra = [] if protect_latest is None else ["protect_latest", "current_affinity"]
        writer.writerow(["layer", "kv_head", *ZONE_BINS.keys(), *extra])
        for idx, ((layer, head), ratios) in enumerate(zip(keys, profiles)):
            row = [layer, head, *(f"{value:.10f}" for value in ratios)]
            if protect_latest is not None:
                row.extend([int(protect_latest[idx]), f"{current_affinity[idx]:.10f}"])
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_csv",
        default="results/myobservations/temporal_topk_profile_10v_fixed448/"
        "raw_pseudo_query_age_bins.csv",
    )
    parser.add_argument(
        "--half_anno",
        default="data/streamingbench/streamingbench_realtime_half_userpaths.json",
    )
    parser.add_argument(
        "--calibration_anno",
        default="results/myobservations/temporal_topk_profile_10v_fixed448/"
        "filtered_anno_10v.json",
    )
    parser.add_argument(
        "--uniform_results",
        default="results/qwen2.5_vl_7b/streamingbench/"
        "fps0.5-kv1000-hermes-ffc448-letterbox-"
        "hermes-half-k1000-canvas448-uniform/results.csv",
    )
    parser.add_argument(
        "--save_dir", default="results/myobservations/temporal_zone_eval50"
    )
    args = parser.parse_args()

    (
        keys, matched, global_profiles, shuffled,
        protect_latest, current_affinity,
        protect_latest_top100, top100_current,
    ) = build_profiles(args.raw_csv)
    os.makedirs(args.save_dir, exist_ok=True)
    write_policy(os.path.join(args.save_dir, "matched_profiles.csv"), keys, matched)
    write_policy(os.path.join(args.save_dir, "global_profiles.csv"), keys, global_profiles)
    write_policy(os.path.join(args.save_dir, "shuffled_profiles_seed2024.csv"), keys, shuffled)
    write_policy(
        os.path.join(args.save_dir, "matched_selective_protect_top1of8.csv"),
        keys,
        matched,
        protect_latest=protect_latest,
        current_affinity=current_affinity,
    )
    write_policy(
        os.path.join(
            args.save_dir,
            "matched_selective_protect_top1of8_top100current.csv",
        ),
        keys,
        matched,
        protect_latest=protect_latest_top100,
        current_affinity=top100_current,
    )

    with open(args.half_anno) as handle:
        half = json.load(handle)
    subset = half[:50]
    with open(os.path.join(args.save_dir, "filtered_anno_first50.json"), "w") as handle:
        json.dump(subset, handle, indent=2, ensure_ascii=True)
    long_question = next(
        (
            question for question in subset[0]["conversations"]
            if float(question.get("end_time", 0.0)) >= 64.0
        ),
        subset[0]["conversations"][-1],
    )
    smoke = [{**subset[0], "conversations": [long_question]}]
    with open(os.path.join(args.save_dir, "filtered_anno_smoke1q.json"), "w") as handle:
        json.dump(smoke, handle, indent=2, ensure_ascii=True)

    with open(args.calibration_anno) as handle:
        calibration = json.load(handle)
    subset_ids = {str(item["video_id"]) for item in subset}
    calibration_ids = {str(item["video_id"]) for item in calibration}
    tasks = {}
    for video in subset:
        for question in video.get("conversations", []):
            task = question.get("task", "unknown")
            tasks[task] = tasks.get(task, 0) + 1
    manifest = {
        "selection": "first 50 entries of StreamingBench half; no reordering",
        "source_annotation": args.half_anno,
        "num_videos": len(subset),
        "num_questions": sum(len(v.get("conversations", [])) for v in subset),
        "video_ids": [item["video_id"] for item in subset],
        "calibration_overlap": sorted(subset_ids & calibration_ids),
        "task_counts": dict(sorted(tasks.items())),
        "profile": {
            "source": args.raw_csv,
            "metric": "mean pseudo-query reuse_score",
            "zones": ZONE_BINS,
            "shrinkage_to_uniform": 0.5,
            "minimum_zone_ratio": 0.15,
            "shuffled_seed": 2024,
            "matched_ratio_min": matched.min(axis=0).tolist(),
            "matched_ratio_mean": matched.mean(axis=0).tolist(),
            "matched_ratio_max": matched.max(axis=0).tolist(),
            "selective_protection": {
                "fraction": "1/8",
                "num_heads": int(protect_latest.sum()),
                "score": "normalized reuse_score[0-4s] minus mean normalized reuse_score[16s+]",
                "heads": [
                    {"layer": int(keys[idx][0]), "kv_head": int(keys[idx][1]),
                     "current_affinity": float(current_affinity[idx])}
                    for idx in np.flatnonzero(protect_latest)
                ],
            },
            "selective_protection_top100_current": {
                "fraction": "1/8",
                "num_heads": int(protect_latest_top100.sum()),
                "score": "absolute mean pseudo-query top100_mean_attention in 0-4s",
                "heads": [
                    {"layer": int(keys[idx][0]), "kv_head": int(keys[idx][1]),
                     "score": float(top100_current[idx])}
                    for idx in np.flatnonzero(protect_latest_top100)
                ],
            },
        },
    }
    with open(os.path.join(args.save_dir, "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)

    if os.path.isfile(args.uniform_results):
        baseline = pd.read_csv(args.uniform_results)
        baseline = baseline[baseline["video_id"].astype(str).isin(subset_ids)]
        baseline.to_csv(os.path.join(args.save_dir, "uniform_hermes_results.csv"), index=False)
        if baseline["video_id"].astype(str).nunique() != 50:
            raise ValueError("Uniform baseline does not contain all first-50 video IDs")
    print(json.dumps(manifest, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
