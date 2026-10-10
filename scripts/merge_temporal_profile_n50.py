#!/usr/bin/env python3
"""Merge the original 10-video profile and two new 20-video shards."""

import json
from pathlib import Path

import pandas as pd


ROOT = Path("results/myobservations/temporal_topk_profile_50v_fixed448")
OLD = Path("results/myobservations/temporal_topk_profile_10v_fixed448")


def main():
    manifest = json.loads((ROOT / "selection_manifest.json").read_text())
    video_order = {
        str(video_id): idx for idx, video_id in enumerate(manifest["combined_video_ids"])
    }
    paths = [
        OLD / "raw_age_bin_attention.csv",
        ROOT / "shard_gpu0" / "raw_age_bin_attention.csv",
        ROOT / "shard_gpu1" / "raw_age_bin_attention.csv",
    ]
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["video_id"] = frame["video_id"].astype(str)
        frame["video_idx"] = frame["video_id"].map(video_order)
        if frame["video_idx"].isna().any():
            raise ValueError(f"Unknown video IDs in {path}")
        frame["video_idx"] = frame["video_idx"].astype(int)
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    if merged["video_id"].nunique() != 50:
        raise ValueError(f"Expected 50 videos, got {merged['video_id'].nunique()}")
    output = ROOT / "raw_age_bin_attention.csv"
    merged.to_csv(output, index=False)
    print({"output": str(output), "rows": len(merged), "videos": 50})


if __name__ == "__main__":
    main()
