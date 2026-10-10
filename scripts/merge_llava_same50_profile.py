#!/usr/bin/env python3
"""Merge the two LLaVA temporal-profile shards using the original video order."""

import json
from pathlib import Path

import pandas as pd


ROOT = Path("results/myobservations/temporal_profile_llavaov7b_same50")


def main():
    manifest = json.loads((ROOT / "input_manifest.json").read_text())
    video_order = {
        str(video_id): index
        for index, video_id in enumerate(manifest["combined_video_ids"])
    }
    frames = []
    for name in ["even", "odd"]:
        path = ROOT / f"shard_{name}" / "raw_age_bin_attention.csv"
        frame = pd.read_csv(path)
        frame["video_id"] = frame["video_id"].astype(str)
        frame["video_idx"] = frame["video_id"].map(video_order)
        if frame["video_idx"].isna().any():
            raise ValueError(f"Unknown video ID in {path}")
        frame["video_idx"] = frame["video_idx"].astype(int)
        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True)
    if merged["video_id"].nunique() != 50:
        raise ValueError(f"Expected 50 videos, got {merged['video_id'].nunique()}")
    if set(merged["video_idx"].unique()) != set(range(50)):
        raise ValueError("Merged global video indices are incomplete")

    output = ROOT / "raw_age_bin_attention.csv"
    merged.to_csv(output, index=False)
    summary = {
        "raw_csv": str(output),
        "num_rows": int(len(merged)),
        "num_videos": int(merged["video_id"].nunique()),
        "num_questions": int(
            merged[["video_id", "question_idx"]].drop_duplicates().shape[0]
        ),
        "num_heads": int(merged[["layer", "kv_head"]].drop_duplicates().shape[0]),
        "even_videos": int(merged.loc[merged.video_idx % 2 == 0, "video_id"].nunique()),
        "odd_videos": int(merged.loc[merged.video_idx % 2 == 1, "video_id"].nunique()),
    }
    (ROOT / "merged_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(summary)


if __name__ == "__main__":
    main()
