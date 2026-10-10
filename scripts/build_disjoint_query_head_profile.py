#!/usr/bin/env python3
"""Remove evaluation-video overlap from an existing query-head trace."""

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--calibration-anno", type=Path, required=True)
    parser.add_argument("--evaluation-anno", type=Path, required=True)
    parser.add_argument("--num-calibration-videos", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    calibration = json.loads(args.calibration_anno.read_text())[: args.num_calibration_videos]
    evaluation = json.loads(args.evaluation_anno.read_text())
    evaluation_ids = {int(item["video_id"]) for item in evaluation}
    excluded_indices = {
        index for index, item in enumerate(calibration)
        if int(item["video_id"]) in evaluation_ids
    }
    frame = pd.read_csv(args.raw_csv)
    filtered = frame[~frame["video_idx"].astype(int).isin(excluded_indices)].copy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(args.output, index=False)
    print(
        f"kept {len(filtered)}/{len(frame)} rows; excluded calibration indices "
        f"{sorted(excluded_indices)}"
    )


if __name__ == "__main__":
    main()
