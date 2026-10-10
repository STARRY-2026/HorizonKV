#!/usr/bin/env python3
"""Build the frozen 51st-100th StreamingBench-half evaluation subset."""

import json
from pathlib import Path

import pandas as pd


HALF = Path("data/streamingbench/streamingbench_realtime_half_userpaths.json")
CALIBRATION = Path(
    "results/myobservations/temporal_topk_profile_10v_fixed448/filtered_anno_10v.json"
)
HERMES = Path(
    "results/qwen2.5_vl_7b/streamingbench/"
    "fps0.5-kv1000-hermes-ffc448-letterbox-hermes-half-k1000-canvas448-uniform/"
    "results.csv"
)
OUTPUT = Path("results/myobservations/temporal_zone_eval51_100")


def main():
    half = json.loads(HALF.read_text())
    subset = half[50:100]
    if len(subset) != 50:
        raise ValueError(f"Expected 50 videos, got {len(subset)}")
    subset_ids = {str(video["video_id"]) for video in subset}
    calibration = json.loads(CALIBRATION.read_text())
    calibration_ids = {str(video["video_id"]) for video in calibration}

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "filtered_anno_51_100.json").write_text(
        json.dumps(subset, indent=2, ensure_ascii=True) + "\n"
    )

    baseline = pd.read_csv(HERMES)
    selected = baseline[baseline["video_id"].astype(str).isin(subset_ids)].copy()
    selected.to_csv(OUTPUT / "uniform_hermes_results.csv", index=False)
    if selected["video_id"].astype(str).nunique() != 50:
        raise ValueError("Hermes baseline does not cover all 50 videos")

    manifest = {
        "selection": "entries 51-100 of StreamingBench half; Python slice [50:100]",
        "source_annotation": str(HALF),
        "num_videos": len(subset),
        "num_questions": sum(len(video.get("conversations", [])) for video in subset),
        "video_ids": [video["video_id"] for video in subset],
        "calibration_overlap": sorted(subset_ids & calibration_ids),
        "hermes_correct": int((selected["qa_acc"] > 0).sum()),
        "hermes_accuracy": float((selected["qa_acc"] > 0).mean() * 100.0),
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
