#!/usr/bin/env python3
"""Extend the frozen 10-video temporal profile with 40 disjoint videos."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from head_analysis.profile_horizonkv_pilot import select_pilot_subset


ANNO = Path("data/streamingbench/streamingbench_realtime.json")
OLD = Path("results/myobservations/temporal_topk_profile_10v_fixed448")
OUTPUT = Path("results/myobservations/temporal_topk_profile_50v_fixed448")
VIDEO_ROOT = Path("/home/yujun/program/streaming_data/streamingbench/videos")


def localize(sample):
    copied = dict(sample)
    copied["video_path"] = str(VIDEO_ROOT / Path(sample["video_path"]).name)
    if not Path(copied["video_path"]).is_file():
        raise FileNotFoundError(copied["video_path"])
    return copied


def main():
    full = json.loads(ANNO.read_text())
    old_manifest = json.loads((OLD / "selection_manifest.json").read_text())
    old_ids = {str(item["video_id"]) for item in old_manifest}
    remaining = [sample for sample in full if str(sample["video_id"]) not in old_ids]
    selected, _ = select_pilot_subset(
        remaining, num_videos=40, max_last_end=320.0, seed=2024
    )
    source_index = {str(sample["video_id"]): idx for idx, sample in enumerate(full)}
    selected = [localize(sample) for sample in selected]
    for sample in selected:
        sample["_source_index"] = source_index[str(sample["video_id"])]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    shards = [selected[::2], selected[1::2]]
    for idx, shard in enumerate(shards):
        (OUTPUT / f"new40_gpu{idx}.json").write_text(
            json.dumps(shard, indent=2, ensure_ascii=True) + "\n"
        )

    old_ids_ordered = [str(item["video_id"]) for item in old_manifest]
    new_ids_ordered = [str(sample["video_id"]) for sample in selected]
    manifest = {
        "selection": "original 10 videos plus 40 duration-stratified disjoint videos",
        "source_annotation": str(ANNO),
        "old_num_videos": 10,
        "new_num_videos": 40,
        "total_num_videos": 50,
        "old_video_ids": old_ids_ordered,
        "new_video_ids": new_ids_ordered,
        "combined_video_ids": old_ids_ordered + new_ids_ordered,
        "new_num_questions": sum(len(sample.get("conversations", [])) for sample in selected),
        "new_shard_sizes": [len(shard) for shard in shards],
    }
    (OUTPUT / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
