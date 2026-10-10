#!/usr/bin/env python3
"""Build the original temporal-profile 50-video split for LLaVA profiling."""

import json
from pathlib import Path


QWEN50 = Path("results/myobservations/temporal_topk_profile_50v_fixed448")
QWEN10 = Path("results/myobservations/temporal_topk_profile_10v_fixed448")
OUTPUT = Path("results/myobservations/temporal_profile_llavaov7b_same50")


def read_json(path):
    return json.loads(path.read_text())


def main():
    manifest = read_json(QWEN50 / "selection_manifest.json")
    video_ids = [str(value) for value in manifest["combined_video_ids"]]
    samples = [
        *read_json(QWEN10 / "filtered_anno_10v.json"),
        *read_json(QWEN50 / "new40_gpu0.json"),
        *read_json(QWEN50 / "new40_gpu1.json"),
    ]
    by_id = {str(sample["video_id"]): sample for sample in samples}
    missing = [video_id for video_id in video_ids if video_id not in by_id]
    if missing:
        raise ValueError(f"Missing original profile videos: {missing}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for parity, name in [(0, "even"), (1, "odd")]:
        split_ids = video_ids[parity::2]
        split = [by_id[video_id] for video_id in split_ids]
        if len(split) != 25:
            raise ValueError(f"Expected 25 {name} videos, got {len(split)}")
        (OUTPUT / f"input_split_{name}.json").write_text(
            json.dumps(split, indent=2, ensure_ascii=True) + "\n"
        )

    output_manifest = {
        "source_manifest": str(QWEN50 / "selection_manifest.json"),
        "combined_video_ids": video_ids,
        "even_video_ids": video_ids[0::2],
        "odd_video_ids": video_ids[1::2],
    }
    (OUTPUT / "input_manifest.json").write_text(
        json.dumps(output_manifest, indent=2) + "\n"
    )
    print({"output": str(OUTPUT), "videos": len(video_ids), "split_sizes": [25, 25]})


if __name__ == "__main__":
    main()
