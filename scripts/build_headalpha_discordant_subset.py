#!/usr/bin/env python3
"""Build a full-video subset around both directions of v8/Head-Alpha disagreement."""

import argparse
import csv
import json
import os


IDENTITY_FIELDS = ("video_id", "question", "task")


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--v8_results", required=True)
    parser.add_argument("--headalpha_results", required=True)
    parser.add_argument("--output_annotation", required=True)
    parser.add_argument("--output_targets", required=True)
    args = parser.parse_args()

    v8_rows = load_csv(args.v8_results)
    headalpha_rows = load_csv(args.headalpha_results)
    if len(v8_rows) != len(headalpha_rows):
        raise ValueError(
            f"Result length mismatch: {len(v8_rows)} vs {len(headalpha_rows)}"
        )

    targets = []
    for row_idx, (v8_row, headalpha_row) in enumerate(zip(v8_rows, headalpha_rows)):
        if any(
            v8_row.get(field) != headalpha_row.get(field)
            for field in IDENTITY_FIELDS
        ):
            raise ValueError(f"Result identity mismatch at row {row_idx}")

        v8_correct = float(v8_row["qa_acc"]) > 0.0
        headalpha_correct = float(headalpha_row["qa_acc"]) > 0.0
        if v8_correct == headalpha_correct:
            continue
        direction = "v8_gain" if v8_correct else "headalpha_gain"
        targets.append({
            "source_row": row_idx,
            "direction": direction,
            "video_id": str(headalpha_row["video_id"]),
            "task": headalpha_row.get("task", ""),
            "question": headalpha_row["question"],
            "correct_choice": headalpha_row.get("correct_choice", ""),
            "v8_pred": v8_row.get("pred_answer", ""),
            "headalpha_pred": headalpha_row.get("pred_answer", ""),
        })

    direction_counts = {
        direction: sum(row["direction"] == direction for row in targets)
        for direction in ("v8_gain", "headalpha_gain")
    }
    if not all(direction_counts.values()):
        raise ValueError(f"Both disagreement directions are required: {direction_counts}")

    target_video_ids = {row["video_id"] for row in targets}
    with open(args.annotation) as f:
        annotation = json.load(f)
    subset = [
        sample
        for sample in annotation
        if str(sample["video_id"]) in target_video_ids
    ]
    found_ids = {str(sample["video_id"]) for sample in subset}
    if found_ids != target_video_ids:
        raise ValueError(
            f"Annotation is missing target videos: {sorted(target_video_ids - found_ids)}"
        )

    os.makedirs(os.path.dirname(args.output_annotation) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_targets) or ".", exist_ok=True)
    with open(args.output_annotation, "w") as f:
        json.dump(subset, f, indent=2, ensure_ascii=False)
        f.write("\n")
    with open(args.output_targets, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(targets[0]))
        writer.writeheader()
        writer.writerows(targets)

    question_count = sum(len(sample["conversations"]) for sample in subset)
    print(
        f"Built {len(subset)}-video/{question_count}-question subset around "
        f"{direction_counts['v8_gain']} v8 gains and "
        f"{direction_counts['headalpha_gain']} Head-Alpha gains"
    )


if __name__ == "__main__":
    main()
