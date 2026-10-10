#!/usr/bin/env python3
"""Build a full-video subset around regressions from v8 to Head-Alpha."""

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
    for row_idx, (v8_row, headalpha_row) in enumerate(
        zip(v8_rows, headalpha_rows)
    ):
        if any(
            v8_row.get(field) != headalpha_row.get(field)
            for field in IDENTITY_FIELDS
        ):
            raise ValueError(f"Result identity mismatch at row {row_idx}")
        if (
            float(v8_row["qa_acc"]) > 0.0
            and float(headalpha_row["qa_acc"]) == 0.0
        ):
            targets.append({
                "source_row": row_idx,
                "video_id": str(headalpha_row["video_id"]),
                "task": headalpha_row.get("task", ""),
                "question": headalpha_row["question"],
                "correct_choice": headalpha_row.get("correct_choice", ""),
                "v8_pred": v8_row.get("pred_answer", ""),
                "headalpha_pred": headalpha_row.get("pred_answer", ""),
            })

    if not targets:
        raise ValueError("No v8-correct/Head-Alpha-wrong targets found")
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
        f"Built {len(subset)}-video/{question_count}-question subset "
        f"around {len(targets)} target regressions"
    )


if __name__ == "__main__":
    main()
