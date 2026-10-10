#!/usr/bin/env python3
"""Build a representative prefix subset for temporal-utility cache diagnosis."""

import argparse
import csv
import json
from pathlib import Path


TARGETS = [
    (12, 3),    # working / spatial / 96s
    (35, 0),    # working / object / 122s
    (57, 1),    # working / action / 139s
    (169, 1),   # working / text / 167s
    (145, 0),   # working / attribute / 181s
    (347, 0),   # working / object / 79s
    (93, 0),    # episodic / prospective / 40s
    (208, 3),   # episodic / action / 135s
    (231, 4),   # episodic / object / 205s
    (448, 2),   # episodic / event / 96s
    (237, 3),   # episodic / counting / 301s
    (330, 2),   # episodic / prospective / 256s
]


def result_map(path):
    counters = {}
    result = {}
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            video_id = int(row["video_id"])
            question_idx = counters.get(video_id, 0)
            counters[video_id] = question_idx + 1
            result[(video_id, question_idx)] = row
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno", required=True)
    parser.add_argument("--hermes", required=True)
    parser.add_argument("--full", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    annotation = json.loads(Path(args.anno).read_text())
    videos = {int(video["video_id"]): video for video in annotation}
    hermes = result_map(args.hermes)
    full = result_map(args.full)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    subset = []
    records = []
    for video_id, question_idx in TARGETS:
        video = videos[video_id]
        question = video["conversations"][question_idx]
        h_row = hermes[(video_id, question_idx)]
        f_row = full[(video_id, question_idx)]
        if not (float(h_row["qa_acc"]) > 0 and float(f_row["qa_acc"]) == 0):
            raise ValueError(f"Target {(video_id, question_idx)} is not Hermes-correct/Full-wrong")
        subset.append({**video, "conversations": video["conversations"][:question_idx + 1]})
        records.append({
            "video_id": video_id,
            "question_index": question_idx,
            "end_time": float(question["end_time"]),
            "task": question.get("task"),
            "ability": question.get("required_ability"),
            "question": question["question"],
            "correct_choice": h_row["correct_choice"],
            "hermes_pred_choice": h_row["pred_choice"],
            "full_pred_choice": f_row["pred_choice"],
        })

    (output / "filtered_anno.json").write_text(json.dumps(subset, indent=2) + "\n")
    manifest = {
        "selection": "representative Hermes-correct and temporal-utility-Full-wrong",
        "num_videos": len(subset),
        "num_prefix_questions": sum(len(v["conversations"]) for v in subset),
        "num_targets": len(records),
        "targets": records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
