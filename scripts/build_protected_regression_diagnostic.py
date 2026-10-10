#!/usr/bin/env python3
"""Build full-prefix traces for Hermes-correct, protected-policy-wrong questions."""

import argparse
import json
from pathlib import Path

import pandas as pd


def load_answers(path):
    frame = pd.read_csv(path)
    frame["video_id"] = frame["video_id"].astype(str)
    return {
        video_id: group.reset_index(drop=True)
        for video_id, group in frame.groupby("video_id", sort=False)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno", required=True)
    parser.add_argument("--hermes", required=True)
    parser.add_argument("--protected", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    annotation = json.loads(Path(args.anno).read_text())
    hermes = load_answers(args.hermes)
    protected = load_answers(args.protected)
    subset = []
    targets = []
    for video in annotation:
        video_id = str(video["video_id"])
        h_rows = hermes[video_id]
        p_rows = protected[video_id]
        questions = video.get("conversations", [])
        if not (len(h_rows) == len(p_rows) == len(questions)):
            raise ValueError(f"Question count mismatch for video {video_id}")
        target_indices = [
            idx
            for idx in range(len(questions))
            if float(h_rows.iloc[idx]["qa_acc"]) > 0.0
            and float(p_rows.iloc[idx]["qa_acc"]) == 0.0
        ]
        if not target_indices:
            continue
        last = max(target_indices)
        subset.append({**video, "conversations": questions[: last + 1]})
        for idx in target_indices:
            question = questions[idx]
            targets.append(
                {
                    "video_id": video["video_id"],
                    "question_index": idx,
                    "end_time": float(question["end_time"]),
                    "task": question.get("task"),
                    "ability": question.get("required_ability"),
                    "question": question["question"],
                    "answer": question.get("answer"),
                    "hermes_pred_choice": h_rows.iloc[idx]["pred_choice"],
                    "protected_pred_choice": p_rows.iloc[idx]["pred_choice"],
                    "correct_choice": h_rows.iloc[idx]["correct_choice"],
                }
            )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "filtered_anno.json").write_text(
        json.dumps(subset, indent=2, ensure_ascii=True) + "\n"
    )
    manifest = {
        "selection": "Hermes correct and protected temporal-zone policy wrong",
        "num_videos": len(subset),
        "num_prefix_questions": sum(len(v["conversations"]) for v in subset),
        "num_targets": len(targets),
        "targets": targets,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
