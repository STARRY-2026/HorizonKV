#!/usr/bin/env python3
"""Build full-prefix diagnostics for Hermes-correct, Matched-wrong questions."""

import argparse
import json
import os

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
    parser.add_argument("--matched", required=True)
    parser.add_argument("--save_dir", required=True)
    args = parser.parse_args()

    with open(args.anno) as handle:
        annotation = json.load(handle)
    hermes = load_answers(args.hermes)
    matched = load_answers(args.matched)

    subset = []
    targets = []
    for video in annotation:
        video_id = str(video["video_id"])
        h_rows = hermes[video_id]
        m_rows = matched[video_id]
        questions = video.get("conversations", [])
        if not (len(h_rows) == len(m_rows) == len(questions)):
            raise ValueError(f"Question count mismatch for video {video_id}")
        target_indices = [
            idx for idx in range(len(questions))
            if float(h_rows.iloc[idx]["qa_acc"]) > 0.0
            and float(m_rows.iloc[idx]["qa_acc"]) == 0.0
        ]
        if not target_indices:
            continue
        last = max(target_indices)
        subset.append({**video, "conversations": questions[: last + 1]})
        for idx in target_indices:
            targets.append(
                {
                    "video_id": video["video_id"],
                    "question_index": idx,
                    "end_time": float(questions[idx]["end_time"]),
                    "task": questions[idx].get("task"),
                    "ability": questions[idx].get("required_ability"),
                    "question": questions[idx]["question"],
                    "hermes_pred_choice": h_rows.iloc[idx]["pred_choice"],
                    "matched_pred_choice": m_rows.iloc[idx]["pred_choice"],
                    "correct_choice": h_rows.iloc[idx]["correct_choice"],
                }
            )

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "filtered_anno.json"), "w") as handle:
        json.dump(subset, handle, indent=2, ensure_ascii=True)
    manifest = {
        "selection": "Hermes correct and no-protection Matched wrong",
        "num_videos": len(subset),
        "num_prefix_questions": sum(len(v["conversations"]) for v in subset),
        "num_targets": len(targets),
        "targets": targets,
    }
    with open(os.path.join(args.save_dir, "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)
    print(json.dumps(manifest, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
