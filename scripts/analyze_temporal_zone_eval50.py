#!/usr/bin/env python3
"""Paired summary for the temporal-zone first-50 feasibility experiment."""

import argparse
import json
import math
import os

import numpy as np
import pandas as pd


def exact_mcnemar_pvalue(left_wins, right_wins):
    discordant = int(left_wins) + int(right_wins)
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(left_wins, right_wins) + 1))
    return min(1.0, 2.0 * tail / (2 ** discordant))


def paired_video_bootstrap_ci(frame, left, right, samples=10000, seed=20260907):
    """Question-weighted accuracy-gap CI with videos resampled as clusters."""
    by_video = frame.assign(
        difference=frame[left].astype(float) - frame[right].astype(float)
    ).groupby(level="video_id")["difference"].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(by_video), size=(int(samples), len(by_video)))
    numerators = by_video["sum"].to_numpy()[draws].sum(axis=1)
    denominators = by_video["count"].to_numpy()[draws].sum(axis=1)
    gaps = 100.0 * numerators / denominators
    return [float(value) for value in np.percentile(gaps, [2.5, 97.5])]


def load_method(path, name):
    frame = pd.read_csv(path)
    required = {"video_id", "question", "qa_acc", "task"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["video_id"] = frame["video_id"].astype(str)
    frame["correct"] = frame["qa_acc"].astype(float) > 0.0
    frame["question_occurrence"] = frame.groupby(
        ["video_id", "question"], sort=False
    ).cumcount()
    return frame.set_index(
        ["video_id", "question", "question_occurrence"]
    )[["correct", "task"]].rename(
        columns={"correct": name}
    )


def annotation_metadata(path):
    with open(path) as handle:
        annotation = json.load(handle)
    rows = []
    for video in annotation:
        for question in video.get("conversations", []):
            rows.append(
                {
                    "video_id": str(video["video_id"]),
                    "question": question["question"],
                    "end_time": float(question["end_time"]),
                    "ability": question.get("required_ability", "unknown"),
                }
            )
    frame = pd.DataFrame(rows)
    frame["question_occurrence"] = frame.groupby(
        ["video_id", "question"], sort=False
    ).cumcount()
    return frame.set_index(["video_id", "question", "question_occurrence"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno", required=True)
    parser.add_argument("--uniform", required=True)
    parser.add_argument("--global_profile", required=True)
    parser.add_argument("--matched", required=True)
    parser.add_argument("--shuffled")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()

    paths = {
        "uniform": args.uniform,
        "global": args.global_profile,
        "matched": args.matched,
    }
    if args.shuffled:
        paths["shuffled"] = args.shuffled
    frames = [load_method(path, name) for name, path in paths.items()]
    joined = annotation_metadata(args.anno)
    for frame in frames:
        joined = joined.join(frame.drop(columns="task"), how="inner")
    task_frame = frames[0][["task"]]
    joined = joined.join(task_frame, how="left")

    result = {
        "num_paired_questions": int(len(joined)),
        "accuracy": {
            name: float(100.0 * joined[name].mean()) for name in paths
        },
        "by_time": {},
        "by_task": {},
        "paired_against_matched": {},
    }
    time_groups = {
        "under_64s": joined["end_time"] < 64.0,
        "64s_or_later": joined["end_time"] >= 64.0,
    }
    for label, mask in time_groups.items():
        result["by_time"][label] = {
            "n": int(mask.sum()),
            **{name: float(100.0 * joined.loc[mask, name].mean()) for name in paths},
        }
    for task, part in joined.groupby("task"):
        result["by_task"][task] = {
            "n": int(len(part)),
            **{name: float(100.0 * part[name].mean()) for name in paths},
        }
    for baseline in (name for name in paths if name != "matched"):
        matched_wins = int((joined["matched"] & ~joined[baseline]).sum())
        baseline_wins = int((~joined["matched"] & joined[baseline]).sum())
        result["paired_against_matched"][baseline] = {
            "matched_only_correct": matched_wins,
            "baseline_only_correct": baseline_wins,
            "accuracy_delta_points": float(
                100.0 * (joined["matched"].mean() - joined[baseline].mean())
            ),
            "video_bootstrap_ci95": paired_video_bootstrap_ci(
                joined, "matched", baseline, args.bootstrap_samples, args.seed
            ),
            "exact_mcnemar_p": exact_mcnemar_pvalue(matched_wins, baseline_wins),
        }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=True)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
