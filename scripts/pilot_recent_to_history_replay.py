#!/usr/bin/env python3
"""Checkpoint-level no-resurrection pilot for recent-to-history promotion.

Uses full-cache HERMES score traces and question readout traces. Every source
frame/content-bin unit is treated as an indivisible 64-token item. Only new
items may enter at a checkpoint; previously evicted items cannot return.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


TRACE = Path("results/myobservations/prospective_lifecycle_pilot6/hermes_trace_0.jsonl")
OUT = Path("results/myobservations/recent_to_history_replay_20260923")
WINDOWS = (0, 1, 2)  # number of newest Qwen temporal units, 256 tokens each
CAPACITIES = (8, 16, 24)  # each item contains 64 visual tokens
HEADS = 112


def flatten(record, field):
    result = {}
    for head in record["heads"]:
        hid = int(head["layer"]) * 4 + int(head["kv_head"])
        units = {}
        for frame in head["frames"]:
            frame_id = int(frame["source_frame_id"])
            for content in frame["content_bins"]:
                key = (frame_id, int(content["content_bin"]))
                if field == "score":
                    value = float(content["attention_mass"]) / max(1, int(content["token_count"]))
                else:
                    value = float(content["effective_readout_mass"])
                units[key] = value
        result[hid] = units
    if len(result) != HEADS:
        raise AssertionError("Unexpected number of physical KV heads")
    return result


def load():
    events = defaultdict(list)
    questions = defaultdict(list)
    for line in TRACE.open(encoding="utf-8"):
        record = json.loads(line)
        if not record["full_cache"]:
            raise AssertionError("Compressed-cache trace cannot be used here")
        video = str(record["video_id"])
        frame = int(record["observation_frame_id"])
        if record["query_source"] == "hermes":
            events[video].append((frame, flatten(record, "score")))
        elif record["query_source"] == "true_question":
            questions[video].append((frame, flatten(record, "utility")))
    if set(events) != set(questions):
        raise AssertionError("Question/score videos do not match")
    return events, questions


def simulate(video_events, video_questions, capacity, windows):
    """Return per-question, per-head numerator and denominator."""
    retained = [set() for _ in range(HEADS)]
    seen = [set() for _ in range(HEADS)]
    qindex = 0
    output = []
    for frame, scores in video_events:
        for head in range(HEADS):
            candidate = scores[head]
            new = set(candidate) - seen[head]
            if not seen[head] and not new:
                raise AssertionError("No initial units")
            # New units are admitted; absent old units are never resurrected.
            pool = (retained[head] | new) & set(candidate)
            seen[head].update(candidate)
            newest = max(key[0] for key in candidate)
            eligible_recent = [key for key in pool if key[0] >= newest - 2 * (windows[head] - 1)] if windows[head] else []
            eligible_recent.sort(key=lambda key: (key[0], candidate[key], key[1]), reverse=True)
            protected = set(eligible_recent[:min(capacity, 4 * windows[head])])
            remainder = sorted(pool - protected, key=lambda key: (candidate[key], key[0], key[1]), reverse=True)
            retained[head] = protected | set(remainder[:capacity - len(protected)])
            if len(retained[head]) > capacity or not retained[head] <= pool:
                raise AssertionError("Invalid capacity or resurrection")
        while qindex < len(video_questions) and video_questions[qindex][0] == frame:
            _, utility = video_questions[qindex]
            numer = np.zeros(HEADS)
            denom = np.zeros(HEADS)
            for head in range(HEADS):
                label = utility[head]
                numer[head] = sum(label.get(key, 0.0) for key in retained[head])
                denom[head] = sum(label.values())
            output.append((numer, denom))
            qindex += 1
    if qindex != len(video_questions):
        raise AssertionError("Unmatched question checkpoint")
    return output


def summarize(outputs):
    num = np.mean([x[0] for x in outputs], axis=0)
    den = np.mean([x[1] for x in outputs], axis=0)
    return num, den


def video_retention(outputs):
    return float(np.mean([num.sum() / den.sum() for num, den in outputs]))


def main():
    events, questions = load()
    videos = sorted(events)
    rng = np.random.default_rng(20260923)
    report = {"protocol": {
        "trace": str(TRACE), "capacities_64_token_items": CAPACITIES,
        "protected_recent_temporal_units": WINDOWS,
        "calibration": "leave-one-video-out; per-head window chosen by training question readout retention",
        "warning": "Full-cache checkpoint scores and readout; not per-chunk online scores, compressed-cache QA, or causal accuracy."
    }, "videos": len(videos), "questions": sum(map(len, questions.values())), "capacities": {}}
    for capacity in CAPACITIES:
        cache = {(vid, w): simulate(events[vid], questions[vid], capacity, [w] * HEADS)
                 for vid in videos for w in WINDOWS}
        rows = []
        assignments = np.zeros((HEADS, len(WINDOWS)), dtype=np.int64)
        for heldout in videos:
            # Sum within video, then average videos equally during fitting.
            train_ret = np.zeros((HEADS, len(WINDOWS)))
            for vid in videos:
                if vid == heldout:
                    continue
                for wi, w in enumerate(WINDOWS):
                    num, den = summarize(cache[vid, w])
                    train_ret[:, wi] += num / np.maximum(den, 1e-12)
            shared_idx = int(np.argmax(train_ret.sum(axis=0)))
            head_idx = np.argmax(train_ret, axis=1)
            np.add.at(assignments, (np.arange(HEADS), head_idx), 1)
            choices = {
                "no_protection": [0] * HEADS,
                "shared_window": [WINDOWS[shared_idx]] * HEADS,
                "head_window": [WINDOWS[wi] for wi in head_idx],
                "shuffled_head_window": [WINDOWS[wi] for wi in rng.permutation(head_idx)],
            }
            scores = {name: video_retention(simulate(events[heldout], questions[heldout], capacity, selection))
                      for name, selection in choices.items()}
            rows.append({"video": heldout, "shared_window": WINDOWS[shared_idx], **scores})
        names = ("no_protection", "shared_window", "head_window", "shuffled_head_window")
        paired = {}
        for a, b in (("head_window", "shared_window"), ("head_window", "no_protection"),
                     ("head_window", "shuffled_head_window")):
            d = np.array([row[a] - row[b] for row in rows])
            boot = rng.choice(d, size=(10000, len(d)), replace=True).mean(axis=1)
            paired[f"{a}_minus_{b}"] = {"mean_pp": float(100 * d.mean()),
                "bootstrap95_pp": (100 * np.quantile(boot, [.025, .975])).tolist(),
                "positive_videos": int(sum(d > 0)), "negative_videos": int(sum(d < 0))}
        report["capacities"][str(capacity * 64)] = {
            "mean_video_retention": {name: float(np.mean([row[name] for row in rows])) for name in names},
            "paired": paired, "chosen_head_windows_counts": assignments.sum(axis=0).tolist(),
            "per_video": rows,
        }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
