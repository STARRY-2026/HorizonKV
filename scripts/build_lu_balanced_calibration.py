#!/usr/bin/env python3
"""Build a deterministic, evaluation-disjoint LU calibration manifest."""

import itertools
import json
import random
from collections import Counter
from pathlib import Path


SOURCE = Path("data/streamingbench/streamingbench_realtime.json")
HELDOUT = Path("data/streamingbench/streamingbench_realtime_half_userpaths.json")
OUT = Path("results/myobservations/lu_kv_balanced100v_20260915/calibration_100v.json")
AUDIT = OUT.with_name("calibration_audit.json")
VIDEO_ROOT = Path("/home/yujun/program/streaming_data/streamingbench/videos")
NUM_VIDEOS = 100
QUESTIONS_PER_VIDEO = 2
SEED = 20260915


source = json.loads(SOURCE.read_text())
heldout = json.loads(HELDOUT.read_text())
heldout_ids = {str(x["video_id"]) for x in heldout}
heldout_names = {Path(x["video_path"]).name for x in heldout}

pool = []
for video in source:
    name = Path(video["video_path"]).name
    if str(video["video_id"]) in heldout_ids or name in heldout_names:
        continue
    copied = dict(video)
    copied["video_path"] = str(VIDEO_ROOT / name)
    if not Path(copied["video_path"]).is_file():
        raise FileNotFoundError(copied["video_path"])
    if len(copied["conversations"]) >= QUESTIONS_PER_VIDEO:
        pool.append(copied)

tasks = sorted({q["task"] for v in pool for q in v["conversations"]})
target = {task: NUM_VIDEOS * QUESTIONS_PER_VIDEO / len(tasks) for task in tasks}
rng = random.Random(SEED)
rng.shuffle(pool)
selected = []
counts = Counter()

# Greedily choose a video and a two-question pair that best reduces global
# task imbalance. A small time-span preference avoids choosing two near-identical
# temporal checkpoints when task balance is tied.
remaining = list(pool)
while len(selected) < NUM_VIDEOS:
    best = None
    for vi, video in enumerate(remaining):
        questions = video["conversations"]
        for pair in itertools.combinations(range(len(questions)), QUESTIONS_PER_VIDEO):
            added = Counter(questions[i]["task"] for i in pair)
            before = sum((counts[t] - target[t]) ** 2 for t in tasks)
            after = sum((counts[t] + added[t] - target[t]) ** 2 for t in tasks)
            times = [float(questions[i].get("end_time", 0)) for i in pair]
            span = abs(times[1] - times[0])
            late = max(times)
            score = (before - after, span / 10000.0 + late / 1000000.0)
            if best is None or score > best[0]:
                best = (score, vi, pair)
    _, vi, pair = best
    video = remaining.pop(vi)
    chosen_questions = []
    for i in pair:
        q = dict(video["conversations"][i])
        q["source_question_index"] = i
        chosen_questions.append(q)
        counts[q["task"]] += 1
    chosen_questions.sort(key=lambda q: float(q.get("end_time", 0)))
    selected.append({**video, "conversations": chosen_questions})

selected_ids = [str(v["video_id"]) for v in selected]
selected_names = [Path(v["video_path"]).name for v in selected]
question_count = sum(len(v["conversations"]) for v in selected)
overlap = (set(selected_ids) & heldout_ids) | (set(selected_names) & heldout_names)
assert len(selected) == NUM_VIDEOS
assert len(set(selected_ids)) == NUM_VIDEOS
assert len(set(selected_names)) == NUM_VIDEOS
assert question_count == NUM_VIDEOS * QUESTIONS_PER_VIDEO
assert not overlap
assert set(counts) == set(tasks)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n")
audit = {
    "source": str(SOURCE),
    "heldout": str(HELDOUT),
    "seed": SEED,
    "pool_videos": len(pool),
    "selected_videos": len(selected),
    "unique_video_ids": len(set(selected_ids)),
    "questions": question_count,
    "questions_per_video": QUESTIONS_PER_VIDEO,
    "evaluation_overlap": len(overlap),
    "task_counts": dict(sorted(counts.items())),
    "video_ids": selected_ids,
}
AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({k: v for k, v in audit.items() if k != "video_ids"}, indent=2))
