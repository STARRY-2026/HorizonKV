#!/usr/bin/env python3
"""Run one StreamingBench method across torchrun ranks and merge exact results."""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
import csv
from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys


def split_videos(data, world_size):
    if world_size < 1 or len(data) < world_size:
        raise ValueError("Need at least one video per rank")
    shards = [[] for _ in range(world_size)]
    loads = [0.0] * world_size
    items = []
    for index, video in enumerate(data):
        weight = max(float(q["end_time"]) for q in video["conversations"])
        items.append((index, weight))
    for index, weight in sorted(items, key=lambda x: (-x[1], x[0])):
        rank = min(range(world_size), key=lambda r: (loads[r], r))
        shards[rank].append(index)
        loads[rank] += weight
    for shard in shards:
        shard.sort()
    assert sorted(i for shard in shards for i in shard) == list(range(len(data)))
    return shards, loads


def question_key(video_id, question):
    return str(video_id), str(question)


def merge_rows(data, shards, shard_rows):
    expected = {}
    for rank, indices in enumerate(shards):
        keys = defaultdict(deque)
        for video_index in indices:
            video = data[video_index]
            for question_index, question in enumerate(video["conversations"]):
                keys[question_key(video["video_id"], question["question"])].append((video_index, question_index))
        if len(shard_rows[rank]) != sum(len(data[i]["conversations"]) for i in indices):
            raise ValueError(f"Rank {rank}: wrong result count")
        for row in shard_rows[rank]:
            key = question_key(row["video_id"], row["question"])
            if not keys[key]:
                raise ValueError(f"Rank {rank}: duplicate or unexpected result {key}")
            location = keys[key].popleft()
            if location in expected:
                raise ValueError(f"Duplicate result at {location}")
            expected[location] = row
        if any(keys.values()):
            raise ValueError(f"Rank {rank}: missing result")
    locations = [(i, j) for i, video in enumerate(data) for j in range(len(video["conversations"]))]
    if set(expected) != set(locations):
        raise ValueError("Global result coverage mismatch")
    return [expected[location] for location in locations]


def merge_outputs(data, shards, output):
    shard_rows = []
    fieldnames = None
    manifests = []
    for rank in range(len(shards)):
        folder = output / "shards" / f"rank-{rank:02d}"
        manifest = json.loads((folder / "experiment_manifest.json").read_text())
        if manifest.get("status") != "completed":
            raise ValueError(f"Rank {rank}: incomplete manifest")
        manifests.append(manifest)
        with (folder / "results.csv").open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            elif reader.fieldnames != fieldnames:
                raise ValueError(f"Rank {rank}: inconsistent CSV fields")
            shard_rows.append(list(reader))
    rows = merge_rows(data, shards, shard_rows)
    with (output / "results.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    accuracy = sum(float(row["qa_acc"]) for row in rows) / len(rows)
    report = {"status": "completed", "videos": len(data), "questions": len(rows),
              "world_size": len(shards), "accuracy_percent": accuracy,
              "shard_videos": [len(s) for s in shards],
              "shard_questions": [len(r) for r in shard_rows],
              "shard_manifests": [str(output / "shards" / f"rank-{r:02d}" / "experiment_manifest.json") for r in range(len(shards))]}
    (output / "merged_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["dp", "hermes"], required=True)
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if rank < 0 or rank >= world_size or local_rank < 0:
        raise ValueError("Invalid torchrun rank")
    data = json.loads(args.annotation.read_text(encoding="utf-8"))
    shards, loads = split_videos(data, world_size)
    output = args.output_dir
    shard_input = output / "inputs" / f"rank-{rank:02d}.json"
    shard_output = output / "shards" / f"rank-{rank:02d}"
    shard_input.parent.mkdir(parents=True, exist_ok=True)
    shard_output.parent.mkdir(parents=True, exist_ok=True)
    with shard_input.open("x", encoding="utf-8") as handle:
        json.dump([data[i] for i in shards[rank]], handle, ensure_ascii=False)
    command = [sys.executable, str(args.runner), "--method", args.method,
               "--budget", str(args.budget), "--annotation", str(shard_input),
               "--model-path", str(args.model_path), "--video-root", str(args.video_root),
               "--output-dir", str(shard_output), "--device", str(local_rank), "--full"]
    if args.dry_run:
        command.append("--dry-run")
    print(f"RANK {rank}/{world_size} LOCAL_RANK {local_rank}: {len(shards[rank])} videos, "
          f"{sum(len(data[i]['conversations']) for i in shards[rank])} questions", flush=True)
    subprocess.run(command, check=True)
    # torchrun provides the rendezvous; Gloo synchronizes only after all rank-local
    # evaluations complete. A failed rank terminates the job without a partial merge.
    if world_size > 1:
        import torch.distributed as dist
        dist.init_process_group("gloo", timeout=timedelta(hours=24))
        dist.barrier()
    if rank == 0:
        if args.dry_run:
            print(json.dumps({"shard_indices": shards, "estimated_loads": loads}), flush=True)
        else:
            print(json.dumps(merge_outputs(data, shards, output), indent=2), flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
