#!/usr/bin/env python3
"""Replay Local/Global selectors and compare retained token timestamps."""

import argparse
import csv
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.run_uniform_selector_auc import _component_rankings


SOURCE = Path("results/myobservations/lu_kv_offline_50v_20260913")
OUTPUT = Path("results/myobservations/local_global_selected_time_50v_20260922")
HEAD_RANKS = Path("results/myobservations/local_global_topk_50v_20260922/local_global_topk_head_ranks.csv")
K_VALUES = (250, 800, 1000)
RECENT = 32


def nanmean_no_warning(values):
    valid = np.isfinite(values)
    count = valid.sum(axis=0)
    return np.divide(np.nansum(values, axis=0), count,
                     out=np.full(values.shape[1:], np.nan, dtype=np.float64),
                     where=count > 0)


def selected_time_metrics(rankings, source_times, utility):
    n = len(source_times)
    history = n - RECENT
    t = source_times[:history]
    span = max(float(source_times.max() - source_times.min()), 1e-12)
    time_position = (t - source_times.min()) / span
    local, global_ = rankings[:2]
    inverse_global = np.empty_like(global_)
    np.put_along_axis(inverse_global, global_, np.arange(history)[None, :], axis=1)
    position_of_local_in_global = np.take_along_axis(inverse_global, local, axis=1)
    keys = ("shared_fraction", "local_mean_time", "global_mean_time",
            "local_current25", "global_current25", "local_current10", "global_current10",
            "local_utility_current25", "global_utility_current25", "temporal_tv_quartiles")
    result = {key: np.full((len(K_VALUES), 112), np.nan, dtype=np.float32) for key in keys}
    for j, k in enumerate(K_VALUES):
        if n < k:
            continue
        count = k - RECENT
        result["shared_fraction"][j] = (
            np.count_nonzero(position_of_local_in_global[:, :count] < count, axis=1) / count
        )
        proportions = []
        for name, indices in (("local", local[:, :count]), ("global", global_[:, :count])):
            selected_t = time_position[indices]
            result[f"{name}_mean_time"][j] = selected_t.mean(axis=1)
            result[f"{name}_current25"][j] = (selected_t >= .75).mean(axis=1)
            result[f"{name}_current10"][j] = (selected_t >= .9).mean(axis=1)
            selected_u = np.take_along_axis(utility[:, :history], indices, axis=1)
            result[f"{name}_utility_current25"][j] = (
                (selected_u * (selected_t >= .75)).sum(axis=1)
                / np.maximum(selected_u.sum(axis=1), 1e-12)
            )
            bins = np.minimum((selected_t * 4).astype(np.int64), 3)
            proportions.append(np.stack([(bins == b).mean(axis=1) for b in range(4)], axis=1))
        result["temporal_tv_quartiles"][j] = (
            .5 * np.abs(proportions[0] - proportions[1]).sum(axis=1)
        )
    return result


def summarize():
    videos = json.loads((SOURCE / "manifest.json").read_text())["videos"]
    by_video = []
    observation_count = 0
    for video in videos:
        observations = []
        for question in video["conversations"]:
            stem = f"v{video['video_id']}_q{question['source_question_index']}"
            path = OUTPUT / f"{stem}.npz"
            if path.exists():
                with np.load(path) as z:
                    part = {key: z[key].astype(np.float64) for key in z.files}
                    part["abs_mean_time_gap"] = np.abs(
                        part["local_mean_time"] - part["global_mean_time"]
                    )
                    part["abs_current25_gap"] = np.abs(
                        part["local_current25"] - part["global_current25"]
                    )
                    part["abs_utility_current25_gap"] = np.abs(
                        part["local_utility_current25"] - part["global_utility_current25"]
                    )
                    observations.append(part)
                observation_count += 1
        if observations:
            by_video.append({key: nanmean_no_warning(np.stack([part[key] for part in observations]))
                             for key in observations[0]})
    if not by_video:
        return {"status": "empty"}
    means = {key: nanmean_no_warning(np.stack([part[key] for part in by_video]))
             for key in by_video[0]}
    with HEAD_RANKS.open(newline="") as handle:
        rank_rows = list(csv.DictReader(handle))
    a = np.array([min(float(row["uniform_local_overlap_auc"]),
                      float(row["uniform_global_overlap_auc"])) for row in rank_rows])
    high = np.array([
        h for h, row in enumerate(rank_rows)
        if int(row["uniform_local_overlap_auc_rank"]) <= 10
        and int(row["uniform_global_overlap_auc_rank"]) <= 10
    ], dtype=np.int64)
    low = np.argsort(a)[:10]
    report = {"status": "complete" if len(by_video) == len(videos) else "partial",
              "videos": len(by_video), "observations": observation_count,
              "K": list(K_VALUES), "time_normalization": "0=earliest, 1=latest observed video time",
              "history_only": True, "recent_excluded": RECENT,
              "high_auc_heads": [[int(h // 4), int(h % 4)] for h in high],
              "metrics": {}}
    for j, k in enumerate(K_VALUES):
        local_mean = means["local_mean_time"][j]
        global_mean = means["global_mean_time"][j]
        local_cur = means["local_current25"][j]
        global_cur = means["global_current25"][j]
        report["metrics"][str(k)] = {
            "all_heads_shared_fraction_mean": float(np.nanmean(means["shared_fraction"][j])),
            "high_auc_shared_fraction_mean": float(np.nanmean(means["shared_fraction"][j, high])),
            "low_auc_shared_fraction_mean": float(np.nanmean(means["shared_fraction"][j, low])),
            "all_heads_mean_time_local": float(np.nanmean(local_mean)),
            "all_heads_mean_time_global": float(np.nanmean(global_mean)),
            "all_heads_abs_mean_time_gap": float(np.nanmean(means["abs_mean_time_gap"][j])),
            "high_auc_abs_mean_time_gap": float(np.nanmean(means["abs_mean_time_gap"][j, high])),
            "all_heads_current25_local": float(np.nanmean(local_cur)),
            "all_heads_current25_global": float(np.nanmean(global_cur)),
            "all_heads_abs_current25_gap": float(np.nanmean(means["abs_current25_gap"][j])),
            "high_auc_abs_current25_gap": float(np.nanmean(means["abs_current25_gap"][j, high])),
            "all_heads_temporal_tv": float(np.nanmean(means["temporal_tv_quartiles"][j])),
            "high_auc_temporal_tv": float(np.nanmean(means["temporal_tv_quartiles"][j, high])),
            "high_auc_utility_current25_local": float(np.nanmean(means["local_utility_current25"][j, high])),
            "high_auc_utility_current25_global": float(np.nanmean(means["global_utility_current25"][j, high])),
            "high_auc_abs_utility_current25_gap": float(np.nanmean(means["abs_utility_current25_gap"][j, high])),
        }
    rows = []
    for h in range(112):
        row = {"layer": h // 4, "kv_head": h % 4, "consensus_topk_auc": float(a[h])}
        for j, k in enumerate(K_VALUES):
            for name, values in means.items():
                row[f"k{k}_{name}"] = float(values[j, h])
        rows.append(row)
    with (OUTPUT / "head_selected_time.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def run(max_videos):
    import torch
    from head_analysis.profile_effective_memory_readout import load_video_tensor
    from head_analysis.profile_horizonkv_pilot import build_pilot_model
    from video_qa.base import BaseVQA

    OUTPUT.mkdir(parents=True, exist_ok=True)
    videos = json.loads((SOURCE / "manifest.json").read_text())["videos"]
    if max_videos:
        videos = videos[:max_videos]
    model_args = SimpleNamespace(model="qwen2.5_vl_7b", device=0,
                                 fixed_frame_canvas_size=448, kv_size=1000000,
                                 sample_fps=0.5, compress_mode="hermes")
    model, processor = build_pilot_model(model_args, "models/Qwen2.5-VL-7B-Instruct")
    analyzer = BaseVQA(anno=videos, save_dir=str(OUTPUT / "unused_qa_output"),
                       sample_fps=0.5, qa_model=model, qa_processor=processor,
                       num_chunks=None, chunk_idx=None)
    for video in videos:
        started = time.time()
        print(f"SELECTED_TIME_VIDEO_START id={video['video_id']}", flush=True)
        clip_start = float(video.get("clip", [0, 0])[0]) if video.get("clip") else 0.0
        load_sample = {**video, "clip": [clip_start, clip_start + max(
            float(q["end_time"]) for q in video["conversations"])]}
        video_tensor = load_video_tensor(analyzer, load_sample)
        model.clear_cache()
        model._position_ids_cache = [None] * model.num_layers
        model._layer_position_ids.clear()
        model.total_processed_frames = 0
        model.encode_init_prompt()
        current = 0
        for question in video["conversations"]:
            end = min(len(video_tensor), math.ceil(float(question["end_time"]) * .5))
            while current < end:
                stop = min(current + 16, end)
                model.encode_video_chunk(video_tensor[current:stop])
                current = stop
            stem = f"v{video['video_id']}_q{question['source_question_index']}"
            path = OUTPUT / f"{stem}.npz"
            if path.exists():
                print(f"SELECTED_TIME_SKIP {stem}", flush=True)
                continue
            with np.load(SOURCE / f"{stem}.npz") as z:
                utility = z["utility"].astype(np.float64)
                source_times = z["source_times"].astype(np.float64)
            if model.kv_cache[0][0].shape[2] - int(model.visual_start_idx) != utility.shape[1]:
                raise AssertionError(stem)
            rankings, _ = _component_rankings(model, processor, RECENT)
            np.savez_compressed(path, **selected_time_metrics(rankings, source_times, utility))
            print(f"SELECTED_TIME_OBSERVATION_DONE {stem}", flush=True)
        print(f"SELECTED_TIME_VIDEO_DONE id={video['video_id']} elapsed={time.time()-started:.1f}s", flush=True)
    print(json.dumps(summarize(), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if args.analyze_only:
        print(json.dumps(summarize(), indent=2))
    else:
        import torch
        torch.manual_seed(20260913)
        torch.set_num_threads(4)
        run(args.max_videos)


if __name__ == "__main__":
    main()
