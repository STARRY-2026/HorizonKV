#!/usr/bin/env python3
"""Replay Local/Global selectors and measure overlap with utility top-k tokens."""

import argparse
import csv
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.run_uniform_selector_auc import COMPRESSION, _component_rankings


POLICIES = ("uniform_local", "uniform_global")
K_VALUES = (250, 500, 800, 1000)


def overlap_curve(utility, ranking, recent):
    """Compare top-m historical sets; exclude the always-kept recent window."""
    heads, n = utility.shape
    history = n - recent
    oracle = np.argsort(-utility[:, :history], axis=1, kind="stable")
    oracle_position = np.empty_like(oracle)
    np.put_along_axis(oracle_position, oracle, np.arange(history)[None, :], axis=1)
    selected_positions = np.take_along_axis(oracle_position, ranking, axis=1)

    budgets = [max(recent, min(n, int(round((1 - c) * n)))) for c in COMPRESSION]
    budgets.extend(K_VALUES)
    overlaps = np.full((heads, len(budgets)), np.nan, dtype=np.float64)
    for i, budget in enumerate(budgets):
        m = min(history, max(0, budget - recent))
        if m:
            overlaps[:, i] = np.count_nonzero(selected_positions[:, :m] < m, axis=1) / m
    return overlaps[:, :len(COMPRESSION)], overlaps[:, len(COMPRESSION):]


def ranks(values):
    order = np.argsort(-values, kind="stable")
    out = np.empty(len(order), dtype=np.int64)
    out[order] = np.arange(1, len(order) + 1)
    return out


def rank_correlation(a, b):
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def nanmean_no_warning(values, axis=0):
    valid = np.isfinite(values)
    return np.divide(np.nansum(values, axis=axis), valid.sum(axis=axis),
                     out=np.full(values.shape[1:], np.nan, dtype=np.float64),
                     where=valid.sum(axis=axis) > 0)


def summarize(source, output):
    videos = json.loads((source / "manifest.json").read_text())["videos"]
    by_video = []
    count = 0
    for video in videos:
        observations = []
        for question in video["conversations"]:
            stem = f"v{video['video_id']}_q{question['source_question_index']}"
            path = output / f"{stem}.npz"
            if path.exists():
                with np.load(path) as z:
                    observations.append((z["overlap_curve"], z["overlap_k"]))
                count += 1
        if observations:
            by_video.append((
                nanmean_no_warning(np.stack([o[0] for o in observations])),
                nanmean_no_warning(np.stack([o[1] for o in observations])),
            ))
    if not by_video:
        return {"status": "empty"}
    mean_curve = nanmean_no_warning(np.stack([o[0] for o in by_video]))
    mean_k = nanmean_no_warning(np.stack([o[1] for o in by_video]))
    overlap_auc = np.trapezoid(np.nan_to_num(mean_curve, nan=0), COMPRESSION, axis=2)

    utility_auc = {}
    with (Path("results/myobservations/uniform_selector_auc_50v_20260920") /
          "uniform_selector_head_auc.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["policy"] in POLICIES:
                utility_auc[(row["policy"], 4 * int(row["layer"]) + int(row["kv_head"]))] = float(row["auc"])
    rows = []
    for h in range(112):
        row = {"layer": h // 4, "kv_head": h % 4}
        for p, name in enumerate(POLICIES):
            ua = np.array([utility_auc[(name, i)] for i in range(112)])
            row.update({
                f"{name}_utility_auc": ua[h],
                f"{name}_utility_auc_rank": int(ranks(ua)[h]),
                f"{name}_overlap_auc": float(overlap_auc[p, h]),
                f"{name}_overlap_auc_rank": int(ranks(overlap_auc[p])[h]),
                f"{name}_overlap_k250": float(mean_k[p, h, 0]),
                f"{name}_overlap_k250_rank": int(ranks(mean_k[p, :, 0])[h]),
                f"{name}_overlap_k500": float(mean_k[p, h, 1]),
                f"{name}_overlap_k800": float(mean_k[p, h, 2]),
                f"{name}_overlap_k1000": float(mean_k[p, h, 3]),
            })
            row[f"{name}_utility_to_overlap_auc_rank_change"] = (
                row[f"{name}_overlap_auc_rank"] - row[f"{name}_utility_auc_rank"]
            )
            row[f"{name}_overlap_auc_to_k250_rank_change"] = (
                row[f"{name}_overlap_k250_rank"] - row[f"{name}_overlap_auc_rank"]
            )
        row["local_to_global_overlap_auc_rank_change"] = (
            row["uniform_global_overlap_auc_rank"] - row["uniform_local_overlap_auc_rank"]
        )
        row["local_to_global_k250_rank_change"] = (
            row["uniform_global_overlap_k250_rank"] - row["uniform_local_overlap_k250_rank"]
        )
        rows.append(row)
    with (output / "local_global_topk_head_ranks.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = {"videos": len(by_video), "observations": count, "K": list(K_VALUES),
              "compression": COMPRESSION.tolist(), "policies": {}}
    for p, name in enumerate(POLICIES):
        ua = np.array([utility_auc[(name, i)] for i in range(112)])
        shifts = np.abs(ranks(overlap_auc[p]) - ranks(ua))
        report["policies"][name] = {
            "overlap_k250_mean": float(mean_k[p, :, 0].mean()),
            "overlap_k250_head_std": float(mean_k[p, :, 0].std()),
            "overlap_k250_p10_p90": np.percentile(mean_k[p, :, 0], [10, 90]).tolist(),
            "overlap_auc_mean": float(overlap_auc[p].mean()),
            "overlap_auc_head_std": float(overlap_auc[p].std()),
            "utility_auc_vs_overlap_auc_rank_corr": rank_correlation(ua, overlap_auc[p]),
            "utility_auc_vs_overlap_auc_median_abs_rank_shift": float(np.median(shifts)),
            "utility_auc_vs_overlap_auc_max_abs_rank_shift": int(shifts.max()),
            "utility_auc_vs_overlap_auc_top10_overlap": int(len(
                set(np.argsort(-ua)[:10]) & set(np.argsort(-overlap_auc[p])[:10])
            )),
            "utility_auc_vs_k250_overlap_rank_corr": rank_correlation(ua, mean_k[p, :, 0]),
        }
    report["local_vs_global"] = {
        "overlap_auc_rank_corr": rank_correlation(*overlap_auc),
        "overlap_auc_top10_overlap": int(len(
            set(np.argsort(-overlap_auc[0])[:10]) & set(np.argsort(-overlap_auc[1])[:10])
        )),
        "k250_overlap_rank_corr": rank_correlation(mean_k[0, :, 0], mean_k[1, :, 0]),
        "k250_overlap_top10_overlap": int(len(
            set(np.argsort(-mean_k[0, :, 0])[:10]) & set(np.argsort(-mean_k[1, :, 0])[:10])
        )),
        "overlap_auc_median_abs_rank_shift": float(np.median(np.abs(ranks(overlap_auc[0]) - ranks(overlap_auc[1])))),
        "overlap_auc_max_abs_rank_shift": int(np.max(np.abs(ranks(overlap_auc[0]) - ranks(overlap_auc[1])))),
    }
    (output / "local_global_topk_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def run(args):
    import torch
    from head_analysis.profile_effective_memory_readout import load_video_tensor
    from head_analysis.profile_horizonkv_pilot import build_pilot_model
    from video_qa.base import BaseVQA

    source = Path(args.source)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    videos = json.loads((source / "manifest.json").read_text())["videos"]
    if args.max_videos:
        videos = videos[:args.max_videos]
    model_args = SimpleNamespace(model="qwen2.5_vl_7b", device=0,
                                 fixed_frame_canvas_size=448, kv_size=1000000,
                                 sample_fps=0.5, compress_mode="hermes")
    model, processor = build_pilot_model(model_args, "models/Qwen2.5-VL-7B-Instruct")
    analyzer = BaseVQA(anno=videos, save_dir=str(output / "unused_qa_output"),
                       sample_fps=0.5, qa_model=model, qa_processor=processor,
                       num_chunks=None, chunk_idx=None)
    for video in videos:
        started = time.time()
        print(f"TOPK_VIDEO_START id={video['video_id']}", flush=True)
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
            path = output / f"{stem}.npz"
            if path.exists():
                print(f"TOPK_SKIP {stem}", flush=True)
                continue
            with np.load(source / f"{stem}.npz") as z:
                utility = z["utility"].astype(np.float64)
            if model.kv_cache[0][0].shape[2] - int(model.visual_start_idx) != utility.shape[1]:
                raise AssertionError(stem)
            rankings, _ = _component_rankings(model, processor, args.recent)
            values = [overlap_curve(utility, rank, args.recent) for rank in rankings[:2]]
            np.savez_compressed(path,
                                overlap_curve=np.stack([v[0] for v in values]).astype(np.float32),
                                overlap_k=np.stack([v[1] for v in values]).astype(np.float32))
            print(f"TOPK_OBSERVATION_DONE {stem}", flush=True)
        print(f"TOPK_VIDEO_DONE id={video['video_id']} elapsed={time.time()-started:.1f}s", flush=True)
    print(json.dumps(summarize(source, output), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="results/myobservations/lu_kv_offline_50v_20260913")
    parser.add_argument("--output", default="results/myobservations/local_global_topk_50v_20260922")
    parser.add_argument("--recent", type=int, default=32)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if args.analyze_only:
        print(json.dumps(summarize(Path(args.source), Path(args.output)), indent=2))
    else:
        import torch
        torch.manual_seed(20260913)
        torch.set_num_threads(4)
        run(args)


if __name__ == "__main__":
    main()
