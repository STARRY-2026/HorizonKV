#!/usr/bin/env python3
"""Replay LU calibration prefixes and compare uniform selector policies.

The source LU traces already contain question-conditioned utility.  This script
only replays the question-free pseudo queries, so it avoids answer generation
and projected-value-norm collection.  Each observation stores compact coverage
curves rather than token-level scores.
"""

import argparse
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np


POLICIES = ("deployed", "uniform_local", "uniform_global", "uniform_joint", "uniform_recency")
COMPRESSION = np.asarray(
    [0, .1, .2, .3, .4, .5, .6, .7, .8, .85, .9, .925, .95, .975, .99],
    dtype=np.float64,
)


def _coverage(utility, ranking, recent):
    heads, tokens = utility.shape
    recent = min(int(recent), tokens)
    history = tokens - recent
    recent_mass = utility[:, history:].sum(axis=1, dtype=np.float64)
    ordered = np.take_along_axis(utility[:, :history], ranking[:, :history], axis=1)
    prefix = np.concatenate(
        [np.zeros((heads, 1), dtype=np.float64), np.cumsum(ordered, axis=1, dtype=np.float64)],
        axis=1,
    )
    total = utility.sum(axis=1, dtype=np.float64)
    result = np.full((heads, len(COMPRESSION)), np.nan, dtype=np.float64)
    valid = total > 1e-12
    for index, compression in enumerate(COMPRESSION):
        budget = max(recent, min(tokens, int(round((1.0 - compression) * tokens))))
        history_budget = min(history, max(0, budget - recent))
        retained = recent_mass + prefix[:, history_budget]
        result[valid, index] = retained[valid] / total[valid]
    return result


def _component_rankings(model, processor, recent):
    import torch
    from head_analysis.hermes_kv_head_ragged import HeadRaggedCache
    from head_analysis.hermes_kv_head_ragged_prefill import _ragged_query_scores

    dense = model.kv_cache
    offsets = model._get_next_global_offset_per_layer()
    local_prompt, global_prompt = model.predict_next_question()
    prompts = (local_prompt, global_prompt, local_prompt + "; " + global_prompt)
    model.kv_cache = HeadRaggedCache.from_dense_cache(model, dense)
    try:
        components = []
        for prompt in prompts:
            ids = torch.tensor([processor.tokenizer(prompt).input_ids], device=model.device)
            components.append(_ragged_query_scores(model, ids, offsets))
        start = int(model.visual_start_idx)
        tokens = dense[0][0].shape[2] - start
        history = max(0, tokens - int(recent))
        rankings = []
        for component in components:
            rows = []
            for layer in component:
                for score in layer:
                    visual = score[start:start + history].float().flatten()
                    rows.append(torch.argsort(visual, descending=True, stable=True).cpu().numpy())
            rankings.append(np.stack(rows))
        return rankings, prompts
    finally:
        model.kv_cache = dense
        model._layer_position_ids.clear()


def _summary(output, source):
    manifest = json.loads((source / "manifest.json").read_text())
    video_for_stem = {}
    for video in manifest["videos"]:
        for question in video["conversations"]:
            video_for_stem[f"v{video['video_id']}_q{question['source_question_index']}"] = str(video["video_id"])

    by_video = defaultdict(list)
    for path in sorted(output.glob("v*_q*.npz")):
        if path.stem not in video_for_stem:
            continue
        with np.load(path) as data:
            by_video[video_for_stem[path.stem]].append(data["coverage"].astype(np.float64))
    if not by_video:
        return {"status": "empty"}

    video_curves = []
    for video in sorted(by_video, key=int):
        video_curves.append(np.nanmean(np.stack(by_video[video]), axis=0))
    curves = np.nanmean(np.stack(video_curves), axis=0)
    auc = np.trapezoid(np.nan_to_num(curves, nan=0.0), COMPRESSION, axis=2)

    rows = []
    for policy_index, policy in enumerate(POLICIES):
        for head in range(112):
            rows.append({
                "policy": policy,
                "layer": head // 4,
                "kv_head": head % 4,
                "auc": float(auc[policy_index, head]),
                **{
                    f"coverage_at_compression_{x:g}": float(curves[policy_index, head, i])
                    for i, x in enumerate(COMPRESSION)
                },
            })
    csv_path = output / "uniform_selector_head_auc.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    policy_summary = {}
    layers = np.repeat(np.arange(28), 4)
    for policy_index, policy in enumerate(POLICIES):
        values = auc[policy_index]
        mean = float(values.mean())
        ss_total = float(np.square(values - mean).sum())
        layer_means = values.reshape(28, 4).mean(axis=1)
        ss_layer = float((4 * np.square(layer_means - mean)).sum())
        segments = {
            "shallow_0_7": values[:32],
            "middle_8_18": values[32:76],
            "deep_19_27": values[76:],
        }
        policy_summary[policy] = {
            "mean_auc": mean,
            "layer_auc_correlation": float(np.corrcoef(layers, values)[0, 1]),
            "layer_eta_squared": ss_layer / ss_total if ss_total > 0 else 0.0,
            "segments": {
                name: {
                    "mean": float(part.mean()),
                    "std": float(part.std()),
                    "min": float(part.min()),
                    "median": float(np.median(part)),
                    "max": float(part.max()),
                }
                for name, part in segments.items()
            },
        }
    report = {
        "status": "complete" if len(by_video) == len(manifest["videos"]) else "partial",
        "videos": len(by_video),
        "observations": sum(len(parts) for parts in by_video.values()),
        "source": str(source),
        "policies": policy_summary,
    }
    (output / "uniform_selector_auc_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def run(args):
    import torch
    from head_analysis.profile_effective_memory_readout import load_video_tensor
    from head_analysis.profile_horizonkv_pilot import build_pilot_model
    from video_qa.base import BaseVQA

    source = Path(args.source)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads((source / "manifest.json").read_text())
    videos = source_manifest["videos"]
    runtime = {
        "source": str(source),
        "recent": int(args.recent),
        "policies": POLICIES,
        "compression": COMPRESSION.tolist(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "manifest.json").write_text(json.dumps(runtime, indent=2) + "\n")

    if args.analyze_only:
        print(json.dumps(_summary(output, source), indent=2))
        return

    model_args = SimpleNamespace(
        model="qwen2.5_vl_7b", device=0, fixed_frame_canvas_size=448,
        kv_size=1000000, sample_fps=0.5, compress_mode="hermes",
    )
    model, processor = build_pilot_model(model_args, "models/Qwen2.5-VL-7B-Instruct")
    analyzer = BaseVQA(
        anno=videos, save_dir=str(output / "unused_qa_output"), sample_fps=0.5,
        qa_model=model, qa_processor=processor, num_chunks=None, chunk_idx=None,
    )
    completed = 0
    for video in videos:
        if args.smoke and completed:
            break
        started = time.time()
        print(f"UNIFORM_SELECTOR_VIDEO_START id={video['video_id']}", flush=True)
        clip_start = float(video.get("clip", [0, 0])[0]) if video.get("clip") else 0.0
        load_sample = {**video, "clip": [clip_start, clip_start + max(
            float(question["end_time"]) for question in video["conversations"]
        )]}
        video_tensor = load_video_tensor(analyzer, load_sample)
        model.clear_cache()
        model._position_ids_cache = [None] * model.num_layers
        model._layer_position_ids.clear()
        model.total_processed_frames = 0
        model.encode_init_prompt()
        current = 0
        for question in video["conversations"]:
            end = min(len(video_tensor), math.ceil(float(question["end_time"]) * 0.5))
            while current < end:
                stop = min(current + 16, end)
                model.encode_video_chunk(video_tensor[current:stop])
                current = stop
            stem = f"v{video['video_id']}_q{question['source_question_index']}"
            source_path = source / f"{stem}.npz"
            output_path = output / f"{stem}.npz"
            if not source_path.exists():
                print(f"UNIFORM_SELECTOR_NO_SOURCE {stem}", flush=True)
                continue
            if output_path.exists():
                print(f"UNIFORM_SELECTOR_RESUME_SKIP {stem}", flush=True)
                continue
            with np.load(source_path) as data:
                utility = data["utility"].astype(np.float64)
                deployed_ranking = data["ranking"].astype(np.int64)
            visual_tokens = model.kv_cache[0][0].shape[2] - int(model.visual_start_idx)
            if visual_tokens != utility.shape[1]:
                raise AssertionError((stem, visual_tokens, utility.shape[1]))
            component_rankings, prompts = _component_rankings(model, processor, args.recent)
            history = max(0, visual_tokens - int(args.recent))
            recency_ranking = np.broadcast_to(
                np.arange(history - 1, -1, -1, dtype=np.int64), (112, history)
            ).copy()
            rankings = [deployed_ranking, *component_rankings, recency_ranking]
            coverage = np.stack([
                _coverage(utility, ranking, args.recent) for ranking in rankings
            ]).astype(np.float32)
            np.savez_compressed(output_path, coverage=coverage)
            (output / f"{stem}.json").write_text(json.dumps({
                "video_id": video["video_id"],
                "source": str(source_path),
                "visual_tokens": int(visual_tokens),
                "local_prompt": prompts[0],
                "global_prompt": prompts[1],
            }, indent=2) + "\n")
            print(f"UNIFORM_SELECTOR_OBSERVATION_DONE {stem} tokens={visual_tokens}", flush=True)
        completed += 1
        print(f"UNIFORM_SELECTOR_VIDEO_DONE id={video['video_id']} elapsed={time.time()-started:.1f}s", flush=True)
        del video_tensor
    del model
    torch.cuda.empty_cache()
    print(json.dumps(_summary(output, source), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="results/myobservations/lu_kv_offline_50v_20260913")
    parser.add_argument("--output", default="results/myobservations/uniform_selector_auc_50v_20260920")
    parser.add_argument("--recent", type=int, default=32)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    torch = __import__("torch")
    torch.manual_seed(20260913)
    torch.set_num_threads(4)
    with torch.inference_mode():
        run(args)


if __name__ == "__main__":
    main()
