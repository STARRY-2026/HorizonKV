"""Bounded Qwen3-VL selector smoke test on native KVBench MCQA annotations."""

import argparse
import json
import time
from pathlib import Path

import torch

from sparsemm_baselines.core import BaselineConfig
from sparsemm_baselines.qwen3_adapter import load_model
from sparsemm_baselines.run_streamingbench import (
    analyze_video, gpu_lease, prepare_annotations, write_json,
)


def run(args):
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    videos = prepare_annotations(args.anno_path, args.video_offset + args.max_videos, args.max_questions)
    videos = videos[args.video_offset:]
    if not videos:
        raise ValueError("No selected videos")
    if len(videos) > 3 or sum(len(v["conversations"]) for v in videos) > 8:
        raise ValueError("Smoke test is limited to three videos and eight questions")
    if any(float(q["end_time"]) > 120 for v in videos for q in v["conversations"]):
        raise ValueError("Question endpoint exceeds 120 seconds")
    output.mkdir(parents=True)
    config = BaselineConfig(method="uniform" if args.method == "full" else args.method,
                            budget=1_000_000 if args.method == "full" else args.budget,
                            budget_scope="visual")
    with gpu_lease(output):
        started = time.perf_counter()
        bridge = load_model(args.model_path, config=config, sample_fps=args.sample_fps,
                            trace_path=output / "cache_trace.jsonl",
                            video_max_pixels=args.video_max_pixels)
        load_seconds = time.perf_counter() - started
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        rows = []
        for video in videos:
            started = time.perf_counter()
            result = analyze_video(bridge, video, args.sample_fps, args.encode_chunk_size, torch)
            rows.extend(result)
            print(json.dumps({"video_id": video["video_id"], "method": args.method,
                              "rows": result, "seconds": time.perf_counter() - started},
                             ensure_ascii=False), flush=True)
        trace = [json.loads(line) for line in (output / "cache_trace.jsonl").read_text().splitlines()]
        prefills = [entry for entry in trace if entry.get("event") == "video_prefill"]
        write_json(output / "results.json", {
            "kind": "qwen3_vl_bounded_selector_smoke",
            "method": args.method, "model_path": str(args.model_path.resolve()),
            "annotation_path": str(args.anno_path.resolve()),
            "budget": config.budget, "sample_fps": args.sample_fps,
            "video_max_pixels": args.video_max_pixels,
            "load_seconds": round(load_seconds, 3), "results": rows,
            "video_input_protocol": "whole_chunk_video; singleton_as_image; no_timestamp_text_markers",
            "compression_events": sum(any(layer.get("compressed") for layer in entry["layers"])
                                      for entry in prefills),
            "visual_tokens_seen": sum(entry["input_tokens"] for entry in prefills),
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--anno-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", choices=("adakv", "pyramidkv", "uniform", "full"), required=True)
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--max-videos", type=int, default=1)
    parser.add_argument("--video-offset", type=int, default=0)
    parser.add_argument("--max-questions", type=int, default=1)
    parser.add_argument("--sample-fps", type=float, default=0.5)
    parser.add_argument("--video-max-pixels", type=int, default=1200000)
    parser.add_argument("--encode-chunk-size", type=int, default=16)
    run(parser.parse_args())
