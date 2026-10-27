"""Two-dataset, four-selector Qwen3-VL-8B smoke matrix (bounded to one question each)."""

import argparse
import ctypes
import gc
import json
import time
from pathlib import Path

import torch

from sparsemm_baselines.core import BaselineConfig
from sparsemm_baselines.qwen3_adapter import SparseMMQwen3, load_model
from sparsemm_baselines.run_streamingbench import (
    analyze_video, gpu_lease, prepare_annotations, write_json,
)

METHODS = ("adakv", "pyramidkv", "uniform", "full")


def trace_summary(path):
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    prefills = [entry for entry in entries if entry.get("event") == "video_prefill"]
    return {
        "video_prefills": len(prefills),
        "visual_tokens": sum(entry["input_tokens"] for entry in prefills),
        "compression_events": sum(any(layer.get("compressed") for layer in entry["layers"])
                                  for entry in prefills),
        "head_lengths_after_each_prefill": [entry["head_lengths"] for entry in prefills],
    }


def main(args):
    root = args.output_dir.resolve()
    if root.exists():
        raise FileExistsError(root)
    ovo = prepare_annotations(args.ovo_annotations, 1, 1)[0]
    streaming = prepare_annotations(args.streaming_annotations, 8, 1)[7]
    for video in (ovo, streaming):
        if float(video["conversations"][0]["end_time"]) > 120:
            raise ValueError("Smoke endpoint exceeds 120 seconds")
    root.mkdir(parents=True)
    with gpu_lease(root):
        loaded = load_model(args.model_path, config=BaselineConfig(method="uniform", budget=args.budget,
                            budget_scope="visual"), sample_fps=args.sample_fps,
                            video_max_pixels=args.video_max_pixels)
        model, processor = loaded.model, loaded.processor
        del loaded
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        torch.cuda.empty_cache()
        for method in METHODS:
            for dataset, video in (("ovobench", ovo), ("streamingbench", streaming)):
                folder = root / f"{method}_{dataset}"
                folder.mkdir()
                config = BaselineConfig(method="uniform" if method == "full" else method,
                                        budget=1_000_000 if method == "full" else args.budget,
                                        budget_scope="visual")
                trace = folder / "cache_trace.jsonl"
                bridge = SparseMMQwen3(model, processor, config,
                                        sample_fps=args.sample_fps, trace_path=trace,
                                        video_max_pixels=args.video_max_pixels)
                started = time.perf_counter()
                try:
                    rows = analyze_video(bridge, video, args.sample_fps, 16, torch)
                    summary = trace_summary(trace)
                    result = {"status": "passed", "method": method, "dataset": dataset,
                              "model_path": str(args.model_path.resolve()),
                              "video_id": video["video_id"], "budget": config.budget,
                              "sample_fps": args.sample_fps,
                              "video_max_pixels": args.video_max_pixels,
                              "video_input_protocol": "whole_chunk_video; singleton_as_image; no_timestamp_text_markers",
                              "seconds": round(time.perf_counter() - started, 3),
                              "rows": rows, **summary}
                except Exception as exc:
                    result = {"status": "failed", "method": method, "dataset": dataset,
                              "error": f"{type(exc).__name__}: {exc}"}
                    write_json(folder / "results.json", result)
                    raise
                write_json(folder / "results.json", result)
                print(json.dumps({k: result[k] for k in ("method", "dataset", "status", "seconds",
                      "visual_tokens", "compression_events")}), flush=True)
                del bridge
                gc.collect()
                ctypes.CDLL("libc.so.6").malloc_trim(0)
                torch.cuda.empty_cache()
        write_json(root / "summary.json", {
            "model_path": str(args.model_path.resolve()),
            "methods": list(METHODS), "datasets": ["ovobench", "streamingbench"],
            "budget": args.budget, "sample_fps": args.sample_fps,
            "video_max_pixels": args.video_max_pixels, "full_benchmarks_paused": True,
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--ovo-annotations", type=Path, required=True)
    parser.add_argument("--streaming-annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--sample-fps", type=float, default=0.5)
    parser.add_argument("--video-max-pixels", type=int, default=1200000)
    main(parser.parse_args())
