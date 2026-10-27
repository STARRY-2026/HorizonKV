"""Bounded two-dataset Qwen3 selector validation with several questions.

This checks execution and cache invariants. Existing LU/DP assets were
calibrated before the window4 vision change, so accuracy is not comparative.
"""

import argparse
import ctypes
import gc
import json
import time
from pathlib import Path

import torch

from sparsemm_baselines.core import BaselineConfig
from sparsemm_baselines.qwen3_adapter import SparseMMQwen3, load_model
from sparsemm_baselines.qwen3_window4 import enable_window4_vision
from sparsemm_baselines.run_streamingbench import analyze_video, gpu_lease, prepare_annotations, sha256, write_json
from sparsemm_native.policy import allocate_head_budgets, load_fixed_head_budgets, load_head_scores
from sparsemm_native.two_bank import load_two_bank_budgets


METHODS = ("adakv", "pyramidkv", "uniform", "full", "lukv", "sparsemm", "dp", "dp_dual")


def selected(path):
    videos = prepare_annotations(path, 3)
    questions = sum(len(video["conversations"]) for video in videos)
    if questions > 8 or any(float(q["end_time"]) > 120 for video in videos for q in video["conversations"]):
        raise ValueError("Extended smoke permits at most 3 videos, 8 questions and 120-second endpoints")
    return videos


def check_run(folder, videos, method, rows):
    entries = [json.loads(line) for line in (folder / "cache_trace.jsonl").read_text().splitlines()]
    prefills = [event for event in entries if event.get("event") == "video_prefill"]
    restored = [event for event in entries if event.get("event") == "answer_restored"]
    expected = [(str(video["video_id"]), q["question"]) for video in videos for q in video["conversations"]]
    if [(str(row["video_id"]), row["question"]) for row in rows] != expected:
        raise RuntimeError("Qwen3 result rows do not cover the selected questions")
    if len(restored) != len(expected) or not prefills:
        raise RuntimeError("Qwen3 trace is missing prefill or question restore")
    last_prefill = {}
    restored_count = 0
    compressed = 0
    inputs = []
    for event in entries:
        if event.get("event") == "video_prefill":
            video_id = str(event["video_id"])
            lengths = event["head_lengths"]
            if len(lengths) != 36 or any(len(heads) != 8 for heads in lengths):
                raise RuntimeError("Qwen3 KV head shape changed")
            last_prefill[video_id] = lengths
            compressed += any(layer.get("compressed", False) for layer in event["layers"])
            inputs.append({"video_id": video_id, "grid_thw": event.get("grid_thw"),
                           "visual_tokens": event["input_tokens"]})
        elif event.get("event") == "answer_restored":
            if event["head_lengths"] != last_prefill.get(str(event["video_id"])):
                raise RuntimeError("Question did not restore the persistent video cache")
            restored_count += 1
    if restored_count != len(expected) or (method == "full" and compressed) or (method != "full" and not compressed):
        raise RuntimeError("Qwen3 restore or compression invariant failed")
    return {"questions": len(expected), "video_prefills": len(prefills),
            "compression_events": compressed, "qa_restore_checks": restored_count,
            "visual_inputs": inputs}


def main(args):
    root = args.output_dir.resolve()
    if root.exists():
        raise FileExistsError(root)
    selections = {"ovobench": selected(args.ovo_annotations),
                  "streamingbench": selected(args.streaming_annotations)}
    if args.budget != 500 or args.video_max_pixels < 3136:
        raise ValueError("This extended validation requires K=500 and a valid pixel cap")
    root.mkdir(parents=True)
    sources = {name: {"path": str(path.resolve()), "sha256": sha256(path),
                      "video_ids": [str(video["video_id"]) for video in selections[name]],
                      "questions": sum(len(video["conversations"]) for video in selections[name])}
               for name, path in (("ovobench", args.ovo_annotations),
                                  ("streamingbench", args.streaming_annotations))}
    assets = {"lukv_profile": args.lukv_profile, "sparsemm_scores": args.sparsemm_scores,
              "dp_allocation": args.dp_allocation}
    write_json(root / "protocol.json", {
        "model_path": str(args.model_path.resolve()), "model_config_sha256": sha256(args.model_path / "config.json"),
        "vision_topology": "window4", "vision_source_sha256": sha256(Path(__file__).resolve().parents[1] / "sparsemm_baselines/qwen3_window4.py"),
        "model_variant": True, "sample_fps": args.sample_fps, "video_max_pixels": args.video_max_pixels,
        "encode_chunk_size": 16, "budget": args.budget, "methods": list(METHODS), "datasets": sources,
        "assets": {name: {"path": str(path.resolve()), "sha256": sha256(path)} for name, path in assets.items()},
        "calibration_transfer_note": "LU and DP tables were calibrated with native Qwen3 vision; window4 online use is interface-only.",
        "full_benchmarks_paused": True,
    })
    if args.dry_run:
        write_json(root / "summary.json", {"status": "dry-run", "methods": list(METHODS)})
        return
    with gpu_lease(root):
        loaded = load_model(args.model_path, config=BaselineConfig(method="uniform", budget=args.budget,
                            budget_scope="visual"), sample_fps=args.sample_fps,
                            video_max_pixels=args.video_max_pixels)
        model, processor = loaded.model, loaded.processor
        layers, query_heads, kv_heads = loaded.num_layers, loaded.num_query_heads, loaded.num_kv_heads
        enable_window4_vision(model)
        del loaded
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        torch.cuda.empty_cache()
        profile = json.loads(args.lukv_profile.read_text())
        scores = load_head_scores(args.sparsemm_scores, layers, query_heads)
        sparsemm_budgets, _ = allocate_head_budgets(scores, budget=args.budget,
                                                     window_size=32, num_kv_heads=kv_heads)
        dp_budgets, _ = load_fixed_head_budgets(args.dp_allocation, layers, kv_heads, args.budget)
        dual_budgets, recent, _ = load_two_bank_budgets(
            args.dp_allocation, layers, kv_heads, args.budget, short_cache_gate="legacy_global")
        capacities = {"sparsemm": sparsemm_budgets, "dp": dp_budgets, "dp_dual": dual_budgets}
        run_summaries = []
        for method in METHODS:
            for dataset, videos in selections.items():
                folder = root / f"{method}_{dataset}"
                folder.mkdir()
                config = BaselineConfig(method=(method if method in ("adakv", "pyramidkv", "uniform") else "uniform"),
                                        budget=(1_000_000 if method == "full" else args.budget),
                                        budget_scope="visual")
                bridge = SparseMMQwen3(
                    model, processor, config, sample_fps=args.sample_fps,
                    trace_path=folder / "cache_trace.jsonl", video_max_pixels=args.video_max_pixels,
                    policy_method=(method if method not in ("full",) else "uniform"),
                    head_budgets=capacities.get(method),
                    recent_budgets=(recent if method == "dp_dual" else None),
                    lukv_profile=(profile if method == "lukv" else None))
                started = time.perf_counter()
                try:
                    rows = []
                    for video in videos:
                        rows.extend(analyze_video(bridge, video, args.sample_fps, 16, torch))
                    audit = check_run(folder, videos, method, rows)
                    result = {"status": "passed", "method": method, "dataset": dataset,
                              "seconds": round(time.perf_counter() - started, 3), "rows": rows, **audit}
                except Exception as exc:
                    write_json(folder / "results.json", {"status": "failed", "method": method,
                                                         "dataset": dataset, "error": f"{type(exc).__name__}: {exc}"})
                    raise
                write_json(folder / "results.json", result)
                run_summaries.append({"method": method, "dataset": dataset,
                                      "questions": len(rows), "correct": sum(row["qa_acc"] == 100.0 for row in rows),
                                      "seconds": result["seconds"], "compression_events": audit["compression_events"],
                                      "qa_restore_checks": audit["qa_restore_checks"]})
                write_json(root / "progress.json", {"status": "running", "runs": run_summaries})
                print(json.dumps(run_summaries[-1]), flush=True)
                del bridge
                gc.collect()
                ctypes.CDLL("libc.so.6").malloc_trim(0)
                torch.cuda.empty_cache()
        for dataset in selections:
            signatures = [[(tuple(item["grid_thw"]), item["visual_tokens"])
                           for item in json.loads((root / f"{method}_{dataset}" / "results.json").read_text())["visual_inputs"]]
                          for method in METHODS]
            if any(signature != signatures[0] for signature in signatures[1:]):
                raise RuntimeError(f"Qwen3 visual grids/tokens differ across methods on {dataset}")
        write_json(root / "summary.json", {"status": "complete", "runs": run_summaries,
                                           "all_audits_passed": True, "full_benchmarks_paused": True,
                                           "comparison_status": "interface_only_mixed_calibration_topology"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--ovo-annotations", type=Path, required=True)
    parser.add_argument("--streaming-annotations", type=Path, required=True)
    parser.add_argument("--lukv-profile", type=Path, required=True)
    parser.add_argument("--sparsemm-scores", type=Path, required=True)
    parser.add_argument("--dp-allocation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--sample-fps", type=float, default=0.5)
    parser.add_argument("--video-max-pixels", type=int, default=1200000)
    parser.add_argument("--dry-run", action="store_true")
    main(parser.parse_args())
