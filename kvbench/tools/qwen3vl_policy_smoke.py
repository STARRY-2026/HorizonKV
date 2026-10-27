"""Qwen3-VL online-policy smoke tests with explicit calibration provenance.

Missing method assets use synthetic 8B-shaped fixtures and are labelled as
such. Bounded real pilots are labelled separately from formal calibration.
"""

import argparse
import csv
import ctypes
import gc
import json
import time
from pathlib import Path

import torch

from sparsemm_baselines.core import BaselineConfig
from sparsemm_baselines.qwen3_adapter import SparseMMQwen3, load_model
from sparsemm_baselines.run_streamingbench import analyze_video, gpu_lease, prepare_annotations, write_json
from sparsemm_native.policy import allocate_head_budgets
from sparsemm_native.policy import load_head_scores
from sparsemm_native.policy import load_fixed_head_budgets
from sparsemm_native.two_bank import load_two_bank_budgets
from tools.qwen3vl_matrix_smoke import trace_summary

METHODS = ("lukv", "sparsemm", "dp", "dp_dual")


def create_assets(root, layers, query_heads, kv_heads, budget):
    scores = torch.tensor([[1 + (layer % 7) / 7 + (head % 5) / 10
                            for head in range(query_heads)] for layer in range(layers)],
                          dtype=torch.float64)
    score_path = root / "synthetic_query_head_scores.csv"
    with score_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("layer", "head", "score_mean"))
        for layer in range(layers):
            for head in range(query_heads):
                writer.writerow((layer, head, float(scores[layer, head])))
    sparsemm_budgets, allocation = allocate_head_budgets(
        scores, budget=budget, window_size=32, num_kv_heads=kv_heads)
    if budget <= 96:
        raise ValueError("Synthetic DP split requires budget > 96")
    balanced = torch.tensor([[budget + (20 if (layer + head) % 2 else -20)
                              for head in range(kv_heads)] for layer in range(layers)], dtype=torch.long)
    dp = torch.tensor([[budget + (30 if (layer + head) % 2 else -30)
                        for head in range(kv_heads)] for layer in range(layers)], dtype=torch.long)
    recent = torch.tensor([[64 if (layer + head) % 2 else 96
                            for head in range(kv_heads)] for layer in range(layers)], dtype=torch.long)
    for name, table in (("synthetic_lukv_capacity.csv", balanced),
                        ("synthetic_dp_capacity.csv", dp)):
        with (root / name).open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("layer", "kv_head", "capacity_n"))
            for layer in range(layers):
                for head in range(kv_heads):
                    writer.writerow((layer, head, int(table[layer, head])))
    with (root / "synthetic_dp_dual_capacity.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("layer", "kv_head", "capacity_n", "recent_r", "history_q"))
        for layer in range(layers):
            for head in range(kv_heads):
                cap, r = int(dp[layer, head]), int(recent[layer, head])
                writer.writerow((layer, head, cap, r, cap - r))
    write_json(root / "asset_manifest.json", {
        "status": "synthetic_smoke_only", "model_shape": [layers, query_heads, kv_heads],
        "budget": budget, "warning": "Tables are deterministic test fixtures, not offline calibration.",
        "sparsemm_allocation": allocation,
    })
    return {"lukv": balanced, "sparsemm": sparsemm_budgets,
            "dp": dp, "dp_dual": dp}, recent


def main(args):
    root = args.output_dir.resolve()
    if root.exists():
        raise FileExistsError(root)
    ovo = prepare_annotations(args.ovo_annotations, 1, 1)[0]
    streaming = prepare_annotations(args.streaming_annotations, 8, 1)[7]
    root.mkdir(parents=True)
    with gpu_lease(root):
        loaded = load_model(args.model_path, config=BaselineConfig(method="uniform", budget=args.budget,
                            budget_scope="visual"), sample_fps=args.sample_fps,
                            video_max_pixels=args.video_max_pixels)
        model, processor = loaded.model, loaded.processor
        if args.vision_topology == "window4":
            from sparsemm_baselines.qwen3_window4 import enable_window4_vision
            enable_window4_vision(model)
        layers = loaded.num_layers
        query_heads = loaded.num_query_heads
        kv_heads = loaded.num_kv_heads
        del loaded
        synthetic_needed = any(
            method == "lukv" and args.lukv_profile is None or
            method == "sparsemm" and args.sparsemm_score_path is None or
            method in ("dp", "dp_dual") and args.dp_allocation_path is None
            for method in args.methods)
        capacities, recent = (create_assets(root, layers, query_heads, kv_heads, args.budget)
                              if synthetic_needed else ({}, None))
        lukv_profile = None
        if args.lukv_profile is not None:
            lukv_profile = json.loads(args.lukv_profile.read_text())
        if args.sparsemm_score_path is not None:
            measured_scores = load_head_scores(args.sparsemm_score_path, layers, query_heads)
            capacities["sparsemm"], measured_allocation = allocate_head_budgets(
                measured_scores, budget=args.budget, window_size=32, num_kv_heads=kv_heads)
            write_json(root / "measured_sparsemm_allocation.json", measured_allocation)
        if args.dp_allocation_path is not None:
            capacities["dp"], dp_allocation = load_fixed_head_budgets(
                args.dp_allocation_path, layers, kv_heads, args.budget)
            capacities["dp_dual"], measured_recent, dual_allocation = load_two_bank_budgets(
                args.dp_allocation_path, layers, kv_heads, args.budget,
                short_cache_gate="legacy_global")
            recent = measured_recent
            write_json(root / "measured_dp_allocation.json", dp_allocation)
            write_json(root / "measured_dp_dual_allocation.json", dual_allocation)
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        torch.cuda.empty_cache()
        for method in args.methods:
            for dataset, video in (("ovobench", ovo), ("streamingbench", streaming)):
                folder = root / f"{method}_{dataset}"
                folder.mkdir()
                config = BaselineConfig(method="uniform", budget=args.budget, budget_scope="visual")
                trace = folder / "cache_trace.jsonl"
                bridge = SparseMMQwen3(model, processor, config,
                                        sample_fps=args.sample_fps, trace_path=trace,
                                        video_max_pixels=args.video_max_pixels,
                                        policy_method=method,
                                        head_budgets=None if method == "lukv" and lukv_profile is not None else capacities[method],
                                        lukv_profile=lukv_profile if method == "lukv" else None,
                                        recent_budgets=recent if method == "dp_dual" else None)
                started = time.perf_counter()
                try:
                    rows = analyze_video(bridge, video, args.sample_fps, 16, torch)
                    result = {"status": "passed", "method": method, "dataset": dataset,
                              "calibration_status": (("formal_10_video_lu_hermes_pseudo" if lukv_profile.get("calibration", {}).get("selector") == "hermes_pseudo"
                                                      else "bounded_real_lu_pilot") if method == "lukv" and lukv_profile is not None
                                                     else "bounded_real_ocr_pilot" if method == "sparsemm" and args.sparsemm_score_path is not None
                                                     else "bounded_attention_dp_pilot" if method in ("dp", "dp_dual") and args.dp_allocation_path is not None
                                                     else "synthetic_smoke_only"),
                              "model_path": str(args.model_path.resolve()),
                              "vision_topology": args.vision_topology,
                              "video_id": video["video_id"], "budget": args.budget,
                              "sample_fps": args.sample_fps,
                              "video_max_pixels": args.video_max_pixels,
                              "video_input_protocol": "whole_chunk_video; singleton_as_image; no_timestamp_text_markers",
                              "seconds": round(time.perf_counter() - started, 3),
                              "rows": rows, **trace_summary(trace)}
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
            "status": "complete", "methods": list(args.methods),
            "datasets": ["ovobench", "streamingbench"],
            "calibration_status": "mixed_or_pilot_see_per_run",
            "model_path": str(args.model_path.resolve()),
            "full_benchmarks_paused": True,
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
    parser.add_argument("--vision-topology", choices=("native", "window4"), default="native")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--lukv-profile", type=Path)
    parser.add_argument("--sparsemm-score-path", type=Path)
    parser.add_argument("--dp-allocation-path", type=Path)
    main(parser.parse_args())
