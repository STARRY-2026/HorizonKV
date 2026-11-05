#!/usr/bin/env python3
"""Preflight and run one frozen StreamingBench protocol without loading a model.

The preflight uses only the Python standard library. Inference runs in a child
process, with this snapshot as its sole PYTHONPATH and external adapters off.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS = ROOT / "configs" / "experiments" / "protocols.json"
HEADS = [(layer, head) for layer in range(28) for head in range(4)]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_heads(path, required):
    """Reject partial, duplicate, out-of-range, or non-finite profile inputs."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = {"layer", "kv_head", *required} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows = list(reader)
    indexed = {}
    for row in rows:
        key = (int(row["layer"]), int(row["kv_head"]))
        if key not in HEADS or key in indexed:
            raise ValueError(f"{path}: duplicate or out-of-range head {key}")
        for column in required:
            if column == "HeadType":
                continue
            if not math.isfinite(float(row[column])):
                raise ValueError(f"{path}: non-finite {column} at {key}")
        indexed[key] = row
    if set(indexed) != set(HEADS):
        raise ValueError(f"{path}: expected all 112 unique physical KV heads")
    return [indexed[key] for key in HEADS]


def sparsemm_budgets(rows, budget, ratio=0.2, window=32):
    """Mirror the runtime sparsemm formula, including round-to-even integers."""
    scores = [abs(float(row["score"])) for row in rows]
    if sum(scores) <= 1e-12:
        scores = [1.0] * len(scores)
    base = max(1, budget - window)
    minimum = int(base * ratio)
    remaining = max(base - minimum, 0) * len(HEADS)
    total = sum(scores)
    return [max(1, round(score / total * remaining + minimum) + window)
            for score in scores]


def budget_preflight(protocol, budget, strict=False):
    options = dict(protocol["options"])
    selected = protocol.get("budgets", {}).get(str(budget))
    if selected is not None:
        options.update(selected)
    supported = protocol.get("supported_budgets")
    if supported and budget not in supported:
        raise ValueError(f"{protocol['name']} supports only budgets {supported}; got {budget}")
    inputs = []
    tables = {}
    for option, columns in {
        "kv_head_budget_scores": ["score"],
        "kv_head_recency_scores": ["recent_ratio", "capacity_n", "recent_r", "history_q"],
        "timestamp_v8_policy": ["long_score", "short_score"],
        "horizonkv_policy": ["HeadType", "RecentBudget", "MemoryBudget", "TotalBudget"],
    }.items():
        if option in options:
            path = ROOT / options[option]
            tables[option] = read_heads(path, columns)
            inputs.append({"option": option, "path": str(path), "sha256": sha256(path),
                           "physical_heads": len(tables[option])})
            options[option] = str(path)
    if "horizonkv_policy" in tables:
        rows = tables["horizonkv_policy"]
        kinds = {"decay_aggressive", "normal_sustained", "sustained_gentle", "rising_far"}
        for row in rows:
            recent, memory, total = (int(row[c]) for c in ("RecentBudget", "MemoryBudget", "TotalBudget"))
            if (row["HeadType"] not in kinds or recent != options["horizonkv_recent_capacity"]
                    or memory < 0 or recent + memory != total):
                raise ValueError("Invalid HorizonKV fixed-region policy")
        budgets = [int(row["TotalBudget"]) for row in rows]
    elif "kv_head_budget_scores" in tables:
        budgets = sparsemm_budgets(tables["kv_head_budget_scores"], budget,
                                  options["kv_head_budget_sparsemm_ratio"],
                                  options["kv_head_budget_sparsemm_window_size"])
    else:
        budgets = [budget] * len(HEADS)
    if "kv_head_recency_scores" in tables:
        for row, actual in zip(tables["kv_head_recency_scores"], budgets):
            capacity = int(row["capacity_n"])
            recent, history = int(row["recent_r"]), int(row["history_q"])
            if capacity != actual or recent < 0 or history < 0 or recent + history != capacity:
                raise ValueError("DP recency profile does not match exact runtime capacity")
            if not math.isclose(float(row["recent_ratio"]), recent / capacity, abs_tol=1e-12):
                raise ValueError("DP recent_ratio does not match recent_r / capacity_n")
    if "timestamp_v8_policy" in tables:
        for row in tables["timestamp_v8_policy"]:
            if float(row["long_score"]) < 0 or float(row["short_score"]) < 0:
                raise ValueError("v8 profile scores must be non-negative")
    delta = sum(budgets) - budget * len(HEADS)
    allowed_delta = protocol.get("rounding_delta", {}).get(str(budget), 0)
    if delta != allowed_delta or (strict and delta):
        raise ValueError(f"Budget mismatch: actual {sum(budgets)}, target {budget * 112}, delta {delta}")
    return options, {
        "target_total_visual_slots": budget * len(HEADS),
        "runtime_total_visual_slots": sum(budgets),
        "rounding_delta": delta,
        "exact_conservation": delta == 0,
        "min_per_head": min(budgets), "max_per_head": max(budgets),
        "physical_heads": len(HEADS), "per_head_visual_slots": budgets,
        "note": ("Historical SparseMM independent integer rounding is preserved."
                 if delta else "Runtime integer budgets match the declared total exactly."),
    }, inputs


def select_smoke(data, videos=1, questions=1, minimum=64.0, maximum=120.0):
    """Keep real questions and timestamps; encode their complete video prefixes."""
    selected = []
    indices = []
    for video_index, source in enumerate(data):
        candidates = [(index, question) for index, question in enumerate(source["conversations"])
                      if minimum <= float(question["end_time"]) <= maximum]
        candidates.sort(key=lambda item: (float(item[1]["end_time"]), item[0]))
        chosen = candidates[:questions]
        if len(chosen) < questions:
            continue
        row = copy.deepcopy(source)
        row["conversations"] = [copy.deepcopy(question) for _, question in chosen]
        # Keep the complete prefix and its original sampling origin. The loader
        # otherwise decodes the entire video, including the unused future suffix.
        suffix_trimmed = "clip" not in row
        if suffix_trimmed:
            row["clip"] = [0, max(float(question["end_time"]) for _, question in chosen)]
        selected.append(row)
        indices.append({"source_video_index": video_index, "video_id": row.get("video_id"),
                        "source_question_indices": [index for index, _ in chosen],
                        "end_times": [float(question["end_time"]) for _, question in chosen],
                        "unused_suffix_trimmed": suffix_trimmed, "clip": row.get("clip")})
        if len(selected) == videos:
            return selected, indices
    raise ValueError(f"Need {videos} videos with {questions} real questions in [{minimum}, {maximum}]s; "
                     "choose a full annotation or adjust --smoke-min/max-seconds")


def model_preflight(model_path):
    path = Path(model_path).expanduser().resolve()
    if not path.is_dir():
        raise ValueError("--model-path must be an existing local Qwen2.5-VL-7B directory")
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    text = config.get("text_config", config)
    shape = [int(text.get(field, 0)) for field in
             ("num_hidden_layers", "num_attention_heads", "num_key_value_heads")]
    if shape != [28, 28, 4] or config.get("model_type") != "qwen2_5_vl":
        raise ValueError(f"Frozen profiles require Qwen2.5-VL-7B [28,28,4], got {shape}")
    for filename in ("tokenizer_config.json", "preprocessor_config.json"):
        if not (path / filename).is_file():
            raise ValueError(f"Model is missing {filename}")
    indexes = list(path.glob("*.safetensors.index.json")) + list(path.glob("pytorch_model.bin.index.json"))
    if indexes:
        weight_map = json.loads(indexes[0].read_text())["weight_map"]
        missing = [name for name in set(weight_map.values()) if not (path / name).is_file()]
        if missing:
            raise ValueError(f"Model weight shards missing: {missing}")
    elif not list(path.glob("*.safetensors")) and not list(path.glob("pytorch_model*.bin")):
        raise ValueError("Model weight files are missing")
    return path, {"path": str(path), "config_sha256": sha256(path / "config.json"),
                  "attention_shape": shape, "weights_present": True}


def compression_evidence(method, log):
    """Count messages emitted after actual cache pruning, not installation."""
    if method == "hermes":
        marker = "Strict-shrunk KV cache. New length:"
    elif method == "horizonkv":
        marker = "[HorizonKV] prune stats:"
    elif method in ("timestamp-v8", "current-ourtop"):
        marker = "[TimestampV8] prune stats:"
    else:
        marker = "[kv_head_ragged_prefill] prune stats:"
    return {"marker": marker, "events": log.count(marker)}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=["hermes", "uniform", "dp", "v8", "memheadkv", "timestamp-v8", "current-ourtop", "horizonkv"], required=True)
    p.add_argument("--budget", type=int, default=None)
    p.add_argument("--model-path", required=True)
    p.add_argument("--annotation", type=Path, default=ROOT / "data/streamingbench/streamingbench_realtime_half.json")
    p.add_argument("--video-root", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="0", help="One physical CUDA device, e.g. 0")
    p.add_argument("--frame-canvas-size", type=int, default=None,
                   help="Explicit resolution override recorded in the manifest (0=native, 448=fixed448)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="Default: one real question after a long prefix")
    mode.add_argument("--full", action="store_true", help="Run every question in the supplied annotation")
    p.add_argument("--smoke-videos", type=int, default=1)
    p.add_argument("--smoke-questions", type=int, default=1)
    p.add_argument("--smoke-min-seconds", type=float, default=64.0)
    p.add_argument("--smoke-max-seconds", type=float, default=120.0)
    p.add_argument("--dry-run", action="store_true", help="Offline CPU-only preflight; write manifest and staged annotation")
    p.add_argument("--strict-budget", action="store_true", help="Reject preserved historical integer rounding differences")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        registry = json.loads(PROTOCOLS.read_text(encoding="utf-8"))
        method = "v8" if args.method == "memheadkv" else args.method
        protocol = registry["methods"][method]
        budget = args.budget if args.budget is not None else protocol["default_budget"]
        if budget < 64 or args.smoke_videos < 1 or args.smoke_questions < 1:
            raise ValueError("Budget must be >=64, and smoke video/question counts must be positive")
        if not args.device.isdigit():
            raise ValueError("--device accepts one non-negative physical CUDA device number")
        if not 0 < args.smoke_min_seconds <= args.smoke_max_seconds:
            raise ValueError("Smoke time bounds must satisfy 0 < min <= max")
        options, budget_report, profiles = budget_preflight(protocol, budget, args.strict_budget)
        overrides = {}
        if args.frame_canvas_size is not None:
            if args.frame_canvas_size < 0:
                raise ValueError("--frame-canvas-size must be >=0")
            if method == "horizonkv" and args.frame_canvas_size:
                raise ValueError("HorizonKV runtime prohibits fixed frame canvas")
            options["fixed_frame_canvas_size"] = args.frame_canvas_size
            options["fixed_frame_pad_mode"] = "letterbox"
            overrides["fixed_frame_canvas_size"] = args.frame_canvas_size
        model_path, model_report = model_preflight(args.model_path)
        source = args.annotation.expanduser().resolve()
        data = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not data:
            raise ValueError("Annotation must be a nonempty list of video records")
        for row in data:
            if not isinstance(row.get("video_path"), str) or not row.get("conversations"):
                raise ValueError("Each annotation record needs video_path and nonempty conversations")
            for q in row["conversations"]:
                if not all(key in q for key in ("question", "answer", "choices", "end_time")):
                    raise ValueError("StreamingBench question fields are incomplete")
                if not math.isfinite(float(q["end_time"])) or float(q["end_time"]) <= 0:
                    raise ValueError("Question end_time must be finite and positive")
        if args.full:
            selected, indices = copy.deepcopy(data), None
        else:
            selected, indices = select_smoke(data, args.smoke_videos, args.smoke_questions,
                                             args.smoke_min_seconds, args.smoke_max_seconds)
        output = args.output_dir.expanduser().resolve()
        if output == ROOT or ROOT.is_relative_to(output):
            raise ValueError("Choose a dedicated experiment output directory")
        if source.is_relative_to(output) or model_path.is_relative_to(output):
            raise ValueError("Output cannot contain the source annotation or model directory")
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError(f"Output directory must be new or empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
        staged = output / "annotation.source.json"
        with staged.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(selected, ensure_ascii=False, indent=2) + "\n")
        annotation = output / "annotation.json"
        if args.video_root:
            subprocess.run([sys.executable, str(ROOT / "tools/relocate_annotation.py"),
                            "--input", str(staged), "--video-root", str(args.video_root.resolve()),
                            "--output", str(annotation)], check=True)
            selected = json.loads(annotation.read_text(encoding="utf-8"))
        else:
            for row in selected:
                video = Path(row["video_path"]).expanduser()
                if not video.is_absolute():
                    video = source.parent / video
                row["video_path"] = str(video.resolve())
            with annotation.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(selected, ensure_ascii=False, indent=2) + "\n")
        missing = [row["video_path"] for row in selected if not Path(row["video_path"]).is_file()]
        if missing:
            raise ValueError(f"Videos missing (use --video-root): {missing[:5]}")
        command = [sys.executable, str(ROOT / "video_qa/run_infer.py"),
                   "--model", "qwen2.5_vl_7b", "--model_path", str(model_path),
                   "--dataset", "streamingbench", "--anno_path", str(annotation),
                   "--num_chunks", "1", "--devices", args.device,
                   "--sample_fps", "0.5", "--kv_size", str(budget),
                   "--compress_mode", "hermes", "--debug", "false", "--output_dir", str(output)]
        for key, value in options.items():
            command.extend([f"--{key}", str(value).lower() if isinstance(value, bool) else str(value)])
        env = os.environ.copy()
        env.update({"PYTHONPATH": str(ROOT), "PROTOKV_ENABLE": "0", "INFINIPOT_V_PUBLIC": "0",
                    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "DECORD_EOF_RETRY_MAX": "40960"})
        manifest = {"method": method, "requested_method": args.method, "budget": budget,
                    "runtime_dtype": "float16", "runtime_dtype_source": "inference/qwenvl_hermes.py load_model torch_dtype=torch.float16",
                    "protocol": protocol, "protocol_sha256": sha256(PROTOCOLS), "overrides": overrides,
                    "model": model_report, "profile_inputs": profiles, "budget_check": budget_report,
                    "annotation": {"source": str(source), "source_sha256": sha256(source),
                                   "prepared": str(annotation), "prepared_sha256": sha256(annotation),
                                   "videos": len(selected), "questions": sum(len(r["conversations"]) for r in selected),
                                   "selection": "full" if args.full else "real long-prefix smoke",
                                   "source_indices": indices},
                    "smoke_compression": {"sample_fps": 0.5, "minimum_prefix_seconds": args.smoke_min_seconds,
                                          "note": "Real prefixes are retained. Verify actual compression in inference.log; dry-run never loads or decodes a model/video."},
                    "command": command, "cwd": str(ROOT),
                    "environment": {key: env[key] for key in ("PYTHONPATH", "PROTOKV_ENABLE", "INFINIPOT_V_PUBLIC", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")},
                    "status": "preflight_passed"}
        manifest_path = output / "experiment_manifest.json"
        with manifest_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        print(f"Preflight passed: {method}, K={budget}, {len(selected)} videos, "
              f"{manifest['annotation']['questions']} questions; visual slots "
              f"{budget_report['runtime_total_visual_slots']} (rounding delta {budget_report['rounding_delta']:+d})")
        print(shlex.join(command), flush=True)
        if args.dry_run:
            return 0
        with (output / "inference.log").open("x", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        expected = manifest["annotation"]["questions"]
        results = output / "results.csv"
        rows = list(csv.DictReader(results.open(encoding="utf-8"))) if results.is_file() else []
        evidence = compression_evidence(method, (output / "inference.log").read_text(encoding="utf-8", errors="replace"))
        complete = completed.returncode == 0 and len(rows) == expected and (args.full or evidence["events"] > 0)
        manifest["status"] = "completed" if complete else "failed"
        manifest["returncode"] = completed.returncode
        manifest["result_rows"] = len(rows)
        manifest["compression_evidence"] = evidence
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if not complete:
            print(f"Inference incomplete: rc={completed.returncode}, rows={len(rows)}/{expected}, "
                  f"compression_events={evidence['events']}; see {output / 'inference.log'}", file=sys.stderr)
            return completed.returncode or 1
        print(f"Completed: {results}")
        return 0
    except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
