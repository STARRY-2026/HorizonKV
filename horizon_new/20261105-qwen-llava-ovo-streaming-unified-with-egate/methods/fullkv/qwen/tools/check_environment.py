#!/usr/bin/env python3
"""Check a release checkout without downloading weights or running inference.

The default requires a GPU and an importable FlashAttention extension. --cpu-only
does not require CUDA or FlashAttention, but checks the installed extension if one
is present. --assets-only runs entirely with the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.metadata
import json
import math
from pathlib import Path
import struct
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODEL_PATHS = {
    "qwen2.5_vl_3b": "models/Qwen2.5-VL-3B-Instruct",
    "qwen2.5_vl_7b": "models/Qwen2.5-VL-7B-Instruct",
    "qwen2.5_vl_32b": "models/Qwen2.5-VL-32B-Instruct",
    "llava_ov_0.5b": "models/llava-onevision-qwen2-0.5b-ov-hf",
    "llava_ov_7b": "models/llava-onevision-qwen2-7b-ov-hf",
    "llava_ov_72b": "models/llava-onevision-qwen2-72b-ov-hf",
}
METHODS = ("all", "hermes", "dp", "timestamp_v8", "horizonkv")
PINNED = {
    "torch": "2.9.1", "torchvision": "0.24.1",
    "transformers": "4.57.1", "tokenizers": "0.22.2",
    "accelerate": "1.11.0", "numpy": "2.3.3", "pandas": "2.3.2",
    "pillow": "11.3.0", "decord": "0.6.0", "av": "15.1.0",
    "qwen-vl-utils": "0.0.8", "einops": "0.8.1", "logzero": "1.7.0",
    "tqdm": "4.67.1", "safetensors": "0.6.2",
}


class Checks:
    def __init__(self):
        self.rows = []

    def add(self, ok, name, detail=""):
        self.rows.append({"ok": bool(ok), "name": name, "detail": str(detail)})

    def run(self, name, callback):
        try:
            detail = callback()
            self.add(True, name, detail or "OK")
        except Exception as exc:
            # Import errors can contain environment values. Show only the class;
            # validation errors below contain only controlled resource messages.
            detail = str(exc) if isinstance(exc, (ValueError, FileNotFoundError)) else type(exc).__name__
            self.add(False, name, detail)

    def finish(self, as_json):
        ok = all(row["ok"] for row in self.rows)
        if as_json:
            print(json.dumps({"ok": ok, "checks": self.rows}, indent=2))
        else:
            for row in self.rows:
                print(f"{'PASS' if row['ok'] else 'FAIL'} {row['name']}: {row['detail']}")
            print(f"{'PASS' if ok else 'FAIL'}: {sum(row['ok'] for row in self.rows)}/{len(self.rows)} checks passed")
        return 0 if ok else 1


def resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def read_json(path):
    if not path.is_file():
        raise FileNotFoundError(f"missing file: {path}")
    return json.loads(path.read_text())


def head_rows(path, columns):
    if not path.is_file():
        raise FileNotFoundError(f"missing profile: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(columns) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"profile columns missing: {sorted(missing)}")
        rows = list(reader)
    keys = [(int(row["layer"]), int(row["kv_head"])) for row in rows]
    expected = {(layer, head) for layer in range(28) for head in range(4)}
    if len(keys) != 112 or set(keys) != expected:
        raise ValueError("profile must have exactly 112 unique heads (28 layers x 4 KV heads)")
    for row in rows:
        for column in columns:
            if column not in {"layer", "kv_head", "HeadType"} and not math.isfinite(float(row[column])):
                raise ValueError(f"profile has non-finite {column}")
    return rows


def check_dp(root, budget, with_runtime):
    scores_path = root / f"dp_k{budget}_exact_budget_scores.csv"
    scores = head_rows(scores_path, ("layer", "kv_head", "score"))
    recent = head_rows(root / "recent_ratio_profile.csv", (
        "layer", "kv_head", "recent_ratio", "capacity_n", "recent_r", "history_q"))
    target = {(int(row["layer"]), int(row["kv_head"])): int(row["capacity_n"]) for row in recent}
    minimum = int((budget - 32) * 0.2) + 32
    for row in scores:
        key = (int(row["layer"]), int(row["kv_head"]))
        if float(row["score"]) != target[key] - minimum:
            raise ValueError("score residual does not match frozen head capacity")
    for row in recent:
        capacity = int(row["capacity_n"])
        recency = int(row["recent_r"])
        history = int(row["history_q"])
        if capacity <= 0 or min(recency, history) < 0 or recency + history != capacity:
            raise ValueError("invalid recent/history/capacity relationship")
        if not math.isclose(float(row["recent_ratio"]), recency / capacity, abs_tol=1e-12):
            raise ValueError("recent_ratio differs from recent_r/capacity_n")
    if sum(target.values()) != budget * 112:
        raise ValueError("frozen capacity sum differs from requested average budget")
    if with_runtime:
        import numpy as np
        from head_analysis.hermes_head_budget import load_head_scores
        from head_analysis.hermes_kv_head_budget import build_kv_head_budget_table
        loaded = load_head_scores(str(scores_path), 28, 28)
        actual = build_kv_head_budget_table(
            loaded, budget, num_layers=28, num_query_heads=28, num_kv_heads=4,
            scheme="sparsemm", sparsemm_ratio=0.2, sparsemm_window_size=32)
        expected = np.array([[target[layer, head] for head in range(4)] for layer in range(28)])
        if not np.array_equal(actual, expected):
            raise ValueError("runtime SparseMM mapping does not reproduce frozen allocation")
    return f"112 heads; total visual slots {budget * 112}; {'runtime mapping checked' if with_runtime else 'CSV consistency checked'}"


def check_v8(assets, with_runtime):
    scores = resolve_path(assets["scores"])
    policy = resolve_path(assets["policy"])
    head_rows(scores, ("layer", "kv_head", "score"))
    head_rows(policy, ("layer", "kv_head", "long_score", "short_score"))
    if with_runtime:
        from head_analysis.timestamp_v8_policy import load_v8_head_profiles
        load_v8_head_profiles(str(policy))
    return "score and timestamp policy: 112 heads each"


def check_horizon(assets, with_runtime):
    path = resolve_path(assets["policy"])
    rows = head_rows(path, ("layer", "kv_head", "HeadType", "RecentBudget", "MemoryBudget", "TotalBudget"))
    recent_capacity = int(assets["recent_capacity"])
    for row in rows:
        r, m, total = (int(row[name]) for name in ("RecentBudget", "MemoryBudget", "TotalBudget"))
        if r != recent_capacity or m < 0 or r + m != total:
            raise ValueError("HorizonKV recent/history budgets are inconsistent")
    if sum(int(row["TotalBudget"]) for row in rows) != 112 * int(assets["average_budget"]):
        raise ValueError("HorizonKV total budget differs from frozen budget")
    if with_runtime:
        from head_analysis.horizonkv_policy import load_horizonkv_policy
        load_horizonkv_policy(str(path), recent_capacity=recent_capacity,
                              expected_average_budget=int(assets["average_budget"]))
    return f"112 heads; mean budget {assets['average_budget']}; recent capacity {recent_capacity}"


def check_model(path, model, with_runtime=False):
    config = read_json(path / "config.json")
    expected = "qwen2_5_vl" if model.startswith("qwen") else "llava_onevision"
    if config.get("model_type") != expected:
        raise ValueError(f"model_type must be {expected}")
    if model == "qwen2.5_vl_7b":
        text_config = config.get("text_config", config)
        if (text_config.get("num_hidden_layers"), text_config.get("num_attention_heads"),
                text_config.get("num_key_value_heads")) != (28, 28, 4):
            raise ValueError("Qwen 7B shape must be 28 layers / 28 query heads / 4 KV heads")
    for name in ("preprocessor_config.json", "tokenizer_config.json"):
        read_json(path / name)
    if not any((path / name).is_file() for name in ("tokenizer.json", "tokenizer.model")):
        raise FileNotFoundError("model tokenizer.json/tokenizer.model is missing")
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        index = read_json(index_path)
        shards = sorted(set(index.get("weight_map", {}).values()))
        if not shards:
            raise ValueError("empty weight_map in model shard index")
    else:
        shards = ["model.safetensors"]
    for shard in shards:
        shard_path = path / shard
        if Path(shard).name != shard:
            raise ValueError("weight shard must be a filename inside the model directory")
        if not shard_path.is_file() or shard_path.stat().st_size < 16:
            raise FileNotFoundError(f"missing/empty weight shard: {shard}")
        with shard_path.open("rb") as handle:
            header_size = struct.unpack("<Q", handle.read(8))[0]
            if not 1 < header_size < min(256 * 1024 * 1024, shard_path.stat().st_size - 8):
                raise ValueError(f"invalid safetensors header: {shard}")
            header = json.loads(handle.read(header_size))
            data_end = max((int(value["data_offsets"][1]) for key, value in header.items()
                            if key != "__metadata__"), default=0)
            if 8 + header_size + data_end != shard_path.stat().st_size:
                raise ValueError(f"weight shard size disagrees with tensor offsets: {shard}")
    if with_runtime:
        from transformers import AutoProcessor
        AutoProcessor.from_pretrained(str(path), local_files_only=True)
    return f"config, tokenizer and {len(shards)} weight shards present (headers/sizes checked; weights not loaded)" + (
        "; processor loaded locally" if with_runtime else "")


def check_annotations(path, decode):
    rows = read_json(path)
    if not isinstance(rows, list) or not rows:
        raise ValueError("annotation must be a non-empty JSON list")
    unique = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("video_path"), str):
            raise ValueError("each annotation requires a video_path")
        video = resolve_path(row["video_path"])
        if not video.is_file():
            raise FileNotFoundError(f"missing annotation video: {video}")
        if not isinstance(row.get("conversations"), list) or not row["conversations"]:
            raise ValueError("each annotation requires non-empty conversations")
        unique.add(video)
    if decode:
        from decord import VideoReader
        reader = VideoReader(str(sorted(unique)[0]), num_threads=1)
        if len(reader) <= 0 or reader[0].asnumpy().size == 0:
            raise ValueError("sample video has no decodable first frame")
    return f"{len(rows)} videos, {len(unique)} unique existing paths" + ("; first video frame decoded" if decode else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, default="all")
    parser.add_argument("--kv-size", type=int, default=1000,
                        help="DP budget; --method all checks all five frozen DP budgets")
    parser.add_argument("--model", choices=MODEL_PATHS, default="qwen2.5_vl_7b")
    parser.add_argument("--model-path", help="local model directory, overrides conventional path")
    parser.add_argument("--anno-path", help="optional relocated annotation; all referenced videos must exist")
    parser.add_argument("--cpu-only", action="store_true", help="skip CUDA requirement; installed FlashAttention is still imported")
    parser.add_argument("--assets-only", action="store_true", help="stdlib-only asset validation, implies no CUDA/import checks")
    parser.add_argument("--skip-model", action="store_true", help="only check code/profile assets, not model files")
    parser.add_argument("--check-video-decode", action="store_true", help="decode first frame of first annotation video")
    parser.add_argument("--json", action="store_true", help="emit structured check results")
    args = parser.parse_args()
    checks = Checks()
    python_ok = sys.version_info >= (3, 10) if args.assets_only else sys.version_info[:2] == (3, 12)
    checks.add(python_ok, "Python version", f"{sys.version_info.major}.{sys.version_info.minor}; " + (
        "asset checks require Python >=3.10" if args.assets_only else "reference environment uses Python 3.12"))
    try:
        assets = read_json(ROOT / "configs/runtime_assets.json")
    except Exception:
        checks.add(False, "asset manifest", "missing or invalid configs/runtime_assets.json")
        return checks.finish(args.json)
    if args.method != "hermes" and args.model != assets["model"]:
        checks.add(False, "profile model", "frozen extension profiles require qwen2.5_vl_7b; use --method hermes for other models")
    if not args.assets_only:
        for package, expected in PINNED.items():
            try:
                actual = importlib.metadata.version(package)
                checks.add(actual.split("+")[0] == expected, f"package {package}", f"{actual}; expected {expected}")
            except importlib.metadata.PackageNotFoundError:
                checks.add(False, f"package {package}", "missing; install requirements-runtime-qwen.txt")
        for module in ("torch", "torchvision", "transformers", "numpy", "pandas", "PIL", "decord",
                       "av", "qwen_vl_utils", "einops", "logzero", "tqdm", "accelerate", "safetensors",
                       "inference.qwenvl_hermes" if args.model.startswith("qwen") else "inference.llavaov_hermes",
                       "video_qa.base", "video_qa.hermes_vqa"):
            checks.run(f"import {module}", lambda module=module: importlib.import_module(module) and "OK")
        if args.method != "hermes":
            for module in ("head_analysis.hermes_kv_head_budget", "head_analysis.hermes_kv_head_ragged_prefill",
                           "head_analysis.timestamp_v8_policy", "head_analysis.horizonkv_policy"):
                checks.run(f"import {module}", lambda module=module: importlib.import_module(module) and "OK")
        flash_present = importlib.util.find_spec("flash_attn") is not None
        if flash_present or not args.cpu_only:
            def check_flash():
                from flash_attn import flash_attn_func, flash_attn_varlen_func, __version__
                if not callable(flash_attn_func) or not callable(flash_attn_varlen_func):
                    raise ValueError("FlashAttention dense/varlen functions are missing")
                return f"{__version__}; dense and varlen extension imports OK"
            checks.run("FlashAttention import", check_flash)
        else:
            checks.add(True, "FlashAttention", "absent; CPU mode does not require the extension")
        if not args.cpu_only:
            def check_cuda():
                import torch
                if not torch.cuda.is_available():
                    raise ValueError("CUDA unavailable; GPU inference requires a working CUDA build and driver")
                return f"torch CUDA {torch.version.cuda}; {torch.cuda.device_count()} visible GPUs; GPU 0: {torch.cuda.get_device_name(0)}"
            checks.run("CUDA availability", check_cuda)
    with_runtime = not args.assets_only
    if args.method in ("all", "dp"):
        budgets = sorted(int(k) for k in assets["dp"]) if args.method == "all" else [args.kv_size]
        for budget in budgets:
            if str(budget) not in assets["dp"]:
                checks.add(False, "DP budget", f"no frozen profile for K={budget}; choose 100, 250, 500, 800 or 1000")
                continue
            directory = resolve_path(assets["dp"][str(budget)])
            checks.run(f"DP K={budget} profiles", lambda directory=directory, budget=budget: check_dp(directory, budget, with_runtime))
    if args.method in ("all", "timestamp_v8"):
        checks.run("Timestamp v8 profiles", lambda: check_v8(assets["timestamp_v8"], with_runtime))
    if args.method in ("all", "horizonkv"):
        checks.run("HorizonKV K=6000 profile", lambda: check_horizon(assets["horizonkv"], with_runtime))
    if not args.skip_model:
        model_path = resolve_path(args.model_path or MODEL_PATHS[args.model])
        checks.run("model files", lambda: check_model(model_path, args.model, with_runtime))
    if args.anno_path:
        if args.assets_only and args.check_video_decode:
            checks.add(False, "video decode", "--check-video-decode requires third-party decord; omit --assets-only")
        else:
            checks.run("annotation/video paths", lambda: check_annotations(resolve_path(args.anno_path), args.check_video_decode))
    elif args.check_video_decode:
        checks.add(False, "video decode", "--check-video-decode requires --anno-path")
    return checks.finish(args.json)


if __name__ == "__main__":
    raise SystemExit(main())
