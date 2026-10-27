"""Bounded controller for the original lmms-eval CLI and streaming plugin.

All inference runs in ``python -m lmms_eval``. This controller prepares local
annotations, records provenance, holds the existing GPU lease, and audits the
native CLI's CSV, metrics and runtime checks. It never resumes full experiments.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import shutil
import sys
from datetime import datetime, timezone
from dataclasses import replace

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
FRAMEWORK = WORKSPACE / "sparsemm_baselines/upstream/lmms-eval"
DEPS = ROOT / "runtime_deps"
TIMING_ANNOTATIONS = WORKSPACE / "hermes/data/streamingbench/streamingbench_realtime_half_userpaths.json"
TIMING_SOURCE_SHA256 = "c7b202726355c95fe26326a9983f70ff784e99ce92a33d970b272816670a9c7b"
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from sparsemm_baselines.run_streamingbench import (
    gpu_lease, load_resume, prepare_annotations, sha256, verify_upstream, write_json,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("protokv", "adakv", "pyramidkv", "lukv", "sparsemm", "uniform"),
                        default=os.environ.get("METHOD", "protokv"))
    parser.add_argument("--anno-path", type=Path, default=WORKSPACE / "hermes/data/streamingbench/streamingbench_realtime_half_userpaths.json")
    parser.add_argument("--model-path", type=Path, default=WORKSPACE / "hermes/models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--max-questions", type=int, help="Per-video limit; videos remain complete evaluation documents")
    parser.add_argument("--budget", type=int, default=os.environ.get("BUDGET", "1000"))
    parser.add_argument("--budget-scope", choices=("visual", "prompt"), default="visual")
    parser.add_argument("--lukv-budget-path", type=Path, help="Explicit historical fixed LU budget CSV")
    parser.add_argument("--lukv-profile-path", type=Path, help="Attention-calibrated LU compression-ratio lookup profile")
    parser.add_argument("--sparsemm-score-path", type=Path, help="Qwen2.5 SparseMM visual-head calibration scores")
    parser.add_argument("--sparsemm-budget-path", type=Path, help="External per-KV-head capacity_n CSV for SparseMM selection")
    parser.add_argument("--sparsemm-cache-policy", choices=("capacity", "two_bank"))
    parser.add_argument("--sparsemm-two-bank-gate", choices=("per_head_capacity", "legacy_global"))
    parser.add_argument("--sparsemm-ratio", type=float, default=None, help="SparseMM uniform history floor ratio (default 0.1)")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--sample-fps", type=float, default=0.5)
    parser.add_argument("--encode-chunk-size", type=int, default=16)
    parser.add_argument("--video-chunk-mode", choices=("temporal_patch", "hermes_chunk"),
                        default="temporal_patch", help="Qwen visual-forward/cache-update granularity")
    visual_inputs = parser.add_mutually_exclusive_group()
    visual_inputs.add_argument("--video-max-pixels", type=int, help="Optional Qwen ProtoKV/SparseMM video-frame pixel cap; changes visual inputs")
    visual_inputs.add_argument("--fixed-frame-size", type=int, default=None,
                               help="Shared Qwen2.5-VL fixed square RGB canvas (896 yields 1024 tokens/temporal unit)")
    parser.add_argument("--config-json", type=Path, help="Explicit StreamingConfig settings; method/budget/scope must match CLI")
    profiles = parser.add_mutually_exclusive_group()
    profiles.add_argument("--small-validation", action="store_true", help="Authorize only this bounded native CLI validation")
    profiles.add_argument("--five-video-timing", action="store_true", help="Authorize only ProtoKV on the original first five complete videos")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_config(args):
    from sparsemm_protokv.config import StreamingConfig

    fixed = {"method": args.method, "budget": args.budget, "budget_scope": args.budget_scope,
             "video_chunk_mode": getattr(args, "video_chunk_mode", "temporal_patch")}
    if getattr(args, "lukv_budget_path", None) is not None:
        fixed["lukv_budget_path"] = str(args.lukv_budget_path.expanduser().resolve())
    if getattr(args, "lukv_profile_path", None) is not None:
        fixed["lukv_profile_path"] = str(args.lukv_profile_path.expanduser().resolve())
    if getattr(args, "sparsemm_score_path", None) is not None:
        fixed["sparsemm_score_path"] = str(args.sparsemm_score_path.expanduser().resolve())
    if getattr(args, "sparsemm_budget_path", None) is not None:
        fixed["sparsemm_budget_path"] = str(args.sparsemm_budget_path.expanduser().resolve())
    if getattr(args, "sparsemm_cache_policy", None) is not None:
        fixed["sparsemm_cache_policy"] = args.sparsemm_cache_policy
    if getattr(args, "sparsemm_two_bank_gate", None) is not None:
        fixed["sparsemm_two_bank_gate"] = args.sparsemm_two_bank_gate
    if getattr(args, "sparsemm_ratio", None) is not None:
        fixed["sparsemm_ratio"] = args.sparsemm_ratio
    extra = {} if args.config_json is None else json.loads(args.config_json.read_text())
    if not isinstance(extra, dict):
        raise ValueError("--config-json must contain an object")
    for key, value in fixed.items():
        if key in extra and extra[key] != value:
            raise ValueError(f"Config {key} conflicts with its explicit CLI value")
    config = StreamingConfig(**dict(extra, **fixed))
    if config.method == "lukv":
        from sparsemm_lukv.policy import DEFAULT_PROFILE_PATH
        if config.lukv_budget_path:
            config = replace(config, lukv_budget_path=str(Path(config.lukv_budget_path).expanduser().resolve()))
        else:
            selected = Path(config.lukv_profile_path or DEFAULT_PROFILE_PATH).expanduser().resolve()
            config = replace(config, lukv_profile_path=str(selected))
    elif config.method == "sparsemm":
        from sparsemm_native.policy import DEFAULT_SCORE_PATH
        selected = Path(config.sparsemm_score_path or DEFAULT_SCORE_PATH).expanduser().resolve()
        config = replace(config, sparsemm_score_path=str(selected))
        if config.sparsemm_budget_path:
            config = replace(config, sparsemm_budget_path=str(Path(config.sparsemm_budget_path).expanduser().resolve()))
    return config


def sparsemm_asset_metadata(config, model_path):
    if config.method != "sparsemm":
        return None
    from sparsemm_native.assets import asset_metadata
    from sparsemm_native.policy import load_head_scores, allocate_head_budgets, load_fixed_head_budgets
    from sparsemm_native.two_bank import load_two_bank_budgets
    metadata = asset_metadata(config.sparsemm_score_path or None)
    model = json.loads((model_path / "config.json").read_text())
    text = model.get("text_config", model)
    if Path(metadata["model_path"]).resolve() != model_path.resolve():
        raise ValueError("SparseMM scores were calibrated for a different model")
    scores = load_head_scores(metadata["path"], int(text["num_hidden_layers"]), int(text["num_attention_heads"]))
    if config.sparsemm_cache_policy == "two_bank":
        _, _, allocation = load_two_bank_budgets(config.sparsemm_budget_path,
            int(text["num_hidden_layers"]), int(text["num_key_value_heads"]),
            config.budget, config.window_size, short_cache_gate=config.sparsemm_two_bank_gate)
    elif config.sparsemm_budget_path:
        _, allocation = load_fixed_head_budgets(config.sparsemm_budget_path,
            int(text["num_hidden_layers"]), int(text["num_key_value_heads"]),
            config.budget, config.window_size)
    else:
        _, allocation = allocate_head_budgets(scores, budget=config.budget, window_size=config.window_size,
                                             ratio=config.sparsemm_ratio, num_kv_heads=int(text["num_key_value_heads"]))
    return dict(metadata, allocation=allocation)


def lukv_asset_metadata(config, model_path):
    if config.method != "lukv":
        return None
    from sparsemm_lukv.policy import configured_asset_metadata
    model = json.loads((model_path / "config.json").read_text())
    text = model.get("text_config", model)
    return configured_asset_metadata(config, num_layers=int(text["num_hidden_layers"]),
                                     num_kv_heads=int(text["num_key_value_heads"]), model_path=model_path)


def check_bounds(args, data):
    if getattr(args, "five_video_timing", False):
        return check_five_video_timing(args, data)
    if not 1 <= args.max_videos <= 3 or not 1 <= len(data) <= 3:
        raise ValueError("Native validation is limited to 1..3 videos; full experiments remain paused")
    questions = sum(len(video["conversations"]) for video in data)
    if not 1 <= questions <= 8:
        raise ValueError("Native validation is limited to at most 8 questions; select fewer per video")
    if any(float(question["end_time"]) > 120 for video in data for question in video["conversations"]):
        raise ValueError("Native validation endpoints must be at most 120 seconds")
    if not math.isfinite(args.sample_fps) or args.sample_fps <= 0 or args.encode_chunk_size < 1:
        raise ValueError("Sampling FPS and chunk size must be positive")
    return questions


def check_five_video_timing(args, data):
    if args.method != "protokv" or args.max_videos != 5 or args.max_questions is not None:
        raise ValueError("Five-video timing requires ProtoKV, exactly five videos and no question truncation")
    if sha256(args.anno_path) != TIMING_SOURCE_SHA256 or sha256(TIMING_ANNOTATIONS) != TIMING_SOURCE_SHA256:
        raise ValueError("Five-video timing requires the pinned original annotation source")
    expected = prepare_annotations(TIMING_ANNOTATIONS, 5)
    if data != expected or [str(v["video_id"]) for v in data] != ["3", "5", "6", "11", "12"]:
        raise ValueError("Five-video timing must exactly match the original first five complete videos")
    if sum(len(v["conversations"]) for v in data) != 25:
        raise ValueError("Five-video timing requires all 25 original questions")
    if not math.isfinite(args.sample_fps) or args.sample_fps <= 0 or args.encode_chunk_size < 1:
        raise ValueError("Sampling FPS and chunk size must be positive")
    return 25


def source_hashes(lukv_metadata=None, sparsemm_metadata=None):
    files = [ROOT / name for name in (
        "__init__.py", "cache.py", "config.py", "monkeypatch.py", "qwen2_5.py", "qwen2.py",
        "model_loader.py", "streaming.py", "run_streamingbench.py",
    )]
    for directory in (ROOT / "models", ROOT / "tasks"):
        files += [path for path in directory.rglob("*") if path.suffix in (".py", ".yaml", ".yml") and not path.name.startswith("test_")]
    files += [WORKSPACE / "protokv/full_state.py", WORKSPACE / "kvbench/fixed_inputs.py"]
    lukv = WORKSPACE / "sparsemm_lukv"
    if lukv.is_dir():
        files += [path for path in lukv.glob("*.py") if not path.name.startswith(("test_", "audit"))]
        if (lukv / "assets").is_dir():
            files += [path for path in (lukv / "assets").rglob("*") if path.is_file() and path.suffix in (".csv", ".json", ".sha256")]
    if lukv_metadata:
        assets = [lukv_metadata["path"], *lukv_metadata.get("calibration_manifests", {}),
                  *lukv_metadata.get("calibration_sources", {})]
        files += [Path(path).resolve() for path in assets if Path(path).resolve().is_relative_to(WORKSPACE)]
    native = WORKSPACE / "sparsemm_native"
    if native.is_dir():
        files += [path for path in native.glob("*.py") if not path.name.startswith(("test_", "audit"))]
    uniform = WORKSPACE / "sparsemm_uniform"
    if uniform.is_dir():
        files += [path for path in uniform.glob("*.py") if not path.name.startswith(("test_", "audit"))]
    if sparsemm_metadata:
        assets = [sparsemm_metadata["path"], sparsemm_metadata.get("allocation", {}).get("budget_path", ""),
                  *sparsemm_metadata.get("calibration_manifests", {}),
                  *sparsemm_metadata.get("calibration_sources", {})]
        files += [Path(path).resolve() for path in assets if path and Path(path).resolve().is_relative_to(WORKSPACE)]
    files += [WORKSPACE / "sparsemm_baselines" / name for name in (
        "core.py", "cache.py", "adapter.py", "run_streamingbench.py", "upstream/sparsemm/sparsemm_utils.py",
    )]
    files += [FRAMEWORK / "lmms_eval" / name for name in (
        "__init__.py", "__main__.py", "models/__init__.py", "api/model.py", "api/task.py", "utils.py", "evaluator.py",
    )]
    return {str(path.relative_to(WORKSPACE)): sha256(path) for path in sorted(set(files)) if path.is_file()}


def build_command(args, config, out, video_count=None):
    from sparsemm_protokv.model_loader import model_type
    backend = model_type(args.model_path)
    video_max_pixels = getattr(args, "video_max_pixels", None)
    fixed_frame_size = getattr(args, "fixed_frame_size", None)
    if fixed_frame_size is not None:
        from kvbench.fixed_inputs import fixed_input_metadata
        fixed_input_metadata(fixed_frame_size)
        if backend != "qwen2_5_vl":
            raise ValueError("--fixed-frame-size currently requires Qwen2.5-VL")
        if video_max_pixels is not None:
            raise ValueError("--fixed-frame-size and --video-max-pixels cannot be combined")
    if video_max_pixels is not None:
        if config.method not in ("protokv", "sparsemm") or backend != "qwen2_5_vl":
            raise ValueError("--video-max-pixels is supported only for Qwen2.5-VL ProtoKV/SparseMM")
        if video_max_pixels < 28 * 28 * 4:
            raise ValueError("--video-max-pixels must be at least 3136")
    if backend == "llava_onevision" and config.video_chunk_mode != "temporal_patch":
        raise ValueError("HERMES chunk mode is Qwen2.5-VL only; LLaVA uses per-frame features")
    native_model = ("llava_onevision_stream" if backend == "llava_onevision"
                    else "qwen2_5_vl_stream")
    five = getattr(args, "five_video_timing", False)
    fields = dict(config.asdict(), pretrained=str(args.model_path.resolve()), dtype=args.dtype,
                  sample_fps=args.sample_fps, encode_chunk_size=args.encode_chunk_size, output_dir=str(out),
                  small_validation=not five, five_video_timing=five)
    if video_max_pixels is not None:
        fields["video_max_pixels"] = video_max_pixels
    if fixed_frame_size is not None:
        fields["fixed_frame_size"] = fixed_frame_size
    if five:
        selected = prepare_annotations(args.anno_path, 5)
        fields.update(timing_source_sha256=TIMING_SOURCE_SHA256,
                      timing_subset_sha256=hashlib.sha256(json.dumps(selected, sort_keys=True, ensure_ascii=False).encode()).hexdigest())
    elif config.method in ("lukv", "sparsemm", "uniform"):
        selected = prepare_annotations(args.anno_path, args.max_videos, args.max_questions)
        fields.update(timing_source_sha256=sha256(args.anno_path),
                      timing_subset_sha256=hashlib.sha256(json.dumps(selected, sort_keys=True, ensure_ascii=False).encode()).hexdigest())
    values = []
    for key, value in fields.items():
        rendered = str(value).lower() if isinstance(value, bool) else str(value)
        if any(char in rendered for char in (",", "\n", "\r")):
            raise ValueError(f"Native lmms model_args cannot safely represent {key}")
        values.append(f"{key}={rendered}")
    command = [sys.executable, "-u", "-m", "lmms_eval", "--model", native_model,
               "--model_args", ",".join(values), "--tasks", "streamingbench_local",
               "--include_path", str(ROOT / "tasks"), "--batch_size", "1",
               "--limit", str(args.max_videos if video_count is None else video_count), "--gen_kwargs", "max_new_tokens=16,temperature=0,num_beams=1",
               "--output_path", str(out / "lmms"), "--log_samples"]
    overrides = {
        "METHOD": config.method, "BUDGET": str(config.budget), "LMMS_EVAL_PLUGINS": "sparsemm_protokv",
        "STREAMINGBENCH_ANNOTATIONS": str(out / "annotations.json"),
        "PYTHONPATH": os.pathsep.join((str(WORKSPACE), str(FRAMEWORK), str(DEPS))),
        "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "DECORD_EOF_RETRY_MAX": "40960",
        "CUDA_VISIBLE_DEVICES": os.environ.get("SPARSEMM_GPU_INDEX", "0"),
        "PROTOKV_ENABLE": "0", "PROTOKV_FULL_ENABLE": "0", "INFINIPOT_V_PUBLIC": "0",
    }
    if five or config.method in ("lukv", "sparsemm", "uniform"):
        # Bound CPU preprocessing/allocator memory under the 32 GiB job limit.
        overrides.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MALLOC_ARENA_MAX="2")
    return command, overrides


def preflight(environment, expected_videos, expected_questions, model_args=None,
              model_name="qwen2_5_vl_stream"):
    """Import the real framework, registry and local task without loading weights."""
    code = '''import importlib.metadata as metadata, json
from pathlib import Path
import lmms_eval
from lmms_eval.models import get_model
from lmms_eval.tasks import TaskManager
model_class = get_model("MODEL_NAME")
model_args = MODEL_ARGUMENTS
model = model_class(device="cpu") if model_args is None else model_class.create_from_arg_string(model_args, {"device": "cpu"})
assert model._bridge is None
manager = TaskManager(include_path="TASK_PATH", include_defaults=False, model_name="MODEL_NAME")
task = manager.load_task_or_group("streamingbench_local")["streamingbench_local"]
docs = task.dataset["test"]
versions = {}
for name in ("torch", "transformers", "flash-attn", "numpy", "pillow", "decord", "accelerate", "datasets", "evaluate", "loguru", "sqlitedict", "pyarrow"):
    try: versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError: versions[name] = None
print(json.dumps({"lmms_eval_file": str(Path(lmms_eval.__file__).resolve()),
    "model_class": type(model).__module__ + "." + type(model).__name__,
    "videos": len(docs), "questions": sum(len(json.loads(doc["video_json"])["conversations"]) for doc in docs),
    "dependencies": versions, "lukv_asset_metadata": model.lukv_asset_metadata,
    "sparsemm_asset_metadata": model.sparsemm_asset_metadata}))
'''.replace('"TASK_PATH"', repr(str(ROOT / "tasks"))).replace('"MODEL_NAME"', repr(model_name)).replace("MODEL_ARGUMENTS", repr(model_args))
    output = subprocess.check_output([sys.executable, "-c", code], cwd=WORKSPACE,
                                     env=environment, text=True)
    runtime = json.loads(output.strip().splitlines()[-1])
    if Path(runtime["lmms_eval_file"]).resolve() != (FRAMEWORK / "lmms_eval/__init__.py").resolve():
        raise RuntimeError("Python imported another lmms-eval instead of the pinned SparseMM framework")
    if runtime["videos"] != expected_videos or runtime["questions"] != expected_questions:
        raise RuntimeError("Native task selection differs from the prepared annotations")
    if runtime["dependencies"]["transformers"] != "4.57.1":
        raise RuntimeError("The Qwen2.5-VL plugin is validated on transformers==4.57.1")
    return runtime


def native_result(out, expected_accuracy):
    matches = []
    for path in (out / "lmms").rglob("*results*.json"):
        data = json.loads(path.read_text())
        metrics = data.get("results", {}).get("streamingbench_local", {})
        if "mcqa_accuracy,none" in metrics:
            value = metrics["mcqa_accuracy,none"]
            if not isinstance(value, (float, int)) or not math.isfinite(value):
                raise RuntimeError("Native lmms accuracy is not finite")
            if not math.isclose(value, expected_accuracy, rel_tol=0, abs_tol=1e-8):
                raise RuntimeError("Native lmms task accuracy disagrees with independent CSV accuracy")
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one native lmms result file; found {len(matches)}")
    return matches[0]


def validate_runtime(out, videos, questions, method=None):
    checks = json.loads((out / "runtime_checks.json").read_text())
    if (checks.get("status") != "complete" or checks.get("failures")
            or checks.get("qa_restore_checks") != questions
            or len(checks.get("qa_checks", [])) != questions):
        raise RuntimeError("Native runtime checks did not finish every question cleanly")
    if any(not check.get("returned") or not check.get("restored") for check in checks["qa_checks"]):
        raise RuntimeError("A question did not return and restore its persistent state")
    for field in ("finite_hidden_checks", "finite_logits_checks"):
        if checks.get(field, 0) < questions:
            raise RuntimeError(f"Missing {field}")
    if (method or checks.get("config", {}).get("method")) == "sparsemm":
        if checks.get("probe_checks") != [] or checks.get("probe_restore_checks") != 0:
            raise RuntimeError("SparseMM must not execute HERMES probes")
        protocol = json.loads((out / "protocol.json").read_text())
        allocation = protocol["sparsemm_allocation"]
        table = allocation["head_budgets"]
        two_bank = protocol.get("sparsemm_cache_policy") == "two_bank"
        if two_bank and (allocation.get("dp_bank_split_used_online") is not True
                         or allocation.get("short_cache_gate") != "legacy_global"):
            raise RuntimeError("DP two-bank allocation was not activated")
        layers, heads = len(table), len(table[0])
        prefixes, prefills = {}, set()
        for line in (out / "cache_trace.jsonl").read_text().splitlines():
            event = json.loads(line)
            if event["event"] == "init":
                prefixes[str(event["video_id"])] = event["text_tokens"]
            elif event["event"] == "video_prefill":
                prefills.add((str(event["video_id"]), event["prefill"]))
        groups = {}
        fields = ("validation_passed", "prefix_preserved", "recent_preserved", "indices_unique", "indices_in_range", "positions_match_kept")
        for item in checks.get("selector_checks", []):
            identity = (str(item["video_id"]), item["prefill_index"])
            layer = item["layer_idx"]
            if (identity not in prefills or not isinstance(layer, int) or isinstance(layer, bool)
                    or not 0 <= layer < layers or any(item.get(field) is not True for field in fields)
                    or layer in groups.setdefault(identity, set()) or item["head_budgets"] != table[layer]):
                raise RuntimeError("SparseMM selector identity, capacities or preservation checks failed")
            groups[identity].add(layer)
            prefix = prefixes[identity[0]]
            before, after = item["before_head_lengths"], item["head_lengths"]
            if len(before) != heads or len(after) != heads:
                raise RuntimeError("SparseMM selector has a wrong head count")
            if two_bank:
                recent = allocation["recent_budgets"][layer]
                if (item.get("allocation") != allocation
                        or item.get("recent_bank_preserved") is not True
                        or item.get("bank_counts_valid") is not True
                        or item.get("quota_or_reference_warmup") is not True
                        or item.get("head_budgets") != table[layer]
                        or any(r < protocol["window_size"] or r > n for r, n in zip(recent, table[layer]))):
                    raise RuntimeError("DP two-bank selector failed recent/history validation")
            bypass = all(max(0, n - prefix - protocol["window_size"]) < protocol["budget"] - protocol["window_size"] for n in before)
            expected = before if bypass else [min(n, prefix + cap) for n, cap in zip(before, table[layer])]
            if item.get("bypassed_short_cache") != bypass or after != expected:
                raise RuntimeError("SparseMM selector violated its original gate or static head quotas")
        if len(prefixes) != videos or set(groups) != prefills or any(group != set(range(layers)) for group in groups.values()):
            raise RuntimeError("SparseMM selector checks do not cover every video patch and layer")
    if (method or checks.get("config", {}).get("method")) == "lukv":
        if checks.get("probe_checks") != [] or checks.get("probe_restore_checks") != 0:
            raise RuntimeError("Visual-attention LU-KV must not execute HERMES probes")
        protocol = json.loads((out / "protocol.json").read_text())
        if protocol.get("lukv_selector") != "visual_attention" or protocol.get("lukv_history") != "none":
            raise RuntimeError("LU-KV runtime must use the visual_attention selector without QA history")
        model_config = json.loads((Path(protocol["model_path"]) / "config.json").read_text())
        text = model_config.get("text_config", model_config)
        layers, heads = int(text["num_hidden_layers"]), int(text["num_key_value_heads"])
        from sparsemm_lukv.policy import load_budget_profile, load_budget_table
        from sparsemm_lukv.profile import allocate_profile
        profile = None
        budgets = None
        if protocol.get("lukv_profile_path"):
            profile = load_budget_profile(protocol["lukv_profile_path"], layers, heads, protocol["window_size"])
        else:
            budgets = load_budget_table(protocol["lukv_budget_path"], layers, heads,
                                        protocol["budget"], protocol["window_size"]).tolist()
        prefixes, prefills, seen, expected_tables, expected_allocations = {}, set(), {}, {}, {}
        for line in (out / "cache_trace.jsonl").read_text().splitlines():
            event = json.loads(line)
            video = str(event["video_id"])
            if event.get("event") == "init":
                prefixes[video] = int(event["text_tokens"])
                seen[video] = 0
            elif event.get("event") == "video_prefill":
                identity = (video, event["prefill"])
                prefills.add(identity)
                seen[video] += int(event["input_tokens"])
                if profile is not None:
                    table, allocation = allocate_profile(profile, seen[video], protocol["budget"], protocol["window_size"])
                    expected_tables[identity] = table.tolist()
                    expected_allocations[identity] = allocation
                else:
                    expected_tables[identity] = budgets
        if len(prefixes) != videos or not prefills:
            raise RuntimeError("LU-KV selector validation is missing video init/prefill events")
        selectors = checks.get("selector_checks", [])
        groups = {}
        fields = ("validation_passed", "prefix_preserved", "recent_preserved", "indices_unique", "indices_in_range", "positions_match_kept")
        for item in selectors:
            video, prefill, layer = str(item["video_id"]), item["prefill_index"], item["layer_idx"]
            if (video not in prefixes or not isinstance(prefill, int) or isinstance(prefill, bool) or prefill < 1
                    or not isinstance(layer, int) or isinstance(layer, bool) or not 0 <= layer < layers
                    or any(item.get(field) is not True for field in fields)):
                raise RuntimeError("LU-KV attention selector identity or preservation validation failed")
            identity = (video, prefill)
            if layer in groups.setdefault(identity, set()):
                raise RuntimeError("LU-KV attention selector duplicated a layer in a prefill")
            groups[identity].add(layer)
            before, after, caps = item.get("before_head_lengths"), item.get("head_lengths"), item.get("head_budgets")
            if (not all(isinstance(values, list) and len(values) == heads for values in (before, after, caps))
                    or not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0
                               for values in (before, after, caps) for value in values)
                    or identity not in expected_tables or caps != expected_tables[identity][layer]):
                raise RuntimeError("LU-KV attention selector lengths or calibrated head capacities disagree")
            if profile is not None and item.get("allocation") != expected_allocations[identity]:
                raise RuntimeError("LU-KV profile allocation does not match uncompressed visual T")
            prefix = prefixes[video]
            expected = [prefix + min(length - prefix, cap) for length, cap in zip(before, caps)]
            if (any(length < prefix for length in before) or after != expected
                    or item.get("compressed") is not any(new < old for old, new in zip(before, after))):
                raise RuntimeError("LU-KV attention selector did not enforce each fixed visual head capacity")
        if set(groups) != prefills or any(group != set(range(layers)) for group in groups.values()):
            raise RuntimeError("LU-KV requires selector checks for every native layer of every visual prefill")
    stats = json.loads((out / "inference_stats.json").read_text())
    if stats.get("status") != "complete" or stats.get("completed_videos") != videos or stats.get("questions") != questions:
        raise RuntimeError("Native inference statistics do not match the selected dataset")
    peak = stats.get("peak_gpu_allocated_gib")
    if not isinstance(peak, (int, float)) or not math.isfinite(peak) or peak <= 0:
        raise RuntimeError("Native inference did not report a valid GPU peak")
    return stats


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.dry_run and not (args.small_validation or args.five_video_timing):
        parser.error("Execution requires an explicit bounded profile; full experiments remain paused")
    if args.max_videos is None:
        args.max_videos = 5 if args.five_video_timing else 2
    config = build_config(args)
    lukv_metadata = lukv_asset_metadata(config, args.model_path)
    sparsemm_metadata = sparsemm_asset_metadata(config, args.model_path)
    data = prepare_annotations(args.anno_path, args.max_videos, args.max_questions)
    question_count = check_bounds(args, data)
    commit = verify_upstream()
    profile = "five_video_timing" if args.five_video_timing else "small_validation"
    out = (args.output_dir or ROOT / f"results/{profile}/{config.method}_k{config.budget}_{config.budget_scope}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ("1_0.csv", "results.csv")):
        raise FileExistsError("Native validation does not resume or overwrite CSVs; use a new --output-dir")
    if (out / "run.status").exists() and (out / "run.status").read_text().strip() == "running":
        raise RuntimeError("Output directory is marked running; use a new output directory")
    subset_hash = hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    command, overrides = build_command(args, config, out, video_count=len(data))
    protocol = dict(config.asdict(), dtype=args.dtype, model_path=str(args.model_path.resolve()),
                    sample_fps=args.sample_fps, encode_chunk_size=args.encode_chunk_size,
                    annotation_source=str(args.anno_path.resolve()), source_sha256=sha256(args.anno_path),
                    subset_sha256=subset_hash,
                    video_max_pixels=getattr(args, "video_max_pixels", None),
                    native_resolution=(command[command.index("--model") + 1] != "llava_onevision_stream"
                                       and getattr(args, "video_max_pixels", None) is None),
                    video_preprocessing=("resize_and_pad_384x384" if command[command.index("--model") + 1] == "llava_onevision_stream"
                                         else "qwen_dynamic_processor_pixel_cap" if getattr(args, "video_max_pixels", None) is not None
                                         else "native_frames_qwen_dynamic_processor"),
                    upstream_commit=commit,
                    reproduction="SparseMM-native-lmms-plugin-streaming-adaptation",
                    native_model=command[command.index("--model") + 1], native_task="streamingbench_local",
                    sampling="rounded_source_fps_integer_stride; endpoint=ceil(end_time*sample_fps)",
                    eviction_granularity=("sampled_frame" if command[command.index("--model") + 1] == "llava_onevision_stream"
                                          else "runner_video_chunk" if config.video_chunk_mode == "hermes_chunk"
                                          else "vision_temporal_patch"),
                    question_cache="temporary_then_restore_video_cache",
                    stderr="none; custom question-weighted aggregation has no registered stderr function",
                    initial_text="outside_visual_budget" if args.budget_scope == "visual" else "inside_prompt_budget",
                    small_validation=not args.five_video_timing, five_video_timing=args.five_video_timing,
                    timing_source_sha256=TIMING_SOURCE_SHA256 if args.five_video_timing else sha256(args.anno_path) if config.method in ("lukv", "sparsemm", "uniform") else None,
                    timing_subset_sha256=subset_hash if args.five_video_timing or config.method in ("lukv", "sparsemm", "uniform") else None,
                    reader_lifecycle="batch_scoped_decord" if args.five_video_timing or config.method in ("lukv", "sparsemm", "uniform") else "per_video_decord",
                    full_benchmarks_paused=True, version=6)
    if getattr(args, "fixed_frame_size", None) is not None:
        from kvbench.fixed_inputs import fixed_input_metadata
        protocol.update(fixed_frame_size=args.fixed_frame_size,
                        fixed_visual_input=fixed_input_metadata(args.fixed_frame_size),
                        native_resolution=False,
                        video_preprocessing="shared_fixed_square_rgb_no_processor_resize",
                        version=7)
    if config.method == "uniform":
        protocol.update(uniform_budget_mode="uniform_per_layer_per_kv_head",
                        uniform_selector="attention_topk", uniform_offline_assets=False,
                        uniform_scoring="FP32 causal Q-to-K attention, avgpool, GQA mean",
                        uniform_probe_count=0,
                        compression_gate="any current budgeted head length >= configured budget",
                        compression_gate_includes_prefix=config.budget_scope == "prompt")
    if lukv_metadata:
        protocol.update(lukv_asset_sha256=lukv_metadata["sha256"],
                        lukv_calibration_manifests=lukv_metadata.get("calibration_manifests", {}),
                        lukv_calibration_sources=lukv_metadata.get("calibration_sources", {}),
                        lukv_calibration_summary=lukv_metadata.get("calibration_summary", {}),
                        compression_gate="per-head visual length > fixed LU head capacity",
                        compression_gate_includes_prefix=False,
                        lukv_scoring="direct visual Q-to-K attention using SparseMM FP32 causal scores, avgpool, GQA mean",
                        lukv_probe_count=0, lukv_recency_bonus=False,
                        lukv_budget_transfer=("attention_calibrated_profile" if config.lukv_profile_path else "legacy_calibrated_transfer"),
                        lukv_budget_recalibrated=bool(config.lukv_profile_path),
                        lukv_budget_mode=("compression_ratio_lookup" if config.lukv_profile_path else "fixed_csv"))
    if sparsemm_metadata:
        protocol.update(sparsemm_asset_sha256=sparsemm_metadata["sha256"],
                        sparsemm_asset_metadata=sparsemm_metadata,
                        sparsemm_allocation=sparsemm_metadata["allocation"],
                        compression_gate="any current physical visual history >= budget-window_size",
                        sparsemm_bypass_gate="all current physical visual histories < budget-window_size",
                        compression_gate_includes_prefix=False,
                        sparsemm_scoring="FP32 causal visual Q-to-K attention, avgpool, GQA mean",
                        sparsemm_budget_mode=("external DP recent_r/history_q two-bank; attention-ranked history"
                                              if config.sparsemm_cache_policy == "two_bank" else
                                              "external DP capacity_n; SparseMM online selector; DP bank split unused"
                                              if config.sparsemm_budget_path else
                                              "static calibrated visual-head scores; GQA sum; upstream rounding"),
                        sparsemm_online_processor="fast", sparsemm_calibration_processor="slow")
    protocol_file = out / "protocol.json"
    if protocol_file.exists() and json.loads(protocol_file.read_text()) != protocol:
        raise RuntimeError("Output directory belongs to another protocol; use a new --output-dir")
    source = source_hashes(lukv_metadata, sparsemm_metadata)
    evaluator = WORKSPACE / "hermes/eval/eval_multiple_choice.py"
    manifest = {"python": sys.executable, "python_version": platform.python_version(),
                "argv": sys.argv if argv is None else [str(Path(__file__)), *argv],
                "upstream_commit": commit, "code_sha256": source,
                "model_files_sha256": {
                    name: sha256(args.model_path / name) for name in (
                        "config.json", "generation_config.json", "preprocessor_config.json",
                        "tokenizer_config.json", "tokenizer.json", "model.safetensors.index.json",
                    ) if (args.model_path / name).is_file()
                }, "evaluator_sha256": sha256(evaluator), "native_command": command,
                "environment": overrides, "full_benchmarks_paused": True}
    if lukv_metadata:
        manifest.update(lukv_asset_metadata=lukv_metadata, lukv_asset_sha256=lukv_metadata["sha256"])
    if sparsemm_metadata:
        manifest.update(sparsemm_asset_metadata=sparsemm_metadata, sparsemm_asset_sha256=sparsemm_metadata["sha256"])
    write_json(protocol_file, protocol)
    write_json(out / "annotations.json", data)
    write_json(out / "dataset_audit.json", {
        "source": str(args.anno_path.resolve()), "source_sha256": protocol["source_sha256"],
        "subset_sha256": subset_hash, "videos": len(data), "questions": question_count,
        "video_ids": [video["video_id"] for video in data],
    })
    write_json(out / "run_manifest.json", manifest)
    print(json.dumps({"protocol": protocol, "videos": len(data), "questions": question_count,
                      "native_command": command}, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        (out / "run.status").write_text("dry-run\n")
        return 0
    environment = dict(os.environ, **overrides)
    try:
        runtime = preflight(environment, len(data), question_count,
                            model_args=command[command.index("--model_args") + 1],
                            model_name=command[command.index("--model") + 1])
        if sparsemm_metadata:
            if runtime.get("sparsemm_asset_metadata") != {key: value for key, value in sparsemm_metadata.items() if key != "allocation"}:
                raise RuntimeError("Native SparseMM preflight asset evidence differs from the controller")
            runtime["sparsemm_asset_metadata"] = sparsemm_metadata
        manifest.update(runtime, started_utc=datetime.now(timezone.utc).isoformat())
        write_json(out / "run_manifest.json", manifest)
        snapshot_root = out / "source_snapshot"
        for name, expected in source.items():
            path = (WORKSPACE / name).resolve()
            target = snapshot_root / name
            if not path.is_relative_to(WORKSPACE) or sha256(path) != expected:
                raise RuntimeError("Runtime source changed before source snapshot capture")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        with gpu_lease(out):
            environment["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
            (out / "run.status").write_text("running\n")
            with (out / "run.log").open("w") as log:
                subprocess.run(command, cwd=WORKSPACE, env=environment,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            rows, completed = load_resume(out / "1_0.csv", data, require_complete=True)
            if completed != len(data):
                raise RuntimeError("Native CLI did not complete every selected video")
            stats = validate_runtime(out, len(data), question_count, method=config.method)
            accuracy = sum(float(row["qa_acc"]) for row in rows) / len(rows)
            native_path = native_result(out, accuracy)
            if source_hashes(lukv_metadata, sparsemm_metadata) != source:
                raise RuntimeError("Runtime source changed during evaluation; results require a new run")
            if lukv_metadata and sha256(lukv_metadata["path"]) != lukv_metadata["sha256"]:
                raise RuntimeError("LU-KV budget asset changed during evaluation")
            (out / "results.csv").write_bytes((out / "1_0.csv").read_bytes())
            with (out / "eval_results.txt").open("w") as log:
                subprocess.run([sys.executable, str(evaluator), "general", "--results_path", str(out / "results.csv")],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            summary = {"videos": len(data), "questions": len(rows),
                       "correct": sum(float(row["qa_acc"]) == 100.0 for row in rows),
                       "accuracy_percent": accuracy, "method": config.method, "budget": config.budget,
                       "budget_scope": config.budget_scope, "peak_gpu_allocated_gib": stats["peak_gpu_allocated_gib"],
                       "is_first250_complete": False, "full_benchmarks_paused": True,
                       "lmms_results_path": str(native_path), "finished_utc": datetime.now(timezone.utc).isoformat()}
            write_json(out / "summary.json", summary)
            (out / "run.status").write_text("complete\n")
            print(json.dumps(summary, indent=2), flush=True)
    except BaseException as error:
        (out / "run.status").write_text(f"failed {type(error).__name__}: {error}\n")
        raise
    return 0


if __name__ == "__main__":
    main()
