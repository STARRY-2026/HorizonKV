"""Bounded, sequential evaluation through the existing SparseMM controller.

This module uses only the standard library. The native controller retains GPU
locking, model loading, cache validation, provenance snapshots and scoring.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from . import paths
from .registry import METHODS, resolve_methods


def _path(value):
    return Path(value).expanduser().resolve()


def _positive_integer(value, name, maximum=None):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1 or (maximum is not None and value > maximum):
        suffix = f" in 1..{maximum}" if maximum is not None else " > 0"
        raise ValueError(f"{name} must be an integer{suffix}")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class RunConfig:
    method: str | None = None
    methods: tuple[str, ...] | None = None
    anno_path: Path = field(default_factory=paths.default_annotation_path)
    model_path: Path = field(default_factory=paths.default_model_path)
    output_dir: Path | None = None
    budget: int = 1000
    max_videos: int = 3
    max_questions: int | None = 1
    sample_fps: float = 0.5
    encode_chunk_size: int = 16
    video_chunk_mode: str = "auto"
    video_max_pixels: int | None = None
    fixed_frame_size: int | None = None
    dtype: str = "bfloat16"
    lukv_profile_path: Path = field(default_factory=paths.default_lukv_profile_path)
    sparsemm_score_path: Path = field(default_factory=paths.default_sparsemm_score_path)
    sparsemm_budget_path: Path | None = None
    dp_budget_path: Path = field(default_factory=paths.default_dp_budget_path)
    dp_dual_budget_path: Path = field(default_factory=paths.default_dp_dual_budget_path)
    config_json: Path | None = None
    dry_run: bool = False
    allow_calibration_transfer: bool = False
    min_compression_events_per_video: int = 0
    python_executable: str = field(default_factory=lambda: sys.executable)

    @property
    def selected_methods(self):
        return resolve_methods(self.method, self.methods)

    @property
    def resolved_video_chunk_mode(self):
        return ("hermes_chunk" if "hermes" in self.selected_methods else "temporal_patch"
                ) if self.video_chunk_mode == "auto" else self.video_chunk_mode

    def validate(self):
        names = self.selected_methods
        if self.video_chunk_mode not in ("auto", "temporal_patch", "hermes_chunk"):
            raise ValueError("video_chunk_mode must be auto, temporal_patch or hermes_chunk")
        if "hermes" in names and len(names) > 1 and self.resolved_video_chunk_mode != "hermes_chunk":
            raise ValueError("A comparison containing HERMES requires hermes_chunk for every Qwen method")
        if "hermes" in names:
            model_kind = json.loads((_path(self.model_path) / "config.json").read_text()).get("model_type")
            if model_kind not in ("qwen2_5_vl", "qwen3_vl"):
                raise ValueError("HERMES requires Qwen2.5-VL or Qwen3-VL")
            if model_kind == "qwen2_5_vl" and self.dtype != "float16":
                raise ValueError("Original HERMES loads float16; select --dtype float16")
        _positive_integer(self.budget, "budget")
        _positive_integer(self.max_videos, "max_videos", 3)
        if self.max_questions is not None:
            _positive_integer(self.max_questions, "max_questions", 8)
        _positive_integer(self.encode_chunk_size, "encode_chunk_size")
        if self.fixed_frame_size is not None:
            from .fixed_inputs import fixed_input_metadata
            fixed_input_metadata(self.fixed_frame_size)
            if self.video_max_pixels is not None:
                raise ValueError("fixed_frame_size and video_max_pixels cannot be combined")
            model_kind = _read_json(_path(self.model_path) / "config.json").get("model_type")
            if model_kind != "qwen2_5_vl":
                raise ValueError("fixed_frame_size currently requires the shared Qwen2.5-VL input contract")
        if self.video_max_pixels is not None:
            _positive_integer(self.video_max_pixels, "video_max_pixels")
            if self.video_max_pixels < 28 * 28 * 4:
                raise ValueError("video_max_pixels must be at least 3136")
            if len(names) != 1 or names[0] not in ("protokv", "sparsemm", "dp", "dp_dual", "hermes"):
                raise ValueError("video_max_pixels requires selecting only protokv, sparsemm, dp, dp_dual or hermes")
            model_type = json.loads((_path(self.model_path) / "config.json").read_text()).get("model_type")
            if not (model_type == "qwen2_5_vl" and names[0] != "hermes"
                    or model_type == "qwen3_vl" and names[0] == "hermes"):
                raise ValueError("video_max_pixels requires Qwen2.5-VL baseline or Qwen3-VL HERMES")
        if isinstance(self.sample_fps, bool) or not isinstance(self.sample_fps, (int, float)) or not math.isfinite(self.sample_fps) or self.sample_fps <= 0:
            raise ValueError("sample_fps must be finite and positive")
        if self.dtype not in ("bfloat16", "float16"):
            raise ValueError("dtype must be bfloat16 or float16")
        if not isinstance(self.min_compression_events_per_video, int) or isinstance(self.min_compression_events_per_video, bool) or self.min_compression_events_per_video < 0:
            raise ValueError("min_compression_events_per_video must be a nonnegative integer")
        if not self.python_executable:
            raise ValueError("python_executable must not be empty")
        if not isinstance(self.dry_run, bool) or not isinstance(self.allow_calibration_transfer, bool):
            raise ValueError("dry_run and allow_calibration_transfer must be explicit booleans")
        if self.config_json is not None:
            extra = _read_json(_path(self.config_json))
            if not isinstance(extra, dict):
                raise ValueError("config_json must contain an object")
            for name in names:
                method = METHODS[name]
                fixed = {"method": method.runtime_method, "budget": method.fixed_budget or self.budget,
                         "budget_scope": "visual", "video_chunk_mode": self.resolved_video_chunk_mode}
                if name == "lukv":
                    fixed["lukv_profile_path"] = str(_path(self.lukv_profile_path))
                if name in ("sparsemm", "dp", "dp_dual"):
                    fixed["sparsemm_score_path"] = str(_path(self.sparsemm_score_path))
                    selected_budget = (self.dp_budget_path if name == "dp" else
                                       self.dp_dual_budget_path if name == "dp_dual" else self.sparsemm_budget_path)
                    if selected_budget is not None:
                        fixed["sparsemm_budget_path"] = str(_path(selected_budget))
                    if name == "dp_dual":
                        fixed["sparsemm_cache_policy"] = "two_bank"
                        fixed["sparsemm_two_bank_gate"] = "legacy_global"
                for key, value in fixed.items():
                    if key in extra and extra[key] != value:
                        raise ValueError(f"config_json {key} conflicts with method {name}")
        return names


def _check_annotation_bounds(config):
    """Reject unsafe requests before creating any output or subprocess.

    The native controller independently validates file paths, MCQA content and
    these same limits; this check makes the unified API fail early.
    """
    data = _read_json(_path(config.anno_path))
    if not isinstance(data, list) or not data:
        raise ValueError("Annotations must be a nonempty list of video records")
    selected = data[:config.max_videos]
    count = 0
    for video in selected:
        if not isinstance(video, dict) or not isinstance(video.get("conversations"), list):
            raise ValueError("Each video requires a conversations list")
        questions = video["conversations"]
        if config.max_questions is not None:
            questions = questions[:config.max_questions]
        if not questions:
            raise ValueError("Each selected video requires at least one question")
        count += len(questions)
        for question in questions:
            endpoint = float(question["end_time"])
            if not math.isfinite(endpoint) or not 0 < endpoint <= 120:
                raise ValueError("Small validation endpoints must be in (0, 120] seconds; full experiments remain paused")
    if not 1 <= count <= 8:
        raise ValueError("Small validation is limited to at most 8 total questions; full experiments remain paused")
    return {"videos": len(selected), "questions": count}


def _check_lu_transfer_opt_in(config):
    if "lukv" not in config.selected_methods or config.allow_calibration_transfer or config.dry_run:
        return
    profile = _read_json(_path(config.lukv_profile_path))
    excluded_path = profile.get("calibration", {}).get("exclude_annotation")
    if not excluded_path:
        raise ValueError("LU profile lacks its original evaluation exclusion source")
    excluded = _read_json(excluded_path)
    evaluation = _read_json(_path(config.anno_path))
    covered_ids = {str(video["video_id"]) for video in excluded}
    covered_names = {Path(video["video_path"]).name for video in excluded}
    if (not {str(video["video_id"]) for video in evaluation} <= covered_ids
            or not {Path(video["video_path"]).name for video in evaluation} <= covered_names):
        raise ValueError("LU profile transfer to this dataset requires explicit --allow-calibration-transfer; no inference has started. The audit will check calibration/evaluation video hashes without recalibration.")


def _interface_hashes():
    files = sorted((paths.PROJECT_ROOT / "kvbench").glob("*.py"))
    if (paths.PROJECT_ROOT / "run.py").is_file():
        files.append(paths.PROJECT_ROOT / "run.py")
    return {str(path.relative_to(paths.PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def _serializable(value):
    if isinstance(value, Path):
        return str(_path(value))
    if isinstance(value, dict):
        return {key: _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    return value


def build_command(config, method, output_dir):
    """Build the native bounded controller argv without importing GPU code."""
    config.validate()
    if method not in config.selected_methods:
        raise ValueError("Method is not selected in RunConfig")
    entry = METHODS[method]
    if method == "hermes":
        command = [config.python_executable, "-u", "-m", "kvbench.hermes_runner",
                   "--anno-path", str(_path(config.anno_path)),
                   "--model-path", str(_path(config.model_path)),
                   "--output-dir", str(_path(output_dir)),
                   "--budget", str(config.budget), "--max-videos", str(config.max_videos),
                   "--sample-fps", str(config.sample_fps),
                   "--encode-chunk-size", str(config.encode_chunk_size), "--dtype", config.dtype]
        if config.max_questions is not None:
            command += ["--max-questions", str(config.max_questions)]
        if config.video_max_pixels is not None:
            command += ["--video-max-pixels", str(config.video_max_pixels)]
        if config.fixed_frame_size is not None:
            command += ["--fixed-frame-size", str(config.fixed_frame_size)]
        if config.dry_run:
            command.append("--dry-run")
        return command
    command = [config.python_executable, "-u", "-m", "sparsemm_protokv.run_streamingbench",
               "--method", entry.runtime_method, "--anno-path", str(_path(config.anno_path)),
               "--model-path", str(_path(config.model_path)), "--output-dir", str(_path(output_dir)),
               "--budget", str(entry.fixed_budget or config.budget), "--budget-scope", "visual",
               "--max-videos", str(config.max_videos), "--sample-fps", str(config.sample_fps),
               "--encode-chunk-size", str(config.encode_chunk_size),
               "--video-chunk-mode", config.resolved_video_chunk_mode, "--dtype", config.dtype,
               "--small-validation"]
    if config.max_questions is not None:
        command += ["--max-questions", str(config.max_questions)]
    if config.video_max_pixels is not None:
        command += ["--video-max-pixels", str(config.video_max_pixels)]
    if config.fixed_frame_size is not None:
        command += ["--fixed-frame-size", str(config.fixed_frame_size)]
    if method == "lukv":
        command += ["--lukv-profile-path", str(_path(config.lukv_profile_path))]
    if method in ("sparsemm", "dp", "dp_dual"):
        command += ["--sparsemm-score-path", str(_path(config.sparsemm_score_path))]
        selected_budget = (config.dp_budget_path if method == "dp" else
                           config.dp_dual_budget_path if method == "dp_dual" else config.sparsemm_budget_path)
        if selected_budget is not None:
            command += ["--sparsemm-budget-path", str(_path(selected_budget))]
        if method == "dp_dual":
            command += ["--sparsemm-cache-policy", "two_bank",
                        "--sparsemm-two-bank-gate", "legacy_global"]
    if config.config_json is not None:
        command += ["--config-json", str(_path(config.config_json))]
    if config.dry_run:
        command.append("--dry-run")
    return command


def _environment():
    # Do not inherit an old source checkout through PYTHONPATH. The controller
    # adds its pinned framework and lightweight runtime dependencies itself.
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(paths.PROJECT_ROOT)
    return environment


def _execute(command, log_path):
    with Path(log_path).open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=paths.PROJECT_ROOT, env=_environment(),
                       stdout=log, stderr=subprocess.STDOUT, check=True, text=True)


def build_audit_command(output_dir, *, method=None, allow_calibration_transfer=False,
                        min_compression_events_per_video=0, python_executable=None):
    out = _path(output_dir)
    protocol = _read_json(out / "protocol.json")
    runtime_method = protocol.get("method")
    if method is None:
        method = runtime_method
    if method not in METHODS or METHODS[method].runtime_method != runtime_method:
        raise ValueError("Selected audit method disagrees with the run protocol")
    if not isinstance(min_compression_events_per_video, int) or isinstance(min_compression_events_per_video, bool) or min_compression_events_per_video < 0:
        raise ValueError("Minimum compression events must be a nonnegative integer")
    command = [python_executable or sys.executable, "-u", "-m", METHODS[method].audit_module,
               str(out), "--min-compression-events-per-video", str(min_compression_events_per_video)]
    if method == "lukv" and allow_calibration_transfer:
        command.append("--allow-calibration-transfer")
    return command


def _audit_one(out, *, method=None, allow_calibration_transfer=False,
               min_compression_events_per_video=0, python_executable=None):
    command = build_audit_command(out, method=method, allow_calibration_transfer=allow_calibration_transfer,
                                  min_compression_events_per_video=min_compression_events_per_video,
                                  python_executable=python_executable)
    _execute(command, out / "kvbench_audit.log")
    report = _read_json(out / "audit_report.json")
    if report.get("passed") is not True:
        raise RuntimeError(f"Audit failed for {out}: {report.get('error', 'see audit_report.json')}")
    if method == "full" and report.get("trace", {}).get("compression_events") != 0:
        raise RuntimeError("The full control evicted KV entries; increase the control capacity before calling it full KV")
    return report


def _compare_visual_inputs(collection, names, budget, *, fixed_frame_size=None):
    """Check that HERMES-chunk arms saw identical resized Qwen video grids."""
    fields = ("raw_frames", "source_hw", "resized_hw", "grid_thw", "visual_tokens")
    if fixed_frame_size is not None:
        fields += ("prepared_rgb_sha256",)
    reference = None
    methods = {}
    for name in names:
        entry = METHODS[name]
        suffix = "native" if name == "hermes" else "visual"
        directory = collection / f"{name}_k{entry.fixed_budget or budget}_{suffix}"
        trace = directory / ("hermes_trace.jsonl" if name == "hermes" else "cache_trace.jsonl")
        expected_event = "video_input" if name == "hermes" else "video_prefill"
        records = []
        for line in trace.read_text().splitlines():
            event = json.loads(line)
            if event.get("event") != expected_event:
                continue
            visual = event if name == "hermes" else event.get("visual_input")
            if not isinstance(visual, dict) or any(field not in visual for field in fields):
                raise RuntimeError(f"{name} did not record complete Qwen visual input metadata")
            records.append((str(event["video_id"]), *(visual[field] for field in fields)))
        if not records:
            raise RuntimeError(f"{name} did not encode any comparable Qwen video chunks")
        methods[name] = len(records)
        if reference is None:
            reference = records
        elif records != reference:
            _write_json(collection / "visual_input_comparison.json",
                        {"passed": False, "reference": names[0], "mismatch": name,
                         "chunks_per_method": methods, "compared_fields": fields})
            raise RuntimeError(f"{name} video grids/tokens differ from {names[0]}; no strict input comparison")
    report = {"passed": True, "methods": list(names), "chunks_per_method": methods,
              "compared_fields": fields,
              "limit": ("Prepared RGB, grid and token counts match; feature values and decoder positions are not compared"
                        if fixed_frame_size is not None else
                        "Grid and token counts match; feature tensor values and decoder positions are not compared")}
    if fixed_frame_size is not None:
        report["fixed_frame_size"] = fixed_frame_size
    _write_json(collection / "visual_input_comparison.json", report)
    return report


def run(config=None, **kwargs):
    """Run selected methods serially, then audit each completed inference.

    All runs use a fresh output collection. There is no resume/full-benchmark
    option; ProtoKV requires an explicit method selection.
    """
    if config is not None and kwargs:
        raise ValueError("Pass RunConfig or keyword arguments, not both")
    config = RunConfig(**kwargs) if config is None else config
    if not isinstance(config, RunConfig):
        raise TypeError("config must be RunConfig")
    names = config.validate()
    out = _path(config.output_dir) if config.output_dir is not None else paths.PROJECT_ROOT / "results" / datetime.now(timezone.utc).strftime("validation_%Y%m%dT%H%M%S_%fZ")
    if os.path.lexists(out):
        raise FileExistsError(f"Output directory already exists; choose a new output directory: {out}")
    selection = _check_annotation_bounds(config)
    _check_lu_transfer_opt_in(config)
    # HERMES's implementation, bounded MCQA adapter and audit now live in this
    # package. Its model/annotation paths are explicit; no external source link
    # is needed for a HERMES-only run.
    if any(name != "hermes" for name in names):
        paths.ensure_resources()
    if "dp" in names and not _path(config.dp_budget_path).is_file():
        raise FileNotFoundError(f"DP budget CSV is missing: {_path(config.dp_budget_path)}")
    if "dp_dual" in names and not _path(config.dp_dual_budget_path).is_file():
        raise FileNotFoundError(f"DP dual-bank budget CSV is missing: {_path(config.dp_dual_budget_path)}")
    if any(name in names for name in ("sparsemm", "dp", "dp_dual")):
        from sparsemm_protokv.model_loader import model_type
        if model_type(_path(config.model_path)) == "llava_onevision":
            from sparsemm_native.assets import asset_metadata
            if _path(config.sparsemm_score_path) == paths.default_sparsemm_score_path():
                raise ValueError("LLaVA-OneVision SparseMM needs model-bound scores; the bundled Qwen2.5-VL scores cannot be transferred. No methods were started.")
            metadata = asset_metadata(config.sparsemm_score_path)
            if (metadata.get("asset_type") != "synthetic_test_scores"
                    or _path(metadata["model_path"]) != _path(config.model_path)):
                raise ValueError("LLaVA SparseMM currently accepts only model-bound synthetic smoke scores; no methods were started")
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"type": "kvbench_collection", "methods": list(names), "selection": selection,
                "created_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
                "dry_run": config.dry_run, "full_benchmarks_paused": True,
                "allow_calibration_transfer": config.allow_calibration_transfer,
                "project_root": str(paths.PROJECT_ROOT), "resource_root": str(paths.resource_root()),
                "interface_sha256": _interface_hashes(), "entry_argv": list(sys.argv),
                "effective_config": _serializable(asdict(config)),
                "annotation_sha256": hashlib.sha256(_path(config.anno_path).read_bytes()).hexdigest()}
    _write_json(out / "suite_manifest.json", manifest)
    try:
        for name in names:
            if _interface_hashes() != manifest["interface_sha256"]:
                raise RuntimeError("Unified interface source changed during this evaluation; use a fresh run")
            entry = METHODS[name]
            suffix = "native" if name == "hermes" else "visual"
            target = out / f"{name}_k{entry.fixed_budget or config.budget}_{suffix}"
            target.mkdir()
            command = build_command(config, name, target)
            _write_json(target / "kvbench_method.json", {"requested_method": name, "runtime_method": entry.runtime_method,
                        "native_controller_command": command, "audit_calibration_transfer_opt_in": config.allow_calibration_transfer if name == "lukv" else False})
            _execute(command, target / "kvbench_controller.log")
            status = (target / "run.status").read_text().strip()
            if status != ("dry-run" if config.dry_run else "complete"):
                raise RuntimeError(f"Native controller did not complete {name}: {status}")
            if not config.dry_run:
                _audit_one(target, method=name, allow_calibration_transfer=config.allow_calibration_transfer,
                           min_compression_events_per_video=config.min_compression_events_per_video,
                           python_executable=config.python_executable)
        if (not config.dry_run and len(names) > 1
                and (config.resolved_video_chunk_mode == "hermes_chunk" or config.fixed_frame_size is not None)):
            _compare_visual_inputs(out, names, config.budget, fixed_frame_size=config.fixed_frame_size)
        if _interface_hashes() != manifest["interface_sha256"]:
            raise RuntimeError("Unified interface source changed during this evaluation; use a fresh run")
        manifest["status"] = "dry-run" if config.dry_run else "complete"
        _write_json(out / "suite_manifest.json", manifest)
        return summarize(out)
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        _write_json(out / "suite_manifest.json", manifest)
        summarize(out)
        raise


def _run_directories(output_dir):
    out = _path(output_dir)
    if not out.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {out}")
    if (out / "protocol.json").is_file():
        return [out], True
    candidates = sorted(path for path in out.iterdir() if path.is_dir() and
                        any((path / name).is_file() for name in ("protocol.json", "kvbench_method.json")))
    if not candidates and not (out / "suite_manifest.json").is_file():
        raise ValueError(f"No native runs found in {out}")
    return candidates, False


def _method_label(out, protocol):
    metadata = _read_json(out / "kvbench_method.json") if (out / "kvbench_method.json").is_file() else {}
    return metadata.get("requested_method", protocol.get("method", out.name))


def audit(output_dir, *, allow_calibration_transfer=False, min_compression_events_per_video=0,
          python_executable=None):
    """Audit one native run or every run in a collection, without inference."""
    directories, _ = _run_directories(output_dir)
    if not summarize(output_dir, write=False)["selection_complete"]:
        raise ValueError("The collection is missing expected method directories; restore the results before auditing")
    for out in directories:
        protocol = _read_json(out / "protocol.json")
        method = _method_label(out, protocol)
        _audit_one(out, method=method, allow_calibration_transfer=allow_calibration_transfer,
                   min_compression_events_per_video=min_compression_events_per_video,
                   python_executable=python_executable)
    return summarize(output_dir)


def summarize(output_dir, *, write=True):
    """Read actual native summaries/audits; missing audits never imply success."""
    out = _path(output_dir)
    directories, direct = _run_directories(out)
    records = []
    for directory in directories:
        protocol = _read_json(directory / "protocol.json") if (directory / "protocol.json").is_file() else {}
        native = _read_json(directory / "summary.json") if (directory / "summary.json").is_file() else {}
        report = _read_json(directory / "audit_report.json") if (directory / "audit_report.json").is_file() else {}
        status = (directory / "run.status").read_text().strip() if (directory / "run.status").is_file() else "not-started"
        record = {"method": _method_label(directory, protocol), "runtime_method": protocol.get("method"),
                  "output_dir": str(directory), "status": status, "audit_passed": report.get("passed"),
                  "videos": native.get("videos"), "questions": native.get("questions"),
                  "correct": native.get("correct"), "accuracy_percent": native.get("accuracy_percent"),
                  "budget": native.get("budget", protocol.get("budget")),
                  "budget_scope": native.get("budget_scope", protocol.get("budget_scope")),
                  "peak_gpu_allocated_gib": native.get("peak_gpu_allocated_gib"),
                  "compression_events": report.get("trace", {}).get("compression_events"),
                  "qa_restore_checks": report.get("trace", {}).get("qa_restore_checks")}
        if protocol.get("fixed_frame_size") is not None:
            record["fixed_frame_size"] = protocol["fixed_frame_size"]
            record["fixed_visual_input"] = protocol.get("fixed_visual_input")
        if protocol.get("sparsemm_asset_metadata", {}).get("asset_type") == "synthetic_test_scores":
            record["calibration_status"] = "synthetic_smoke_only"
        if report.get("error"):
            record["audit_error"] = report["error"]
        if record["method"] == "full" and status == "complete" and record["compression_events"] != 0:
            record["audit_passed"] = False
            record["audit_error"] = "Full KV control requires an audited compression_events count of zero; the recorded count is missing or nonzero"
        lmms_path = native.get("lmms_results_path")
        if lmms_path and Path(lmms_path).is_file():
            value = _read_json(lmms_path).get("total_evaluation_time_seconds")
            if value is not None:
                record["native_evaluation_seconds"] = float(value)
        records.append(record)
    manifest = _read_json(out / "suite_manifest.json") if (out / "suite_manifest.json").is_file() else {}
    expected = manifest.get("methods")
    actual = [record["method"] for record in records]
    selection_complete = expected is None or (len(actual) == len(expected) and set(actual) == set(expected))
    all_complete = selection_complete and all(record["status"] == "complete" for record in records)
    all_audits = selection_complete and all(record["audit_passed"] is True for record in records)
    status = ("failed" if manifest.get("status") == "failed" else "complete" if all_complete and all_audits
              else "dry-run" if selection_complete and all(record["status"] == "dry-run" for record in records)
              else "incomplete")
    result = {"type": "kvbench_summary", "output_dir": str(out), "status": status,
              "full_benchmarks_paused": True, "all_audits_passed": all_audits,
              "expected_methods": expected, "selection_complete": selection_complete,
              "missing_methods": sorted(set(expected or ()) - set(actual)),
              "runs": records,
              "limitations": ["Bounded integration validation; not official full benchmark reproduction.",
                              "Small sample accuracy does not establish a method ranking.",
                              "The full method is a large-budget control and must show zero compression events."]}
    if any(record.get("calibration_status") == "synthetic_smoke_only" for record in records):
        result["limitations"].append("SparseMM used synthetic constant scores for interface testing; no OCR calibration or paper-method result is claimed.")
    if any(record["method"] == "dp" for record in records):
        result["limitations"].append("DP here injects capacity_n into SparseMM attention selection; the DP recent/history bank split is not used online.")
    if any(record["method"] == "dp_dual" for record in records):
        result["limitations"].append("DP dual-bank uses recent_r/history_q online; its bounded auditor checks recorded quotas and cache lengths but cannot replay QK rankings.")
    if any(record["method"] == "hermes" for record in records):
        result["limitations"].append("HERMES uses a per-layer shared visual-position target; DP uses per-KV-head visual capacities. Match actual KV occupancy before treating budgets as equal.")
    if len({(record["budget"], record["budget_scope"]) for record in records}) > 1:
        result["limitations"].append("This collection mixes budget values or scopes; compare measured visual KV occupancy before interpreting accuracy differences.")
    if manifest.get("error"):
        result["error"] = manifest["error"]
    if write:
        _write_json(out / ("kvbench_summary.json" if direct else "summary.json"), result)
        rows = ["# Unified KV evaluation", "", f"Status: **{status}**. Full benchmarks remain paused.", "",
                "| Method | Run status | Audit | Correct / questions | Compression events | Peak GPU GiB |",
                "|---|---|---|---:|---:|---:|"]
        for record in records:
            accuracy = f"{record['correct']} / {record['questions']}" if record["questions"] is not None else "—"
            peak = f"{record['peak_gpu_allocated_gib']:.3f}" if record["peak_gpu_allocated_gib"] is not None else "—"
            verdict = "passed" if record["audit_passed"] is True else "failed" if record["audit_passed"] is False else "not run"
            rows.append(f"| {record['method']} | {record['status']} | {verdict} | {accuracy} | {record['compression_events'] if record['compression_events'] is not None else '—'} | {peak} |")
        rows += ["", *result["limitations"], "", "Native results, logs, source snapshots and audit reports are kept in each method directory.", ""]
        (out / ("kvbench_report.md" if direct else "report.md")).write_text("\n".join(rows), encoding="utf-8")
    return result
