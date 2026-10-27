"""Native lmms-eval model plugin for a complete StreamingBench video.

One request contains every question for one video. The streaming bridge keeps
its visual memory between those questions and starts fresh for the next video.
The response is a JSON array of answer strings, scored by streamingbench_local.
"""

import json
import math
import os
import copy
import hashlib
import time
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path

import torch

from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

from sparsemm_baselines.run_streamingbench import analyze_video, write_json, write_results, prepare_annotations, sha256


METHODS = ("protokv", "adakv", "pyramidkv", "lukv", "sparsemm", "uniform")
PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "hermes/models/Qwen2.5-VL-7B-Instruct"
TIMING_ANNOTATIONS = PACKAGE.parent / "hermes/data/streamingbench/streamingbench_realtime_half_userpaths.json"
TIMING_SOURCE_SHA256 = "c7b202726355c95fe26326a9983f70ff784e99ce92a33d970b272816670a9c7b"


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise ValueError(f"Expected a boolean, received {value!r}")


def _validate_generation(kwargs):
    """Do not silently change the established MCQA generation protocol."""
    if float(kwargs.get("temperature", 0)) != 0:
        raise ValueError("StreamingBench comparison requires temperature=0")
    if int(kwargs.get("num_beams", 1)) != 1:
        raise ValueError("Streaming memory currently requires num_beams=1")
    if _as_bool(kwargs.get("do_sample", False)):
        raise ValueError("StreamingBench comparison uses greedy decoding")
    if int(kwargs.get("max_new_tokens", 16)) != 16:
        raise ValueError("StreamingBench MCQA uses max_new_tokens=16 per question")


def _close_decord_reader(reader):
    """Join Decord's native decoder immediately and prevent a second free."""
    try:
        reader.__del__()
    finally:
        # Decord 0.6.0's destructor frees the handle without clearing it.
        reader._handle = None


def _install_fixed_frame_input(bridge, canvas_size):
    """Change RGB pixels only, and check every existing native video forward.

    Keep temporal-patch/whole-chunk encoding, positions and cache updates in the
    original bridge. Its per-forward metadata lets us reject grid/token drift
    before the language model consumes the features.
    """
    from kvbench.fixed_inputs import (
        assert_qwen_grid, fixed_input_metadata, frame_rgb_sha256,
        prepare_fixed_frames,
    )

    metadata = fixed_input_metadata(canvas_size)
    if not metadata["enabled"]:
        return bridge
    if bridge.model_config.model_type != "qwen2_5_vl" or bridge.is_llava:
        raise ValueError("Fixed-frame inputs currently require Qwen2.5-VL")
    if getattr(bridge, "video_max_pixels", None) is not None:
        raise ValueError("Fixed-frame inputs cannot be combined with video_max_pixels")
    vision = bridge.model_config.vision_config
    if (int(vision.patch_size), int(vision.spatial_merge_size),
            int(vision.temporal_patch_size)) != (14, 2, 2):
        raise ValueError("Fixed-frame contract requires Qwen patch=14, spatial merge=2, temporal patch=2")
    # The supplied canvas is already exact. Disable the processor's optional
    # spatial resize; its normalization and temporal padding remain unchanged.
    bridge.processor.video_processor.do_resize = False
    bridge.fixed_input_metadata = metadata
    bridge.fixed_input_checks = []
    original_encode = bridge.encode_video_chunk

    def encode_fixed(video_chunk, is_last_chunk=False):
        if video_chunk is None or video_chunk.shape[0] == 0:
            return original_encode(video_chunk, is_last_chunk=is_last_chunk)
        original_hw = ([int(video_chunk.shape[1]), int(video_chunk.shape[2])]
                       if video_chunk.shape[-1] == 3 else
                       [int(video_chunk.shape[-2]), int(video_chunk.shape[-1])])
        prepared = prepare_fixed_frames(video_chunk, canvas_size=canvas_size)
        original_forward = bridge.forward_chunk
        consumed = 0

        def checked_forward(inputs_embeds, positions, *args, **kwargs):
            nonlocal consumed
            visual = bridge._pending_visual_input
            if not isinstance(visual, dict):
                raise AssertionError("Missing per-forward video input metadata")
            raw_frames = int(visual["raw_frames"])
            if raw_frames < 1 or consumed + raw_frames > len(prepared):
                raise AssertionError("Visual forwards do not match the prepared frame batch")
            audit = assert_qwen_grid(
                visual["grid_thw"], int(inputs_embeds.shape[1]), raw_frames,
                canvas_size=canvas_size,
            )
            if (visual["visual_tokens"] != int(inputs_embeds.shape[1])
                    or visual["resized_hw"] != metadata["resized_hw"]
                    or visual["source_hw"] != metadata["resized_hw"]):
                raise AssertionError("Processor changed the fixed canvas or visual token count")
            visual.update(
                original_source_hw=original_hw,
                prepared_rgb_sha256=frame_rgb_sha256(prepared[consumed:consumed + raw_frames]),
                fixed_frame_size=canvas_size,
                tokens_per_temporal_unit=metadata["tokens_per_temporal_unit"],
            )
            consumed += raw_frames
            bridge.fixed_input_checks.append(dict(
                audit, video_id=getattr(bridge, "_active_video_id", None),
                **{key: visual[key] for key in ("prepared_rgb_sha256", "original_source_hw")},
            ))
            return original_forward(inputs_embeds, positions, *args, **kwargs)

        # Restore the original bound method even when preprocessing/inference
        # fails; question prefill and later calls must retain their own path.
        bridge.forward_chunk = checked_forward
        try:
            result = original_encode(prepared, is_last_chunk=is_last_chunk)
            if consumed != len(prepared):
                raise AssertionError("A prepared input frame was omitted by native encoding")
            return result
        finally:
            bridge.forward_chunk = original_forward

    bridge.encode_video_chunk = encode_fixed
    return bridge


class _CopiedVideoBatch:
    def __init__(self, array):
        self.array = array

    def asnumpy(self):
        return self.array


class BatchScopedVideoReader:
    """Retain only NumPy pixels while the GPU processes a sampled batch.

    Decord's background decoder can continue allocating during slow GPU work.
    Reopen the same video for each batch and release its native reader before
    returning; sampling indices, native resolution and decoded pixels stay the
    same. This changes decoder lifetime, not temporal-patch inference order.
    """
    def __init__(self, video_path, num_threads=1):
        from decord import VideoReader

        self.video_path = video_path
        self.num_threads = num_threads
        reader = VideoReader(video_path, num_threads=num_threads)
        try:
            self.frame_count = len(reader)
            self.avg_fps = reader.get_avg_fps()
        finally:
            _close_decord_reader(reader)

    def __len__(self):
        return self.frame_count

    def get_avg_fps(self):
        return self.avg_fps

    def get_batch(self, indices):
        from decord import VideoReader

        reader = VideoReader(self.video_path, num_threads=self.num_threads)
        try:
            # NDArray.asnumpy copies into independent NumPy-owned storage.
            array = reader.get_batch(indices).asnumpy()
        finally:
            _close_decord_reader(reader)
        return _CopiedVideoBatch(array)


@register_model("qwen2_5_vl_stream")
class Qwen25VLStream(lmms):
    """Qwen2.5-VL with native attention/cache patches selected by METHOD."""

    def __init__(
        self,
        pretrained=str(DEFAULT_MODEL),
        device="cuda",
        device_map=None,
        batch_size=1,
        max_batch_size=None,
        use_cache=True,
        method=None,
        budget=None,
        budget_scope="visual",
        sample_fps=0.5,
        encode_chunk_size=16,
        video_max_pixels=None,
        fixed_frame_size=None,
        dtype="bfloat16",
        trace_path=None,
        output_dir=None,
        small_validation=False,
        five_video_timing=False,
        timing_source_sha256=None,
        timing_subset_sha256=None,
        **config_kwargs,
    ):
        super().__init__()
        from sparsemm_protokv.config import StreamingConfig

        selected = method if method is not None else os.environ.get("METHOD", "protokv")
        selected = str(selected).strip().lower()
        if selected not in METHODS:
            raise ValueError(f"Unknown METHOD={selected!r}; expected one of {METHODS}")
        if int(batch_size) != 1 or max_batch_size is not None and int(max_batch_size) != 1:
            raise ValueError("A streaming request is one complete video; batch_size must be 1")
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError("This streaming plugin currently supports a single evaluation process")
        if not _as_bool(use_cache):
            raise ValueError("Streaming evaluation requires use_cache=True")
        if device_map not in (None, "", device):
            raise ValueError("Use one explicit device; automatic model sharding is unsupported")
        self.sample_fps = float(sample_fps)
        self.encode_chunk_size = int(encode_chunk_size)
        self.video_max_pixels = None if video_max_pixels is None else int(video_max_pixels)
        self.fixed_frame_size = None if fixed_frame_size is None else int(fixed_frame_size)
        if self.fixed_frame_size is not None:
            from kvbench.fixed_inputs import fixed_input_metadata
            fixed_input_metadata(self.fixed_frame_size)
            if self.video_max_pixels is not None:
                raise ValueError("fixed_frame_size and video_max_pixels cannot be combined")
        if self.video_max_pixels is not None:
            if selected not in ("protokv", "sparsemm") or self.video_max_pixels < 28 * 28 * 4:
                raise ValueError("video_max_pixels requires ProtoKV/SparseMM and at least 3136 pixels")
        if not math.isfinite(self.sample_fps) or self.sample_fps <= 0 or self.encode_chunk_size < 1:
            raise ValueError("Sampling FPS and encode chunk size must be positive")
        if dtype not in ("bfloat16", "float16"):
            raise ValueError("dtype must be bfloat16 or float16")
        allowed = {field.name for field in fields(StreamingConfig)} - {"method", "budget", "budget_scope"}
        unknown = set(config_kwargs) - allowed
        if unknown:
            raise TypeError(f"Unknown streaming model arguments: {sorted(unknown)}")
        value = budget if budget is not None else os.environ.get("BUDGET", "1000")
        self.streaming_config = StreamingConfig(
            method=selected, budget=int(value), budget_scope=budget_scope, **config_kwargs
        )
        self.model_path = str(Path(pretrained).expanduser().resolve())
        self.lukv_asset_metadata = None
        self.sparsemm_asset_metadata = None
        if selected == "sparsemm":
            from sparsemm_native.assets import asset_metadata
            self.sparsemm_asset_metadata = asset_metadata(self.streaming_config.sparsemm_score_path or None)
            if Path(self.sparsemm_asset_metadata["model_path"]).resolve() != Path(self.model_path):
                raise ValueError("SparseMM visual-head scores were calibrated for a different model")
        if selected == "lukv":
            from sparsemm_lukv.policy import configured_asset_metadata
            model_config = json.loads((Path(self.model_path) / "config.json").read_text())
            text = model_config.get("text_config", model_config)
            self.lukv_asset_metadata = configured_asset_metadata(
                self.streaming_config, num_layers=int(text["num_hidden_layers"]),
                num_kv_heads=int(text["num_key_value_heads"]), model_path=self.model_path,
            )
        self.dtype_name = dtype
        self.small_validation = _as_bool(small_validation)
        self.five_video_timing = _as_bool(five_video_timing)
        self.timing_requested = selected in ("protokv", "lukv", "sparsemm", "uniform")
        if self.small_validation and self.five_video_timing:
            raise ValueError("Small validation and five-video timing are mutually exclusive")
        if self.five_video_timing and selected != "protokv":
            raise ValueError("Five-video timing is authorized only for ProtoKV")
        self.timing_source_sha256 = timing_source_sha256
        self.timing_subset_sha256 = timing_subset_sha256
        self._five_selection_verified = False
        self._evaluation_started = None
        self._model_load_seconds = None
        self._video_timings = []
        self.output_dir = None if output_dir is None else Path(output_dir).expanduser().resolve()
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if any((self.output_dir / name).exists() for name in ("1_0.csv", "results.csv")):
                raise FileExistsError("Native streaming CLI does not resume CSVs; choose a new output_dir")
        if trace_path is None and self.output_dir is not None:
            trace_path = self.output_dir / "cache_trace.jsonl"
        self.trace_path = None if trace_path is None else str(Path(trace_path).expanduser().resolve())
        self._device = torch.device(device)
        self.batch_size_per_gpu = 1
        self.use_cache = True
        self._bridge = None
        self.last_video_rows = []
        self.completed_video_rows = []

    @property
    def device(self):
        return self._device

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def bridge(self):
        if self._bridge is None:
            paused = (PACKAGE / "results/PAUSED", PACKAGE.parent / "protokv/results/PAUSED",
                      PACKAGE.parent / "sparsemm_baselines/results/PAUSED")
            if self.five_video_timing and not self._five_selection_verified:
                raise RuntimeError("Five-video timing selection must be verified before loading the model")
            if not (self.small_validation or self.five_video_timing) and any(path.exists() for path in paused):
                raise RuntimeError("Experiments remain paused; native live inference requires small_validation=true")
            if self.streaming_config.method == "lukv":
                from sparsemm_lukv.streaming import load_model
            elif self.streaming_config.method == "sparsemm":
                from sparsemm_native.streaming import load_model
            else:
                from sparsemm_protokv.streaming import load_model

            if self.timing_requested and self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            self._bridge = load_model(
                self.model_path, self.streaming_config, sample_fps=self.sample_fps,
                trace_path=self.trace_path, device=str(self.device), dtype=self.dtype_name,
                video_max_pixels=self.video_max_pixels,
            )
            if self.fixed_frame_size is not None:
                _install_fixed_frame_input(self._bridge, self.fixed_frame_size)
            if self.timing_requested:
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                self._model_load_seconds = time.perf_counter() - started
                self._bridge.timing_enabled = True
                self._persist_timing("running")
        return self._bridge

    def _check_request_bounds(self, requests):
        if self.five_video_timing:
            return self._check_five_video_timing(requests)
        if not self.small_validation:
            raise RuntimeError("Native live inference requires small_validation=true; full experiments remain paused")
        questions = len(self.completed_video_rows)
        ids = {str(row["video_id"]) for row in self.completed_video_rows}
        for request in requests:
            _context, kwargs, _visual, doc_id, task, split = request.args
            _validate_generation(kwargs)
            doc = self.task_dict[task][split][doc_id]
            if "video_json" not in doc:
                raise ValueError("qwen2_5_vl_stream requires the streamingbench_local video-level task")
            video = json.loads(doc["video_json"])
            if str(video["video_id"]) in ids:
                raise ValueError("Each video may occur only once in a streaming evaluation")
            ids.add(str(video["video_id"]))
            conversations = video["conversations"]
            if not conversations:
                raise ValueError("Every streaming video requires a question")
            questions += len(conversations)
            for question in conversations:
                end = float(question["end_time"])
                if not math.isfinite(end) or not 0 < end <= 120:
                    raise ValueError("Native small validation endpoints must be in (0,120] seconds")
        if len(ids) > 3 or questions > 8:
            raise ValueError("Native small validation permits at most 3 videos and 8 questions")

    def _check_five_video_timing(self, requests):
        if self.streaming_config.method != "protokv" or len(requests) != 5 or self.completed_video_rows:
            raise ValueError("Five-video timing requires one request batch containing exactly five ProtoKV videos")
        if self.timing_source_sha256 != TIMING_SOURCE_SHA256 or sha256(TIMING_ANNOTATIONS) != TIMING_SOURCE_SHA256:
            raise ValueError("Five-video timing annotation source hash does not match the pinned original")
        expected = prepare_annotations(TIMING_ANNOTATIONS, 5)
        expected_hash = hashlib.sha256(json.dumps(expected, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if self.timing_subset_sha256 != expected_hash:
            raise ValueError("Five-video timing subset hash is not the original first five complete videos")
        selected = []
        for request in requests:
            _context, kwargs, _visual, doc_id, task, split = request.args
            _validate_generation(kwargs)
            selected.append(json.loads(self.task_dict[task][split][doc_id]["video_json"]))
        if selected != expected or sum(len(v["conversations"]) for v in selected) != 25:
            raise ValueError("Five-video timing requests differ from the original five complete videos / 25 questions")
        self._five_selection_verified = True

    def _persist_timing(self, status, error=None):
        if not self.timing_requested or self.output_dir is None:
            return
        report = {"status": status, "profile": "five_video_timing" if self.five_video_timing else self.streaming_config.method + "_small_validation",
                  "method": self.streaming_config.method,
                  "source_sha256": self.timing_source_sha256, "subset_sha256": self.timing_subset_sha256,
                  "model_load_seconds": self._model_load_seconds, "videos": self._video_timings,
                  "evaluation_seconds": None if self._evaluation_started is None else time.perf_counter() - self._evaluation_started,
                  "component_totals": copy.deepcopy(getattr(self._bridge, "timings", {})),
                  "full_benchmarks_paused": True}
        if error is not None:
            report["runner_error"] = f"{type(error).__name__}: {error}"
        write_json(self.output_dir / "timing.json", report)

    def _persist_runtime(self, status, error=None):
        if self.output_dir is None:
            return
        # Preserve every cumulative failure/check, including on exception paths.
        checks = dict(getattr(self._bridge, "validation_checks", {}))
        checks.update(status=status, updated_utc=datetime.now(timezone.utc).isoformat(),
                      config=self.streaming_config.asdict())
        if self.fixed_frame_size is not None:
            from kvbench.fixed_inputs import fixed_input_metadata
            checks.update(fixed_input=fixed_input_metadata(self.fixed_frame_size),
                          fixed_input_checks=getattr(self._bridge, "fixed_input_checks", []))
        if error is not None:
            checks["runner_error"] = f"{type(error).__name__}: {error}"
        write_json(self.output_dir / "runtime_checks.json", checks)
        peak = 0.0
        if self.device.type == "cuda" and self._bridge is not None:
            peak = torch.cuda.max_memory_allocated(self.device) / 2**30
        write_json(self.output_dir / "inference_stats.json", {
            "status": status, "peak_gpu_allocated_gib": peak,
            "completed_videos": len({str(row["video_id"]) for row in self.completed_video_rows}),
            "questions": len(self.completed_video_rows),
        })

    def generate_until(self, requests):
        responses = []
        try:
            self._evaluation_started = time.perf_counter()
            self._persist_runtime("running")
            requests = list(requests)
            self._check_request_bounds(requests)
            for request in requests:
                context, gen_kwargs, _doc_to_visual, doc_id, task, split = request.args
                _validate_generation(gen_kwargs)
                doc = self.task_dict[task][split][doc_id]
                if "video_json" not in doc:
                    raise ValueError("qwen2_5_vl_stream requires the streamingbench_local video-level task")
                video = json.loads(doc["video_json"])
                if str(video["video_id"]) in {str(row["video_id"]) for row in self.completed_video_rows}:
                    raise ValueError("Each video may occur only once in a streaming evaluation")
                bridge = self.bridge
                before = copy.deepcopy(getattr(bridge, "timings", {}))
                if self.timing_requested and self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                video_started = time.perf_counter()
                try:
                    rows = analyze_video(
                        bridge, video, self.sample_fps, self.encode_chunk_size, torch,
                        reader_factory=BatchScopedVideoReader if self.streaming_config.method in ("protokv", "lukv", "sparsemm", "uniform") else None,
                    )
                finally:
                    # analyze_video resets on success; also reset after an exception.
                    bridge.clear_cache()
                if self.timing_requested:
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    after = copy.deepcopy(getattr(bridge, "timings", {}))
                    delta = {key: value[len(before.get(key, [])):] if isinstance(value, list)
                             else value - before.get(key, 0) for key, value in after.items()}
                    self._video_timings.append({"video_id": video["video_id"], "questions": len(rows),
                                                "wall_seconds": time.perf_counter() - video_started,
                                                "components": delta})
                if len(rows) != len(video["conversations"]):
                    raise RuntimeError("Streaming model did not return every question in the video")
                self.last_video_rows = rows
                self.completed_video_rows.extend(rows)
                if self.output_dir is not None:
                    write_results(self.output_dir / "1_0.csv", self.completed_video_rows)
                self._persist_runtime("running")
                self._persist_timing("running")
                answer = json.dumps([row["pred_answer"] for row in rows], ensure_ascii=False)
                responses.append(answer)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), answer)
            self._persist_runtime("complete")
            self._persist_timing("complete")
        except BaseException as error:
            self._persist_runtime("failed", error=error)
            self._persist_timing("failed", error=error)
            raise
        return responses

    def generate_until_multi_round(self, requests):
        raise NotImplementedError("Use streamingbench_local: each request already includes all video questions")

    def loglikelihood(self, requests):
        raise NotImplementedError("This plugin implements StreamingBench multiple-choice generation")
