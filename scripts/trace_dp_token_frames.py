#!/usr/bin/env python3
"""Run the unchanged HermesVQA entrypoint with a visual-token source manifest.

All ordinary video_qa/base.py arguments are accepted. Add
--source_manifest_path PATH and BOTH --ragged_token_trace_path and
--ragged_readout_trace_path. This script observes the existing loader and the
actual get_video_features call; it never runs a second processor or changes
frames, features, cache selection, positions, or model generation.
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import os
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def install_source_manifest(path):
    import numpy as np
    from decord import VideoReader
    from video_qa import base
    from video_qa.hermes_vqa import HermesVQA

    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep each run separate: existing trace files are also reset by the core.
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"type": "run", "record_type": "run", "schema_version": 1,
                                 "argv": sys.argv, "project_root": str(ROOT)},
                                ensure_ascii=False) + "\n")

    def emit(record):
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    original_analyze = HermesVQA.analyze_a_video

    @functools.wraps(original_analyze)
    def analyze(self, sample, *args, **kwargs):
        self._source_manifest_video_id = sample.get("video_id")
        self.qa_model._source_manifest_video_id = sample.get("video_id")
        self.qa_model._source_manifest_encode_call = 0
        self.qa_model._source_manifest_samples = None
        return original_analyze(self, sample, *args, **kwargs)

    HermesVQA.analyze_a_video = analyze
    original_load = base.BaseVQA.load_video

    @functools.wraps(original_load)
    def load_video(self, video_path, clip=None):
        # Preserve the original loader's exact output and call exactly once.
        video = original_load(self, video_path, clip=clip)
        fps = float(self.sample_fps)
        if str(video_path).endswith(".npy"):
            source = np.load(video_path, mmap_mode="r")
            count = len(source)
            indices = np.linspace(0, count - 1, int(count * fps), dtype=int).tolist()
            pts = [None] * len(indices)
            source_fps = rounded_fps = None
            sampling_step = None
            timestamp_basis = "npy has no decoder timestamps"
        else:
            # A separate reader inspects metadata only; it does not decode or
            # replace any input frame returned by the original loader.
            reader = VideoReader(video_path, num_threads=1)
            source_fps = float(reader.get_avg_fps())
            rounded_fps = round(source_fps)
            count = len(reader)
            first = max(0, int(clip[0] * rounded_fps)) if clip is not None else 0
            stop = min(count, int(clip[1] * rounded_fps) + 1) if clip is not None else count
            sampling_step = int(rounded_fps / fps)
            indices = list(range(first, stop, sampling_step))
            pts = reader.get_frame_timestamp(indices).tolist() if indices else []
            timestamp_basis = "decord get_frame_timestamp: [start_s, end_s]"
        if len(indices) != len(video):
            raise RuntimeError("Manifest sample count differs from original loader output")
        samples = [{"sample_ordinal": ordinal, "source_frame_index": int(index),
                    "decoder_pts": interval, "nominal_time_s": ordinal / fps}
                   for ordinal, (index, interval) in enumerate(zip(indices, pts))]
        self.qa_model._source_manifest_samples = samples
        self.qa_model._source_manifest_input_shape = list(video.shape)
        emit({"type": "video", "record_type": "video", "video_id": self._source_manifest_video_id,
              "sample_indices": indices, "sample_pts": pts,
              "source_shape": list(video.shape[1:]), "fps": source_fps,
              "video_path": str(Path(video_path).resolve()), "clip": clip,
              "sample_fps": fps, "source_fps": source_fps,
              "loader_rounded_fps": rounded_fps, "source_frame_count": count,
              "sampling_step": sampling_step, "loaded_shape": list(video.shape),
              "timestamp_basis": timestamp_basis, "samples": samples})
        return video

    base.BaseVQA.load_video = load_video
    original_qwen_load = base.qwenvl_hermes_load_model

    @functools.wraps(original_qwen_load)
    def load_model(*args, **kwargs):
        model, processor = original_qwen_load(*args, **kwargs)
        original_features = model.get_video_features

        def get_video_features(self, *feature_args, **feature_kwargs):
            # The core passes the actual preprocessed grid to this function.
            grid_tensor = feature_kwargs.get("video_grid_thw")
            if grid_tensor is None and len(feature_args) >= 2:
                grid_tensor = feature_args[1]
            if grid_tensor is None:
                raise RuntimeError("get_video_features call has no observable video_grid_thw")
            token_config = getattr(self, "_ragged_token_trace_config", None)
            if token_config is None:
                raise RuntimeError("Source manifest requires --ragged_token_trace_path")
            # Read the original encoder's local frame count without processing
            # the video again. Retain no frame reference after recording shape.
            caller = sys._getframe(1)
            try:
                chunk = caller.f_locals.get("video_chunk")
                if caller.f_code.co_name != "_encode_ragged_frames" or chunk is None:
                    raise RuntimeError("Expected the standard ragged _encode_ragged_frames caller")
                chunk_shape = list(chunk.shape)
            finally:
                del caller
            grids = grid_tensor.detach().cpu().tolist()
            uid_start = int(token_config["next_visual_token_uid"])
            frame_start = int(token_config["next_frame_id"])
            # Call the unmodified feature extractor once and return its object.
            features = original_features(*feature_args, **feature_kwargs)
            first_features = features[0]
            q_len = int(first_features.shape[-2])
            raw_count = int(chunk_shape[0])
            if len(grids) != 1:
                raise RuntimeError("Manifest currently requires one video grid per encode call")
            t, h, w = map(int, grids[0])
            vc = self.config.vision_config
            merge = int(vc.spatial_merge_size)
            patch = int(vc.patch_size)
            if q_len != t * (h // merge) * (w // merge):
                raise RuntimeError("Observed Qwen visual output length does not match merged grid")
            all_samples = getattr(self, "_source_manifest_samples", None)
            if all_samples is None or frame_start + raw_count > len(all_samples):
                raise RuntimeError("Encoded frames do not fit observed original loader timeline")
            call = int(getattr(self, "_source_manifest_encode_call", 0))
            emit({"type": "encode", "record_type": "encode", "video_id": self._source_manifest_video_id,
                  "grid_thw": [t, h, w], "first_sample_ordinal": frame_start,
                  "raw_frame_count": raw_count, "merge_size": merge,
                  "question_index": getattr(self, "_active_question_index", None),
                  "encode_call": call, "source_sample_start": frame_start,
                  "source_sample_count": raw_count, "video_chunk_shape": chunk_shape,
                  "video_grid_thw": [t, h, w], "q_len": q_len,
                  "uid_start": uid_start, "uid_stop_exclusive": uid_start + q_len,
                  "patch_size": patch, "spatial_merge_size": merge,
                  "temporal_patch_size": int(vc.temporal_patch_size),
                  "llm_grid_thw": [t, h // merge, w // merge],
                  "processor_resized_hw": [h * patch, w * patch],
                  "source_loaded_shape": self._source_manifest_input_shape,
                  "fixed_frame_canvas_size": int(getattr(self, "fixed_frame_canvas_size", 0) or 0),
                  "fixed_frame_pad_mode": str(getattr(self, "fixed_frame_pad_mode", "letterbox")),
                  "samples": all_samples[frame_start:frame_start + raw_count]})
            self._source_manifest_encode_call = call + 1
            return features

        model.get_video_features = types.MethodType(get_video_features, model)
        return model, processor

    base.qwenvl_hermes_load_model = load_model
    # MODELS contains function references captured when base was imported.
    for spec in base.MODELS.values():
        if spec["load_func"] is original_qwen_load:
            spec["load_func"] = load_model
    return base, HermesVQA


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source_manifest_path", required=True)
    own, remaining = parser.parse_known_args()
    if "--ragged_token_trace_path" not in remaining or "--ragged_readout_trace_path" not in remaining:
        raise SystemExit("Pass both --ragged_token_trace_path and --ragged_readout_trace_path")
    sys.argv = [sys.argv[0]] + remaining
    base, qa_class = install_source_manifest(own.source_manifest_path)
    base.work(qa_class)


if __name__ == "__main__":
    main()
