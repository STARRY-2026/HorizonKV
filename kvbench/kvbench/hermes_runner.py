"""Bounded adapter for original Qwen2.5 HERMES and a Qwen3 adaptation.

This intentionally keeps HERMES's predictive compression and native kv_size
semantics: a visual-position target per layer, shared by KV heads, with the
initial text prefix retained separately. It shares KVBench's selected videos,
frame indices, questions, GPU lease and result format.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from kvbench import paths
from sparsemm_baselines.run_streamingbench import (
    analyze_video, gpu_lease, prepare_annotations, sha256, write_json, write_results,
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--anno-path", type=Path, required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--budget", type=int, required=True)
    p.add_argument("--max-videos", type=int, required=True)
    p.add_argument("--max-questions", type=int)
    p.add_argument("--sample-fps", type=float, required=True)
    p.add_argument("--encode-chunk-size", type=int, required=True)
    p.add_argument("--dtype", choices=("float16", "bfloat16"), required=True)
    p.add_argument("--vision-topology", choices=("native", "window4"), default="window4")
    p.add_argument("--video-max-pixels", type=int)
    p.add_argument("--fixed-frame-size", type=int,
                   help="Qwen2.5-only square letterbox canvas; 896 gives 1024 visual tokens per temporal unit")
    p.add_argument("--dry-run", action="store_true")
    return p


def _digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def validate_fixed_request(frame_size, model_kind, video_max_pixels=None):
    """Keep the fixed-input contract explicit and separate from pixel caps."""
    if frame_size is None:
        return
    if (isinstance(frame_size, bool) or not isinstance(frame_size, int)
            or frame_size < 28 or frame_size % 28):
        raise ValueError("fixed_frame_size must be a positive multiple of 28")
    if model_kind != "qwen2_5_vl":
        raise ValueError("fixed_frame_size currently supports Qwen2.5 HERMES only")
    if video_max_pixels is not None:
        raise ValueError("fixed_frame_size and video_max_pixels are mutually exclusive")


def configure_fixed_processor(model, frame_size):
    """Pin only this model's processor; keep the HERMES model loader unchanged."""
    validate_fixed_request(frame_size, "qwen2_5_vl")
    if getattr(model, "last_frame_stride", 1.0) != 1.0:
        raise ValueError("Fixed HERMES inputs require last_frame_stride=1")
    vision = model.config.vision_config
    if (int(vision.patch_size) != 14 or int(vision.spatial_merge_size) != 2
            or int(vision.temporal_patch_size) != 2):
        raise ValueError("Fixed HERMES inputs require Qwen2.5's 14x14, merge-2, temporal-2 vision grid")
    seen = set()
    for name in ("image_processor", "video_processor"):
        processor = getattr(model.processor, name, None)
        if processor is None or id(processor) in seen:
            continue
        seen.add(id(processor))
        pixels = frame_size ** 2
        processor.min_pixels = processor.max_pixels = pixels
        processor.size = {"shortest_edge": pixels, "longest_edge": pixels}
    if not seen:
        raise ValueError("HERMES processor has no image/video preprocessing component")


def validate_fixed_visual_output(raw_frames, frame_size, grid, visual_tokens=None):
    """Check the actual processor grid and merged features, including odd padding."""
    if raw_frames < 1:
        raise ValueError("A fixed HERMES chunk must contain at least one real frame")
    expected_grid = [(raw_frames + 1) // 2, frame_size // 14, frame_size // 14]
    expected_tokens = expected_grid[0] * (frame_size // 28) ** 2
    if list(grid) != expected_grid:
        raise RuntimeError(f"HERMES changed the fixed input grid: {grid} != {expected_grid}")
    if visual_tokens is not None and visual_tokens != expected_tokens:
        raise RuntimeError(f"HERMES fixed feature count changed: {visual_tokens} != {expected_tokens}")
    return {"expected_grid_thw": expected_grid, "expected_visual_tokens": expected_tokens}


def main(argv=None):
    args = parser().parse_args(argv)
    if not 1 <= args.max_videos <= 3 or not 1 <= args.budget or args.encode_chunk_size < 1:
        raise ValueError("HERMES adapter permits only 1..3 videos and positive budget/chunk size")
    model_config = args.model_path.resolve() / "config.json"
    model_info = json.loads(model_config.read_text())
    model_kind = model_info.get("model_type")
    if model_kind not in ("qwen2_5_vl", "qwen3_vl"):
        raise ValueError("HERMES adapter supports Qwen2.5-VL and Qwen3-VL only")
    validate_fixed_request(args.fixed_frame_size, model_kind, args.video_max_pixels)
    if model_kind == "qwen2_5_vl" and args.dtype != "float16":
        raise ValueError("Original Qwen2.5 HERMES requires float16")
    if args.video_max_pixels is not None and (model_kind != "qwen3_vl" or args.video_max_pixels < 3136):
        raise ValueError("video_max_pixels requires Qwen3 HERMES and at least 3136 pixels")
    data = prepare_annotations(args.anno_path, args.max_videos, args.max_questions)
    questions = sum(len(video["conversations"]) for video in data)
    if questions > 8 or any(float(q["end_time"]) > 120 for v in data for q in v["conversations"]):
        raise ValueError("HERMES adapter is limited to eight questions and 120-second endpoints")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    backend = Path(__file__).resolve().parent / "hermes_backend"
    implementation = (backend / "qwenvl_hermes.py" if model_kind == "qwen2_5_vl"
                      else Path(__file__).resolve().parents[1] / "sparsemm_baselines" / "qwen3_hermes.py")
    abstract = backend / "abstract_hermes.py"
    reindex = backend / "reindex_3d.py"
    adapter = Path(__file__).resolve()
    num_layers = int(model_info.get("text_config", model_info)["num_hidden_layers"])
    protocol = {
        "method": "hermes", "budget": args.budget,
        "model_type": model_kind, "num_layers": num_layers,
        "budget_scope": "visual_positions_per_layer_shared_across_kv_heads_text_prefix_extra",
        "dtype": args.dtype, "sample_fps": args.sample_fps,
        "video_max_pixels": args.video_max_pixels,
        "encode_chunk_size": args.encode_chunk_size, "model_path": str(args.model_path.resolve()),
        "model_config_sha256": sha256(model_config), "annotation_path": str(args.anno_path.resolve()),
        "annotation_sha256": sha256(args.anno_path), "subset_sha256": _digest_json(data),
        "hermes_source": str(implementation.resolve()), "hermes_source_sha256": sha256(implementation),
        "adapter_source": str(adapter), "adapter_sha256": sha256(adapter),
        "sampling": "rounded_source_fps_integer_stride; endpoint=ceil(end_time*sample_fps)",
        "video_preprocessing": ("original_HERMES_Qwen_processor_dynamic_video_frames" if model_kind == "qwen2_5_vl"
                                else "Qwen3_VL_processor_dynamic_video_frames_existing_streaming_bridge"),
        "video_chunk_mode": "hermes_chunk", "eviction_granularity": "runner_video_chunk",
        "compression": ("original_HERMES_predict_next_question_then_pseudo_forward" if model_kind == "qwen2_5_vl"
                        else "Qwen3_adaptation_predict_next_question_then_pseudo_forward_original_mrope_positions"),
        "comparison_status": "same_questions_and_frame_indices; input_tokens_and_budget_semantics_not_equated",
        "videos": len(data), "questions": questions, "full_benchmarks_paused": True,
    }
    if model_kind == "qwen2_5_vl":
        protocol.update({"hermes_abstract_source": str(abstract.resolve()),
                         "hermes_abstract_sha256": sha256(abstract),
                         "hermes_reindex_source": str(reindex.resolve()),
                         "hermes_reindex_sha256": sha256(reindex)})
    else:
        from sparsemm_baselines.qwen3_hermes import layer_groups
        shallow_end, deep_start = layer_groups(num_layers)
        bridge = implementation.with_name("qwen3_adapter.py")
        protocol["qwen3_bridge_source"] = str(bridge.resolve())
        protocol["qwen3_bridge_sha256"] = sha256(bridge)
        protocol["vision_topology"] = args.vision_topology
        if args.vision_topology == "window4":
            window4 = implementation.with_name("qwen3_window4.py")
            protocol["vision_adapter_source"] = str(window4.resolve())
            protocol["vision_adapter_sha256"] = sha256(window4)
        protocol["layer_groups"] = {"shallow": [0, shallow_end - 1],
                                    "middle": [shallow_end, deep_start - 1],
                                    "deep": [deep_start, num_layers - 1],
                                    "ratios": [0.1, 0.6, 0.3]}
        protocol["position_policy"] = "retain_original_mrope_positions_after_eviction"
    if args.fixed_frame_size is not None:
        from kvbench.fixed_inputs import fixed_input_metadata
        fixed_source = Path(__file__).with_name("fixed_inputs.py")
        protocol.update(
            fixed_frame_size=args.fixed_frame_size,
            fixed_visual_input=fixed_input_metadata(args.fixed_frame_size),
            fixed_visual_tokens_per_temporal_unit=(args.fixed_frame_size // 28) ** 2,
            fixed_frame_pad_value=127,
            fixed_input_source=str(fixed_source), fixed_input_source_sha256=sha256(fixed_source),
            video_preprocessing="shared_bicubic_antialias_letterbox_fixed_canvas",
            last_frame_stride=1.0,
        )
    write_json(out / "protocol.json", protocol)
    write_json(out / "annotations.json", data)
    if args.dry_run:
        (out / "run.status").write_text("dry-run\n")
        return 0
    if (out / "1_0.csv").exists():
        raise FileExistsError("HERMES result path already contains predictions")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    for key in ("PROTOKV_ENABLE", "PROTOKV_FULL_ENABLE", "INFINIPOT_V_PUBLIC"):
        os.environ[key] = "0"
    with gpu_lease(out):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("HERMES requires CUDA; use --dry-run to inspect the protocol")
        if model_kind == "qwen2_5_vl":
            from kvbench.hermes_backend.qwenvl_hermes import load_model
            model, _ = load_model(str(args.model_path.resolve()), kv_size=args.budget,
                                  sample_fps=args.sample_fps, compress_mode="hermes")
            feature_owner = model
        else:
            from sparsemm_baselines.qwen3_hermes import load_qwen3_hermes
            model = load_qwen3_hermes(str(args.model_path.resolve()), kv_size=args.budget,
                                      sample_fps=args.sample_fps, dtype=args.dtype,
                                      vision_topology=args.vision_topology,
                                      video_max_pixels=args.video_max_pixels)
            feature_owner = model.model
        if args.fixed_frame_size is not None:
            configure_fixed_processor(model, args.fixed_frame_size)
        trace = []
        native_encode = model.encode_video_chunk
        native_features = feature_owner.get_video_features
        native_image_features = (feature_owner.get_image_features if model_kind == "qwen3_vl" else None)
        native_compress = model.predict_and_compress
        native_answer = model.question_answering

        def traced_features(pixels, grid, *args, **kwargs):
            if parsed_fixed_size is not None:
                validate_fixed_visual_output(model._fixed_input_raw_frames,
                                             parsed_fixed_size, grid[0].tolist())
            result = native_features(pixels, grid, *args, **kwargs)
            features = result[0]
            model._last_visual_grid = [int(value) for value in grid[0].tolist()]
            model._last_visual_tokens = int(features[0].shape[0] if isinstance(features, (list, tuple)) else features.shape[0])
            return result

        def traced_image_features(pixels, grid, *args, **kwargs):
            result = native_image_features(pixels, grid, *args, **kwargs)
            features = result[0]
            model._last_visual_grid = [int(value) for value in grid[0].tolist()]
            model._last_visual_tokens = int(features[0].shape[0] if isinstance(features, (list, tuple)) else features.shape[0])
            return result

        def traced_encode(frames, is_last_chunk=False):
            source_hw = ([int(frames.shape[1]), int(frames.shape[2])] if frames.shape[-1] == 3
                         else [int(frames.shape[-2]), int(frames.shape[-1])])
            fixed_audit = None
            if parsed_fixed_size is not None:
                from kvbench.fixed_inputs import (
                    fixed_input_metadata, frame_rgb_sha256, prepare_fixed_frames,
                )
                frames = prepare_fixed_frames(frames, canvas_size=parsed_fixed_size, pad_value=127)
                fixed_audit = fixed_input_metadata(canvas_size=parsed_fixed_size, pad_value=127)
                fixed_audit["prepared_rgb_sha256"] = frame_rgb_sha256(frames)
                model._fixed_input_raw_frames = int(frames.shape[0])
            model._last_visual_grid = None
            model._last_visual_tokens = None
            native_encode(frames, is_last_chunk=is_last_chunk)
            grid = model._last_visual_grid
            count = model._last_visual_tokens
            if not isinstance(grid, list) or count is None:
                raise RuntimeError("HERMES did not report the encoded video chunk")
            vision_config = (model.config if model_kind == "qwen2_5_vl" else model.model_config).vision_config
            visual = {"raw_frames": int(frames.shape[0]), "source_hw": source_hw,
                      "grid_thw": grid,
                      "resized_hw": [grid[1] * int(vision_config.patch_size),
                                     grid[2] * int(vision_config.patch_size)],
                      "visual_tokens": count}
            if fixed_audit is not None:
                visual["fixed_input"] = fixed_audit
                visual["original_source_hw"] = source_hw
                visual["source_hw"] = [parsed_fixed_size, parsed_fixed_size]
                visual["prepared_rgb_sha256"] = fixed_audit["prepared_rgb_sha256"]
                visual.update(validate_fixed_visual_output(
                    int(frames.shape[0]), parsed_fixed_size, grid, count))
            trace.append({"event": "video_input", "video_id": str(model._active_video_id),
                          **visual})

        def traced_compress():
            before = model._get_cache_seq_len_per_layer()
            native_compress()
            after = model._get_cache_seq_len_per_layer()
            trace.append({"event": "compression", "video_id": str(model._active_video_id),
                          "before": before, "after": after, "evicted": any(a < b for a, b in zip(after, before))})

        def traced_answer(*a, **kw):
            before = model._get_cache_seq_len_per_layer()
            result = native_answer(*a, **kw)
            after = model._get_cache_seq_len_per_layer()
            trace.append({"event": "question", "video_id": str(model._active_video_id),
                          "before": before, "after": after, "restored": before == after})
            if before != after:
                raise RuntimeError("HERMES question changed the persistent video cache")
            return result

        parsed_fixed_size = args.fixed_frame_size
        model.encode_video_chunk = traced_encode
        feature_owner.get_video_features = traced_features
        if native_image_features is not None:
            feature_owner.get_image_features = traced_image_features
        model.predict_and_compress = traced_compress
        model.question_answering = traced_answer
        rows = []
        (out / "run.status").write_text("running\n")
        for video in data:
            rows.extend(analyze_video(model, video, args.sample_fps, args.encode_chunk_size, torch))
            write_results(out / "1_0.csv", rows)
            (out / "hermes_trace.jsonl").write_text("".join(json.dumps(event) + "\n" for event in trace))
        correct = sum(float(row["qa_acc"]) == 100.0 for row in rows)
        write_json(out / "summary.json", {
            "method": "hermes", "budget": args.budget, "budget_scope": protocol["budget_scope"],
            "videos": len(data), "questions": len(rows), "correct": correct,
            "accuracy_percent": 100 * correct / len(rows),
            "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        })
        (out / "run.status").write_text("complete\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
