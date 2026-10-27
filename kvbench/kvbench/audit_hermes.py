"""Read-only audit of bounded Qwen2.5 HERMES and Qwen3 adaptation output."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from sparsemm_baselines.run_streamingbench import load_resume, prepare_annotations, sha256, write_json


def _fixed_input_contract(protocol):
    """Validate the optional fixed-pixel source and its recorded model contract."""
    size = protocol.get("fixed_frame_size")
    if size is None:
        return None
    from kvbench.fixed_inputs import fixed_input_metadata
    try:
        metadata = fixed_input_metadata(size, protocol.get("fixed_frame_pad_value"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid HERMES fixed input metadata") from exc
    if (protocol.get("model_type") != "qwen2_5_vl"
            or protocol.get("video_max_pixels") is not None
            or protocol.get("last_frame_stride") != 1.0
            or protocol.get("fixed_frame_pad_value") != 127
            or protocol.get("fixed_visual_tokens_per_temporal_unit") != metadata["tokens_per_temporal_unit"]):
        raise ValueError("HERMES fixed input protocol conflicts with the selected model/input settings")
    source = Path(protocol.get("fixed_input_source", ""))
    if not source.is_file() or sha256(source) != protocol.get("fixed_input_source_sha256"):
        raise ValueError("HERMES fixed input source changed since inference")
    return metadata


def _validate_fixed_input_event(event, metadata):
    """Check dimensions, actual features and comparable prepared-RGB evidence."""
    from kvbench.fixed_inputs import assert_qwen_grid
    size = metadata["canvas_size"]
    digest = event.get("prepared_rgb_sha256")
    if (not isinstance(digest, str) or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)):
        raise ValueError("HERMES fixed input has an invalid prepared RGB hash")
    expected_metadata = dict(metadata, prepared_rgb_sha256=digest)
    original = event.get("original_source_hw")
    if (event.get("fixed_input") != expected_metadata
            or event.get("source_hw") != [size, size]
            or event.get("resized_hw") != [size, size]
            or not isinstance(original, list) or len(original) != 2
            or any(type(value) is not int or value <= 0 for value in original)):
        raise ValueError("HERMES fixed input canvas/metadata differs from its protocol")
    try:
        actual = assert_qwen_grid(event.get("grid_thw", []), event.get("visual_tokens"),
                                 event.get("raw_frames"), canvas_size=size, kind="video")
    except (AssertionError, TypeError, ValueError) as exc:
        raise ValueError("HERMES fixed input grid/token count differs from its real frame count") from exc
    if (event.get("expected_grid_thw") != actual["grid_thw"][0]
            or event.get("expected_visual_tokens") != actual["visual_tokens"]):
        raise ValueError("HERMES fixed input expected-grid evidence disagrees with actual features")


def audit(out, min_compression_events_per_video=0):
    out = Path(out).resolve()
    protocol = json.loads((out / "protocol.json").read_text())
    if protocol.get("method") != "hermes" or (out / "run.status").read_text().strip() != "complete":
        raise ValueError("HERMES run is not complete")
    source = Path(protocol["hermes_source"])
    abstract = Path(protocol["hermes_abstract_source"]) if "hermes_abstract_source" in protocol else None
    reindex = Path(protocol["hermes_reindex_source"]) if "hermes_reindex_source" in protocol else None
    adapter = Path(protocol["adapter_source"])
    legacy_adapter = (Path(__file__).resolve().parent / "hermes_backend" /
                      "legacy_runner_source_20261023.txt")
    adapter_historical = (sha256(adapter) != protocol["adapter_sha256"]
                          and protocol.get("video_chunk_mode") is None
                          and legacy_adapter.is_file()
                          and sha256(legacy_adapter) == protocol["adapter_sha256"])
    model = Path(protocol["model_path"]) / "config.json"
    annotation = Path(protocol["annotation_path"])
    if (sha256(source) != protocol["hermes_source_sha256"]
            or (abstract is not None and sha256(abstract) != protocol["hermes_abstract_sha256"])
            or (reindex is not None and sha256(reindex) != protocol["hermes_reindex_sha256"])
            or ("qwen3_bridge_source" in protocol and
                sha256(Path(protocol["qwen3_bridge_source"])) != protocol["qwen3_bridge_sha256"])
            or ("vision_adapter_source" in protocol and
                sha256(Path(protocol["vision_adapter_source"])) != protocol["vision_adapter_sha256"])
            or (sha256(adapter) != protocol["adapter_sha256"] and not adapter_historical)
            or sha256(model) != protocol["model_config_sha256"]
            or sha256(annotation) != protocol["annotation_sha256"]):
        raise ValueError("HERMES source, model or annotation changed since inference")
    fixed_metadata = _fixed_input_contract(protocol)
    data = json.loads((out / "annotations.json").read_text())
    model_info = json.loads(model.read_text())
    num_layers = int(model_info.get("text_config", model_info)["num_hidden_layers"])
    if protocol.get("num_layers", num_layers) != num_layers:
        raise ValueError("HERMES protocol layer count differs from model")
    if protocol.get("model_type") == "qwen3_vl":
        from sparsemm_baselines.qwen3_hermes import layer_groups
        shallow_end, deep_start = layer_groups(num_layers)
        if protocol.get("layer_groups") != {"shallow": [0, shallow_end - 1],
                                             "middle": [shallow_end, deep_start - 1],
                                             "deep": [deep_start, num_layers - 1],
                                             "ratios": [0.1, 0.6, 0.3]}:
            raise ValueError("HERMES Qwen3 depth groups differ from requested ratios")
    expected = prepare_annotations(annotation, protocol["videos"])
    expected = [dict(video, conversations=video["conversations"][:len(data[i]["conversations"])])
                for i, video in enumerate(expected)]
    digest = hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if expected != data or digest != protocol["subset_sha256"]:
        raise ValueError("HERMES selected subset disagrees with its pinned source")
    rows, videos = load_resume(out / "1_0.csv", data, require_complete=True)
    if videos != len(data) or len(rows) != protocol["questions"]:
        raise ValueError("HERMES result CSV does not cover the selected questions")
    summary = json.loads((out / "summary.json").read_text())
    correct = sum(float(row["qa_acc"]) == 100.0 for row in rows)
    if (summary.get("method") != "hermes" or summary.get("budget") != protocol["budget"]
            or summary.get("correct") != correct or summary.get("questions") != len(rows)
            or summary.get("accuracy_percent") != 100 * correct / len(rows)):
        raise ValueError("HERMES summary differs from its checked answer rows")
    trace = [json.loads(line) for line in (out / "hermes_trace.jsonl").read_text().splitlines()]
    ids = {str(video["video_id"]): len(video["conversations"]) for video in data}
    observed_questions = {key: 0 for key in ids}
    observed_compressions = {key: 0 for key in ids}
    observed_inputs = {key: 0 for key in ids}
    pending_input = {key: 0 for key in ids}
    require_inputs = protocol.get("video_chunk_mode") == "hermes_chunk"
    for event in trace:
        video_id = event.get("video_id")
        if video_id not in ids or event.get("event") not in ("video_input", "compression", "question"):
            raise ValueError("HERMES trace has unknown video or event")
        if event["event"] == "video_input":
            grid = event.get("grid_thw")
            source = event.get("source_hw")
            resized = event.get("resized_hw")
            if (not isinstance(grid, list) or len(grid) != 3
                    or not isinstance(source, list) or len(source) != 2
                    or not isinstance(resized, list) or len(resized) != 2
                    or any(type(value) is not int or value <= 0
                           for value in grid + source + resized + [event.get("raw_frames"), event.get("visual_tokens")])):
                raise ValueError("HERMES trace has invalid video input dimensions")
            if fixed_metadata is not None:
                _validate_fixed_input_event(event, fixed_metadata)
            observed_inputs[video_id] += 1
            pending_input[video_id] += 1
            continue
        before, after = event.get("before"), event.get("after")
        if (not isinstance(before, list) or len(before) != num_layers or len(after or []) != num_layers
                or any(type(n) is not int or n < 0 for n in before + after)):
            raise ValueError("HERMES trace has invalid layer lengths")
        if event["event"] == "question":
            if event.get("restored") is not True or before != after:
                raise ValueError("HERMES question did not restore persistent KV")
            observed_questions[video_id] += 1
        else:
            if require_inputs and pending_input[video_id] != 1:
                raise ValueError("HERMES compression must follow exactly one video chunk")
            pending_input[video_id] = 0
            if event.get("evicted") != any(a < b for a, b in zip(after, before)):
                raise ValueError("HERMES compression count differs from layer lengths")
            observed_compressions[video_id] += bool(event["evicted"])
    if (observed_questions != ids or (require_inputs and (any(pending_input.values()) or any(n == 0 for n in observed_inputs.values())))
            or any(n < min_compression_events_per_video for n in observed_compressions.values())):
        raise ValueError("HERMES trace does not cover every question or required compression")
    report = {"passed": True, "method": "hermes", "videos": videos, "questions": len(rows),
              "adapter_historical_source_verified": adapter_historical,
              "trace": {"video_chunks": sum(observed_inputs.values()),
                        "compression_events": sum(observed_compressions.values()),
                        "qa_restore_checks": sum(observed_questions.values())},
              "budget_scope": protocol["budget_scope"],
              "limitations": ["HERMES selects visual positions per layer shared by KV heads; DP allocates per KV head. Text prefix is extra in both.",
                              "Length checks do not independently replay predictive attention or tensor contents."]}
    if protocol.get("model_type") == "qwen3_vl":
        report["limitations"].append("Qwen3 adaptation retains original M-RoPE positions after eviction; original Qwen2.5 HERMES compacts and rerotates them.")
    if fixed_metadata is not None:
        report["fixed_inputs_verified"] = True
        report["fixed_input"] = fixed_metadata
        report["limitations"].append("Prepared RGB hashes identify the pixels recorded at inference; this audit does not decode the videos again.")
    write_json(out / "audit_report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--min-compression-events-per-video", type=int, default=0)
    args = parser.parse_args(argv)
    print(json.dumps(audit(args.output_dir, args.min_compression_events_per_video), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
