"""CPU safeguards for HERMES's optional fixed-canvas input contract."""

from types import SimpleNamespace
from contextlib import nullcontext
import json
import sys

import pytest

from kvbench.hermes_runner import (
    configure_fixed_processor, parser, validate_fixed_request, validate_fixed_visual_output,
)


def _model():
    return SimpleNamespace(
        last_frame_stride=1.0,
        config=SimpleNamespace(vision_config=SimpleNamespace(
            patch_size=14, spatial_merge_size=2, temporal_patch_size=2)),
        processor=SimpleNamespace(
            image_processor=SimpleNamespace(size={"shortest_edge": 3136, "longest_edge": 602112}),
            video_processor=SimpleNamespace(size={"shortest_edge": 100352, "longest_edge": 602112})),
    )


def test_fixed_option_defaults_to_none():
    args = parser().parse_args([
        "--anno-path", "annotation.json", "--model-path", "model", "--output-dir", "out",
        "--budget", "500", "--max-videos", "1", "--sample-fps", "0.5",
        "--encode-chunk-size", "16", "--dtype", "float16",
    ])
    assert args.fixed_frame_size is None


def test_fixed_request_refuses_model_and_cap_mismatches():
    validate_fixed_request(None, "qwen3_vl", 151200)
    validate_fixed_request(896, "qwen2_5_vl")
    for invalid in (0, -28, 895, True):
        with pytest.raises(ValueError, match="multiple of 28"):
            validate_fixed_request(invalid, "qwen2_5_vl")
    with pytest.raises(ValueError, match="Qwen2.5"):
        validate_fixed_request(896, "qwen3_vl")
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_fixed_request(896, "qwen2_5_vl", 896 ** 2)


def test_processor_limits_change_only_the_selected_model():
    fixed, untouched = _model(), _model()
    configure_fixed_processor(fixed, 896)
    for name in ("image_processor", "video_processor"):
        processor = getattr(fixed.processor, name)
        assert processor.min_pixels == processor.max_pixels == 896 ** 2
        assert processor.size == {"shortest_edge": 896 ** 2, "longest_edge": 896 ** 2}
    assert untouched.processor.video_processor.size["longest_edge"] == 602112
    assert fixed.last_frame_stride == 1.0


def test_last_frame_resolution_override_fails_before_encoding():
    model = _model()
    model.last_frame_stride = 2.0
    with pytest.raises(ValueError, match="last_frame_stride=1"):
        configure_fixed_processor(model, 896)


@pytest.mark.parametrize("frames,units", [(1, 1), (2, 1), (3, 2), (16, 8)])
def test_actual_grid_and_feature_counts_include_qwen_temporal_padding(frames, units):
    audit = validate_fixed_visual_output(frames, 896, [units, 64, 64], units * 1024)
    assert audit["expected_visual_tokens"] == units * 1024
    with pytest.raises(RuntimeError, match="fixed input grid"):
        validate_fixed_visual_output(frames, 896, [units, 56, 56], units * 784)
    with pytest.raises(RuntimeError, match="feature count"):
        validate_fixed_visual_output(frames, 896, [units, 64, 64], units * 1024 - 1)


def test_real_video_processor_preserves_prepared_canvas_without_second_resize():
    import torch
    from transformers.models.qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor

    processor = Qwen2VLVideoProcessor()
    model = _model()
    model.processor.video_processor = processor
    configure_fixed_processor(model, 896)
    pixels = torch.full((3, 3, 896, 896), 127, dtype=torch.uint8)
    result = processor(pixels, return_tensors="pt")
    grid = result["video_grid_thw"][0].tolist()
    validate_fixed_visual_output(3, 896, grid)
    assert result["pixel_values_videos"].shape[0] == 2 * 64 * 64


@pytest.mark.parametrize("nested_text_config", [False, True])
def test_runner_passes_shared_pixels_to_original_encode_and_records_actual_output(tmp_path, monkeypatch,
                                                                                  nested_text_config):
    """Exercise the runner wrappers, using a tiny model instead of GPU weights."""
    import torch
    from kvbench import hermes_runner
    from kvbench.fixed_inputs import frame_rgb_sha256, prepare_fixed_frames

    model_path = tmp_path / "model"
    model_path.mkdir()
    model_config = {"model_type": "qwen2_5_vl"}
    model_config.update({"text_config": {"num_hidden_layers": 1}} if nested_text_config
                        else {"num_hidden_layers": 1})
    (model_path / "config.json").write_text(json.dumps(model_config))
    annotation = tmp_path / "annotation.json"
    annotation.write_text("[]")
    output = tmp_path / "out"
    frames = torch.arange(3 * 28 * 56 * 3).remainder(256).to(torch.uint8).reshape(3, 28, 56, 3)
    model = _model()
    model._get_cache_seq_len_per_layer = lambda: [0]
    model.get_video_features = lambda pixels, grid: (torch.zeros((int(grid[0, 0]) * 1024, 4)),)
    model.predict_and_compress = lambda: None
    model.question_answering = lambda *_a, **_kw: "A"
    encoded = []

    def encode(pixels, is_last_chunk=False):
        encoded.append((pixels.clone(), is_last_chunk))
        grid = torch.tensor([[(len(pixels) + 1) // 2, pixels.shape[1] // 14, pixels.shape[2] // 14]])
        model.get_video_features(torch.zeros(1), grid)

    model.encode_video_chunk = encode
    loads = []

    def load(*args, **kwargs):
        loads.append((args, kwargs))
        return model, model.processor

    monkeypatch.setitem(sys.modules, "kvbench.hermes_backend.qwenvl_hermes", SimpleNamespace(load_model=load))
    monkeypatch.setattr(hermes_runner, "gpu_lease", lambda _out: nullcontext())
    monkeypatch.setattr(hermes_runner, "prepare_annotations", lambda *_args: [
        {"video_id": "v", "conversations": [{"end_time": 6.0}]}])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)

    def analyze(bridge, video, fps, chunk_size, _torch):
        assert fps == 0.5 and chunk_size == 16
        bridge._active_video_id = video["video_id"]
        bridge.encode_video_chunk(frames, is_last_chunk=True)
        bridge.predict_and_compress()
        assert bridge.question_answering("unchanged question") == "A"
        return [{"qa_acc": 100.0}]

    monkeypatch.setattr(hermes_runner, "analyze_video", analyze)
    monkeypatch.setattr(hermes_runner, "write_results", lambda *_args: None)
    assert hermes_runner.main([
        "--anno-path", str(annotation), "--model-path", str(model_path), "--output-dir", str(output),
        "--budget", "500", "--max-videos", "1", "--sample-fps", "0.5",
        "--encode-chunk-size", "16", "--dtype", "float16", "--fixed-frame-size", "896",
    ]) == 0
    assert loads == [((str(model_path),), {"kv_size": 500, "sample_fps": 0.5, "compress_mode": "hermes"})]
    expected = prepare_fixed_frames(frames, canvas_size=896, pad_value=127)
    assert torch.equal(encoded[0][0], expected) and encoded[0][1] is True
    trace = [json.loads(line) for line in (output / "hermes_trace.jsonl").read_text().splitlines()]
    visual = trace[0]
    assert visual["source_hw"] == visual["resized_hw"] == [896, 896]
    assert visual["original_source_hw"] == [28, 56]
    assert visual["prepared_rgb_sha256"] == frame_rgb_sha256(expected)
    assert visual["grid_thw"] == [2, 64, 64] and visual["visual_tokens"] == 2048
    assert [row["event"] for row in trace] == ["video_input", "compression", "question"]
    from kvbench import audit_hermes
    selected = json.loads((output / "annotations.json").read_text())
    monkeypatch.setattr(audit_hermes, "prepare_annotations", lambda *_a: selected)
    monkeypatch.setattr(audit_hermes, "load_resume", lambda *_a, **_kw: ([{"qa_acc": 100.0}], 1))
    assert audit_hermes.audit(output)["fixed_inputs_verified"] is True


def test_audit_pins_fixed_input_source_and_keeps_legacy_protocol_compatible():
    import hashlib
    from pathlib import Path
    from kvbench.audit_hermes import _fixed_input_contract
    from kvbench import fixed_inputs

    assert _fixed_input_contract({"model_type": "qwen3_vl"}) is None
    source = Path(fixed_inputs.__file__)
    protocol = dict(
        fixed_frame_size=896, model_type="qwen2_5_vl", video_max_pixels=None,
        last_frame_stride=1.0, fixed_frame_pad_value=127,
        fixed_visual_tokens_per_temporal_unit=1024, fixed_input_source=str(source),
        fixed_input_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    assert _fixed_input_contract(protocol)["canvas_size"] == 896
    with pytest.raises(ValueError, match="source changed"):
        _fixed_input_contract(dict(protocol, fixed_input_source_sha256="0" * 64))
    with pytest.raises(ValueError, match="conflicts"):
        _fixed_input_contract(dict(protocol, last_frame_stride=2.0))


def test_audit_rejects_modified_fixed_canvas_grid_count_and_rgb_evidence():
    from copy import deepcopy
    from kvbench.audit_hermes import _validate_fixed_input_event
    from kvbench.fixed_inputs import fixed_input_metadata

    metadata = fixed_input_metadata(896, 127)
    digest = "a" * 64
    event = dict(
        raw_frames=3, source_hw=[896, 896], resized_hw=[896, 896],
        original_source_hw=[720, 1280], prepared_rgb_sha256=digest,
        fixed_input=dict(metadata, prepared_rgb_sha256=digest),
        grid_thw=[2, 64, 64], visual_tokens=2048,
        expected_grid_thw=[2, 64, 64], expected_visual_tokens=2048,
    )
    _validate_fixed_input_event(event, metadata)
    for field, value in (
        ("raw_frames", 2), ("grid_thw", [2, 56, 56]), ("visual_tokens", 1024),
        ("source_hw", [720, 1280]), ("prepared_rgb_sha256", "z" * 64),
        ("expected_visual_tokens", 1024),
        ("fixed_input", dict(metadata, pad_value=0, prepared_rgb_sha256=digest)),
    ):
        changed = deepcopy(event)
        changed[field] = value
        with pytest.raises(ValueError, match="HERMES fixed input"):
            _validate_fixed_input_event(changed, metadata)
