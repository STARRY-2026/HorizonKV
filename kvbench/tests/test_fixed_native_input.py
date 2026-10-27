"""Fixed-canvas integration checks without weights, CUDA or lmms imports."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from kvbench.fixed_inputs import frame_rgb_sha256, prepare_fixed_frames
from sparsemm_protokv.config import StreamingConfig
from sparsemm_protokv.run_streamingbench import build_command, build_parser, source_hashes
from sparsemm_protokv.streaming import StreamingQwen


def installer():
    source = Path(__file__).resolve().parents[1] / "sparsemm_protokv/models/qwen2_5_vl_stream.py"
    node = next(node for node in ast.parse(source.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == "_install_fixed_frame_input")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[node.name]


class Inputs(dict):
    def to(self, *_args):
        return self


class Processor:
    def __init__(self, wrong_grid=False):
        self.video_processor = SimpleNamespace(do_resize=True)
        self.calls = []
        self.wrong_grid = wrong_grid

    def __call__(self, *, text, videos, return_tensors, **kwargs):
        assert text == [""] and return_tensors == "pt" and kwargs == {}
        assert not self.video_processor.do_resize
        assert tuple(videos.shape[1:]) == (3, 896, 896)
        self.calls.append(videos.clone())
        grid = [(len(videos) + 1) // 2, 32 if self.wrong_grid else 64, 64]
        return Inputs(video_grid_thw=torch.tensor([grid]), pixel_values_videos=videos)


def make_bridge(mode, wrong_grid=False):
    bridge = StreamingQwen.__new__(StreamingQwen)
    bridge.is_llava = False
    bridge.config = SimpleNamespace(video_chunk_mode=mode)
    bridge.model_config = SimpleNamespace(model_type="qwen2_5_vl", vision_config=SimpleNamespace(
        temporal_patch_size=2, patch_size=14, spatial_merge_size=2))
    bridge.processor = Processor(wrong_grid)
    bridge.model = SimpleNamespace(
        device=torch.device("cpu"), parameters=lambda: iter([torch.zeros(1)]),
        get_video_features=lambda _pixels, grid: [torch.zeros(int(grid[0].prod()) // 4, 8)])
    bridge.video_max_pixels = None
    bridge._measure = lambda _name: nullcontext()
    bridge._video_positions = lambda _grid, count: torch.arange(count).view(1, 1, -1).expand(3, 1, -1)
    bridge.calls = []
    bridge.forward_chunk = lambda values, positions, **kwargs: bridge.calls.append(
        (values.shape, positions.clone(), dict(bridge._pending_visual_input), kwargs))
    bridge._pending_visual_input = None
    bridge._seen_visual_tokens = bridge._next_position = bridge.total_processed_frames = 0
    bridge.timings = dict(raw_frames=0, visual_tokens=0, video_prefills=0)
    bridge._active_video_id = "synthetic"
    return bridge


@pytest.mark.parametrize("mode,counts", [("temporal_patch", [2, 2, 1]), ("hermes_chunk", [5])])
def test_fixed_pixels_and_audit_preserve_original_chunk_cadence(mode, counts):
    frames = torch.stack([torch.full((56, 112, 3), color, dtype=torch.uint8)
                          for color in (11, 33, 77, 129, 211)])
    source = frames.clone()
    expected = prepare_fixed_frames(frames)
    bridge = make_bridge(mode)
    original_forward = bridge.forward_chunk
    installer()(bridge, 896)
    bridge.encode_video_chunk(frames, is_last_chunk=True)
    assert bridge.forward_chunk is original_forward
    assert torch.equal(frames, source)
    assert [len(call) for call in bridge.processor.calls] == counts
    assert bridge.total_processed_frames == bridge.last_encoded_frames == 5
    assert bridge.timings["visual_tokens"] == 3072
    cursor = 0
    for count, (shape, positions, audit, kwargs) in zip(counts, bridge.calls):
        tokens = ((count + 1) // 2) * 1024
        assert shape == (1, tokens, 8) and positions.shape == (3, 1, tokens)
        assert kwargs["compress"] is True
        assert audit["source_hw"] == audit["resized_hw"] == [896, 896]
        assert audit["original_source_hw"] == [56, 112]
        assert audit["grid_thw"] == [(count + 1) // 2, 64, 64]
        assert audit["visual_tokens"] == tokens
        assert audit["prepared_rgb_sha256"] == frame_rgb_sha256(expected[cursor:cursor + count])
        cursor += count
    assert len(bridge.fixed_input_checks) == len(counts)


def test_wrong_processor_grid_fails_before_language_forward_and_restores_method():
    bridge = make_bridge("temporal_patch", wrong_grid=True)
    original_forward = bridge.forward_chunk
    installer()(bridge, 896)
    with pytest.raises(AssertionError, match="spatial grid"):
        bridge.encode_video_chunk(torch.zeros(2, 56, 112, 3, dtype=torch.uint8))
    assert bridge.calls == [] and bridge.forward_chunk is original_forward


def test_real_checkpoint_processor_keeps_896_canvas_despite_a_smaller_pixel_cap():
    from transformers import Qwen2_5_VLProcessor
    from kvbench.paths import default_model_path

    model_path = default_model_path()
    if not (model_path / "preprocessor_config.json").is_file():
        pytest.skip("Local Qwen checkpoint processor is not installed")
    bridge = make_bridge("hermes_chunk")
    bridge.processor = Qwen2_5_VLProcessor.from_pretrained(model_path, local_files_only=True)
    # Even if a caller previously selected a lower processor cap, turning off
    # resize must preserve 64x64 patches rather than silently shrinking them.
    bridge.processor.video_processor.size = dict(
        bridge.processor.video_processor.size, longest_edge=602112)
    installer()(bridge, 896)
    bridge.encode_video_chunk(torch.full((3, 56, 112, 3), 37, dtype=torch.uint8))
    assert len(bridge.calls) == 1
    assert bridge.calls[0][2]["grid_thw"] == [2, 64, 64]
    assert bridge.calls[0][2]["visual_tokens"] == 2048
    assert bridge.calls[0][2]["resized_hw"] == [896, 896]


def test_inference_error_also_restores_original_forward():
    bridge = make_bridge("hermes_chunk")
    def fail(*_args, **_kwargs):
        raise RuntimeError("synthetic inference failure")
    bridge.forward_chunk = fail
    installer()(bridge, 896)
    with pytest.raises(RuntimeError, match="synthetic"):
        bridge.encode_video_chunk(torch.zeros(1, 56, 112, 3, dtype=torch.uint8))
    assert bridge.forward_chunk is fail


def test_disabled_profile_is_no_op_and_pixel_cap_conflict_fails():
    bridge = make_bridge("hermes_chunk")
    original = bridge.encode_video_chunk
    assert installer()(bridge, None) is bridge
    assert bridge.encode_video_chunk == original
    bridge.video_max_pixels = 151200
    with pytest.raises(ValueError, match="cannot be combined"):
        installer()(bridge, 896)
    bridge.video_max_pixels = None
    bridge.model_config.model_type = "qwen3_vl"
    with pytest.raises(ValueError, match="Qwen2.5"):
        installer()(bridge, 896)


@pytest.mark.parametrize("method", ["adakv", "pyramidkv", "lukv", "sparsemm", "uniform", "protokv"])
def test_every_native_method_receives_fixed_size_outside_streaming_config(tmp_path, method):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen2_5_vl"}')
    anno = tmp_path / "annotations.json"
    anno.write_text("[]")
    args = SimpleNamespace(model_path=model, dtype="bfloat16", sample_fps=.5,
                           encode_chunk_size=16, fixed_frame_size=896, video_max_pixels=None,
                           max_videos=1, max_questions=1, five_video_timing=False, anno_path=anno)
    config = StreamingConfig(method=method)
    with patch("sparsemm_protokv.run_streamingbench.prepare_annotations", return_value=[]):
        command, _env = build_command(args, config, tmp_path / "out")
    assert "fixed_frame_size=896" in command[command.index("--model_args") + 1]
    assert "fixed_frame_size" not in config.asdict()
    args.video_max_pixels = 151200
    with pytest.raises(ValueError, match="cannot be combined"):
        build_command(args, config, tmp_path / "out")


def test_controller_rejects_invalid_size_and_parser_rejects_conflicting_overrides(tmp_path):
    parser = build_parser()
    assert parser.parse_args(["--fixed-frame-size", "896"]).fixed_frame_size == 896
    with pytest.raises(SystemExit):
        parser.parse_args(["--fixed-frame-size", "896", "--video-max-pixels", "151200"])
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"llava_onevision"}')
    args = SimpleNamespace(model_path=model, fixed_frame_size=896, video_max_pixels=None)
    with pytest.raises(ValueError, match="Qwen2.5"):
        build_command(args, StreamingConfig(method="adakv"), tmp_path)
    (model / "config.json").write_text('{"model_type":"qwen2_5_vl"}')
    args.fixed_frame_size = 895
    with pytest.raises(ValueError, match="divisible"):
        build_command(args, StreamingConfig(method="adakv"), tmp_path)


def test_native_source_manifest_pins_shared_pixel_preparation():
    assert "kvbench/fixed_inputs.py" in source_hashes()
