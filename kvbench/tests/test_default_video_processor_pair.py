"""Both recent-window paths preserve the checkpoint's default resize policy.

Exercise the real encoding methods with CPU-only processor/model doubles. The
default result has more features than the explicit low-pixel configuration,
making a misplaced size override visible in the constructed visual units.
"""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from sparsemm_native.decoupled import DecoupledRecent4StreamingQwen
from sparsemm_native.mild_pseudo_history_fps05 import MildPseudoHistoryQwen


class Inputs(dict):
    def to(self, device, dtype):
        assert torch.device(device).type == 'cpu'
        assert dtype == torch.float32
        return self


class Processor:
    def __init__(self):
        self.video_processor = SimpleNamespace(size={
            'shortest_edge': 128 * 28 * 28,
            'longest_edge': 768 * 28 * 28,
        })
        self.calls = []

    def __call__(self, *, text, videos, return_tensors, **kwargs):
        assert text == ['']
        assert return_tensors == 'pt'
        assert videos.shape[1] == 3
        self.calls.append((videos.clone(), kwargs))
        # Two different grids ensure the feature/position lengths follow the
        # actual processor result rather than assuming the old pixel limit.
        grid = (1, 2, 4) if 'size' in kwargs else (1, 4, 6)
        return Inputs(video_grid_thw=torch.tensor([grid]),
                      pixel_values_videos=videos)


def fake_encoder(cls, pixels):
    encoder = object.__new__(cls)
    encoder.processor = Processor()
    encoder.video_max_pixels = pixels
    encoder._measure = lambda name: nullcontext()
    encoder.model = SimpleNamespace(
        device=torch.device('cpu'),
        parameters=lambda: iter([torch.zeros(1, dtype=torch.float32)]),
        get_video_features=lambda values, grid: [
            torch.zeros(int(grid[0].prod()) // 4, 8)])
    encoder.model_config = SimpleNamespace(vision_config=SimpleNamespace(
        spatial_merge_size=2, tokens_per_second=2))
    encoder.sample_fps = .5
    encoder.visual_start_idx = 11
    return encoder


def assert_resize_call(encoder, pixels):
    for _, kwargs in encoder.processor.calls:
        if pixels is None:
            assert kwargs == {}  # No size=None or partial size dictionary.
        else:
            assert kwargs == {'size': {
                'shortest_edge': encoder.processor.video_processor.size['shortest_edge'],
                'longest_edge': pixels,
            }}
    assert encoder.processor.video_processor.size['longest_edge'] == 768 * 28 * 28


@pytest.mark.parametrize('pixels', [None, 151200])
def test_two_frame_stream_uses_default_or_explicit_resize(pixels):
    encoder = fake_encoder(DecoupledRecent4StreamingQwen, pixels)
    units = []
    encoder.kv_cache = SimpleNamespace(push_unit=lambda unit: units.append(unit))
    encoder.total_processed_frames = encoder._next_position = encoder._seen_visual_tokens = 0
    encoder.timings = dict(raw_frames=0, visual_tokens=0, video_prefills=0)
    encoder._video_positions = lambda grid, count: torch.zeros(3, 1, count, dtype=torch.long)
    frames = torch.arange(4 * 8 * 10 * 3).reshape(4, 8, 10, 3)

    encoder.encode_video_chunk(frames)

    assert len(units) == len(encoder.processor.calls) == 2
    assert_resize_call(encoder, pixels)
    tokens = 6 if pixels is None else 2
    assert [unit.tokens for unit in units] == [tokens, tokens]
    assert [unit.frame_start for unit in units] == [0, 2]
    assert [unit.real_frames for unit in units] == [2, 2]
    for offset, (passed, _) in zip((0, 2), encoder.processor.calls):
        assert torch.equal(passed, frames[offset:offset + 2].permute(0, 3, 1, 2))
    assert encoder.total_processed_frames == encoder.last_encoded_frames == 4
    assert encoder.timings['visual_tokens'] == tokens * 2


@pytest.mark.parametrize('pixels', [None, 151200])
def test_single_frame_recent_rebuild_uses_same_resize_policy(pixels):
    encoder = fake_encoder(MildPseudoHistoryQwen, pixels)
    encoder.total_processed_frames = 8
    frame = torch.arange(8 * 10 * 3).reshape(1, 8, 10, 3)

    unit = encoder._last_frame_unit(SimpleNamespace(latest_raw_frame=frame))

    assert len(encoder.processor.calls) == 1
    assert_resize_call(encoder, pixels)
    assert torch.equal(encoder.processor.calls[0][0], frame.permute(0, 3, 1, 2))
    assert unit.tokens == (6 if pixels is None else 2)
    assert unit.positions.shape == (3, 1, unit.tokens)
    assert bool((unit.positions[0] == 39).all())  # (7 / fps .5) * 2 + prefix 11.
    assert unit.frame_start == 7
    assert unit.real_frames == 1
