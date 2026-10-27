"""CPU regression checks for sparse-tail input without changing frame pairing."""

from contextlib import nullcontext
import hashlib
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from sparsemm_baselines.adapter import SparseMMQwen
from sparsemm_baselines.run_streamingbench import sampled_indices
from sparsemm_native import recent_tail as tail
from sparsemm_native.decoupled import VisualUnit, dense_prefix
from sparsemm_protokv.config import StreamingConfig


@pytest.mark.parametrize('target', range(1, 10))
def test_plan_retains_original_final_two_units_and_global_ordinals(target):
    plan = tail.sampling_plan(1200, 29.97, target * 2, .5)
    original_units = [list(range(start, min(start + 2, target)))
                      for start in range(0, target, 2)]
    last_units = original_units[-2:]
    assert plan['unit_start_ordinals'] == [unit[0] for unit in last_units]
    assert plan['tail_ordinals'] == [index for unit in last_units for index in unit]
    assert plan['tail_raw_indices'] == [60 * index for index in plan['tail_ordinals']]
    assert plan['start_ordinal'] % 2 == 0
    assert len(plan['tail_ordinals']) <= 4


def test_sampler_rounding_and_source_end_are_unchanged():
    # The established sampler rounds 29.97 to 30 before deriving step 60.
    plan = tail.sampling_plan(601, 29.97, 1000, .5)
    assert plan['prefix_indices'] == sampled_indices(601, 29.97, .5)
    assert plan['prefix_indices'] == list(range(0, 601, 60))
    assert plan['sampled_prefix_frames'] == 11
    assert plan['tail_raw_indices'] == [480, 540, 600]
    assert tail.sampling_plan(1000, 29.97, 20.1, .5)['sampled_prefix_frames'] == 11
    with pytest.raises(ValueError, match='no sampled frame'):
        tail.sampling_plan(1000, 29.97, 0, .5)


def prefix_cache():
    cache = tail.RecentOnlyCache(
        StreamingConfig(method='sparsemm', budget=500, budget_scope='visual'),
        28, 4, 4, 'cpu', torch.float32)
    positions = torch.arange(5).float().expand(3, -1)
    cache.set_context('init', positions)
    keys = torch.zeros(1, 4, 5, 4)
    for layer in range(28):
        cache.append_layer(layer, keys, keys.clone())
    cache.finish_init(5)
    cache.prefix_dense = dense_prefix(cache)
    return cache


def test_recent_cache_discards_old_units_without_altering_prefix_or_phase():
    cache = prefix_cache()
    prefix = {name: (value, value.clone()) for name, value in cache.state_tensors().items()}
    for index in range(8):
        unit = VisualUnit(torch.full((1, 6, 4), float(index)),
                          torch.full((3, 1, 6), float(index * 8 + 5)),
                          (1, 4, 6), index * 2, 2)
        assert cache.push_unit(unit) is None
    assert [unit.frame_start for unit in cache.units] == [12, 14]
    assert cache.history_units == 0
    assert cache.phase == 'init'
    assert cache.prefix_tokens == cache.logical_tokens == 5
    current = cache.state_tensors()
    # Visual features are tracked by state_tensors too; only the saved prefix
    # tensors must retain both their ownership and byte contents.
    for name, (original, saved) in prefix.items():
        assert current[name] is original
        assert torch.equal(current[name], saved)
    assert all(int(length) == 5 for layer in cache.layers for length in layer.head_lens)


class ProcessorInputs(dict):
    def to(self, *args, **kwargs):
        return self


class Processor:
    def __call__(self, *, text, videos, return_tensors, **kwargs):
        assert text == [''] and return_tensors == 'pt'
        # No size override is allowed for the KVBench-default input test.
        assert not kwargs
        return ProcessorInputs(video_grid_thw=torch.tensor([[1, 4, 6]]),
                               pixel_values_videos=videos.float())


def fake_bridge():
    bridge = object.__new__(tail.RecentTailQwen)
    bridge.video_max_pixels = None
    bridge.sample_fps = .5
    bridge.model_config = SimpleNamespace(vision_config=SimpleNamespace(
        spatial_merge_size=2, temporal_patch_size=2, tokens_per_second=2., patch_size=2))
    bridge.processor = Processor()
    bridge.model = SimpleNamespace(device=torch.device('cpu'),
                                  parameters=lambda: iter([torch.zeros(1)]),
                                  get_video_features=lambda pixels, grid:
                                  [torch.full((6, 4), float(pixels.mean()))])
    bridge.timings = dict(raw_frames=0, visual_tokens=0, video_prefills=0)
    bridge._measure = lambda field: nullcontext()
    bridge.predict_and_compress = lambda: None
    bridge.forward_chunk = Mock(side_effect=AssertionError('History forward must never run'))

    def initialize():
        bridge.kv_cache = prefix_cache()
        bridge._tail_rgb = None
        bridge.visual_start_idx = bridge._next_position = 5
        bridge._seen_visual_tokens = bridge.total_processed_frames = 0
        bridge.last_encoded_frames = bridge._prefills = bridge._compressions = 0

    def answer(query, max_new_tokens):
        units = bridge.kv_cache.units
        one = bridge._last_frame_unit(bridge.kv_cache)
        return {'response': 'A', 'tail_audit': {
            'unit_start_ordinals': [unit.frame_start for unit in units],
            'positions': torch.cat([unit.positions for unit in units], dim=2),
            'single_positions': one.positions,
            'question_position': max(bridge._next_position, int(one.positions.max()) + 1),
            'features': torch.cat([unit.features for unit in units], dim=1),
            'single_features': one.features}}

    bridge.encode_init_prompt = initialize
    bridge.answer_recent = answer
    bridge.clear_cache = lambda: None
    initialize()
    return bridge


class Reader:
    def __init__(self, path, num_threads=1):
        assert num_threads == 1

    def __len__(self):
        return 1200

    def get_avg_fps(self):
        return 29.97

    def get_batch(self, indices):
        frames = np.stack([np.full((4, 6, 3), index % 251, dtype=np.uint8)
                           for index in indices])
        return SimpleNamespace(asnumpy=lambda: frames)


@pytest.mark.parametrize('target', range(1, 10))
def test_tail_matches_full_float_positions_features_and_one_frame_without_history(monkeypatch, target):
    monkeypatch.setattr(tail, 'BatchScopedVideoReader', Reader)
    bridge = fake_bridge()
    query = {'video_path': 'synthetic.mp4', 'clip_end': target * 2}
    full = tail.run_recent_query(bridge, query, 'full_prefix')['tail_audit']
    sparse = tail.run_recent_query(bridge, query, 'tail_only')['tail_audit']
    for name in ('positions', 'single_positions', 'features', 'single_features'):
        assert torch.equal(full[name], sparse[name])
        assert tail.tensor_digest(full[name]) == tail.tensor_digest(sparse[name])
    assert full['positions'].dtype == sparse['positions'].dtype == torch.float32
    assert full['single_positions'].dtype == sparse['single_positions'].dtype == torch.int64
    assert full['question_position'] == sparse['question_position']
    assert full['tail_raw_indices'] == sparse['tail_raw_indices']
    assert full['unit_start_ordinals'] == sparse['unit_start_ordinals']
    assert full['encoded_real_frames'] == target
    assert sparse['encoded_real_frames'] == min(target, 3 if target % 2 else 4)
    bridge.forward_chunk.assert_not_called()


def test_native_video_positions_use_absolute_ordinal_not_retained_length():
    bridge = fake_bridge()
    bridge.total_processed_frames = 100
    positions = SparseMMQwen._video_positions(bridge, [1, 4, 6], 6)
    assert positions.dtype == torch.float32
    assert positions[0, 0].tolist() == [405.] * 6
    assert positions[1, 0].tolist() == [5., 5., 5., 6., 6., 6.]
    assert positions[2, 0].tolist() == [5., 6., 7., 5., 6., 7.]


def test_tensor_digest_supports_bfloat16_and_noncontiguous_tensors_without_casting():
    value = torch.tensor([[1., -2.], [3., 4.]], dtype=torch.bfloat16).T
    expected = hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    assert tail.tensor_digest(value) == expected
    assert tail.tensor_digest(value.clone()) == expected
    assert tail.tensor_digest(value.float()) != expected
