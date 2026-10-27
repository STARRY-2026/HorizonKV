"""CPU-only checks for the precise continuous-prefix position intervention."""

from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel

from sparsemm_native.decoupled import VisualUnit
from sparsemm_native.epm_position_probe import position_variant


def fixture(count=2):
    config = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2, tokens_per_second=2),
        vision_start_token_id=100, image_token_id=101,
        video_token_id=102, vision_end_token_id=103)
    multimodal = SimpleNamespace(config=config)
    multimodal.get_rope_index = lambda **kwargs: Qwen2_5_VLModel.get_rope_index(
        multimodal, **kwargs)
    bridge = SimpleNamespace(
        model_config=config, model=SimpleNamespace(model=multimodal),
        init_prompt_ids=torch.tensor([[1, 2, 100]]), visual_start_idx=3,
        _next_position=250)
    units = [VisualUnit(torch.full((1, 6, 4), float(index)),
                        torch.full((3, 1, 6), float(200 + index)),
                        (1, 4, 6), 100 + 2 * index, 2)
             for index in range(count)]
    one = VisualUnit(torch.ones(1, 6, 4), torch.full((3, 1, 6), 300),
                     (1, 4, 6), 100 + 2 * count - 1, 1)
    return bridge, one, units


def test_native_preserves_exact_cursor_dtype_and_value_without_tensor_aliases():
    bridge, one, units = fixture()
    copied_one, copied_units, cursor = position_variant(bridge, one, units, 'native')
    assert cursor == 301
    bridge._next_position = 400
    assert position_variant(bridge, one, units, 'native')[2] == 400
    for old, copied in zip([one, *units], [copied_one, *copied_units]):
        assert copied is not old
        assert copied.features is old.features
        assert copied.positions is not old.positions
        assert copied.positions.dtype == old.positions.dtype
        assert torch.equal(copied.positions, old.positions)


def test_legacy_has_one_3d_grid_then_contiguous_1d_visuals_and_shared_one_slice():
    bridge, one, units = fixture()
    copied_one, copied_units, cursor = position_variant(
        bridge, one, units, 'simplestream_contiguous')
    expected_first = torch.tensor([
        [3, 3, 3, 3, 3, 3], [3, 3, 3, 4, 4, 4], [3, 4, 5, 3, 4, 5]])[:, None]
    expected_last = torch.arange(6, 12).view(1, 1, -1).expand(3, 1, -1)
    assert torch.equal(copied_units[0].positions, expected_first)
    assert torch.equal(copied_units[1].positions, expected_last)
    assert torch.equal(copied_one.positions, expected_last)
    assert copied_one.positions.data_ptr() != copied_units[-1].positions.data_ptr()
    assert cursor == 12
    for old, copied in zip([one, *units], [copied_one, *copied_units]):
        assert copied.features is old.features
        assert copied.grid is old.grid
        assert copied.frame_start == old.frame_start
        assert copied.real_frames == old.real_frames
        assert copied is not old
    assert bool((units[0].positions == 200).all())
    assert bool((units[1].positions == 201).all())
    assert bool((one.positions == 300).all())
    assert bridge._next_position == 250


def test_legacy_single_unit_has_only_3d_grid_and_one_uses_entire_grid():
    bridge, one, units = fixture(count=1)
    copied_one, copied_units, cursor = position_variant(
        bridge, one, units, 'simplestream_contiguous')
    assert cursor == 6
    assert torch.equal(copied_one.positions, copied_units[0].positions)
    assert not torch.equal(copied_one.positions[0], copied_one.positions[1])


def test_legacy_positions_ignore_global_frame_ordinals_and_absolute_native_cursor():
    bridge, one, units = fixture()
    first = position_variant(bridge, one, units, 'simplestream_contiguous')
    for unit in [one, *units]:
        unit.frame_start += 1000
        unit.positions.add_(10000)
    bridge._next_position += 10000
    other = position_variant(bridge, one, units, 'simplestream_contiguous')
    assert first[2] == other[2]
    assert torch.equal(first[0].positions, other[0].positions)
    assert all(torch.equal(left.positions, right.positions)
               for left, right in zip(first[1], other[1]))


@pytest.mark.parametrize('error', ('missing_start', 'double_start', 'wrong_prefix',
                                 'bad_unit_grid', 'different_one_grid', 'unknown_variant'))
def test_invalid_probe_boundaries_are_rejected(error):
    bridge, one, units = fixture()
    if error == 'missing_start':
        bridge.init_prompt_ids[0, -1] = 9
    elif error == 'double_start':
        bridge.init_prompt_ids[0, 0] = 100
    elif error == 'wrong_prefix':
        bridge.visual_start_idx += 1
    elif error == 'bad_unit_grid':
        units[0].grid = (2, 4, 6)
    elif error == 'different_one_grid':
        one.grid = (1, 6, 4)
    with pytest.raises(ValueError):
        position_variant(bridge, one, units,
                         'invalid' if error == 'unknown_variant' else 'simplestream_contiguous')
