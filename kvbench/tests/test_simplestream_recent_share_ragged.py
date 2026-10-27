"""Physical two/four-frame head-cache invariants for the SimpleStream ablation."""

import csv
import sys
from pathlib import Path

import torch

from sparsemm_native.simplestream_recent_share_ragged import (
    HEADS, retained_position_indices,
)
from sparsemm_native.simplestream_recent_share_ragged_reverse import HEADS as REVERSED_HEADS
from sparsemm_native.simplestream_recent_share_ragged_gt70 import (
    HEADS as GT70_TWO_FRAME_HEADS, SOURCE_TABLE as GT70_SOURCE_TABLE,
)
from sparsemm_native.simplestream_recent_share_independent_last1 import (
    HEADS as SINGLE_FRAME_HEADS, _combine_prefix_caches,
    single_frame_position_indices,
)
from sparsemm_native.simplestream_recent_share_independent_last1_reverse import (
    HEADS as REVERSE_SINGLE_FRAME_HEADS,
)


def test_recent_share_split_is_disjoint_and_complete():
    assert len(HEADS) == len(set(HEADS)) == 56
    assert all(0 <= layer < 28 and 0 <= head < 4 for layer, head in HEADS)
    assert len({(layer, head) for layer in range(28) for head in range(4)} - set(HEADS)) == 56


def test_reversed_split_swaps_every_head_once():
    assert len(REVERSED_HEADS) == len(set(REVERSED_HEADS)) == 56
    assert not set(HEADS) & set(REVERSED_HEADS)
    assert set(HEADS) | set(REVERSED_HEADS) == {
        (layer, head) for layer in range(28) for head in range(4)}


def test_gt70_split_uses_strict_recent_share_threshold():
    with GT70_SOURCE_TABLE.open(newline='') as source:
        rows = [row for row in csv.DictReader(source) if int(row['frames']) == 4]
    two_frame = {(int(row['layer']), int(row['kv_head'])) for row in rows
                 if 10 * int(row['recent_r']) <= 7 * int(row['capacity_n'])}
    four_frame = {(int(row['layer']), int(row['kv_head'])) for row in rows
                  if 10 * int(row['recent_r']) > 7 * int(row['capacity_n'])}
    assert set(GT70_TWO_FRAME_HEADS) == two_frame
    assert len(two_frame) == 87
    assert len(four_frame) == 25
    assert not two_frame & four_frame


def test_independent_single_frame_branch_uses_its_own_prefix_kv():
    assert set(SINGLE_FRAME_HEADS) == set(HEADS)
    assert single_frame_position_indices(11, 2, 8) == [0, 1, 8, 9, 10]
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'hermes'))
    from head_analysis.hermes_kv_head_ragged import HeadRaggedLayerCache

    full = torch.arange(44, dtype=torch.float32).view(1, 4, 11, 1)
    one = (100 + torch.arange(20, dtype=torch.float32)).view(1, 4, 5, 1)
    layer = _combine_prefix_caches([(full, full)], [(one, one)],
                                   {(0, 0), (0, 2)}, HeadRaggedLayerCache)[0]
    assert layer.head_lens.tolist() == [5, 11, 5, 11]
    assert layer.get_segment(0)[0][:, 0].tolist() == one[0, 0, :, 0].tolist()
    assert layer.get_segment(1)[0][:, 0].tolist() == full[0, 1, :, 0].tolist()


def test_reverse_independent_split_is_exact_complement():
    assert len(REVERSE_SINGLE_FRAME_HEADS) == 56
    assert not set(SINGLE_FRAME_HEADS) & set(REVERSE_SINGLE_FRAME_HEADS)
    assert set(SINGLE_FRAME_HEADS) | set(REVERSE_SINGLE_FRAME_HEADS) == {
        (layer, head) for layer in range(28) for head in range(4)}


def test_ragged_cache_physically_omits_first_two_visual_frames():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'hermes'))
    from head_analysis.hermes_kv_head_ragged import HeadRaggedLayerCache

    # Two text-prefix tokens, four image frames with two tokens each, one text tail.
    low = retained_position_indices(11, 2, 6, True)
    high = retained_position_indices(11, 2, 6, False)
    values = torch.arange(44, dtype=torch.float32).view(1, 4, 11, 1)
    cache = HeadRaggedLayerCache.from_dense_layer(values, values, [low, high, low, high])
    assert cache.head_lens.tolist() == [7, 11, 7, 11]
    low_k, _ = cache.get_segment(0)
    high_k, _ = cache.get_segment(1)
    assert low_k[:, 0].tolist() == [0, 1, 6, 7, 8, 9, 10]
    assert high_k[:, 0].tolist() == list(range(11, 22))
    assert cache.flat_k.numel() == 36
