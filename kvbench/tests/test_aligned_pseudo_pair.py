"""Paired visual-budget, cache-ownership and released-decoder invariants.

Synthetic inputs exercise the same per-head assembly as the small OVO run,
without loading a checkpoint or allocating CUDA memory.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sparsemm_native import aligned_pseudo_pair as paired
from sparsemm_native.decoupled import VisualUnit
from sparsemm_native.simplestream_recent_share_attention import HEADS


def fixture(one_tokens=8, four_tokens=16, historical=True):
    prefix, dim = 2, 4
    one = [(torch.full((1, 4, prefix + one_tokens, dim), 11.),
            torch.full((1, 4, prefix + one_tokens, dim), 12.)) for _ in range(28)]
    four = [(torch.full((1, 4, prefix + four_tokens, dim), 41.),
             torch.full((1, 4, prefix + four_tokens, dim), 42.)) for _ in range(28)]
    unit = VisualUnit(torch.zeros(1, one_tokens, dim),
                      torch.full((3, 1, one_tokens), 200), (1, 1, one_tokens), 9, 1)
    full = VisualUnit(torch.zeros(1, four_tokens, dim),
                      torch.full((3, 1, four_tokens), 100), (1, 1, four_tokens), 6, 2)
    layers = []
    for layer in range(28):
        segments, positions = [], []
        for head in range(4):
            h = int(paired.ROWS[layer, head]['history_quota']) if historical else 0
            keys = torch.full((prefix + h, dim), -1.)
            values = torch.full((prefix + h, dim), -2.)
            segments.append((keys, values))
            positions.append(torch.full((3, prefix + h), 10))
        layers.append(SimpleNamespace(
            segments=segments, positions=positions,
            get_segment=lambda head, s=segments: s[head],
            get_position_segment=lambda head, p=positions: p[head]))
    history = SimpleNamespace(prefix_tokens=prefix, num_kv_heads=4,
                              layers=layers, units=[full])
    probes = [(torch.ones(8, 2, dim), torch.ones(4, 2, dim)) for _ in range(28)]
    return history, one, four, unit, probes


def test_allocation_preserves_old_head_split_and_global_mean():
    for arm in paired.ARMS:
        rows = paired.allocation(arm)
        assert sum(int(row['total_quota']) for row in rows.values()) == 56000
        assert {key for key, row in rows.items() if row['recent_group'] == 'one_frame'} == set(HEADS)
    assert all(int(row['recent_quota']) == 500 and int(row['history_quota']) == 0
               for row in paired.allocation('recent_only').values())


def test_underfilled_recent_is_identical_and_history_is_the_only_added_kv():
    data = fixture(one_tokens=192, four_tokens=384)
    recent, r = paired.assemble(*data, 'recent_only')
    historical, h = paired.assemble(*data, 'mild_history')
    assert r['actual_history_visual_slots'] == 0
    assert r['actual_recent_visual_slots'] == 56 * (192 + 384)
    assert h['actual_recent_visual_slots'] == r['actual_recent_visual_slots']
    assert h['actual_history_visual_slots'] == sum(int(row['history_quota']) for row in paired.ROWS.values())
    assert r['recent_selection_sha256'] == h['recent_selection_sha256']
    assert r['recent_pruned_heads'] == h['recent_pruned_heads'] == 0
    assert h['actual_visual_slots'] - r['actual_visual_slots'] == h['actual_history_visual_slots']
    for index, (left, right) in enumerate(zip(recent.layers, historical.layers)):
        for head in range(4):
            lk, _ = left.get_segment(head)
            hk, _ = right.get_segment(head)
            count = int(paired.ROWS[index, head]['history_quota'])
            assert torch.equal(lk[2:], hk[2 + count:])
            expected = 11. if (index, head) in HEADS else 41.
            assert bool((lk[2:] == expected).all())


def test_ragged_arm_owns_storage_and_does_not_modify_shared_state():
    data = fixture()
    first, _ = paired.assemble(*data, 'recent_only')
    second, _ = paired.assemble(*data, 'mild_history')
    second_before = second.layers[0].flat_k.clone()
    first.layers[0].flat_k.fill_(999.)
    first.layers[0].flat_v.fill_(999.)
    assert torch.equal(second.layers[0].flat_k, second_before)
    assert bool((data[0].layers[0].get_segment(0)[0] == -1.).all())
    assert bool((data[1][0][0] == 11.).all())
    assert bool((data[2][0][0] == 41.).all())


@pytest.mark.parametrize('historical', [False, True])
def test_large_candidates_obey_quotas_and_history_shortage_returns_to_recent(historical):
    data = fixture(one_tokens=600, four_tokens=1200, historical=historical)
    for arm in paired.ARMS:
        cache, audit = paired.assemble(*data, arm)
        assert audit['actual_visual_slots'] == 56000
        assert audit['recent_pruned_heads'] == 112
        for layer_index, layer in enumerate(cache.layers):
            for head, total in enumerate(audit['per_head_visual_slots'][layer_index]):
                row = paired.allocation(arm)[layer_index, head]
                assert total == int(row['total_quota'])
                h = int(row['history_quota']) if historical else 0
                k, _ = layer.get_segment(head)
                assert k.shape[0] == 2 + h + (total - h)


def test_temporal_overlap_is_rejected():
    data = fixture()
    for layer in data[0].layers:
        for positions in layer.positions:
            positions.fill_(10000)
    with pytest.raises(AssertionError, match='time ranges overlap'):
        paired.assemble(*data, 'mild_history')


def test_released_decoder_penalizes_prompt_and_stops_on_either_configured_eos():
    # Token 2 is largest before the prompt-scope penalty; token 7 becomes the
    # winner afterwards and is the second EOS in the checkpoint configuration.
    logits = torch.tensor([[[0., 0., 10., 0., 0., 0., 0., 9.8]]])
    model = SimpleNamespace(
        generation_config=SimpleNamespace(repetition_penalty=1.05, eos_token_id=[5, 7]),
        get_input_embeddings=lambda: lambda ids: torch.zeros(1, ids.shape[1], 4),
        lm_head=lambda hidden: logits)
    bridge = SimpleNamespace(
        model=model, device='cpu', num_layers=28,
        processor=SimpleNamespace(batch_decode=lambda ids, **kwargs: [str(ids[0])]))
    with patch.object(paired, '_prefill_question_suffix', return_value=torch.zeros(1, 2, 4)), \
         patch.object(paired, '_ragged_helpers', return_value=(None, None, None)):
        response, audit = paired.AlignedPseudoPairQwen.decode_arm(
            bridge, object(), [1, 3], 101, [2], max_new_tokens=4)
    assert response == '[7]'
    assert audit['generated_tokens'] == 1
    assert audit['eos_token_ids'] == [5, 7]
    assert audit['hit_generation_limit'] is False
