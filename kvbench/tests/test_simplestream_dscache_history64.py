import pytest
import torch

from sparsemm_native.simplestream_dscache_history64 import (
    allocation,
    anchor_positions_to_recent,
    recent_probe_position,
)
from sparsemm_native.dscache_pae_history import compact_positions
from sparsemm_native import run_simplestream_dscache_history64 as runner
from sparsemm_native import simplestream_recent_share_attention as recent
from sparsemm_native import simplestream_headmix_attention as headmix
from pathlib import Path
import json


@pytest.mark.parametrize('old_frames', [0, 1, 127, 128, 600])
def test_recent_probe_follows_virtual_image_positions(old_frames):
    prefix = 4
    history_tokens = old_frames * 1024
    history_end = prefix + history_tokens
    recent_length = prefix + 4 * 1024
    positions = torch.zeros((3, 1, history_end + 1), dtype=torch.long)
    positions[:, :, history_end] = prefix + old_frames * 32

    probe = recent_probe_position(positions, prefix, history_end, recent_length)

    assert probe == recent_length + old_frames * 32
    assert probe - positions[0, 0, history_end].item() == 4 * 1024


def test_recent_probe_rejects_invalid_multimodal_start():
    positions = torch.tensor([[[4]], [[5]], [[4]]])
    with pytest.raises(RuntimeError, match='M-RoPE'):
        recent_probe_position(positions, 4, 0, 4100)


@pytest.mark.parametrize('old_frames', [0, 1, 127, 128, 600])
def test_anchor_restores_recent_positions_and_preserves_history_spacing(old_frames):
    prefix = 4
    old_positions = torch.arange(old_frames * 32, dtype=torch.long)
    first_recent = prefix + old_frames * 32
    visual = torch.cat((old_positions + prefix,
                        torch.arange(first_recent, first_recent + 128)))
    positions = torch.cat((torch.arange(prefix), visual,
                           torch.arange(first_recent + 128, first_recent + 138)))
    positions = positions.view(1, 1, -1).expand(3, -1, -1).clone()
    recent_start = prefix + old_frames * 32

    anchored, shift = anchor_positions_to_recent(positions, prefix, recent_start)

    assert shift == old_frames * 32
    assert anchored[0, 0, recent_start].item() == prefix
    assert anchored[0, 0, -1].item() == prefix + 137
    assert torch.equal(anchored[:, :, :prefix], positions[:, :, :prefix])
    if old_frames:
        assert anchored[0, 0, prefix].item() == prefix - shift
        assert torch.equal(anchored[:, :, prefix:] - anchored[:, :, prefix:prefix+1],
                           positions[:, :, prefix:] - positions[:, :, prefix:prefix+1])


def test_stable_arm_keeps_the_strict_visual_budget():
    pure = allocation('strict_k500')
    stable = allocation('strict_k500_stable')
    assert stable == pure
    assert sum(v['total_quota'] for v in stable.values()) == 56_000


def test_position_agnostic_recent_control_has_no_history_and_k500():
    recent = allocation('pae_recent_control')
    assert recent == allocation('recent_control')
    assert sum(v['total_quota'] for v in recent.values()) == 56_000


def test_compact_history_arms_share_k500_and_differ_only_in_history():
    control = allocation('pae_compact_control')
    history = allocation('pae_history_k500')
    assert len(control) == len(history) == 112
    assert sum(v['total_quota'] for v in control.values()) == 56_000
    assert sum(v['total_quota'] for v in history.values()) == 56_000
    assert sum(v['history_quota'] for v in control.values()) == 0
    assert sum(v['history_quota'] for v in history.values()) == 56 * 64


def test_compact_history_diagnostic_adds_history_without_eviction():
    control = allocation('pae_compact_control')
    diagnostic = allocation('pae_history_k532_diagnostic')
    assert sum(v['total_quota'] for v in diagnostic.values()) == 59_584
    for head, row in diagnostic.items():
        assert row['recent_quota'] == control[head]['recent_quota'] == 500
        assert row['history_quota'] in (0, 64)


def test_release_uses_local_frozen_tables_and_ragged_helper():
    root = Path(__file__).resolve().parents[1]
    assert recent.SOURCE_TABLE == root / 'assets/recent1or4/per_head_allocations.csv'
    assert headmix.SOURCE_TABLE == root / 'assets/dp/per_head_allocation_dual_k500.csv'
    assert runner.old.RAGGED == root / 'sparsemm_native/hermes_kv_head_ragged.py'


def test_baseline_must_be_explicit_and_complete(tmp_path):
    query = {'query_id': '1117'}
    missing = tmp_path / 'missing.jsonl'
    with pytest.raises(FileNotFoundError, match='Baseline'):
        runner.load_baseline(missing, [query])
    baseline = tmp_path / 'queries.jsonl'
    baseline.write_text(json.dumps({'query_id': '1117', 'input_audit': {
        'selected_raw_frame_indices': [0], 'selected_chunk_ids': [0],
        'prompt_sha256': 'abc'}}) + '\n')
    assert runner.load_baseline(baseline, [query])['1117']['input_audit']['prompt_sha256'] == 'abc'
    with pytest.raises(ValueError, match='Missing completed'):
        runner.load_baseline(baseline, [{'query_id': 'other'}])


def test_compact_positions_share_frame_ranks_across_heads():
    frames = [1, 9, 600]
    first = compact_positions([-2, -1, 9 * 1024 + 7, 600 * 1024 + 1023],
                              2, device='cpu', frame_order=frames)
    second = compact_positions([-2, -1, 1 * 1024, 9 * 1024 + 7],
                               2, device='cpu', frame_order=frames)
    assert torch.equal(first[:, :, 2], second[:, :, 3])
    assert first[:, :, 3].max().item() == 2 + 2 * 32 + 31
    assert first.max().item() < 128000
