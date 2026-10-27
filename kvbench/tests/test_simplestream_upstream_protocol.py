"""Checks for the protocol wrapper; inference remains in the original tree."""
import json
from types import SimpleNamespace

from sparsemm_native.supervise_simplestream_upstream import (
    full_window_indices, read_jsonl, worker_env,
)


def test_recent_chunks_can_contain_more_than_four_frames():
    # A frame-count truncation would silently change the released chunk window.
    plan = SimpleNamespace(raw_fps=10, full_indices=[0, 10, 20, 30, 40, 41],
                           tail_indices=[20, 30, 40, 41])
    indices, chunks = full_window_indices(plan)
    assert chunks == [1, 2, 3, 4]
    assert indices == [10, 20, 30, 40, 41]
    assert indices != plan.tail_indices  # Must fall back to the original path.


def test_short_clip_keeps_all_frames_even_when_in_one_chunk():
    plan = SimpleNamespace(raw_fps=30, full_indices=[0, 5, 10, 15])
    indices, chunks = full_window_indices(plan)
    assert indices == [0, 5, 10, 15]
    assert chunks == [0]


def test_sampling_uses_source_frame_timestamps_instead_of_ordinals():
    plan = SimpleNamespace(raw_fps=29.97, full_indices=[0, 31, 62, 93, 124, 155])
    indices, chunks = full_window_indices(plan)
    assert indices == [62, 93, 124, 155]
    assert chunks == [2, 3, 4, 5]


def test_previous_experiment_overrides_are_removed(monkeypatch):
    for key in ('MIN_PIXELS', 'MAX_PIXELS', 'MODEL_SEQ_LEN', 'WORLD_SIZE',
                'PYTHONPATH', 'FORCE_QWENVL_VIDEO_READER'):
        monkeypatch.setenv(key, 'wrong_previous_setting')
    env = worker_env()
    assert all(key not in env for key in ('MIN_PIXELS', 'MAX_PIXELS', 'MODEL_SEQ_LEN',
               'WORLD_SIZE', 'PYTHONPATH', 'FORCE_QWENVL_VIDEO_READER'))
    assert env['HF_HUB_OFFLINE'] == '1'


def test_interrupted_checkpoint_tail_is_not_counted(tmp_path):
    path = tmp_path / 'checkpoint.jsonl'
    path.write_text(json.dumps({'id': 1}) + '\n' + '{"id":')
    assert read_jsonl(path) == [{'id': 1}]
