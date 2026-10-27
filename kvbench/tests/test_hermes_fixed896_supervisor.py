"""Protect HERMES-specific workload, scoring and smoke acceptance semantics."""

from collections import Counter
import json

import pytest

from kvbench.fixed_inputs import fixed_input_metadata
from sparsemm_native import supervise_ovobench_hermes_fixed896 as supervisor


def query(qid='0', frames=17, task='EPM', video='source.mp4'):
    return dict(query_id=qid, video_path=video, clip_end=frames*2, task=task, target='A')


def row(qid='0', frames=17, task='EPM', response='A', units=9, seconds=23.):
    return dict(query_id=qid, frames=frames, task=task, target='A', response=response,
        encoded_temporal_units=units, seconds=seconds,
        input_audit=dict(fixed_input=fixed_input_metadata(896), tokens_per_temporal_unit=1024,
            video_temporal_units=(frames+1)//2, prefix_encoded_visual_tokens=((frames+1)//2)*1024,
            native_video_forward_audits=[dict(raw_frames=16, grid_thw=[8, 64, 64], visual_tokens=8192),
                dict(raw_frames=1, grid_thw=[1, 64, 64], visual_tokens=1024)]),
        budget_audit=dict(actual_tokens_per_head=500., actual_visual_slots=56000,
            per_head_visual_slots=[[500]*4 for _ in range(28)], individual_quota_cap=500,
            actual_includes_question_tokens=False),
        validation=dict({name: True for name in supervisor.VALIDATION_FIELDS},
            history_mode='native_previous_real_qa', history_length_before=0, history_length_after=1))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(value)+'\n' for value in rows))


def test_official_scoring_preserves_raw_case_and_defers_all_macros_until_full_coverage():
    expected = Counter({task: 1 for task in supervisor.TASKS})
    values = [row(str(i), task=task, response='A' if task in supervisor.helpers.BACKWARD else 'B')
        for i, task in enumerate(supervisor.TASKS)]
    partial = supervisor.score(values[:-1], expected)
    assert partial['backward_macro'] is partial['nine_task_macro'] is partial['official_two_group_macro'] is None
    full = supervisor.score(values, expected)
    assert full['correct'] == 3 and full['question_accuracy'] == 1/3
    assert full['nine_task_macro'] == 1/3 and full['official_two_group_macro'] == .5
    assert supervisor.score([row(response='a')], expected)['correct'] == 0
    assert supervisor.score([row(response='A or B')], expected)['correct'] == 1


def test_workload_commits_every_endpoint_and_counts_padding_at_each_new_question_boundary():
    queries = [query(str(i), frames) for i, frames in enumerate((15, 16, 17, 31, 32))]
    all_work = supervisor.estimate_remaining(queries, [])
    # New frame counts 15,1,1,14,1 become 8,1,1,7,1 temporal units. Every
    # endpoint is committed; odd boundaries increase total above ceil(32/2).
    assert all_work['remaining_temporal_units'] == 18
    done = [row('0', frames=15, units=8), row('1', frames=16, units=8)]
    rest = supervisor.estimate_remaining(queries, done)
    assert rest['remaining_temporal_units'] == 9 and rest['remaining_questions'] == 3
    assert rest['seconds'] is None  # Still fewer than three calibration questions.


def test_real_source_metadata_caps_targets_before_native_increment_accounting():
    queries = [query(str(i), frames) for i, frames in enumerate((15, 16, 17, 31, 32))]
    metadata = {'source.mp4': dict(source_frames=30, source_fps=1.)}
    estimate = supervisor.estimate_remaining(queries, [], source_metadata=metadata)
    # Thirty raw frames at 1fps yield only 15 sampled frames at .5fps. The first
    # question encodes 8 units; every later capped endpoint reuses that prefix.
    assert estimate['remaining_temporal_units'] == 8
    assert estimate['known_source_lengths'] == 1


def test_eta_fits_new_units_plus_qa_overhead_and_requires_32_measured_units():
    queries = [query('future', 32)]
    calibration = [row(str(i), units=units, seconds=2*units+5)
        for i, units in enumerate((1, 8, 24))]
    estimate = supervisor.estimate_remaining(queries, [], calibration=calibration)
    assert estimate['remaining_temporal_units'] == 16
    assert estimate['seconds_per_temporal_unit'] == pytest.approx(2.)
    assert estimate['seconds_per_question'] == pytest.approx(5.)
    assert estimate['seconds'] == pytest.approx(37.)
    calibration[-1]['encoded_temporal_units'] = 22
    assert supervisor.estimate_remaining(queries, [], calibration=calibration)['seconds'] is None


def test_smoke_accepts_native_hermes_shared_layer_budget_without_sparsemm_no_cap_rule(tmp_path):
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps([query()]))
    write_rows(tmp_path/'queries.jsonl', [row()])
    checks = supervisor.validate_smoke(tmp_path, manifest)
    assert len(checks) == 1 and all(checks[0]['checks'].values())
    assert 'no_individual_cap' not in checks[0]['checks']


@pytest.mark.parametrize('corruption', ['history', 'history_count', 'positions', 'unequal_heads', 'grid', 'feature_count'])
def test_smoke_rejects_state_leaks_and_invalid_real_visual_inputs(tmp_path, corruption):
    value = row()
    if corruption == 'history':
        value['validation']['real_qa_history_preserved'] = False
    elif corruption == 'history_count':
        value['validation']['history_length_after'] = 0
    elif corruption == 'positions':
        value['validation']['positions_restored'] = False
    elif corruption == 'unequal_heads':
        value['budget_audit']['per_head_visual_slots'][0] = [499, 500, 500, 500]
        value['budget_audit']['actual_visual_slots'] -= 1
        value['budget_audit']['actual_tokens_per_head'] = 55999/112
    elif corruption == 'grid':
        value['input_audit']['native_video_forward_audits'][0]['grid_thw'] = [8, 56, 56]
    else:
        value['input_audit']['native_video_forward_audits'][0]['visual_tokens'] -= 1
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps([query()]))
    write_rows(tmp_path/'queries.jsonl', [value])
    with pytest.raises((ValueError, AssertionError)):
        supervisor.validate_smoke(tmp_path, manifest)


def test_smoke_accepts_extra_temporal_padding_from_committed_odd_question_boundaries(tmp_path):
    value = row()
    value['input_audit']['native_video_forward_audits'] = [
        dict(raw_frames=15, grid_thw=[8, 64, 64], visual_tokens=8192),
        dict(raw_frames=1, grid_thw=[1, 64, 64], visual_tokens=1024),
        dict(raw_frames=1, grid_thw=[1, 64, 64], visual_tokens=1024)]
    value['input_audit']['video_temporal_units'] = 10
    value['input_audit']['prefix_encoded_visual_tokens'] = 10240
    value['validation'].update(history_length_before=2, history_length_after=3)
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps([query()]))
    write_rows(tmp_path/'queries.jsonl', [value])
    assert all(supervisor.validate_smoke(tmp_path, manifest)[0]['checks'].values())


def test_comparison_uses_exact_question_intersections_and_states_precision_limit(tmp_path):
    references = {'sparsemm': tmp_path/'sparsemm', 'simplestream': tmp_path/'simplestream'}
    write_rows(references['sparsemm']/'queries.jsonl', [row('0', response='B'), row('1')])
    write_rows(references['simplestream']/'queries.jsonl', [row('0')])
    output = tmp_path/'output'
    output.mkdir()
    expected = Counter({task: 1 for task in supervisor.TASKS})
    supervisor.report_comparison(output, [row('0')], references, expected)
    result = json.loads((output/'comparison_summary.json').read_text())
    paired = result['comparisons']['sparsemm']
    assert paired['matched_questions'] == 1
    assert paired['hermes']['question_accuracy'] == 1 and paired['baseline']['question_accuracy'] == 0
    assert paired['hermes']['official_two_group_macro'] is None
    assert 'FP16' in result['limitation'] and 'BF16' in result['limitation']
    assert result['causal_cache_comparison'] is False
    altered = row('0')
    altered['input_audit']['fixed_input'] = fixed_input_metadata(448)
    write_rows(references['sparsemm']/'queries.jsonl', [altered])
    with pytest.raises(ValueError, match='metadata differs'):
        supervisor.report_comparison(output, [row('0')], references, expected)


def test_paused_stage_stops_verified_worker_without_spawning_another(tmp_path, monkeypatch):
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps([query()]))
    (tmp_path/'PAUSE').touch()
    record, stopped = dict(pid=123, start_ticks='456', command=['verified']), []
    monkeypatch.setattr(supervisor, 'find_worker', lambda *_args: record)
    monkeypatch.setattr(supervisor, 'stop_worker', lambda value: stopped.append(value))
    monkeypatch.setattr(supervisor.subprocess, 'Popen', lambda *_a, **_kw: pytest.fail('PAUSE launched a worker'))
    assert supervisor.run_stage(tmp_path, 'full', manifest, {}, lambda: False) is False
    assert stopped == [record]
    status = json.loads((tmp_path/'supervisor_status.json').read_text())
    assert status['state'] == 'paused' and status['done'] == 0


def process_identity(command=None, **updates):
    value = dict(pid=123, start_ticks='456', session=123, process_group=123,
        command=supervisor.normalized_command(command or ['/python', '-u', '-m', supervisor.MODULE]))
    value.update(updates)
    return value


def test_worker_exec_empty_cmdline_is_not_frozen_as_a_changed_identity(monkeypatch):
    ready = process_identity()
    before_exec = dict(ready, command=[])
    observations = iter((before_exec, ready))
    monkeypatch.setattr(supervisor, 'process_record', lambda _pid: next(observations))
    monkeypatch.setattr(supervisor.time, 'sleep', lambda _seconds: None)
    assert supervisor.verified_process(123, ready['command']) == ready


def test_empty_exec_snapshot_can_be_upgraded_only_after_live_command_and_session_match(tmp_path, monkeypatch):
    ready = process_identity()
    old = {key: ready[key] for key in ('pid', 'start_ticks')}
    old['command'] = []
    (tmp_path/'worker_identity.json').write_text(json.dumps(old))
    monkeypatch.setattr(supervisor, 'process_record', lambda _pid: ready)
    assert supervisor.find_worker(tmp_path, ready['command']) == ready
    assert json.loads((tmp_path/'worker_identity.json').read_text()) == ready


@pytest.mark.parametrize('changed', [dict(start_ticks='reused'), dict(session=999),
    dict(process_group=999), dict(command=['/python', '-m', 'an.unrelated.job'])])
def test_reused_pid_or_wrong_command_session_never_receives_a_signal(monkeypatch, changed):
    saved = process_identity()
    current = dict(saved, **changed)
    monkeypatch.setattr(supervisor, 'process_record', lambda _pid: current)
    monkeypatch.setattr(supervisor.os, 'killpg', lambda *_args: pytest.fail('Signalled an unverified process'))
    with pytest.raises(RuntimeError):
        supervisor.stop_worker(saved)


def test_pid_reuse_while_waiting_for_exec_cmdline_is_rejected(monkeypatch):
    ready = process_identity()
    observations = iter((dict(ready, command=[]), dict(ready, start_ticks='reused')))
    monkeypatch.setattr(supervisor, 'process_record', lambda _pid: next(observations))
    monkeypatch.setattr(supervisor.time, 'sleep', lambda _seconds: None)
    with pytest.raises(RuntimeError, match='reused'):
        supervisor.verified_process(123, ready['command'])


def test_empty_cmdline_timeout_does_not_accept_or_signal_unverifiable_worker(monkeypatch):
    ready = process_identity()
    monkeypatch.setattr(supervisor, 'process_record', lambda _pid: dict(ready, command=[]))
    with pytest.raises(RuntimeError, match='remained empty'):
        supervisor.verified_process(123, ready['command'], timeout=0.)


def test_monitored_worker_exit_is_normal_not_an_identity_mismatch(monkeypatch):
    saved = process_identity()
    monkeypatch.setattr(supervisor, 'process_record', lambda _pid: None)
    assert supervisor.verified_process(123, saved['command'], saved) is None
