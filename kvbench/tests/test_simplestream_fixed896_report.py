"""Scoring must use raw official answers and same-question controlled pairs."""

import json

import pytest

from sparsemm_native import report_simplestream_fixed896 as report


def row(qid='0', task='EPM', response='A', target='A'):
    return dict(query_id=qid, task=task, target=target, response=response, seconds=2.,
        input_audit=dict(prompt_sha256='identical-prompt', selected_raw_frame_indices=[0, 5, 10, 15]),
        budget_audit=dict(frame_visual_tokens=[1024] * 4, actual_visual_slots=56000,
            selector=report.SELECTOR, visual_budget_per_head=500, low_head_count=56,
            independent_single_frame_prefill=True, pruned_before_question=True))


def test_raw_official_scoring_does_not_extract_letters_or_change_case():
    values = [row(response='A or B', target='B'), row('1', response='b', target='B'),
              row('2', response='A', target='C')]
    result = report.score(values)
    assert result['correct'] == 1
    assert result['question_accuracy'] == 1 / 3
    assert result['per_task']['EPM']['total'] == 3
    assert result['official_two_group_macro_average'] is None
    assert result['nine_task_macro_average'] is None


def test_macro_matches_official_two_group_equal_weighting():
    values = [row(str(i), task=task, response='A' if task in report.BACKWARD else 'B')
              for i, task in enumerate(report.TASKS)]
    values.append(row('extra', task='EPM'))
    result = report.score(values)
    assert result['question_accuracy'] == 4 / 10
    assert result['nine_task_macro_average'] == 1 / 3
    assert result['official_two_group_macro_average'] == .5


def test_partial_pair_uses_intersection_instead_of_baseline_full_score():
    current = {'0': row('0'), '1': row('1', response='B')}
    baseline = {'0': row('0', response='B'), '1': row('1'), '2': row('2')}
    result, table = report.compare(current, baseline, controlled=True)
    assert result['matched_questions'] == 2
    assert result['current_only'] == result['baseline_only'] == 1
    assert result['current']['question_accuracy'] == result['baseline']['question_accuracy'] == .5
    assert len(table) == 2


@pytest.mark.parametrize('section,key,value', [
    ('input_audit', 'prompt_sha256', 'different-question'),
    ('input_audit', 'selected_raw_frame_indices', [1, 5, 10, 15]),
    ('input_audit', 'position_layout', 'different-position-implementation'),
    ('budget_audit', 'selector', 'uniform'),
    ('budget_audit', 'independent_single_frame_prefill', False),
    ('budget_audit', 'question_text_used_for_selection', True),
    ('budget_audit', 'actual_history_visual_slots', 1),
])
def test_input_or_algorithm_drift_is_rejected(section, key, value):
    current, baseline = row(), row()
    current[section][key] = value
    with pytest.raises(ValueError):
        report.validate_controlled_pair(current, baseline)


def test_different_spatial_grid_and_budget_underfill_are_allowed():
    current, baseline = row(), row()
    baseline['input_audit']['processor_resize_hw_per_frame'] = [[644, 840]] * 4
    current['input_audit']['processor_resize_hw_per_frame'] = [[896, 896]] * 4
    baseline['budget_audit']['frame_visual_tokens'] = [221] * 4
    baseline['budget_audit']['actual_visual_slots'] = 40484
    report.validate_controlled_pair(current, baseline)


def test_interrupted_append_is_ignored_but_earlier_json_corruption_fails(tmp_path):
    path = tmp_path / 'queries.jsonl'
    path.write_text('{"query_id":"0"}\n{"query_id":')
    assert report.read_rows(path) == [dict(query_id='0')]
    path.write_text('{"query_id":\n{"query_id":"1"}\n')
    with pytest.raises(json.JSONDecodeError):
        report.read_rows(path)


def test_legacy_results_join_physical_audit_and_observation_by_order(tmp_path):
    worker = tmp_path / 'worker'
    worker.mkdir()
    contents = {
        'results_incremental.jsonl': dict(id=0, task='EPM', ground_truth='A', response='A'),
        'ragged_query_audit.jsonl': dict(prompt_sha256='same', frame_visual_tokens=[720]*4, actual_visual_slots=56000),
        'query_observations.jsonl': dict(query_id='0', prompt_sha256='same', total_query_seconds=3.),
    }
    for name, value in contents.items():
        (worker / name).write_text(json.dumps(value) + '\n')
    loaded = report.load_run(tmp_path)
    assert loaded['0']['correct'] is True
    assert loaded['0']['seconds'] == 3.
    contents['query_observations.jsonl']['query_id'] = '1'
    (worker / 'query_observations.jsonl').write_text(json.dumps(contents['query_observations.jsonl']) + '\n')
    with pytest.raises(ValueError, match='order'):
        report.load_run(tmp_path)


def test_fifth_mcqa_option_in_official_data_is_valid(tmp_path):
    value = row(target='E', response='E')
    (tmp_path / 'queries.jsonl').write_text(json.dumps(value) + '\n')
    assert report.load_run(tmp_path)['0']['correct'] is True
