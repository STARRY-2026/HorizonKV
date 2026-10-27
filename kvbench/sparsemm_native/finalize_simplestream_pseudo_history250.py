"""Officially score 250/250 history against the recent-only pseudo K500 control."""

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path


ROOT = Path('/home/yujun/program/kvbench/results/ovobench_simplestream_pseudo_history250_20261025/full')
CONTROL = Path('/home/yujun/program/kvbench/results/ovobench_simplestream_pseudo_budget500_20261025/full')
SCORE = Path('/home/yujun/program/SimpleStream/scoring/score_ovo_bench.py')
BACKWARD = {'EPM', 'ASI', 'HLD'}


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(values, scorer):
    grouped = {'backward': [row for row in values if row['task'] in BACKWARD],
               'realtime': [row for row in values if row['task'] not in BACKWARD]}
    result = {}
    for group, section in grouped.items():
        _, scores = scorer.calculate_score_backward_realtime(section)
        result[group] = {task: {'correct':sum(outcomes), 'total':len(outcomes),
                                'accuracy':sum(outcomes)/len(outcomes)}
                         for task, outcomes in scores.items()}
    tasks = {**result['backward'], **result['realtime']}
    result['correct'] = sum(row['correct'] for row in tasks.values())
    result['question_accuracy'] = result['correct']/len(values)
    result['nine_task_macro_average'] = statistics.mean(row['accuracy'] for row in tasks.values())
    result['backward_macro_average'] = statistics.mean(row['accuracy'] for row in result['backward'].values())
    result['realtime_macro_average'] = statistics.mean(row['accuracy'] for row in result['realtime'].values())
    return result


def main():
    target = ROOT/'worker/results_incremental.jsonl'
    while not target.exists() or len(target.read_text().splitlines()) < 1468:
        time.sleep(15)
    values = rows(target)
    control = rows(CONTROL/'worker/results_incremental.jsonl')
    audit = rows(ROOT/'worker/ragged_query_audit.jsonl')
    control_audit = rows(CONTROL/'worker/ragged_query_audit.jsonl')
    if not (len(values) == len(control) == len(audit) == len(control_audit) == 1468):
        raise RuntimeError('Incomplete or unmatched run')
    if any(row['actual_visual_slots'] > 56000 or
           row.get('actual_history_visual_slots', -1) > 14000 or
           row.get('history_recent_positions_separate') is not True or
           row.get('selector') != 'pseudo_history16_recent250_history250_lowheads_recent500_highheads'
           for row in audit):
        raise RuntimeError('Budget or selector audit failed')
    spec = importlib.util.spec_from_file_location('official_ovo_score', SCORE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scorer = module.OVOBenchOfflineScore(argparse.Namespace(model='Qwen2.5-VL',mode='offline'), {})
    recent = summarize(values, scorer)
    recent_only = summarize(control, scorer)
    control_by_id = {str(row['id']):row for row in control}
    control_audit_by_id = {str(row['id']):item for row, item in zip(control, control_audit)}
    paired = {task:{'both_correct':0,'history_only':0,'recent_only':0,'both_wrong':0}
              for task in {row['task'] for row in values}}
    for index, row in enumerate(values):
        other = control_by_id[str(row['id'])]
        if row['task'] != other['task'] or row['ground_truth'] != other['ground_truth']:
            raise RuntimeError('Different questions or targets')
        if (row['final_chunk_ids'] != other['final_chunk_ids'] or
                row['num_vision_tokens'] != other['num_vision_tokens'] or
                audit[index]['prompt_sha256'] != control_audit_by_id[str(row['id'])]['prompt_sha256']):
            raise RuntimeError('Recent video frames or question prompt differ')
        a = row['ground_truth'] in row['response']
        b = other['ground_truth'] in other['response']
        key = ('both_correct' if a and b else 'history_only' if a else
               'recent_only' if b else 'both_wrong')
        paired[row['task']][key] += 1
    output = {'version':'pseudo_history250_vs_pseudo_recentonly500','complete':True,
              'query_count':1468,'pseudo_history250':recent,'pseudo_recentonly500_control':recent_only,
              'paired_by_task':paired,
              'actual_visual_tokens_per_head_mean':statistics.mean(
                  row['actual_visual_slots']/112 for row in audit),
              'actual_history_tokens_per_head_mean':statistics.mean(
                  row['actual_history_visual_slots']/112 for row in audit),
              'actual_recent_tokens_per_head_mean':statistics.mean(
                  row['actual_recent_visual_slots']/112 for row in audit),
              'actual_visual_tokens_per_head_max':max(
                  row['actual_visual_slots']/112 for row in audit)}
    (ROOT/'summary_official.json').write_text(json.dumps(output,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
