"""Resume physical head-ragged SimpleStream on OVO's nine non-forward tasks."""

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

from sparsemm_native import supervise_simplestream_upstream as base
from sparsemm_native import supervise_simplestream_recent_share_independent_last1_reverse as ragged


BACKWARD = ('EPM', 'ASI', 'HLD')
REALTIME = ('OCR', 'ACR', 'ATR', 'STU', 'FPD', 'OJR')
TASKS = set(BACKWARD + REALTIME)
EXPECTED_QUESTIONS = 1468
ROOT = base.ROOT
SOURCE = ROOT / 'sparsemm_native/supervise_simplestream_recent_share_independent_last1_reverse_nine.py'


def prepare(output):
    protocol_file = output / 'protocol.json'
    if protocol_file.exists():
        protocol = base.check_pins(output)
        if protocol.get('version') != 'simplestream_recent_share_independent_last1_reverse_nine_v1':
            raise RuntimeError('Existing output uses a different task subset')
        return protocol
    protocol = ragged.prepare(output)
    counts = []
    for rank in range(2):
        rows = json.loads((output / f'annotations_shard{rank}.json').read_text())
        filtered = [row for row in rows if row['task'] in TASKS]
        path = output / f'nine_annotations_shard{rank}.json'
        base.write(path, filtered)
        protocol['input_sha256'][str(path)] = base.sha(path)
        counts.append(len(filtered))
    if sum(counts) != EXPECTED_QUESTIONS:
        raise RuntimeError(f'Unexpected nine-task annotation count: {counts}')
    protocol.update(version='simplestream_recent_share_independent_last1_reverse_nine_v1',
                    query_count=EXPECTED_QUESTIONS,
                    annotation_count=EXPECTED_QUESTIONS,
                    excluded_tasks=['REC', 'SSR', 'CRR'],
                    included_tasks=list(BACKWARD + REALTIME),
                    shard_query_counts=counts,
                    score_scope='official backward/realtime task scoring; no 12-task official total')
    protocol['code_sha256'][str(SOURCE)] = base.sha(SOURCE)
    target = output / 'source_snapshot' / ROOT.name / SOURCE.relative_to(ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE, target)
    base.write(protocol_file, protocol)
    return protocol


class SubprocessProxy:
    def __getattr__(self, name):
        return getattr(subprocess, name)

    @staticmethod
    def Popen(command, *args, **kwargs):
        if isinstance(command, (list, tuple)) and '-m' in command:
            command = list(command)
            module_index = command.index('-m') + 1
            if command[module_index] == 'sparsemm_native.supervise_simplestream_upstream' and '--worker' in command:
                command[module_index] = 'sparsemm_native.supervise_simplestream_recent_share_independent_last1_reverse'
                anno_index = command.index('--annotation') + 1
                full = Path(command[anno_index])
                command[anno_index] = str(full.with_name('nine_' + full.name))
        return subprocess.Popen(command, *args, **kwargs)


def report(output, state):
    results = base.collect(output)
    if results['forward']:
        raise RuntimeError('Forward-task results found in the nine-task run')
    completed = len(results['backward']) + len(results['realtime'])
    state.update(updated_at=base.now(), completed_queries=completed,
                 completed_annotations=completed, total_queries=EXPECTED_QUESTIONS)
    base.write(output / 'status.json', state)
    summary = state.get('summary')
    content = (f"状态：{state['state']}；进度：{completed}/{EXPECTED_QUESTIONS} 题。\n\n"
               "范围：EPM、ASI、HLD、OCR、ACR、ATR、STU、FPD、OJR；"
               "REC、SSR、CRR 未运行。\n\n")
    if summary:
        content += (f"九类逐题正确率：{summary['question_accuracy']:.2%}；"
                    f"九类等权平均：{summary['nine_task_macro_average']:.2%}。\n")
    (output / 'progress.md').write_text(content)
    return results, completed


def finish(output, results):
    base.check_pins(output)
    if results['forward']:
        raise RuntimeError('Forward tasks unexpectedly present')
    annotations = {}
    for rank in range(2):
        for row in json.loads((output / f'nine_annotations_shard{rank}.json').read_text()):
            annotations[row['id']] = row
    rows = results['backward'] + results['realtime']
    if len(rows) != EXPECTED_QUESTIONS or len({row['id'] for row in rows}) != EXPECTED_QUESTIONS:
        raise RuntimeError('Incomplete or duplicate nine-task results')
    if {row['id'] for row in rows} != set(annotations):
        raise RuntimeError('Nine-task result IDs do not match annotations')
    for row in rows:
        source = annotations[row['id']]
        if row['task'] != source['task'] or row['video'] != source['video']:
            raise RuntimeError('Nine-task result identity differs from annotation')
        if not isinstance(row.get('response'), str) or not row['response'].strip():
            raise RuntimeError('Missing/empty model response')
    obs = {}
    for rank in range(2):
        for record in base.read_jsonl(output / f'shards/shard{rank}/query_observations.jsonl'):
            obs[record['query_id']] = record
    if len(obs) != EXPECTED_QUESTIONS or {str(row['id']) for row in rows} != set(obs):
        raise RuntimeError('Incomplete/different query observations')
    if any(record['initial_cache_tokens'] != 0 for record in obs.values()):
        raise RuntimeError('Historical language KV found')

    spec = importlib.util.spec_from_file_location('upstream_official_ovo_score', base.UPSTREAM / 'scoring/score_ovo_bench.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scorer = module.OVOBenchOfflineScore(argparse.Namespace(model='Qwen2.5-VL', mode='offline'), results)
    _, scores_b = scorer.calculate_score_backward_realtime(results['backward'])
    _, scores_r = scorer.calculate_score_backward_realtime(results['realtime'])
    scores = {**scores_b, **scores_r}
    if set(scores) != TASKS:
        raise RuntimeError('Expected all nine official backward/realtime task scores')
    tasks = {task: dict(correct=sum(values), total=len(values), accuracy=sum(values) / len(values))
             for task, values in scores.items()}
    correct = sum(item['correct'] for item in tasks.values())
    summary = dict(complete=True, total_queries=EXPECTED_QUESTIONS,
                   correct=correct, question_accuracy=correct / EXPECTED_QUESTIONS,
                   nine_task_macro_average=sum(item['accuracy'] for item in tasks.values()) / 9,
                   backward_macro_average=sum(tasks[task]['accuracy'] for task in BACKWARD) / 3,
                   realtime_macro_average=sum(tasks[task]['accuracy'] for task in REALTIME) / 6,
                   excluded_tasks=['REC', 'SSR', 'CRR'], tasks=tasks)
    base.write(output / 'summary.json', summary)
    base.write(output / 'qwen25vl_results_nine.json',
               dict(config=json.loads((output / 'protocol.json').read_text()), **results))
    with (output / 'per_query.csv').open('w') as handle:
        writer = csv.writer(handle)
        writer.writerow(('query_id', 'task', 'response', 'visual_tokens', 'query_seconds'))
        for row in sorted(rows, key=lambda row: row['id']):
            record = obs[str(row['id'])]
            writer.writerow((row['id'], row['task'], row['response'],
                             record['result']['num_vision_tokens'], record['total_query_seconds']))
    base.write(output / 'final_audit.json', dict(passed=True, completed_at=base.now(),
               unique_queries=EXPECTED_QUESTIONS, excluded_tasks=['REC', 'SSR', 'CRR']))
    return summary


if __name__ == '__main__':
    base.subprocess = SubprocessProxy()
    base.prepare = prepare
    base.report = report
    base.finish = finish
    base.main()
