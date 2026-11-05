"""Validate and summarize complete H2SVC StreamingBench or OVO-Bench results."""
import argparse
import json
from pathlib import Path

from methods.horizonkv_egate.summarize import STREAM, RT, BACK
from .data import read_rows, validate_rows, write_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest.get('backend') != 'h2svc':
        raise ValueError('Not an H2SVC output directory')
    rows = []
    for index in range(len(manifest['gpus'])):
        data = json.loads((root/'annotations'/f'shard_{index:02}.json').read_text())
        actual = read_rows(root/f'shard_{index:02}'/'results.csv')
        validate_rows(actual, data, manifest['accept_equivalent_choices'], complete=True)
        rows.extend(actual)
    if len(rows) != manifest['questions']:
        raise ValueError('Question count differs from manifest')
    tasks = {}
    for row in rows:
        entry = tasks.setdefault(row['task'], dict(count=0, correct=0))
        entry['count'] += 1
        entry['correct'] += float(row['qa_acc']) == 100
    for entry in tasks.values():
        entry['accuracy'] = 100*entry['correct']/entry['count']
    micro = sum(float(row['qa_acc']) for row in rows)/len(rows)
    if manifest['benchmark'] == 'streamingbench':
        table = {short:tasks[name]['accuracy'] if name in tasks else None for short, name in STREAM}
        table['Overall'] = micro
    else:
        table = {name:tasks[name]['accuracy'] if name in tasks else None for name in RT+BACK}
        table['RT'] = sum(tasks[name]['accuracy'] for name in RT)/len(RT) if all(name in tasks for name in RT) else None
        table['Back'] = sum(tasks[name]['accuracy'] for name in BACK)/len(BACK) if all(name in tasks for name in BACK) else None
        table['Avg'] = (table['RT']+table['Back'])/2 if table['RT'] is not None and table['Back'] is not None else None
    result = dict(benchmark=manifest['benchmark'], model=manifest['model'], method='h2svc',
                  questions=len(rows), correct=sum(float(row['qa_acc']) == 100 for row in rows),
                  micro_accuracy=micro, tasks=tasks, table_columns=table,
                  frame_budgets=manifest['frame_budgets'], head_map_sha256=manifest['head_map_sha256'])
    write_rows(root/'results.csv', rows)
    (root/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    labels = ['Model', 'Method', '#Frames'] + list(table)
    values = [manifest['model'], 'H2SVC', '1 fps'] + [f'{value:.2f}' if value is not None else '—'
                                                     for value in table.values()]
    markdown = '| '+' | '.join(labels)+' |\n| '+' | '.join(['---']*len(labels))+' |\n| '+' | '.join(values)+' |\n'
    (root/'table.md').write_text(markdown)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(markdown)


if __name__ == '__main__':
    main()
