"""Validate the entire EGate result ledger and report benchmark metrics."""
import argparse
from collections import Counter
import json
from pathlib import Path
from .data import read_rows, validate_rows, write_rows

STREAM = [('OP', 'Object Recognition'), ('CR', 'Causal Reasoning'), ('CS', 'Clips Summarize'),
          ('ATP', 'Attribute Recognition'), ('EU', 'Event Understanding'), ('TR', 'Text-Rich Understanding'),
          ('PR', 'Prospective Reasoning'), ('SU', 'Spatial Understanding'), ('ACP', 'Action Recognition'), ('CT', 'Counting')]
RT = ('OJR', 'ATR', 'ACR', 'OCR', 'STU', 'FPD')
BACK = ('EPM', 'ASI', 'HLD')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir', type=Path)
    a = parser.parse_args()
    root = a.output_dir.resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    rows = []
    for i in range(len(manifest['gpus'])):
        annotations = json.loads((root / 'annotations' / f'shard_{i:02}.json').read_text())
        actual = read_rows(root / f'shard_{i:02}' / 'results.csv')
        validate_rows(actual, annotations, manifest['accept_equivalent_choices'], complete=True)
        rows += actual
    if len(rows) != manifest['questions']:
        raise ValueError('Question count differs from manifest')
    tasks = {}
    for row in rows:
        entry = tasks.setdefault(row['task'], dict(count=0, correct=0))
        entry['count'] += 1
        entry['correct'] += float(row['qa_acc']) == 100
    for entry in tasks.values():
        entry['accuracy'] = 100 * entry['correct'] / entry['count']
    result = dict(model=manifest['model'], method=manifest['method'], benchmark=manifest['benchmark'],
                  questions=len(rows), correct=sum(float(r['qa_acc']) == 100 for r in rows),
                  micro_accuracy=sum(float(r['qa_acc']) for r in rows)/len(rows), tasks=tasks,
                  gate_sources=dict(Counter(r['gate_source'] for r in rows)),
                  allocation_sha256=manifest['allocation_sha256'], calibration_note=manifest['calibration_note'])
    table = {}
    if manifest['benchmark'] == 'streamingbench':
        table = {short: tasks[name]['accuracy'] if name in tasks else None for short, name in STREAM}
        table['Overall'] = result['micro_accuracy']
    else:
        table = {task: tasks[task]['accuracy'] if task in tasks else None for task in RT + BACK}
        table['RT'] = sum(tasks[t]['accuracy'] for t in RT)/len(RT) if all(t in tasks for t in RT) else None
        table['Back'] = sum(tasks[t]['accuracy'] for t in BACK)/len(BACK) if all(t in tasks for t in BACK) else None
        table['Avg'] = (table['RT']+table['Back'])/2 if table['RT'] is not None and table['Back'] is not None else None
    result['table_columns'] = table
    write_rows(root/'results.csv', rows)
    (root/'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    labels = ['Model', 'Method', '#Frames'] + list(table)
    values = [manifest['model'], 'HorizonKV-EGate', '1 fps'] + [f'{x:.2f}' if x is not None else '—' for x in table.values()]
    markdown = '| '+' | '.join(labels)+' |\n| '+' | '.join(['---']*len(labels))+' |\n| '+' | '.join(values)+' |\n'
    (root/'table.md').write_text(markdown)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(markdown)


if __name__ == '__main__':
    main()
