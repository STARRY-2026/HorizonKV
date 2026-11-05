#!/usr/bin/env python3
"""Single entry point for the existing StreamingBench / OVO evaluation adapters.

Examples (execute from any working directory):
  python unified_workflow.py --benchmark ovobench --methods adakv pyramidkv --gpus 0,1,2,3 --output-root /path/to/results --dry-run
  python unified_workflow.py --benchmark ovobench --methods hermes --gpus 0 --output-root /path/to/results --max-videos 1 --execute

The method implementations remain in their existing adapter files. This script
orchestrates them; it does not implement DSCache or reproduce each baseline.
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
METHODS = ('adakv', 'pyramidkv', 'sparsemm', 'lukv', 'hermes', 'fullkv')

def run(cmd):
    print('Running:', ' '.join(map(str, cmd)), flush=True)
    return subprocess.run([str(x) for x in cmd], check=False).returncode

def validate_smoke(path):
    results = list(path.glob('shard_*/results.csv'))
    rows = []
    for result in results:
        with result.open(encoding='utf-8-sig', newline='') as f:
            rows.extend(csv.DictReader(f))
    if len(rows) != 1 or rows[0].get('pred_choice', '') not in tuple('ABCDEFGH'):
        raise RuntimeError('Smoke must produce exactly one valid answer; inspect shard logs.')

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--benchmark', choices=['streamingbench', 'ovobench'], required=True)
    p.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    p.add_argument('--gpus', default='0,1,2,3', help='Comma-separated visible PPU indexes')
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--annotation', type=Path)
    p.add_argument('--max-videos', type=int)
    p.add_argument('--cohort', choices=['heldout', 'full'], default='heldout', help='StreamingBench only')
    p.add_argument('--allow-calibration-overlap', action='store_true')
    p.add_argument('--accept-equivalent-choices', action='store_true')
    p.add_argument('--skip-smoke', action='store_true')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true', help='Run evaluation; omitted by default')
    mode.add_argument('--dry-run', action='store_true', help='Inspect delegated configurations only')
    a = p.parse_args()
    if a.max_videos is not None and a.max_videos < 1:
        p.error('--max-videos must be positive')
    if a.benchmark == 'ovobench' and (a.allow_calibration_overlap or a.accept_equivalent_choices):
        p.error('These options apply to StreamingBench only')
    failures = []
    for method in a.methods:
        launcher = HERE / ('ovo_run.py' if a.benchmark == 'ovobench' else 'run.py')
        base = [sys.executable, launcher]
        if a.benchmark == 'streamingbench':
            base += ['--benchmark', 'streamingbench']
        base += ['--model', 'llavaov', '--method', method, '--input-profile', 'dscache']
        if a.benchmark == 'streamingbench':
            base += ['--cohort', a.cohort]
            if a.allow_calibration_overlap: base.append('--allow-calibration-overlap')
            if a.accept_equivalent_choices: base.append('--accept-equivalent-choices')
        if a.annotation: base += ['--annotation', a.annotation]
        output = a.output_root / method
        if not a.execute:
            cmd = base + ['--gpus', a.gpus, '--output-dir', output, '--dry-run']
            if a.max_videos: cmd += ['--max-videos', str(a.max_videos)]
            if run(cmd): failures.append(method)
            continue
        if not a.skip_smoke:
            smoke = a.output_root / (method + '_smoke')
            gpu = a.gpus.split(',')[0]
            if run(base + ['--gpus', gpu, '--output-dir', smoke, '--max-videos', '1']):
                failures.append(method + ':smoke'); continue
            try:
                # OVO samples contain one question; StreamingBench may have several.
                if a.benchmark == 'ovobench': validate_smoke(smoke)
                elif not list(smoke.glob('shard_*/results.csv')): raise RuntimeError('No smoke results')
            except (OSError, RuntimeError) as exc:
                print(f'{method}: {exc}', file=sys.stderr); failures.append(method + ':smoke'); continue
        cmd = base + ['--gpus', a.gpus, '--output-dir', output]
        if a.max_videos: cmd += ['--max-videos', str(a.max_videos)]
        if run(cmd): failures.append(method + ':inference'); continue
        if run([sys.executable, HERE / 'summarize.py', output]):
            failures.append(method + ':summary')
    print(json.dumps({'failures': failures, 'mode': 'execute' if a.execute else 'dry-run'}, ensure_ascii=False), flush=True)
    return int(bool(failures))

if __name__ == '__main__':
    sys.exit(main())
