"""Detached, resumable supervisor for the nine-task fps=0.5 mild-history run."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from sparsemm_native.supervise_ovobench_dual import now, repair_jsonl_tail, write_json


ROOT = Path(__file__).resolve().parents[1]
ARM = 'mild_pseudo_history_native_fps05_k500'
TOTAL = 1468


def count_rows(path):
    if not path.exists():
        return 0
    with path.open('rb') as handle:
        return sum(1 for line in handle if line.endswith(b'\n') and line.strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    arm = root / ARM
    arm.mkdir(exist_ok=True)
    lock = (root / 'supervisor.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    log_path = root / 'worker.log'
    progress = arm / 'queries.jsonl'
    pause = root / 'PAUSE'
    command = [sys.executable, '-u', '-m',
               'sparsemm_native.run_ovobench_mild_pseudo_history_fps05',
               '--output-root', str(root)]
    environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                       TRANSFORMERS_OFFLINE='1', MALLOC_ARENA_MAX='2')
    attempt = 0
    while True:
        rows = count_rows(progress)
        if rows == TOTAL:
            write_json(root / 'status.json', dict(state='complete', progress=rows,
                                                  total=TOTAL, updated_at=now()))
            return
        if pause.exists():
            write_json(root / 'status.json', dict(state='paused', progress=rows,
                                                  total=TOTAL, updated_at=now()))
            return
        if attempt:
            repair_jsonl_tail(progress, validate_complete=False)
        attempt += 1
        with log_path.open('a') as log:
            log.write(f'\n{now()} START attempt={attempt} progress={rows}/{TOTAL}\n')
            log.flush()
            worker = subprocess.Popen(command, cwd=ROOT, env=environment,
                                      stdin=subprocess.DEVNULL, stdout=log,
                                      stderr=subprocess.STDOUT)
            while worker.poll() is None:
                rows = count_rows(progress)
                state = 'stopping' if pause.exists() else 'running'
                write_json(root / 'status.json', dict(state=state, progress=rows,
                           total=TOTAL, attempt=attempt, worker_pid=worker.pid,
                           updated_at=now(), log=str(log_path)))
                if pause.exists():
                    worker.terminate()
                time.sleep(10)
            code = worker.returncode
            log.write(f'{now()} EXIT attempt={attempt} code={code} progress={count_rows(progress)}/{TOTAL}\n')
            log.flush()
        rows = count_rows(progress)
        if code == 0 and rows == TOTAL:
            write_json(root / 'status.json', dict(state='complete', progress=rows,
                                                  total=TOTAL, updated_at=now()))
            return
        if pause.exists():
            write_json(root / 'status.json', dict(state='paused', progress=rows,
                                                  total=TOTAL, updated_at=now()))
            return
        if attempt >= 5 or (code == 0 and rows < TOTAL):
            write_json(root / 'status.json', dict(state='failed', progress=rows,
                                                  total=TOTAL, exit_code=code,
                                                  attempt=attempt, updated_at=now(),
                                                  log=str(log_path)))
            raise RuntimeError('OVO mild-history worker did not complete')
        write_json(root / 'status.json', dict(state='retrying', progress=rows,
                                              total=TOTAL, exit_code=code,
                                              attempt=attempt, updated_at=now()))
        time.sleep(30)


if __name__ == '__main__':
    main()
