"""Detached supervisor for the native aligned K=500 two-bank OVO arm only."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path("/home/yujun/program/kvbench")
OUTPUT = ROOT / "results/ovobench_sparsemm_aligned_k500_20261025/full"
ARM = "native_two_bank_250250_k500"
DIRECTORY = OUTPUT / ARM
PYTHON = Path("/home/yujun/.conda/envs/hermes-qwen/bin/python")
PYTHONPATH = ":".join((str(ROOT), str(ROOT / "sparsemm_baselines/upstream/lmms-eval"),
                       str(ROOT / "sparsemm_protokv/runtime_deps")))
TOTAL = 1468


def progress():
    path = DIRECTORY / "queries.jsonl"
    if not path.exists():
        return 0
    with path.open() as handle:
        return sum(bool(line.strip()) for line in handle)


def write_status(state, **extra):
    state_file = DIRECTORY / "background_status.json"
    info = dict(state=state, completed=progress(), total=TOTAL,
                updated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                supervisor_pid=os.getpid(), **extra)
    temp = state_file.with_suffix(".tmp")
    temp.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n")
    os.replace(temp, state_file)


def main():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(PYTHONPATH=PYTHONPATH, SPARSEMM_GPU_LEASE_SHARED="1",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    command = [str(PYTHON), "-m", "sparsemm_native.run_ovobench_sparsemm_aligned_k500",
               "--output-root", str(OUTPUT), "--arm", ARM]
    attempt = 0
    while progress() < TOTAL:
        attempt += 1
        write_status("starting", attempt=attempt)
        with (DIRECTORY / "background_run.log").open("a") as log:
            log.write(f"\n=== attempt {attempt} at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ===\n")
            log.flush()
            child = subprocess.Popen(command, cwd=ROOT, env=environment,
                                     stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            write_status("running", attempt=attempt, worker_pid=child.pid)
            while child.poll() is None:
                write_status("running", attempt=attempt, worker_pid=child.pid)
                time.sleep(15)
            code = child.returncode
        if code == 0 and progress() == TOTAL:
            write_status("complete", attempt=attempt)
            return
        tail = (DIRECTORY / "background_run.log").read_text(errors="replace")[-4000:]
        if code and ("GPU is busy" in tail or "MiB free; requires" in tail or
                     "Both GPU process slots are busy" in tail):
            write_status("waiting_for_gpu", attempt=attempt, last_exit_code=code)
            time.sleep(30)
            continue
        write_status("failed", attempt=attempt, last_exit_code=code,
                     error_tail=tail[-1200:])
        raise RuntimeError(f"Aligned two-bank experiment failed with exit {code}")
    write_status("complete", attempt=attempt)


if __name__ == "__main__":
    main()
