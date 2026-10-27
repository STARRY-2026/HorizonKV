"""Keep the two K=500 OVO-Bench arms paired and resumable on one GPU."""

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ.get("OVO_OUTPUT_ROOT", ROOT / "results/ovobench_official_full_k500_fps05_p151200_20261019"))
DATASET = os.environ.get("OVO_DATASET", "official_full")
GROUPED = os.environ.get("OVO_GROUPED", "0") == "1"
TOTAL = {"hermes9": 1468, "forward3": 1567, "official_full": 3035}[DATASET]
PYTHON = "/home/yujun/.conda/envs/hermes-qwen/bin/python"


def count(arm):
    path = OUTPUT / arm / "queries.jsonl"
    return len(path.read_text().splitlines()) if path.exists() else 0


def wait_for_gpu():
    lock = Path("/tmp/hermes-gpu0.lock")
    while True:
        with lock.open("a") as stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
                return
        time.sleep(10)


def main():
    env = dict(os.environ)
    env.update(PYTHONPATH=os.pathsep.join((str(ROOT), str(ROOT / "sparsemm_baselines/upstream/lmms-eval"),
                                           str(ROOT / "sparsemm_protokv/runtime_deps"))),
               HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MALLOC_ARENA_MAX="2")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    while min(count("dp"), count("native")) < TOTAL:
        wait_for_gpu()
        dp, native = count("dp"), count("native")
        arm = "native" if native < dp else "dp"
        batch = min(2 if max(dp, native) <= 2 else 5, TOTAL - count(arm))
        if batch <= 0:
            arm = "native" if arm == "dp" else "dp"
            batch = min(5, TOTAL - count(arm))
        command = [PYTHON, "-u", "-m", "sparsemm_native.run_ovobench", "--arm", arm,
                   "--dataset", DATASET,
                   "--output-root", str(OUTPUT), "--max-new", str(batch)]
        if GROUPED:
            command.append("--grouped")
        with (OUTPUT / "schedule.log").open("a") as report:
            report.write(json.dumps({"event": "start", "arm": arm, "dp": dp, "native": native,
                                     "batch": batch, "grouped": GROUPED,
                                     "epoch": time.time()}) + "\n")
            report.flush()
        with (OUTPUT / f"{arm}.log").open("a") as log:
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            (OUTPUT / "schedule.status").write_text(f"failed arm={arm} exit={completed.returncode}\n")
            return completed.returncode
        (OUTPUT / "schedule.status").write_text(f"running dp={count('dp')} native={count('native')}\n")
    (OUTPUT / "schedule.status").write_text("complete\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
