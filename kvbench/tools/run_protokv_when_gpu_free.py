"""Run one bounded ProtoKV check after the shared GPU becomes available."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/protokv_native_wait_20261016"
ANNOTATIONS = ROOT / "hermes/data/streamingbench/streamingbench_realtime_half_userpaths.json"
GPU_LOCK = Path("/tmp/hermes-gpu0.lock")
MIN_FREE_MIB = 28000
POLL_SECONDS = 60
child = None
cancelled = False


def write_status(output, **values):
    payload = dict(values, updated_utc=datetime.now(timezone.utc).isoformat())
    temporary = output / "wait_status.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(output / "wait_status.json")
    return payload


def gpu_available():
    with GPU_LOCK.open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False, "shared GPU lock is held"
        gpu = os.environ.get("SPARSEMM_GPU_INDEX", "0")
        free = int(subprocess.check_output(
            ["nvidia-smi", "-i", gpu, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True, timeout=15,
        ).strip())
        minimum = int(os.environ.get("SPARSEMM_MIN_FREE_MIB", MIN_FREE_MIB))
        return free >= minimum, f"GPU {gpu}: {free} MiB free, requires {minimum} MiB"


def on_stop(_signum, _frame):
    global cancelled
    cancelled = True
    if child is not None and child.poll() is None:
        child.terminate()


def main():
    global child
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--probe-once", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not args.probe_once and (output / "watcher.pid").exists():
        raise RuntimeError("Watcher PID file exists; use a fresh output directory")
    if not args.probe_once:
        (output / "watcher.pid").write_text(str(os.getpid()) + "\n")
    signal.signal(signal.SIGTERM, on_stop)
    signal.signal(signal.SIGINT, on_stop)
    attempt = 0
    while not cancelled:
        available, detail = gpu_available()
        if not available:
            write_status(output, status="waiting_for_gpu", detail=detail, attempts=attempt,
                         full_benchmarks_paused=True)
            if args.probe_once:
                print(detail, flush=True)
                return 0
            time.sleep(POLL_SECONDS)
            continue
        if args.probe_once:
            print(detail, flush=True)
            return 0
        attempt += 1
        run_dir = output / f"attempt_{attempt:03d}"
        command = [sys.executable, "-u", "-m", "kvbench", "run", "--method", "protokv",
                   "--anno-path", str(ANNOTATIONS), "--max-videos", "2", "--max-questions", "1",
                   "--output-dir", str(run_dir)]
        write_status(output, status="running", detail=detail, attempts=attempt,
                     run_dir=str(run_dir), command=command, full_benchmarks_paused=True)
        environment = dict(os.environ, PYTHONPATH=str(ROOT))
        with (output / f"attempt_{attempt:03d}.log").open("w") as log:
            child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                     stderr=subprocess.STDOUT)
            code = child.wait()
            child = None
        if cancelled:
            break
        summary = run_dir / "summary.json"
        if code == 0 and summary.is_file():
            result = json.loads(summary.read_text())
            if result.get("status") == "complete" and result.get("all_audits_passed") is True:
                write_status(output, status="complete", attempts=attempt, run_dir=str(run_dir),
                             summary=str(summary), full_benchmarks_paused=True)
                return 0
        controller_log = run_dir / "protokv_k1000_visual/kvbench_controller.log"
        busy = controller_log.is_file() and (
            "Output directory or GPU is busy" in controller_log.read_text()
            or "MiB free; requires" in controller_log.read_text()
        )
        if not busy:
            write_status(output, status="failed", attempts=attempt, run_dir=str(run_dir),
                         exit_code=code, full_benchmarks_paused=True)
            return 1
    write_status(output, status="cancelled", attempts=attempt, full_benchmarks_paused=True)
    return 130


if __name__ == "__main__":
    raise SystemExit(main())
