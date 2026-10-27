"""Detached watchdog for the long, resumable OVO-Bench experiment."""

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ.get("OVO_OUTPUT_ROOT", ROOT / "results/ovobench_official_full_k500_fps05_p151200_20261019"))
DATASET = os.environ.get("OVO_DATASET", "official_full")
TOTAL = {"hermes9": 1468, "forward3": 1567, "official_full": 3035}[DATASET]


def running(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
        return b"sparsemm_native.schedule_ovobench_full" in command
    except (OSError, ValueError):
        return False


def count(arm):
    path = OUTPUT / arm / "queries.jsonl"
    return len(path.read_text().splitlines()) if path.exists() else 0


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / ".watchdog.lock").open("w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        while True:
            dp, native = count("dp"), count("native")
            status_path = OUTPUT / "schedule.status"
            status = status_path.read_text().strip() if status_path.exists() else "missing"
            pid_path = OUTPUT / "schedule.pid"
            pid = pid_path.read_text().strip() if pid_path.exists() else ""
            alive = running(pid)
            now = datetime.now(timezone.utc).isoformat()
            write_json(OUTPUT / "watchdog.json", {
                "checked_utc": now, "watchdog_pid": os.getpid(),
                "schedule_pid": int(pid) if pid.isdigit() else None,
                "schedule_alive": alive, "dp_completed": dp,
                "native_completed": native, "schedule_status": status,
            })
            if min(dp, native) >= TOTAL or status.startswith("failed"):
                return 0
            if not alive:
                # Give a still-running inference child time to exit before
                # starting another scheduler; the GPU lease enforces this too.
                children = subprocess.check_output(["ps", "-eo", "args="], text=True)
                if not any("python -u -m sparsemm_native.run_ovobench --arm" in line
                           for line in children.splitlines()):
                    with (OUTPUT / "schedule_daemon.log").open("a") as log:
                        child = subprocess.Popen(
                            [sys.executable, "-u", "-m", "sparsemm_native.schedule_ovobench_full"],
                            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)
                    pid_path.write_text(str(child.pid) + "\n")
                    with (OUTPUT / "watchdog_events.jsonl").open("a") as events:
                        events.write(json.dumps({"utc": now, "event": "scheduler_restarted",
                                                 "pid": child.pid, "dp": dp, "native": native}) + "\n")
            time.sleep(30)


if __name__ == "__main__":
    sys.exit(main())
