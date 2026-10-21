"""Run one resumable 3,035-query DP dual-bank experiment in two pinned protocols.

The two output parts preserve the completed controls' prompts and scorers.
Workers are detached; restarting this supervisor attaches to a surviving worker.
Create OUTPUT_ROOT/PAUSE to stop the verified worker and exit without retrying;
remove that file and relaunch this supervisor to resume.
"""

import argparse
from datetime import datetime
import fcntl
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from sparsemm_native import run_ovobench as runner
from sparsemm_native.compare_ovobench_three_way import (
    ARMS, PROTOCOL_FIELDS, _digest, _read_rows, _validate_rows, compare, markdown,
)


ROOT = Path(__file__).resolve().parents[1]
BASELINES = {
    "hermes9": ROOT / "results/ovobench_hermes9_k500_fps05_p151200_20261019",
    "forward3": ROOT / "results/ovobench_forward3_k500_fps05_p151200_20261020",
}
TOTALS = {"hermes9": 1468, "forward3": 1567}
SOURCE_FILES = (
    "sparsemm_native/supervise_ovobench_dual.py", "sparsemm_native/run_ovobench.py",
    "sparsemm_native/two_bank.py", "sparsemm_native/policy.py", "sparsemm_native/cache.py",
    "sparsemm_native/streaming.py", "sparsemm_native/assets.py",
    "sparsemm_native/compare_ovobench_three_way.py", "sparsemm_protokv/config.py",
    "sparsemm_protokv/cache.py", "sparsemm_protokv/streaming.py",
    "sparsemm_protokv/monkeypatch.py", "sparsemm_protokv/model_loader.py",
    "sparsemm_protokv/models/qwen2_5_vl_stream.py", "sparsemm_baselines/cache.py",
    "sparsemm_baselines/core.py", "sparsemm_baselines/adapter.py",
)
GPU_LOCK = Path("/tmp/hermes-gpu0.lock")


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def write_json(path, value):
    atomic_write(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def event(root, kind, **details):
    record = {"time": now(), "event": kind, **details}
    print(json.dumps(record, ensure_ascii=False), flush=True)
    with (root / "supervisor_events.jsonl").open("a") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def worker_command(root, part, python=sys.executable):
    return [python, "-u", "-m", "sparsemm_native.run_ovobench", "--arm", "dp_dual",
            "--dataset", part, "--output-root", str(Path(root).resolve() / part),
            "--grouped", "--max-new", "100000", "--budget", "500", "--fps", "0.5",
            "--chunk-size", "16", "--max-new-tokens", "16", "--video-max-pixels", "151200"]


def worker_environment():
    env = dict(os.environ)
    env.update(PYTHONPATH=os.pathsep.join((str(ROOT), str(ROOT / "sparsemm_baselines/upstream/lmms-eval"),
                                         str(ROOT / "sparsemm_protokv/runtime_deps"))),
               HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MALLOC_ARENA_MAX="2")
    return env


def process_record(pid):
    try:
        directory = Path("/proc") / str(pid)
        command = [p.decode() for p in directory.joinpath("cmdline").read_bytes().split(b"\0") if p]
        fields = directory.joinpath("stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"pid": int(pid), "start_ticks": fields[19], "command": command}
    except (OSError, ValueError, IndexError, UnicodeError):
        return None


def find_worker(root, part):
    expected = worker_command(root, part)
    found = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        record = process_record(directory.name)
        if record is None:
            continue
        command = record["command"]
        if ("sparsemm_native.run_ovobench" not in command
                or "--output-root" not in command
                or str(Path(root).resolve() / part) not in command):
            continue
        if command[1:] != expected[1:]:
            raise ValueError(f"Conflicting worker for {part}: PID {record['pid']} {command}")
        found.append(record)
    if len(found) > 1:
        raise ValueError(f"Multiple workers already running for {part}: {found}")
    return found[0] if found else None


def gpu_available():
    with GPU_LOCK.open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        return True


def stop_worker(record):
    """Stop only the verified detached worker, never another process group."""
    current = process_record(record["pid"])
    if current is None:
        return
    if (current != record or os.getpgid(record["pid"]) != record["pid"]
            or os.getsid(record["pid"]) != record["pid"]):
        raise ValueError("Refusing to stop an unverified/non-detached worker")
    os.killpg(record["pid"], signal.SIGTERM)
    deadline = time.monotonic() + 20
    while process_record(record["pid"]) is not None and time.monotonic() < deadline:
        time.sleep(1)
    if process_record(record["pid"]) is not None:
        os.killpg(record["pid"], signal.SIGKILL)


def progress_rows(path):
    """Read live results, allowing only a currently incomplete final append."""
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open("rb") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if not line.endswith(b"\n") and not stream.read(1):
                    break
                raise ValueError(f"Malformed complete result record: {path}")
            if not isinstance(row, dict):
                raise ValueError(f"Non-object result record: {path}")
            rows.append(row)
    return rows


def repair_jsonl_tail(path, *, validate_complete=True):
    """Before resuming a stopped worker, preserve then remove an incomplete tail.

    Complete records, including a valid record lacking its newline, survive.
    Small result files validate all records. Set validate_complete=False for
    large traces: inspect only their final append; full trace auditing remains
    separate. Malformed complete records that are inspected are never removed.
    """
    path = Path(path)
    if not path.exists():
        return None
    with path.open("rb") as stream:
        if not validate_complete:
            stream.seek(0, os.SEEK_END)
            end = stream.tell()
            if not end:
                return None
            stream.seek(end - 1)
            if stream.read(1) == b"\n":
                return None
            cursor = end
            while cursor:
                start = max(0, cursor - 65536)
                stream.seek(start)
                block = stream.read(cursor - start)
                newline = block.rfind(b"\n")
                if newline >= 0:
                    stream.seek(start + newline + 1)
                    break
                cursor = start
            else:
                stream.seek(0)
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                return None
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Non-object JSONL record: {path}")
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                if line.endswith(b"\n") or stream.read(1) or not line.lstrip().startswith(b"{"):
                    raise ValueError(f"Malformed complete/middle JSONL record: {path}") from error
                if isinstance(error, json.JSONDecodeError):
                    suffix = error.doc[error.pos:].strip()
                    partial_literal = suffix in ("t", "tr", "tru", "f", "fa", "fal", "fals", "n", "nu", "nul")
                    if (not error.msg.startswith("Unterminated string")
                            and error.pos < len(error.doc.rstrip()) - 1 and not partial_literal):
                        raise ValueError(f"Corrupt JSONL tail is not an incomplete append: {path}") from error
                backup = path.with_name(f"{path.name}.incomplete.{time.time_ns()}.bin")
                backup.write_bytes(line)
                with path.open("r+b") as output:
                    output.truncate(offset)
                    output.flush()
                    os.fsync(output.fileno())
                return {"path": str(path), "removed_bytes": len(line), "backup": str(backup)}
            if not line.endswith(b"\n"):
                with path.open("ab") as output:
                    output.write(b"\n")
                    output.flush()
                    os.fsync(output.fileno())
                return {"path": str(path), "added_final_newline": True}


def preflight(root):
    if Path(runner.__file__).resolve() != ROOT / "sparsemm_native/run_ovobench.py":
        raise ValueError("Wrong run_ovobench module imported; run with cwd=kvbench")
    expected = {"hermes9": runner.hermes_queries(), "forward3": runner.forward_queries()}
    files = {str(ROOT / name): runner.sha256(ROOT / name) for name in SOURCE_FILES}
    assets = [runner.DP_BUDGET, runner.DP_DUAL_BUDGET, runner.ANNOTATIONS, runner.HERMES_ANNOTATIONS]
    assets.extend(runner.MODEL / name for name in (
        "config.json", "preprocessor_config.json", "tokenizer_config.json", "tokenizer.json",
        "chat_template.json", "generation_config.json"))
    files.update({str(path.resolve()): runner.sha256(path) for path in assets})
    for part, baseline in BASELINES.items():
        reference = json.loads((baseline / "dp/protocol.json").read_text())
        for arm in ("native", "dp"):
            directory = baseline / arm
            protocol_path = directory / "protocol.json"
            files[str(protocol_path)] = runner.sha256(protocol_path)
            protocol = json.loads(protocol_path.read_text())
            source = runner.HERMES_ANNOTATIONS if part == "hermes9" else runner.ANNOTATIONS
            if (protocol.get("dataset") != part or protocol.get("arm") != arm
                    or protocol.get("query_count") != TOTALS[part]
                    or protocol.get("query_list_sha256") != _digest(expected[part])
                    or protocol.get("source_sha256") != runner.sha256(source)
                    or protocol.get("model_config_sha256") != runner.sha256(runner.MODEL / "config.json")):
                raise ValueError(f"Baseline protocol/query drift: {part}/{arm}")
            for field in PROTOCOL_FIELDS:
                if protocol.get(field) != reference.get(field):
                    raise ValueError(f"Baseline protocol mismatch: {part}/{arm}/{field}")
            fixed = {"budget": 500, "fps": 0.5, "chunk_size": 16, "max_new_tokens": 16,
                     "video_max_pixels": 151200, "dtype": "bfloat16"}
            if any(protocol.get(name) != value for name, value in fixed.items()):
                raise ValueError(f"Baseline differs from pinned inference settings: {part}/{arm}")
            rows_path = directory / "queries.jsonl"
            rows = _read_rows(rows_path)
            _validate_rows(rows, expected[part], arm, part, require_complete=True)
            files[str(rows_path)] = runner.sha256(rows_path)
        if reference.get("dp_budget_sha256") != runner.sha256(runner.DP_BUDGET):
            raise ValueError(f"Baseline DP capacity asset changed: {part}")
    manifest = {"experiment": "one_dp_dual_experiment_3035_queries", "total_queries": 3035,
                "parts": TOTALS, "baseline_roots": {k: str(v) for k, v in BASELINES.items()},
                "query_hashes": {k: _digest(v) for k, v in expected.items()}, "files_sha256": files,
                "python": str(Path(sys.executable).resolve()),
                "dependencies": {name: version(name) for name in ("torch", "transformers", "numpy", "decord", "Pillow")}}
    destination = root / "supervision_manifest.json"
    if destination.exists():
        if json.loads(destination.read_text()) != manifest:
            raise ValueError("Supervision manifest drift; refusing to resume with changed code/protocol/assets")
    else:
        write_json(destination, manifest)
    return manifest


def check_pins(manifest):
    for name, digest in manifest["files_sha256"].items():
        if runner.sha256(name) != digest:
            raise ValueError(f"Pinned code/protocol/asset changed: {name}")


def publish(root, counts):
    reports = {}
    for part in TOTALS:
        directory = root / part
        if not (directory / "dp_dual/protocol.json").exists():
            continue
        try:
            report = compare(BASELINES[part], directory)
        except ValueError as error:
            # A writer can be between appending JSON and its newline. No file
            # is repaired while that worker is alive; try the next report tick.
            if "Malformed result at" in str(error) and find_worker(root, part):
                event(root, "comparison_deferred_for_live_append", part=part)
                return
            raise
        reports[part] = report
        write_json(directory / "comparison.json", report)
        atomic_write(directory / "comparison.md", markdown(report))
    lines = ["# OVO-Bench DP 双缓存实验实时总表", "",
             f"更新时间：{now()}。一项实验，覆盖进度 **{sum(counts.values())}/3035**。", "",
             "K=500，FPS=0.5，chunk=16，像素上限151200；仅运行新双缓存，SparseMM 和旧 DP 复用已有成绩。", "",
             "前九类沿用 HERMES 提示词，前瞻三类沿用官方提示词；3035 是覆盖题数，合计不作为官方总分。", "",
             "| 协议 | 同题数 / 全量 | SparseMM 答对 / 正确率 | DP 原容量版 | DP 双缓存 | 双缓存 − SparseMM | 双缓存 − 原 DP |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    combined = {arm: 0 for arm in ARMS}
    matched = 0
    for part in TOTALS:
        if part not in reports:
            lines.append(f"| {part} | 0 / {TOTALS[part]} | — | — | — | — | — |")
            continue
        values = reports[part]["subtotal"]
        n = values["matched_queries"]
        matched += n
        cells = []
        for arm in ARMS:
            correct = values[f"{arm}_correct"]
            combined[arm] += correct
            cells.append(f"{correct} / {100 * correct / n:.2f}%" if n else "0 / —")
        delta = [f"{values[f'dp_dual_minus_{arm}_percentage_points']:+.2f} pp" if n else "—"
                 for arm in ("native", "dp")]
        lines.append(f"| {part} | {n} / {TOTALS[part]} | " + " | ".join(cells + delta) + " |")
    if matched:
        cells = [f"{combined[arm]} / {100 * combined[arm] / matched:.2f}%" for arm in ARMS]
        delta = [f"{100 * (combined['dp_dual'] - combined[arm]) / matched:+.2f} pp" for arm in ("native", "dp")]
        lines.append(f"| 混合协议覆盖合计（非官方分数） | {matched} / 3035 | " + " | ".join(cells + delta) + " |")
    for part, report in reports.items():
        lines.extend(["", markdown(report)])
    atomic_write(root / "live_report.md", "\n".join(lines) + "\n")


def supervise(root, poll_seconds=15, report_seconds=60, max_no_progress_retries=3):
    root.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    with (root / ".supervisor.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("A supervisor already owns this experiment", flush=True)
            return 0
        started = now()
        status = {"supervisor_pid": os.getpid(), "started_at": started, "total_queries": 3035}
        try:
            manifest = preflight(root)
            child = None
            active = None
            started_count = 0
            no_progress = 0
            retry_at = 0
            last_report = 0
            last_report_counts = None
            while True:
                counts = {part: len(progress_rows(root / part / "dp_dual/queries.jsonl")) for part in TOTALS}
                if any(counts[part] > TOTALS[part] for part in TOTALS):
                    raise ValueError("Result count exceeds pinned experiment size")
                unfinished = next((name for name in TOTALS if counts[name] < TOTALS[name]), None)
                live = [(name, record) for name in TOTALS
                        if (record := find_worker(root, name)) is not None]
                if len(live) > 1:
                    raise ValueError("More than one inference worker exists for this experiment")
                part, record = live[0] if live else (unfinished, None)
                if record and part != unfinished and counts[part] < TOTALS[part]:
                    raise ValueError("A worker is running a later part before the earlier part completed")
                # A stopped writer's incomplete result tail must be repaired
                # before compare() reads it. Never repair a live writer's file.
                for name in TOTALS:
                    if not record or name != part:
                        repair = repair_jsonl_tail(root / name / "dp_dual/queries.jsonl")
                        if repair:
                            event(root, "incomplete_tail_repaired", **repair)
                status.update(updated_at=now(), completed_queries=sum(counts.values()), counts=counts,
                              current_part=part, worker_pid=record["pid"] if record else None,
                              no_progress_retries=no_progress,
                              state="running" if record else "waiting", error=None)
                if (root / "PAUSE").exists():
                    if record:
                        stop_worker(record)
                    if child is not None:
                        child.wait(timeout=30)
                    status.update(state="paused", worker_pid=None, updated_at=now())
                    write_json(root / "status.json", status)
                    event(root, "paused", completed_queries=sum(counts.values()))
                    return 0
                if sum(counts.values()) == 3035:
                    # Wait for any worker's final model teardown before marking
                    # the sole experiment complete and releasing supervision.
                    if record is not None or (child is not None and child.poll() is None):
                        status["state"] = "finishing_worker_teardown"
                        write_json(root / "status.json", status)
                        time.sleep(poll_seconds)
                        continue
                    check_pins(manifest)
                    publish(root, counts)
                    status.update(state="complete", current_part=None, worker_pid=None, updated_at=now())
                    write_json(root / "status.json", status)
                    event(root, "complete", completed_queries=3035)
                    return 0
                if active is not None and record is None:
                    exit_code = child.poll() if child is not None else None
                    progressed = counts[active] > started_count
                    no_progress = 0 if progressed else no_progress + 1
                    event(root, "worker_stopped", part=active, exit_code=exit_code,
                          progressed=progressed, completed=counts[active], no_progress_retries=no_progress)
                    if no_progress > max_no_progress_retries:
                        raise RuntimeError("Worker repeatedly exited without progress; retry limit reached")
                    retry_at = time.monotonic() + (min(30 * 2 ** (no_progress - 1), 300) if no_progress else 0)
                    child, active = None, None
                if record is not None and active is None:
                    active, started_count = part, counts[part]
                    write_json(root / "worker.json", {**record, "part": part, "attached_at": now()})
                    event(root, "worker_attached", part=part, pid=record["pid"])
                if (time.monotonic() - last_report >= report_seconds
                        and counts != last_report_counts):
                    check_pins(manifest)
                    publish(root, counts)
                    last_report, last_report_counts = time.monotonic(), counts.copy()
                if record is None and time.monotonic() >= retry_at:
                    check_pins(manifest)
                    if gpu_available():
                        directory = root / part / "dp_dual"
                        directory.mkdir(parents=True, exist_ok=True)
                        for name in ("queries.jsonl", "cache_trace.jsonl"):
                            repair = repair_jsonl_tail(directory / name,
                                                       validate_complete=name == "queries.jsonl")
                            if repair:
                                event(root, "incomplete_tail_repaired", **repair)
                        with (root / part / "dp_dual.log").open("a") as log:
                            child = subprocess.Popen(worker_command(root, part), cwd=ROOT,
                                                     env=worker_environment(), stdin=subprocess.DEVNULL,
                                                     stdout=log, stderr=subprocess.STDOUT,
                                                     start_new_session=True, close_fds=True)
                        active, started_count = part, counts[part]
                        record = process_record(child.pid) or {"pid": child.pid}
                        write_json(root / "worker.json", {**record, "part": part, "launched_at": now()})
                        event(root, "worker_launched", part=part, pid=child.pid, completed=started_count)
                        status.update(worker_pid=child.pid, state="running")
                    else:
                        status["state"] = "waiting_for_gpu_lease"
                elif record is None:
                    status["state"] = "retry_backoff"
                status["no_progress_retries"] = no_progress
                write_json(root / "status.json", status)
                time.sleep(poll_seconds)
        except Exception as error:
            status.update(state="failed", updated_at=now(), error=f"{type(error).__name__}: {error}")
            write_json(root / "status.json", status)
            event(root, "failed", error=status["error"])
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=15)
    parser.add_argument("--report-seconds", type=float, default=60)
    parser.add_argument("--max-no-progress-retries", type=int, default=3)
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0 or args.report_seconds <= 0 or args.max_no_progress_retries < 0:
        parser.error("Polling/report intervals must be positive and retry count nonnegative")
    return supervise(args.output_root.resolve(), args.poll_seconds, args.report_seconds,
                     args.max_no_progress_retries)


if __name__ == "__main__":
    sys.exit(main())
