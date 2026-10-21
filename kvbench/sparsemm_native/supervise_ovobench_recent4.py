"""Detached two-worker supervision of one all-official, three-arm OVO run.

Each arm covers the same 3035 queries. Validated official controls are reused;
all missing predictions retain global same-video prefix reuse. Create PAUSE
inside the output root to stop the verified workers and checkpoint progress.
"""

import argparse
import collections
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from sparsemm_native import run_ovobench_recent4 as runner
from sparsemm_native import report_ovobench_recent4 as reporting
from sparsemm_native.supervise_ovobench_dual import (
    atomic_write, check_pins, event, now, process_record, progress_rows,
    repair_jsonl_tail, stop_worker, worker_environment, write_json,
)


ROOT = runner.ROOT
ORDER = ("dp_recent4", "native", "dp")
REUSE_ROOTS = (
    ROOT / "results/ovobench_official_full_k500_fps05_p151200_20261019",
    ROOT / "results/ovobench_forward3_k500_fps05_p151200_20261020",
)
EXTRA_SOURCES = (
    "sparsemm_native/supervise_ovobench_recent4.py",
    "sparsemm_native/supervise_ovobench_dual.py",
    "sparsemm_native/report_ovobench_recent4.py",
)


def worker_command(root, arm, python=sys.executable):
    if arm not in runner.ARMS:
        raise ValueError("Unknown comparison arm")
    command = [python, "-u", "-m", "sparsemm_native.run_ovobench_recent4",
               "--arm", arm, "--output-root", str(Path(root).resolve()),
               "--budget", "500", "--fps", "0.5", "--chunk-size", "16",
               "--max-new-tokens", "16", "--video-max-pixels", "151200",
               "--grouped", "--concurrent-processes", "2", "--max-new", "100000"]
    if arm != "dp_recent4":
        for source in REUSE_ROOTS:
            command += ["--reuse-root", str(source.resolve())]
    return command


def find_workers(root):
    """Adopt surviving detached children after a supervisor restart."""
    expected = {arm: worker_command(root, arm) for arm in runner.ARMS}
    found = {}
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        record = process_record(directory.name)
        if record is None:
            continue
        command = record["command"]
        if ("sparsemm_native.run_ovobench_recent4" not in command
                or "--output-root" not in command
                or str(Path(root).resolve()) not in command):
            continue
        arm = next((a for a, wanted in expected.items() if command[1:] == wanted[1:]), None)
        if arm is None or arm in found:
            raise ValueError(f"Conflicting/duplicate worker for this experiment: {command}")
        try:
            detached = (os.getpgid(record["pid"]) == record["pid"]
                        and os.getsid(record["pid"]) == record["pid"])
        except ProcessLookupError:
            # A worker can finish between reading /proc and inspecting its
            # session. The next loop will collect its committed results.
            continue
        if not detached:
            raise ValueError("A comparison worker is not detached")
        found[arm] = record
    if len(found) > 2:
        raise ValueError("More than two inference processes belong to this experiment")
    return found


def preflight(root):
    if Path(runner.__file__).resolve() != ROOT / "sparsemm_native/run_ovobench_recent4.py":
        raise ValueError("Wrong module imported; run from the kvbench directory")
    official = runner.legacy.queries()
    selected = runner.schedule_queries(official)
    if len(selected) != 3035 or len({q["video_path"] for q in selected}) != 644:
        raise ValueError("The official query/video manifest changed")
    protocols, reused = {}, {}
    for arm in runner.ARMS:
        args = runner.parse_args(worker_command(root, arm)[4:])
        protocols[arm] = runner.build_protocol(args, selected, official)
        reused[arm] = runner.reusable_rows(args.reuse_root, arm, official, protocols[arm])
    files = {str(ROOT / name): runner.legacy.sha256(ROOT / name)
             for name in runner.SOURCE_FILES + EXTRA_SOURCES}
    source = protocols["native"]
    assets = [runner.legacy.ANNOTATIONS, runner.legacy.DP_BUDGET, runner.legacy.DP_DUAL_BUDGET]
    assets.extend(runner.legacy.MODEL / name for name in source["processor_files_sha256"])
    from sparsemm_native.policy import DEFAULT_SCORE_PATH
    assets.append(DEFAULT_SCORE_PATH)
    files.update({str(path.resolve()): runner.legacy.sha256(path) for path in assets})
    for protocol in protocols.values():
        files.update(protocol["reuse_source_files_sha256"])
    manifest = dict(
        experiment="official_ovo_recent4_three_arm_comparison", created_protocol_version=1,
        total_queries_per_arm=3035, unique_source_videos=644, task_counts=dict(collections.Counter(
            q["task"] for q in official)), concurrent_inference_processes=2,
        launch_order=list(ORDER), protocols=protocols, files_sha256=files,
        reused_counts={arm: len(rows) for arm, rows in reused.items()},
        predictions_to_compute={arm: 3035 - len(rows) for arm, rows in reused.items()},
        python=str(Path(sys.executable).resolve()), query_list_sha256=runner.digest(selected),
        full_official_query_list_sha256=runner.digest(official),
    )
    destination = root / "supervision_manifest.json"
    if destination.exists() and json.loads(destination.read_text()) != manifest:
        raise ValueError("Pinned supervisor/code/assets/protocol changed; refusing to resume")
    # Validate every existing arm before writing anything into its namespace.
    for arm, protocol in protocols.items():
        directory = root / arm
        protocol_path = directory / "protocol.json"
        if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
            raise ValueError(f"Existing protocol differs: {arm}")
        rows = progress_rows(directory / "queries.jsonl")
        if rows and not protocol_path.exists():
            raise ValueError(f"Existing results have no protocol: {arm}")
        validate_progress(rows, selected)
    write_json(destination, manifest)
    write_json(root / "query_manifest.json", selected)
    for arm, protocol in protocols.items():
        directory = root / arm
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / "protocol.json", protocol)
        write_json(directory / "query_manifest.json", selected)
        write_json(directory / "reusable_predictions.json", reused[arm])
        for name in runner.SOURCE_FILES:
            target = directory / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
    for name in EXTRA_SOURCES:
        target = root / "supervision_source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    return manifest, selected, official, reused


def validate_progress(rows, selected):
    if len(rows) > len(selected):
        raise ValueError("Progress exceeds the full official manifest")
    for row, query in zip(rows, selected):
        runner.validate_row(row, query)


def collect(root, selected, reused):
    written, available, inferred = {}, {}, {}
    for arm in runner.ARMS:
        rows = progress_rows(root / arm / "queries.jsonl")
        validate_progress(rows, selected)
        written[arm] = len(rows)
        available[arm] = dict(reused[arm])
        inferred[arm] = sum(not bool(row.get("reused_from")) for row in rows)
        for row in rows:
            old = available[arm].get(row["query_id"])
            if old is not None and old["response"] != row["response"]:
                raise ValueError("Reused and appended predictions disagree")
            available[arm][row["query_id"]] = row
    return written, available, inferred


def publish(root, official, available, status):
    matched = []
    for query in official:
        qid = query["query_id"]
        if all(qid in available[arm] for arm in runner.ARMS):
            item = {arm: available[arm][qid] for arm in runner.ARMS}
            if len({row["frames"] for row in item.values()}) != 1:
                raise ValueError(f"Different frame counts for matched query {qid}")
            matched.append(item)
    grouped = collections.defaultdict(list)
    for item in matched:
        grouped[item["native"]["task"]].append(item)
    totals = collections.Counter(q["task"] for q in official)
    complete = len(matched) == 3035
    summaries = {arm: runner.legacy.summary([item[arm] for item in matched], 3035)
                 for arm in runner.ARMS}
    report = dict(
        dataset="official_full", budget=500, fps=0.5, total_queries=3035,
        completed_queries={arm: len(available[arm]) for arm in runner.ARMS}, complete=complete,
        tasks={task: dict(expected=totals[task], **reporting.metrics(grouped[task])) for task in sorted(totals)},
        matched=reporting.metrics(matched),
        group_accuracy={arm: summaries[arm]["group_accuracy"] for arm in runner.ARMS},
        official_total_average={arm: summaries[arm]["official_total_average"] if complete else None
                                for arm in runner.ARMS},
        aggregation="Mean of backward (3-task mean), realtime (6-task mean), forward (3-task mean)",
        updated_at=now(),
        note="Only identical official queries are compared; verified prior official control predictions are included. "
             "The official total is shown after all 3035 matched queries complete.",
    )
    write_json(root / "comparison.json", report)
    introduction = (f"更新时间：{now()}。官方12类、3035题；K500、FPS0.5。\n\n"
                    "三个方法统一官方提示词、标签与评分；最多两个推理进程，同视频多题共享对应时间点的前缀。\n\n"
                    f"可用题数：{status['available_queries']}；本次新增推理题数：{status['new_inference_queries']}。\n\n")
    atomic_write(root / "live_report.md", introduction + reporting.markdown(report) + "\n")
    return report


def supervise(root, poll_seconds=10, report_seconds=60, max_no_progress_retries=3):
    root.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    with (root / ".supervisor.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("A supervisor already owns this experiment")
        status = dict(supervisor_pid=os.getpid(), started_at=now(), state="preflight",
                      total_queries_per_arm=3035, concurrent_processes=2)
        children, active, failures, retry_at = {}, {}, collections.Counter(), {}
        last_report, last_counts = 0, None
        try:
            manifest, selected, official, reused = preflight(root)
            event(root, "preflight_passed", reuse_counts=manifest["reused_counts"],
                  remaining_inference=manifest["predictions_to_compute"])
            while True:
                live = find_workers(root)
                for arm in runner.ARMS:
                    if arm not in live:
                        for filename in ("queries.jsonl", "cache_trace.jsonl"):
                            repair = repair_jsonl_tail(root / arm / filename,
                                                       validate_complete=filename == "queries.jsonl")
                            if repair:
                                event(root, "incomplete_tail_repaired", arm=arm, **repair)
                written, available, inferred = collect(root, selected, reused)
                for arm in list(active):
                    if arm in live:
                        continue
                    child = children.pop(arm, None)
                    if child is not None and child.poll() is None:
                        continue
                    progressed = written[arm] > active.pop(arm)["started_count"]
                    if written[arm] < 3035:
                        failures[arm] = 0 if progressed else failures[arm] + 1
                        if failures[arm] > max_no_progress_retries:
                            raise RuntimeError(f"{arm} repeatedly exited without progress; see its log")
                        retry_at[arm] = time.monotonic() + min(15 * 2 ** max(0, failures[arm] - 1), 120)
                    event(root, "worker_exited", arm=arm, exit_code=child.returncode if child else None,
                          progressed=progressed, written_queries=written[arm], retries=failures[arm])
                for arm, record in live.items():
                    if arm not in active:
                        active[arm] = dict(record, started_count=written[arm])
                        event(root, "worker_attached", arm=arm, pid=record["pid"])
                status.update(updated_at=now(), written_queries=written,
                              available_queries={a: len(v) for a, v in available.items()},
                              new_inference_queries=inferred, reused_queries=manifest["reused_counts"],
                              worker_pids={a: r["pid"] for a, r in live.items()},
                              no_progress_retries=dict(failures), error=None)
                if (root / "PAUSE").exists():
                    for record in live.values():
                        stop_worker(record)
                    status.update(state="paused", worker_pids={}, updated_at=now())
                    write_json(root / "status.json", status)
                    publish(root, official, available, status)
                    event(root, "paused", counts=status["available_queries"])
                    return 0
                if all(n == 3035 for n in written.values()) and not live:
                    check_pins(manifest)
                    status.update(state="complete", worker_pids={}, updated_at=now())
                    write_json(root / "status.json", status)
                    publish(root, official, available, status)
                    event(root, "complete", counts=status["available_queries"])
                    return 0
                for arm in ORDER:
                    if len(live) >= 2:
                        break
                    if arm in live or arm in active or written[arm] == 3035 or time.monotonic() < retry_at.get(arm, 0):
                        continue
                    check_pins(manifest)
                    with (root / f"{arm}.log").open("a") as log:
                        child = subprocess.Popen(
                            worker_command(root, arm), cwd=ROOT, env=worker_environment(),
                            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True, close_fds=True)
                    children[arm] = child
                    record = process_record(child.pid)
                    if record is None:
                        record = dict(pid=child.pid)
                    active[arm] = dict(record, started_count=written[arm])
                    live[arm] = record
                    event(root, "worker_launched", arm=arm, pid=child.pid, written=written[arm])
                status.update(state="running" if live else "retry_backoff",
                              worker_pids={a: r["pid"] for a, r in live.items()})
                write_json(root / "status.json", status)
                counts = (written, {arm: len(rows) for arm, rows in available.items()})
                if time.monotonic() - last_report >= report_seconds and counts != last_counts:
                    check_pins(manifest)
                    result = publish(root, official, available, status)
                    status["matched_queries"] = result["matched"]["matched_queries"]
                    write_json(root / "status.json", status)
                    last_report, last_counts = time.monotonic(), counts
                time.sleep(poll_seconds)
        except Exception as error:
            # Stop only children verified against this experiment's exact CLI.
            for record in find_workers(root).values():
                stop_worker(record)
            status.update(state="failed", error=f"{type(error).__name__}: {error}", updated_at=now(), worker_pids={})
            write_json(root / "status.json", status)
            event(root, "failed", error=status["error"])
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=10)
    parser.add_argument("--report-seconds", type=float, default=60)
    parser.add_argument("--max-no-progress-retries", type=int, default=3)
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0 or args.report_seconds <= 0 or args.max_no_progress_retries < 0:
        parser.error("Polling/report intervals must be positive and retry count nonnegative")
    return supervise(args.output_root.expanduser().resolve(), args.poll_seconds,
                     args.report_seconds, args.max_no_progress_retries)


if __name__ == "__main__":
    sys.exit(main())
