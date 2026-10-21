"""Run only recent-four DP on two whole-video shards; read existing controls.

Both workers are detached and resumable. No SparseMM or old-DP inference is
launched. Existing official controls are used only for matched-query reports.
Create OUTPUT/PAUSE to checkpoint and stop both verified workers.
"""

import argparse
import collections
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from sparsemm_native import supervise_ovobench_recent4 as base


runner = base.runner
ROOT = base.ROOT
JOBS = ("shard0", "shard1")


def split_videos(selected):
    """Balance estimated encoded frames, never divide one source video's QA."""
    groups = runner.legacy.grouped_batches(selected, len(selected))
    costs = []
    for i, group in enumerate(groups):
        targets = [math.ceil(q["clip_end"] * 0.5) for q in group]
        frames = max(t - t % 2 for t in targets) + sum(t % 2 for t in targets)
        costs.append((frames + len(group), i, frames))
    bins, weights, frames = [[], []], [0, 0], [0, 0]
    for cost, index, count in sorted(costs, key=lambda item: (-item[0], item[1])):
        slot = min(range(2), key=lambda x: (weights[x], x))
        bins[slot].append(index)
        weights[slot] += cost
        frames[slot] += count
    jobs = {}
    for slot, job in enumerate(JOBS):
        items = [q for index in sorted(bins[slot]) for q in groups[index]]
        jobs[job] = dict(queries=items, query_ids=[q["query_id"] for q in items],
                         videos=len(bins[slot]), expected_encoded_frames=frames[slot])
    if (set(jobs["shard0"]["query_ids"]) & set(jobs["shard1"]["query_ids"])
            or sum(len(j["queries"]) for j in jobs.values()) != len(selected)
            or ({q["video_path"] for q in jobs["shard0"]["queries"]}
                & {q["video_path"] for q in jobs["shard1"]["queries"]})):
        raise AssertionError("Video-disjoint sharding did not preserve the official query set")
    return jobs


def worker_command(root, name, job, python=sys.executable):
    return [python, "-u", "-m", "sparsemm_native.run_ovobench_recent4",
            "--arm", "dp_recent4", "--output-root", str((Path(root) / "shards" / name).resolve()),
            "--ids", ",".join(job["query_ids"]), "--budget", "500", "--fps", "0.5",
            "--chunk-size", "16", "--max-new-tokens", "16", "--video-max-pixels", "151200",
            "--grouped", "--concurrent-processes", "2", "--max-new", "100000"]


def find_workers(root, jobs):
    expected = {name: worker_command(root, name, job) for name, job in jobs.items()}
    found = {}
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        record = base.process_record(directory.name)
        if record is None or "sparsemm_native.run_ovobench_recent4" not in record["command"]:
            continue
        command = record["command"]
        if "--output-root" not in command:
            continue
        destination = command[command.index("--output-root") + 1]
        if not destination.startswith(str((Path(root) / "shards").resolve()) + os.sep):
            continue
        name = next((n for n, wanted in expected.items() if command[1:] == wanted[1:]), None)
        if name is None or name in found:
            raise ValueError("Conflicting/duplicate new-method shard worker")
        try:
            detached = os.getpgid(record["pid"]) == record["pid"] == os.getsid(record["pid"])
        except ProcessLookupError:
            continue
        if not detached:
            raise ValueError("An inference shard is not detached")
        found[name] = record
    return found


def prepare(root):
    manifest, selected, official, reused = base.preflight(root)
    jobs = split_videos(selected)
    for name, job in jobs.items():
        args = runner.parse_args(worker_command(root, name, job)[4:])
        protocol = runner.build_protocol(args, job["queries"], official)
        directory = root / "shards" / name / "dp_recent4"
        path = directory / "protocol.json"
        if path.exists() and json.loads(path.read_text()) != protocol:
            raise ValueError("Shard protocol/code changed; refusing to resume")
        base.validate_progress(base.progress_rows(directory / "queries.jsonl"), job["queries"])
        directory.mkdir(parents=True, exist_ok=True)
        base.write_json(path, protocol)
        base.write_json(directory / "query_manifest.json", job["queries"])
    plan = dict(
        phase="new_method_only", run_baseline_inference=False,
        total_queries=3035, unique_source_videos=644, concurrent_processes=2,
        budget=500, fps=0.5, official_protocol=True, grouped_by_video=True,
        source_sha256=runner.legacy.sha256(__file__),
        jobs={name: {key: value for key, value in job.items() if key != "queries"}
              for name, job in jobs.items()},
        baseline_reused_counts=manifest["reused_counts"],
        expected_encoded_frames=sum(job["expected_encoded_frames"] for job in jobs.values()),
        encoded_frame_estimate_note="FPS0.5 prefix targets, including odd branches; source-end clamping may lower this.",
    )
    destination = root / "recent4_only_manifest.json"
    if destination.exists() and json.loads(destination.read_text()) != plan:
        raise ValueError("New-method shard plan changed; refusing to resume")
    base.write_json(destination, plan)
    target = root / "supervision_source_snapshot/sparsemm_native/supervise_ovobench_recent4_only.py"
    target.write_bytes(Path(__file__).read_bytes())
    manifest["files_sha256"][str(Path(__file__).resolve())] = plan["source_sha256"]
    return manifest, selected, official, reused, jobs, plan


def collect(root, jobs, reused):
    available = {arm: dict(rows) for arm, rows in reused.items()}
    progress, measurements = {}, {}
    for name, job in jobs.items():
        rows = base.progress_rows(root / "shards" / name / "dp_recent4/queries.jsonl")
        base.validate_progress(rows, job["queries"])
        progress[name] = len(rows)
        for row in rows:
            if row["query_id"] in available["dp_recent4"]:
                raise ValueError("Duplicate prediction across shards")
            available["dp_recent4"][row["query_id"]] = dict(row, worker_shard=name)
        frames = sum(row["encoded_frames"] for row in rows)
        seconds = sum(row["seconds"] for row in rows)
        remaining = max(0, job["expected_encoded_frames"] - frames)
        measurements[name] = dict(
            completed_queries=len(rows), total_queries=len(job["queries"]),
            encoded_frames=frames, expected_encoded_frames=job["expected_encoded_frames"],
            measured_inference_seconds=seconds,
            frames_per_second=frames / seconds if seconds else None,
            estimated_remaining_seconds=remaining * seconds / frames if frames >= 100 and seconds > 0 else None)
    return progress, available, measurements


def publish(root, selected, official, available, status):
    report = base.publish(root, official, available, status)
    rows = [available["dp_recent4"][q["query_id"]] for q in selected
            if q["query_id"] in available["dp_recent4"]]
    summary = runner.legacy.summary(rows, 3035)
    base.write_json(root / "dp_recent4/summary.json", summary)
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    base.atomic_write(root / "dp_recent4/predictions_live.jsonl", content)
    if len(rows) == 3035:
        base.atomic_write(root / "dp_recent4/queries.jsonl", content)
    text = (root / "live_report.md").read_text()
    base.atomic_write(root / "live_report.md", "**当前只运行新方案，两个基线不启动推理；对照仅使用已验证的官方结果。**\n\n" + text)
    return report


def supervise(root, poll_seconds=10, report_seconds=60, max_no_progress_retries=3):
    root.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    with (root / ".supervisor.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("A supervisor already owns this experiment")
        status = dict(supervisor_pid=os.getpid(), started_at=base.now(), state="preflight",
                      phase="new_method_only", baseline_inference_running=False,
                      total_queries=3035, concurrent_processes=2)
        jobs, active, children, failures, retry_at = {}, {}, {}, collections.Counter(), {}
        last_report = 0
        try:
            manifest, selected, official, reused, jobs, plan = prepare(root)
            base.event(root, "new_method_only_preflight_passed", expected_frames=plan["expected_encoded_frames"],
                       shards={name: len(job["queries"]) for name, job in jobs.items()},
                       run_baseline_inference=False)
            while True:
                live = find_workers(root, jobs)
                for name in jobs:
                    if name not in live:
                        for filename in ("queries.jsonl", "cache_trace.jsonl"):
                            repair = base.repair_jsonl_tail(root / "shards" / name / "dp_recent4" / filename,
                                                           validate_complete=filename == "queries.jsonl")
                            if repair:
                                base.event(root, "incomplete_tail_repaired", shard=name, **repair)
                progress, available, measurements = collect(root, jobs, reused)
                for name in list(active):
                    if name in live:
                        continue
                    child = children.pop(name, None)
                    if child is not None and child.poll() is None:
                        continue
                    progressed = progress[name] > active.pop(name)["started_count"]
                    if progress[name] < len(jobs[name]["queries"]):
                        failures[name] = 0 if progressed else failures[name] + 1
                        if failures[name] > max_no_progress_retries:
                            raise RuntimeError(f"{name} repeatedly exited without progress; see its log")
                        retry_at[name] = time.monotonic() + min(15 * 2 ** max(0, failures[name] - 1), 120)
                    base.event(root, "worker_exited", shard=name, progressed=progressed,
                               exit_code=child.returncode if child else None, completed=progress[name])
                for name, record in live.items():
                    if name not in active:
                        active[name] = dict(record, started_count=progress[name])
                        base.event(root, "worker_attached", shard=name, pid=record["pid"])
                estimates = [m["estimated_remaining_seconds"] for m in measurements.values()]
                status.update(
                    updated_at=base.now(), completed_queries=sum(progress.values()), shard_queries=progress,
                    available_queries={arm: len(rows) for arm, rows in available.items()},
                    new_inference_queries=dict(native=0, dp=0, dp_recent4=sum(progress.values())),
                    reused_queries=plan["baseline_reused_counts"], worker_pids={n: r["pid"] for n, r in live.items()},
                    measurements=measurements, estimated_remaining_seconds=max(estimates)
                    if estimates and all(x is not None for x in estimates) else None,
                    no_progress_retries=dict(failures), error=None)
                paused = (root / "PAUSE").exists()
                complete = sum(progress.values()) == 3035 and not live
                if paused or complete:
                    for record in live.values():
                        base.stop_worker(record)
                    base.check_pins(manifest)
                    status.update(state="paused" if paused else "complete", worker_pids={}, updated_at=base.now())
                    base.write_json(root / "status.json", status)
                    publish(root, selected, official, available, status)
                    base.event(root, status["state"], completed_queries=sum(progress.values()))
                    return 0
                for name, job in jobs.items():
                    if name in live or name in active or progress[name] == len(job["queries"]) or time.monotonic() < retry_at.get(name, 0):
                        continue
                    base.check_pins(manifest)
                    with (root / f"{name}.log").open("a") as log:
                        child = subprocess.Popen(
                            worker_command(root, name, job), cwd=ROOT, env=base.worker_environment(),
                            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True, close_fds=True)
                    children[name] = child
                    record = base.process_record(child.pid) or dict(pid=child.pid)
                    active[name] = dict(record, started_count=progress[name])
                    live[name] = record
                    base.event(root, "worker_launched", shard=name, pid=child.pid,
                               completed_queries=progress[name], arm="dp_recent4")
                status.update(state="running" if live else "retry_backoff",
                              worker_pids={n: r["pid"] for n, r in live.items()})
                base.write_json(root / "status.json", status)
                if time.monotonic() - last_report >= report_seconds:
                    base.check_pins(manifest)
                    report = publish(root, selected, official, available, status)
                    status["matched_queries"] = report["matched"]["matched_queries"]
                    base.write_json(root / "status.json", status)
                    last_report = time.monotonic()
                time.sleep(poll_seconds)
        except Exception as error:
            for record in find_workers(root, jobs).values():
                base.stop_worker(record)
            status.update(state="failed", worker_pids={}, updated_at=base.now(), error=f"{type(error).__name__}: {error}")
            base.write_json(root / "status.json", status)
            base.event(root, "failed", error=status["error"])
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
