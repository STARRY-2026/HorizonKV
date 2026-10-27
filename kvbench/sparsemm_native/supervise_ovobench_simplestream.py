"""Detached, resumable, two-process full OVO SimpleStream four-frame control.

Only this new control performs inference. The completed official DP recent-four
predictions are read and validated for an identical-query comparison. Creating
OUTPUT_ROOT/PAUSE checkpoints and stops the two verified worker sessions.
"""

import argparse
import collections
import csv
import fcntl
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from sparsemm_native import run_ovobench_simplestream as runner
from sparsemm_native import supervise_ovobench_dual as base


ROOT = runner.ROOT
JOBS = ("shard0", "shard1")


def split_videos(selected):
    """Balance fresh-prefill QA and distinct image windows, never split video."""
    groups = runner.official.grouped_batches(selected, len(selected))
    costs = []
    for index, group in enumerate(groups):
        previous, images = set(), 0
        for query in group:
            target = math.ceil(query["clip_end"] * 0.5)
            window = set(range(max(0, target - 4), target))
            images += len(window - previous)
            previous = window
        costs.append((images + 2 * len(group), index, images))
    bins, loads, frames = [[], []], [0, 0], [0, 0]
    for cost, index, images in sorted(costs, key=lambda x: (-x[0], x[1])):
        slot = min(range(2), key=lambda x: (loads[x], x))
        bins[slot].append(index)
        loads[slot] += cost
        frames[slot] += images
    jobs = {}
    for slot, name in enumerate(JOBS):
        items = [query for index in sorted(bins[slot]) for query in groups[index]]
        jobs[name] = dict(queries=items, query_ids=[q["query_id"] for q in items],
                          source_videos=len(bins[slot]), expected_encoded_frames=frames[slot],
                          estimated_work_units=loads[slot])
    left, right = jobs.values()
    if ({q["video_path"] for q in left["queries"]} & {q["video_path"] for q in right["queries"]}
            or len(set(left["query_ids"]) | set(right["query_ids"])) != len(selected)
            or len(left["queries"]) + len(right["queries"]) != len(selected)):
        raise ValueError("Whole-video shards did not preserve the exact query set")
    return jobs


def worker_command(root, name, job, python=sys.executable):
    return [python, "-u", "-m", "sparsemm_native.run_ovobench_simplestream",
            "--output-root", str((Path(root) / "shards" / name).resolve()),
            "--ids", ",".join(job["query_ids"]), "--concurrent-processes", "2"]


def find_workers(root, jobs):
    expected = {name: worker_command(root, name, job) for name, job in jobs.items()}
    found = {}
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        record = base.process_record(directory.name)
        if record is None:
            continue
        command = record["command"]
        if "sparsemm_native.run_ovobench_simplestream" not in command or "--output-root" not in command:
            continue
        destination = command[command.index("--output-root") + 1]
        if not destination.startswith(str((Path(root) / "shards").resolve()) + os.sep):
            continue
        name = next((n for n, wanted in expected.items() if wanted[1:] == command[1:]), None)
        if name is None or name in found:
            raise ValueError("Conflicting/duplicate SimpleStream shard worker")
        try:
            detached = os.getsid(record["pid"]) == record["pid"] == os.getpgid(record["pid"])
        except ProcessLookupError:
            continue
        if not detached:
            raise ValueError("Inference worker is not detached from SSH/terminal")
        found[name] = record
    return found


def load_reference(official):
    reference = runner.REFERENCE
    manifest = json.loads((reference / "query_manifest.json").read_text())
    rows = runner.read_progress(reference / "queries.jsonl", manifest)
    if len(rows) != 3035 or len({r["query_id"] for r in rows}) != 3035:
        raise ValueError("The DP reference is not a complete unique official run")
    indexed = {q["query_id"]: q for q in official}
    if {q["query_id"]: q for q in manifest} != indexed:
        raise ValueError("DP reference and new control do not have identical official queries/prompts")
    protocol = json.loads((reference / "protocol.json").read_text())
    expected = dict(arm="dp_recent4", fps=0.5, budget=500, max_new_tokens=16, video_max_pixels=151200,
                    score_mode="official_OVOBenchOfflineScore", dtype="bfloat16", repetition_penalty=1.1)
    if any(protocol.get(k) != value for k, value in expected.items()):
        raise ValueError("DP reference inference/scoring settings differ")
    if (protocol.get("source_sha256") != runner.official.sha256(runner.official.ANNOTATIONS)
            or Path(protocol["model_path"]).resolve() != runner.official.MODEL.resolve()):
        raise ValueError("DP reference uses a different model or dataset")
    for name, checksum in protocol["processor_files_sha256"].items():
        if runner.official.sha256(runner.official.MODEL / name) != checksum:
            raise ValueError("DP reference model/processor asset changed")
    return {row["query_id"]: row for row in rows}


def prepare(root):
    expected = runner.official.queries()
    selected = runner.schedule_queries(expected)
    if len(selected) != 3035 or len({q["video_path"] for q in selected}) != 644:
        raise ValueError("Official full query/video manifest changed")
    reference = load_reference(expected)
    jobs = split_videos(selected)
    protocols = {}
    for name, job in jobs.items():
        args = runner.parse_args(worker_command(root, name, job)[4:])
        protocols[name] = runner.build_protocol(args, job["queries"], expected)
    sources = runner.SOURCE_FILES + ("sparsemm_native/supervise_ovobench_simplestream.py",)
    assets = [ROOT / name for name in sources]
    assets += [runner.official.MODEL / name for name in runner.PROCESSOR_FILES]
    assets += [runner.official.ANNOTATIONS]
    assets += [runner.UPSTREAM / name for name in protocols["shard0"]["upstream_sha256"]]
    assets += [runner.REFERENCE / name for name in ("protocol.json", "query_manifest.json", "queries.jsonl")]
    manifest = dict(
        experiment="official_ovo_simplestream_recent4_fresh_prefill", version=1,
        total_queries=3035, source_videos=644, concurrent_processes=2,
        task_counts=dict(collections.Counter(q["task"] for q in expected)),
        upstream_commit=runner.UPSTREAM_COMMIT, reference_dp_path=str(runner.REFERENCE.resolve()),
        reference_dp_inference_running=False, run_sparsemm_baseline=False,
        files_sha256={str(path.resolve()): runner.official.sha256(path) for path in assets},
        protocols=protocols, query_list_sha256=runner.digest(selected),
        full_official_query_list_sha256=runner.digest(expected), jobs=jobs,
        expected_encoded_frames=sum(job["expected_encoded_frames"] for job in jobs.values()),
        python=str(Path(sys.executable).resolve()),
    )
    destination = root / "supervision_manifest.json"
    if destination.exists() and json.loads(destination.read_text()) != manifest:
        raise ValueError("Pinned code/assets/shards changed; refusing to resume")
    for name, job in jobs.items():
        directory = root / "shards" / name / runner.ARM
        path = directory / "protocol.json"
        if path.exists() and json.loads(path.read_text()) != protocols[name]:
            raise ValueError("Worker protocol changed")
        rows = base.progress_rows(directory / "queries.jsonl")
        if len(rows) > len(job["queries"]):
            raise ValueError("Too many shard predictions")
        for row, query in zip(rows, job["queries"]):
            runner.validate_result(row, query)
        base.write_json(path, protocols[name])
        base.write_json(directory / "query_manifest.json", job["queries"])
    base.write_json(destination, manifest)
    base.write_json(root / "query_manifest.json", selected)
    for name in sources:
        path = root / "source_snapshot" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    for name in protocols["shard0"]["upstream_sha256"]:
        path = root / "source_snapshot/SimpleStream" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((runner.UPSTREAM / name).read_bytes())
    return manifest, selected, expected, reference, jobs


def collect(root, jobs, reference):
    available, progress, measurements = {}, {}, {}
    for name, job in jobs.items():
        rows = base.progress_rows(root / "shards" / name / runner.ARM / "queries.jsonl")
        if len(rows) > len(job["queries"]):
            raise ValueError("Too many shard predictions")
        for row, query in zip(rows, job["queries"]):
            runner.validate_result(row, query)
            qid = query["query_id"]
            if qid in available or row["frames"] != reference[qid]["frames"]:
                raise ValueError("Duplicate query or different DP/control source-prefix frames")
            available[qid] = dict(row, worker_shard=name)
        progress[name] = len(rows)
        seconds = sum(row["seconds"] for row in rows)
        images = sum(row["encoded_frames"] for row in rows)
        work = images + 2 * len(rows)
        left = max(0, job["estimated_work_units"] - work)
        measurements[name] = dict(completed_queries=len(rows), total_queries=len(job["queries"]),
            encoded_frames=images, expected_encoded_frames=job["expected_encoded_frames"],
            measured_seconds=seconds, mean_query_seconds=seconds / len(rows) if rows else None,
            estimated_remaining_seconds=left * seconds / work if len(rows) >= 15 and seconds else None)
    return progress, available, measurements


def describe(values):
    values = sorted(values)
    if not values:
        return dict(count=0, mean=None, minimum=None, maximum=None, p50=None, p95=None)
    def percentile(p):
        offset = (len(values) - 1) * p
        lo = math.floor(offset)
        hi = math.ceil(offset)
        return values[lo] + (values[hi] - values[lo]) * (offset - lo)
    return dict(count=len(values), mean=sum(values) / len(values), minimum=values[0], maximum=values[-1],
                p50=percentile(0.5), p95=percentile(0.95))


def comparison_metrics(rows, reference):
    n = len(rows)
    a = sum(reference[row["query_id"]]["correct"] for row in rows)
    b = sum(row["correct"] for row in rows)
    dp_only = sum(reference[row["query_id"]]["correct"] and not row["correct"] for row in rows)
    ss_only = sum(row["correct"] and not reference[row["query_id"]]["correct"] for row in rows)
    return dict(n=n, dp_correct=a, simplestream_correct=b,
                dp_accuracy=a / n if n else None, simplestream_accuracy=b / n if n else None,
                simplestream_minus_dp=(b - a) / n if n else None,
                dp_only_correct=dp_only, simplestream_only_correct=ss_only)


def publish(root, official, selected, available, reference, status):
    rows = [available[q["query_id"]] for q in official if q["query_id"] in available]
    total_by_task = collections.Counter(q["task"] for q in official)
    tasks = {task: dict(expected=total_by_task[task], **comparison_metrics(
        [row for row in rows if row["task"] == task], reference)) for task in sorted(total_by_task)}
    complete = len(rows) == 3035
    summaries = {
        "simplestream": runner.official.summary(rows, 3035),
        "dp_matched": runner.official.summary([reference[r["query_id"]] for r in rows], 3035),
        "dp_full": runner.official.summary(list(reference.values()), 3035),
    }
    budgets = {field: describe([r[field] for r in rows]) for field in (
        "visual_tokens", "input_tokens", "non_visual_input_tokens", "window_real_frames")}
    budgets["prefill_kv_bytes"] = describe([r["prefill_kv"]["bytes"] for r in rows])
    budgets["final_kv_tokens_per_head"] = describe([r["final_kv"]["tokens_per_head"] for r in rows])
    report = dict(updated_at=base.now(), complete=complete, matched_queries=len(rows), total_queries=3035,
                  tasks=tasks, matched=comparison_metrics(rows, reference), summaries=summaries,
                  token_budget=budgets, status=status,
                  comparison_note="Same official prompts/queries, Qwen2.5, source-prefix FPS0.5 sampling, "
                  "pixel cap151200, greedy max-new16, repetition penalty1.1 and official scorer. "
                  "SimpleStream uses four independent image embeddings with fresh LM prefill; DP uses "
                  "K500 streaming KV, two temporal units in recent area and weighted historical quotas. "
                  "Actual budgets differ and are measured; this is not an equal-token-budget comparison.")
    base.write_json(root / "comparison.json", report)
    base.write_json(root / runner.ARM / "summary.json", summaries["simplestream"])
    base.write_json(root / "token_budget_summary.json", budgets)
    content = "".join(json.dumps(available[q["query_id"]], ensure_ascii=False) + "\n"
                      for q in selected if q["query_id"] in available)
    base.atomic_write(root / runner.ARM / "predictions_live.jsonl", content)
    if complete:
        base.atomic_write(root / runner.ARM / "queries.jsonl", content)
        base.write_json(root / runner.ARM / "query_manifest.json", selected)
        base.write_json(root / "final_audit.json", dict(
            completed_at=base.now(), unique_queries=len({r["query_id"] for r in rows}),
            matching_dp_source_prefixes=True, official_scores_revalidated=True,
            official_prompt_hashes_revalidated=True, empty_lm_cache_before_every_question=True,
            all_queries_fresh_full_prefill=True, code_and_asset_pins_checked=True,
            source_videos=len({q["video_path"] for q in official}),
            output_sha256=runner.official.sha256(root / runner.ARM / "queries.jsonl")))
    def percent(value):
        return "—" if value is None else f"{100 * value:.2f}%"
    text = (
        f"更新时间：{base.now()}。进度 **{len(rows)}/3035**，644 个源视频，12 类。\n\n"
        "只启动 SimpleStream 四帧重新 prefill 对照的两个分片进程；DP 使用已完成的官方全量结果。\n\n"
        "两边采用相同 Qwen2.5-VL-7B、FPS0.5、源视频采样位置、像素上限151200、官方提示词和评分。"
        "SimpleStream 每题对最近四张独立图片重新 prefill，只有视觉特征复用，无跨题语言 KV。"
        "复现发布的 Qwen2.5 连续视觉前缀；不是将视觉 token 强制截断到500。\n\n"
        "| 题型 | 同题数/全量 | 我们 DP | SimpleStream 四帧重算 | 差值（后者−DP） |\n"
        "|---|---:|---:|---:|---:|\n")
    for task, values in tasks.items():
        text += (f"| {task} | {values['n']}/{values['expected']} | {percent(values['dp_accuracy'])} | "
                 f"{percent(values['simplestream_accuracy'])} | {percent(values['simplestream_minus_dp'])} |\n")
    m = report["matched"]
    text += (f"| 逐题加权 | {len(rows)}/3035 | {percent(m['dp_accuracy'])} | "
             f"{percent(m['simplestream_accuracy'])} | {percent(m['simplestream_minus_dp'])} |\n\n")
    if complete:
        text += (f"官方总分（三组先各自题型等权，再组间等权）：DP "
                 f"{percent(summaries['dp_full']['official_total_average'])}；SimpleStream "
                 f"{percent(summaries['simplestream']['official_total_average'])}。\n\n")
    else:
        text += "未完成全量，官方总分暂不显示；表中仅比较双方都已作答的相同题。\n\n"
    if rows:
        v, k = budgets["visual_tokens"], budgets["input_tokens"]
        text += (f"当前实测：四帧视觉 token 平均 {v['mean']:.1f}，范围 {v['minimum']}–{v['maximum']}；"
                 f"prefill 实际 KV 每头平均 {k['mean']:.1f} token（含文本）。"
                 "我们的名义视觉预算为每头平均500；本对照的视觉预算随实际图片尺寸变化。\n\n")
    base.atomic_write(root / "live_report.md", text)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["query_id", "task", "target", "dp_response", "dp_correct", "simplestream_response",
                     "simplestream_correct", "prefix_frames", "window_frames", "visual_tokens", "input_tokens",
                     "prefill_kv_bytes", "seconds"])
    for row in rows:
        dp = reference[row["query_id"]]
        writer.writerow([row["query_id"], row["task"], row["target"], dp["response"], dp["correct"],
                         row["response"], row["correct"], row["frames"], row["window_real_frames"],
                         row["visual_tokens"], row["input_tokens"], row["prefill_kv"]["bytes"], row["seconds"]])
    base.atomic_write(root / "per_query_comparison.csv", output.getvalue())
    return report


def supervise(root, poll_seconds=10, report_seconds=60, max_no_progress_retries=3):
    root.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    with (root / ".supervisor.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("A supervisor already owns this experiment")
        existing = json.loads((root / "status.json").read_text()) if (root / "status.json").exists() else {}
        status = dict(supervisor_pid=os.getpid(), started_at=existing.get("started_at", base.now()),
                      resumed_at=base.now() if existing else None, state="preflight",
                      total_queries=3035, concurrent_processes=2, dp_inference_running=False)
        jobs, active, children = {}, {}, {}
        failures, retry_at = collections.Counter(), {}
        last_report = 0
        try:
            manifest, selected, official, reference, jobs = prepare(root)
            base.event(root, "preflight_passed", shards={n: len(j["queries"]) for n, j in jobs.items()},
                       dp_reference_queries=len(reference), source_videos=644)
            while True:
                live = find_workers(root, jobs)
                for name in jobs:
                    if name not in live:
                        repair = base.repair_jsonl_tail(root / "shards" / name / runner.ARM / "queries.jsonl")
                        if repair:
                            base.event(root, "incomplete_tail_repaired", shard=name, **repair)
                progress, available, measurements = collect(root, jobs, reference)
                for name in list(active):
                    if name in live:
                        continue
                    child = children.get(name)
                    if child is not None and child.poll() is None:
                        continue
                    children.pop(name, None)
                    previous = active.pop(name)
                    progressed = progress[name] > previous["started_count"]
                    if progress[name] < len(jobs[name]["queries"]):
                        failures[name] = 0 if progressed else failures[name] + 1
                        if failures[name] > max_no_progress_retries:
                            raise RuntimeError(f"{name} repeatedly exited without progress; see worker log")
                        retry_at[name] = time.monotonic() + min(15 * 2 ** max(0, failures[name] - 1), 120)
                    base.event(root, "worker_exited", shard=name, progressed=progressed,
                               completed=progress[name], exit_code=child.returncode if child else None)
                for name, record in live.items():
                    if name not in active:
                        active[name] = dict(record, started_count=progress[name])
                        base.event(root, "worker_attached", shard=name, pid=record["pid"])
                estimates = [m["estimated_remaining_seconds"] for m in measurements.values()]
                status.update(updated_at=base.now(), completed_queries=sum(progress.values()),
                              shard_queries=progress, worker_pids={n: r["pid"] for n, r in live.items()},
                              measurements=measurements, no_progress_retries=dict(failures), error=None,
                              estimated_remaining_seconds=max(estimates)
                              if estimates and all(v is not None for v in estimates) else None)
                paused = (root / "PAUSE").exists()
                complete = len(available) == 3035 and not live
                if paused or complete:
                    for record in live.values():
                        base.stop_worker(record)
                    # A pause can stop after the latest collect; reread committed
                    # results after workers exit so no valid answers are hidden.
                    progress, available, measurements = collect(root, jobs, reference)
                    base.check_pins(manifest)
                    status.update(state="paused" if paused else "complete", worker_pids={},
                                  completed_queries=len(available), shard_queries=progress,
                                  measurements=measurements, updated_at=base.now(),
                                  estimated_remaining_seconds=0 if complete else status["estimated_remaining_seconds"])
                    base.write_json(root / "status.json", status)
                    publish(root, official, selected, available, reference, status)
                    base.event(root, status["state"], completed_queries=len(available))
                    return 0
                for name, job in jobs.items():
                    if (name in live or name in active or progress[name] == len(job["queries"])
                            or time.monotonic() < retry_at.get(name, 0)):
                        continue
                    base.check_pins(manifest)
                    with (root / f"{name}.log").open("a") as log:
                        child = subprocess.Popen(worker_command(root, name, job), cwd=ROOT,
                            env=base.worker_environment(), stdin=subprocess.DEVNULL,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
                    children[name] = child
                    record = base.process_record(child.pid) or dict(pid=child.pid)
                    active[name] = dict(record, started_count=progress[name])
                    live[name] = record
                    base.event(root, "worker_launched", shard=name, pid=child.pid, completed=progress[name])
                status.update(state="running" if live else "retry_backoff",
                              worker_pids={n: r["pid"] for n, r in live.items()})
                base.write_json(root / "status.json", status)
                if time.monotonic() - last_report >= report_seconds:
                    base.check_pins(manifest)
                    publish(root, official, selected, available, reference, status)
                    last_report = time.monotonic()
                time.sleep(poll_seconds)
        except Exception as error:
            for record in find_workers(root, jobs).values():
                base.stop_worker(record)
            status.update(state="failed", worker_pids={}, updated_at=base.now(),
                          error=f"{type(error).__name__}: {error}")
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
        parser.error("Polling/report periods must be positive and retry count nonnegative")
    return supervise(args.output_root.expanduser().resolve(), args.poll_seconds,
                     args.report_seconds, args.max_no_progress_retries)


if __name__ == "__main__":
    sys.exit(main())
