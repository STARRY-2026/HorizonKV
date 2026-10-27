"""Detached, resumable fixed-896 SimpleStream one/four-frame K500 experiment.

Only the independent three-question validation and the requested nine-task full
experiment run.  A root-level PAUSE marker stops verified workers and prevents
new launches.  Importing this module never starts inference.
"""

import argparse
from collections import Counter
from datetime import datetime, timedelta
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from zoneinfo import ZoneInfo

from sparsemm_native import supervise_simplestream_k500_input_only as old


ROOT = old.ROOT
DEFAULT_OUTPUT = ROOT / 'results/ovobench_simplestream_onefour_pseudo_k500_fps05_fixed896_20261026'
DEFAULT_REFERENCE = ROOT / 'results/ovobench_simplestream_k500_input_only_fps05_native_20261026/validation_new'
MODULE = 'sparsemm_native.run_simplestream_k500_fixed896'
TASKS = old.BACKWARD + old.REALTIME
MAX_RETRIES = 3


def worker_command(output, manifest):
    return [str(old.base.ENV_PYTHON), '-u', '-m', MODULE,
            '--output-root', str(output), '--manifest', str(manifest)]


def optional_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def count_rows(output):
    return len(old.base.read_jsonl(output / 'queries.jsonl'))


def percent(value):
    return f'{value:.2%}' if value is not None else '待相关任务全部完成'


def completed_means(per_task, expected_counts):
    """Never present an incomplete task as a completed official task average."""
    def macro(names):
        if not all(per_task.get(name, {}).get('total', 0) == expected_counts.get(name, -1)
                   and expected_counts.get(name, 0) > 0 for name in names):
            return None
        return sum(per_task[name]['accuracy'] for name in names) / len(names)
    backward, realtime = macro(old.BACKWARD), macro(old.REALTIME)
    return dict(backward_macro=backward, realtime_macro=realtime,
                nine_task_macro=macro(TASKS),
                official_two_group_macro=(backward + realtime) / 2
                if backward is not None and realtime is not None else None)


def write_progress(root, phase, state, output, manifest, **extra):
    queries = json.loads(manifest.read_text())
    expected = Counter(row['task'] for row in queries)
    rows = old.base.read_jsonl(output / 'queries.jsonl')
    counts = Counter(row['task'] for row in rows)
    corrects = Counter(row['task'] for row in rows if row.get('correct', row['target'] in row['response']))
    per_task = {task: dict(correct=corrects[task], total=counts[task], expected_total=expected[task],
                           accuracy=corrects[task] / counts[task] if counts[task] else None)
                for task in TASKS if task in expected}
    summary = optional_json(output / 'summary.json')
    total = len(queries)
    seconds = [float(row['seconds']) for row in rows if row.get('seconds') is not None]
    # Median-free arithmetic mean includes every measured query and excludes
    # model loading; per-row timings remain stable across supervisor restarts.
    mean_seconds = sum(seconds) / len(seconds) if seconds else None
    remaining = (total - len(rows)) * mean_seconds if mean_seconds is not None else None
    now = datetime.now(ZoneInfo('Asia/Shanghai'))
    eta = (now + timedelta(seconds=remaining)).isoformat(timespec='seconds') if remaining is not None else None
    means = completed_means(per_task, expected) if phase == 'full' else completed_means({}, {})
    tokens = [float(row['budget_audit']['actual_tokens_per_head']) for row in rows]
    token_mean = sum(tokens) / len(tokens) if tokens else None
    question_accuracy = sum(corrects.values()) / len(rows) if rows else None
    status = dict(state=state, phase=phase, done=len(rows), total=total,
                  updated_at=now.isoformat(timespec='seconds'), supervisor_pid=os.getpid(),
                  stage_output=str(output), per_task=per_task, official_means=means,
                  question_accuracy=question_accuracy, actual_tokens_per_head_mean=token_mean,
                  query_seconds_mean=mean_seconds, estimated_remaining_seconds=remaining,
                  estimated_completion_beijing=eta, **extra)
    lines = ['# SimpleStream 一帧／四帧固定分辨率实验进度', '',
             f'更新时间（北京时间）：{status["updated_at"]}', '',
             f'状态：**{state}**；阶段：**{phase}**；已完成 **{len(rows)}/{total}** 题。', '',
             '配置：fps=0.5；896×896 等比缩放后灰色补边；逐图编码每帧1024视觉 token；'
             '原一帧／四帧独立重新 prefill；固定伪查询裁剪；每头视觉 K500；不加入历史 KV。', '',
             '流程：新固定输入3题验证 → 九类1468题全量。前瞻 REC、SSR、CRR 不运行。', '',
             '| 任务 | 正确/已答 | 全量题数 | 已答正确率 |', '|---|---:|---:|---:|']
    for task in TASKS:
        if task not in per_task:
            continue
        item = per_task[task]
        accuracy = f'{item["accuracy"]:.2%}' if item['accuracy'] is not None else '尚未作答'
        lines.append(f'| {task} | {item["correct"]}/{item["total"]} | {item["expected_total"]} | {accuracy} |')
    lines += ['', f'已答逐题正确率：{percent(question_accuracy)}。',
              f'实际平均视觉 token/头：{token_mean if token_mean is not None else "待产出"}。',
              '候选输入 token 与回答前 K500 缓存预算分别记录。']
    if phase == 'full':
        lines += ['', f'官方两组宏均值：{percent(means["official_two_group_macro"])}。',
                  f'九类等权均值：{percent(means["nine_task_macro"])}。',
                  f'回溯三类均值：{percent(means["backward_macro"])}；'
                  f'实时六类均值：{percent(means["realtime_macro"])}。',
                  '只在相关任务全量完成后展示对应宏均值；局部已答准确率不能当作全量成绩。']
    if mean_seconds is not None:
        lines += ['', f'实测平均每题：{mean_seconds:.2f}秒（不含模型加载）。']
    if remaining is not None and state not in ('paused', 'failed', 'complete'):
        lines += [f'按已完成题目估计剩余：{remaining / 3600:.2f}小时；'
                  f'预计完成（北京时间）：{eta}。视频差异和后续题型会改变估计。']
    if extra.get('error'):
        lines += ['', f'错误：{extra["error"]}']
    lines += ['', f'结果目录：`{output}`', '',
              '根目录创建 `PAUSE` 可暂停并保留断点；删除后重新启动守护可续跑。']
    temporary = root / 'progress.md.tmp'
    temporary.write_text('\n'.join(lines) + '\n')
    temporary.replace(root / 'progress.md')
    old.base.write(root / 'supervisor_status.json', status)


def validate_smoke(output, manifest, reference):
    rows = old.base.read_jsonl(output / 'queries.jsonl')
    queries = json.loads(manifest.read_text())
    expected = {str(row['query_id']): row for row in queries}
    references = {str(row['query_id']): row for row in old.base.read_jsonl(reference / 'queries.jsonl')}
    if len(rows) != len(expected) or {str(row['query_id']) for row in rows} != set(expected):
        raise ValueError('Fixed-input validation query coverage differs from manifest')
    checked = []
    for row in rows:
        qid = str(row['query_id'])
        baseline = references[qid]
        inputs, previous = row['input_audit'], baseline['input_audit']
        budget = row['budget_audit']
        frame_tokens = inputs['per_frame_visual_tokens']
        head_tokens = [int(token) for layer in budget['per_head_visual_slots'] for token in layer]
        checks = dict(
            target_exact=row['target'] == baseline['target'] == expected[qid]['target'],
            normal_answer=bool(re.fullmatch(r'[A-Z]', row['response'].strip())),
            original_prompt_exact=inputs['prompt_sha256'] == previous['prompt_sha256'],
            original_image_encoder=inputs['visual_encoder'] == previous['visual_encoder'] == 'original_per_image',
            original_position_layout=inputs['position_layout'] == previous['position_layout'],
            fps_half=inputs['requested_fps'] == .5,
            original_sample_indices_exact=inputs['selected_raw_frame_indices'] == previous['selected_raw_frame_indices'],
            fixed_1024_each_image=bool(frame_tokens) and all(value == 1024 for value in frame_tokens),
            fixed_grid_each_image=inputs['image_grid_thw'] == [[1, 64, 64]] * len(frame_tokens),
            fixed_processor_size=inputs['processor_resize_hw_per_frame'] == [[896, 896]] * len(frame_tokens),
            physical_budget_500_each_head=len(head_tokens) == 112 and set(head_tokens) == {500},
            visual_budget_exact=budget['actual_visual_slots'] == 56000
                and budget['actual_tokens_per_head'] == 500,
            no_history=budget['actual_history_visual_slots'] == 0,
            question_free_selection=budget['question_text_used_for_selection'] is False,
            original_selector=budget['selector'] == baseline['budget_audit']['selector'],
            independent_single_prefill=budget['independent_single_frame_prefill'] is True,
        )
        if not all(checks.values()):
            raise ValueError(f'Fixed-input validation failed for {qid}: {checks}')
        checked.append(dict(query_id=qid, checks=checks, response=row['response'],
                            baseline_response=baseline['response'], actual_tokens_per_head=500))
    return checked


def refresh_report(output, *, complete=False):
    from sparsemm_native.report_simplestream_fixed896 import generate
    generate(output, allow_partial=not complete)


def run_stage(root, phase, manifest, stop_requested):
    output = root / phase
    output.mkdir(parents=True, exist_ok=True)
    total = len(json.loads(manifest.read_text()))
    command = worker_command(output, manifest)
    failures, last_report_count = 0, -20
    while True:
        record = old.find_worker(output, command)
        if (root / 'PAUSE').exists() or stop_requested():
            if record:
                old.stop_worker(record)
            write_progress(root, phase, 'paused', output, manifest)
            return False
        if count_rows(output) == total and optional_json(output / 'status.json').get('state') == 'complete':
            return True
        child = None
        if record is None:
            for path in output.glob('*.jsonl'):
                old.repair_tail(path)
            with (output / 'worker.log').open('a') as log:
                child = subprocess.Popen(command, cwd=ROOT, env=old.worker_environment(),
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True)
            record = old.process_record(child.pid)
            if record is None:
                child.wait()
            else:
                old.base.write(output / 'worker_identity.json', record)
                (output / 'worker.pid').write_text(str(child.pid) + '\n')
        while record and old.process_record(record['pid']) is not None:
            if (root / 'PAUSE').exists() or stop_requested():
                old.stop_worker(record)
                if child:
                    child.wait()
                write_progress(root, phase, 'paused', output, manifest)
                return False
            write_progress(root, phase, 'running', output, manifest,
                           worker_pid=record['pid'], retry=failures)
            done = count_rows(output)
            if phase == 'full' and done and done - last_report_count >= 20:
                try:
                    refresh_report(output)
                except Exception as exc:
                    # Reporting is read-only to worker predictions. Preserve
                    # inference on a report error and make the error visible.
                    old.base.write(root / 'report_error.json', dict(error=f'{type(exc).__name__}: {exc}',
                                   done=done, updated_at=old.base.now()))
                last_report_count = done
            time.sleep(5)
        code = child.wait() if child else None
        status = optional_json(output / 'status.json')
        if code in (None, 0) and status.get('state') == 'complete' and count_rows(output) == total:
            return True
        failures += 1
        error = f'Worker stopped: exit={code}, status={status.get("state")}, {status.get("error", "")}'
        if failures > MAX_RETRIES:
            raise RuntimeError(error + '; three retries exhausted')
        write_progress(root, phase, 'retrying', output, manifest, retry=failures, error=error)
        for _ in range(10):
            if (root / 'PAUSE').exists() or stop_requested():
                break
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--reference-root', type=Path, default=DEFAULT_REFERENCE)
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    full_manifest = (args.manifest or root / 'full_nine_manifest.json').resolve()
    smoke_manifest = root / 'smoke_manifest.json'
    stages = [('validation_fixed896', smoke_manifest), ('full', full_manifest)]
    stopped = [False]
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    for name in (signal.SIGTERM, signal.SIGINT):
        signal.signal(name, lambda *_: stopped.__setitem__(0, True))
    with (root / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        (root / 'supervisor.pid').write_text(str(os.getpid()) + '\n')
        phase, manifest = stages[0]
        try:
            if (root / 'PAUSE').exists():
                for stage, stage_manifest in stages:
                    output = root / stage
                    record = old.find_worker(output, worker_command(output, stage_manifest))
                    if record:
                        old.stop_worker(record)
                        phase, manifest = stage, stage_manifest
                write_progress(root, phase, 'paused', root / phase, manifest)
                return
            for phase, manifest in stages:
                if not run_stage(root, phase, manifest, lambda: stopped[0]):
                    return
                if phase == 'validation_fixed896':
                    checked = validate_smoke(root / phase, manifest, args.reference_root)
                    old.base.write(root / 'validation_checks.json', dict(passed=True, checks=checked,
                                   reference_root=str(args.reference_root), updated_at=old.base.now()))
            summary = optional_json(root / 'full/summary.json')
            total = len(json.loads(full_manifest.read_text()))
            if summary.get('done') != total or summary.get('total') != total:
                raise ValueError('Full summary coverage differs from requested manifest')
            refresh_report(root / 'full', complete=True)
            old.base.write(root / 'final_summary.json', summary)
            write_progress(root, 'full', 'complete', root / 'full', full_manifest)
        except BaseException as exc:
            write_progress(root, phase, 'failed', root / phase, manifest,
                           error=f'{type(exc).__name__}: {exc}')
            raise


if __name__ == '__main__':
    main()
