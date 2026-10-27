"""Detached, resumable supervision of the old SimpleStream K500 input control.

Run old-protocol smoke reproduction, new-input smoke validation, then one full
fps=.5/native-resolution experiment.  OUTPUT/PAUSE prevents new work and stops
only the verified worker belonging to this supervisor.  Nothing runs on import.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from sparsemm_native import supervise_simplestream_upstream as base


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'results/ovobench_simplestream_k500_input_only_fps05_native_20261026'
MODULE = 'sparsemm_native.run_simplestream_k500_input_only'
BACKWARD = ('EPM', 'ASI', 'HLD')
REALTIME = ('STU', 'OJR', 'ATR', 'ACR', 'OCR', 'FPD')
MAX_RETRIES = 3


def worker_environment():
    env = base.worker_env()
    env['PYTHONPATH'] = os.pathsep.join((str(ROOT), str(base.UPSTREAM), str(ROOT.parent / 'hermes')))
    return env


def worker_command(output, manifest, fps, resolution):
    return [str(base.ENV_PYTHON), '-u', '-m', MODULE,
            '--output-root', str(output), '--manifest', str(manifest),
            '--fps', str(fps), '--resolution', resolution]


def process_record(pid):
    try:
        directory = Path('/proc') / str(pid)
        fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return dict(pid=int(pid), start_ticks=fields[19],
                    command=[word.decode() for word in (directory / 'cmdline').read_bytes().split(b'\0') if word])
    except (OSError, ValueError, UnicodeError, IndexError):
        return None


def find_worker(output, command):
    path = output / 'worker_identity.json'
    if not path.exists():
        return None
    saved = json.loads(path.read_text())
    current = process_record(saved['pid'])
    if current is None:
        return None
    if current != saved or current['command'] != command:
        raise RuntimeError(f'Worker identity changed; refusing attachment or signals: {path}')
    return current


def stop_worker(record):
    if process_record(record['pid']) is None:
        return
    if (process_record(record['pid']) != record
            or os.getpgid(record['pid']) != record['pid']
            or os.getsid(record['pid']) != record['pid']):
        raise RuntimeError('Refusing to signal an unverified/non-detached worker')
    os.killpg(record['pid'], signal.SIGTERM)
    deadline = time.monotonic() + 15
    while process_record(record['pid']) is not None and time.monotonic() < deadline:
        time.sleep(.5)
    if process_record(record['pid']) is not None:
        os.killpg(record['pid'], signal.SIGKILL)


def repair_tail(path):
    """Preserve malformed final fragments; reject malformed completed records."""
    if not path.exists():
        return
    raw = path.read_bytes()
    if not raw or raw.endswith(b'\n'):
        base.read_jsonl(path)
        return
    offset = raw.rfind(b'\n') + 1
    try:
        json.loads(raw[offset:])
    except (json.JSONDecodeError, UnicodeDecodeError):
        backup = path.with_name(path.name + f'.incomplete.{time.time_ns()}')
        backup.write_bytes(raw[offset:])
        path.write_bytes(raw[:offset])
    else:
        path.write_bytes(raw + b'\n')
    base.read_jsonl(path)


def optional_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def count_rows(output):
    return len(base.read_jsonl(output / 'queries.jsonl'))


def official_means(summary):
    tasks = summary.get('per_task', {})
    def macro(names):
        return (sum(tasks[name]['accuracy'] for name in names) / len(names)
                if all(name in tasks and tasks[name]['total'] for name in names) else None)
    backward, realtime = macro(BACKWARD), macro(REALTIME)
    return dict(backward_macro=backward, realtime_macro=realtime,
                official_two_group_macro=(backward + realtime) / 2
                if backward is not None and realtime is not None else None,
                nine_task_macro=macro(BACKWARD + REALTIME),
                question_accuracy=summary.get('question_accuracy'))


def percent(value):
    return f'{value:.2%}' if value is not None else '待覆盖所有相关题型'


def write_progress(root, phase, state, output, total, **extra):
    rows = count_rows(output)
    summary = optional_json(output / 'summary.json')
    status = dict(state=state, phase=phase, done=rows, total=total,
                  updated_at=base.now(), supervisor_pid=os.getpid(),
                  stage_output=str(output), **extra)
    if phase == 'full':
        means = official_means(summary)
        status['official_means'] = means
    else:
        means = official_means({})
    lines = ['# SimpleStream K500 输入设置对照进度', '',
             f'更新时间：{status["updated_at"]}', '',
             f'状态：**{state}**；阶段：**{phase}**；已完成 **{rows}/{total}** 题。', '',
             '顺序：旧3题严格复现 → 新设置3题验证 → 唯一一组全量实验。', '',
             '全量设置：旧 SimpleStream 一帧／四帧独立重新 prefill、原伪查询裁剪 K500，'
             '仅调整 fps=0.5 与 native 输入分辨率。前瞻 REC、SSR、CRR 不运行。', '',
             '当前结果只覆盖已完成题目，不能提前作为全量成绩。', '']
    if summary:
        lines += ['| 任务 | 正确数/题数 | 正确率 |', '|---|---:|---:|']
        for task, value in sorted(summary.get('per_task', {}).items()):
            lines.append(f'| {task} | {value["correct"]}/{value["total"]} | {value["accuracy"]:.2%} |')
        lines += ['', f'逐题均值：{percent(summary.get("question_accuracy"))}。',
                  f'实际平均视觉 token/头：{summary.get("actual_tokens_per_head_mean", "待产出")}。']
    if phase == 'full':
        lines += ['', f'官方两组宏均值：{percent(means["official_two_group_macro"])}。',
                  f'九类等权均值：{percent(means["nine_task_macro"])}。',
                  f'回溯三类均值：{percent(means["backward_macro"])}；'
                  f'实时六类均值：{percent(means["realtime_macro"])}。']
    if extra.get('error'):
        lines += ['', f'错误：{extra["error"]}']
    lines += ['', f'详细结果目录：`{output}`', '',
              '创建根目录 `PAUSE` 可暂停并保留逐题断点；删除后重新启动守护可续跑。']
    temporary = root / 'progress.md.tmp'
    temporary.write_text('\n'.join(lines) + '\n')
    temporary.replace(root / 'progress.md')
    # Include computed official aggregates in the machine-readable status too.
    base.write(root / 'supervisor_status.json', status)


def validate_smoke(output, expected, *, old):
    rows = base.read_jsonl(output / 'queries.jsonl')
    if len(rows) != len(expected) or {str(row['query_id']) for row in rows} != set(expected):
        raise ValueError('Smoke query coverage differs from expected reproduction set')
    checked = []
    for row in rows:
        qid = str(row['query_id'])
        truth = expected[qid]
        if row['target'] != truth['target'] or not re.fullmatch(r'[A-Z]', row['response'].strip()):
            raise ValueError(f'Smoke target or answer format invalid: {qid}')
        budget = row['budget_audit']
        counts = [int(value) for layer in budget['per_head_visual_slots'] for value in layer]
        if (len(counts) != 112 or min(counts) < 0 or max(counts) > 500
                or sum(counts) != budget['actual_visual_slots']
                or abs(sum(counts) / 112 - budget['actual_tokens_per_head']) > 1e-8):
            raise ValueError(f'Invalid physical K500 budget: {qid}')
        inputs = row['input_audit']
        checks = dict(normal_answer=True, visual_budget_valid=True,
                      original_prompt_exact=inputs['prompt_sha256'] == truth['old_prompt_sha256'],
                      original_image_encoder=inputs['visual_encoder'] == 'original_per_image')
        if old:
            checks.update(
                raw_response_exact=row['response'] == truth['old_response'],
                candidate_tokens_exact=inputs['per_frame_visual_tokens'] == truth['old_frame_visual_tokens'],
                sampled_indices_exact=inputs['selected_raw_frame_indices'] == truth['old_selected_raw_frame_indices'],
                resize_exact=inputs['resize_hw'] == truth['old_resize_hw'],
                image_grids_exact=inputs['image_grid_thw'] == truth['old_image_grid_thw'])
        else:
            checks.update(requested_fps_exact=inputs['requested_fps'] == .5,
                native_frame_sizes_exact=inputs['processor_resize_hw_per_frame'] ==
                    [truth['current_default_video_resize_hw']] * len(inputs['per_frame_visual_tokens']),
                no_external_video_resize=not inputs['external_video_resize'])
        if not all(checks.values()):
            raise ValueError(f'SimpleStream validation failed for {qid}: {checks}')
        checked.append(dict(query_id=qid, checks=checks,
                            response=row['response'], actual_tokens_per_head=budget['actual_tokens_per_head']))
    return checked


def run_stage(root, phase, manifest, fps, resolution, stop_requested):
    output = root / phase
    output.mkdir(parents=True, exist_ok=True)
    total = len(json.loads(manifest.read_text()))
    command = worker_command(output, manifest, fps, resolution)
    failures = 0
    while True:
        record = find_worker(output, command)
        if (root / 'PAUSE').exists() or stop_requested():
            if record:
                stop_worker(record)
            write_progress(root, phase, 'paused', output, total)
            return False
        if count_rows(output) == total and optional_json(output / 'status.json').get('state') == 'complete':
            return True
        child = None
        if record is None:
            for path in output.glob('*.jsonl'):
                repair_tail(path)
            with (output / 'worker.log').open('a') as log:
                child = subprocess.Popen(command, cwd=ROOT, env=worker_environment(),
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True)
            record = process_record(child.pid)
            if record is None:
                child.wait()
            else:
                base.write(output / 'worker_identity.json', record)
                (output / 'worker.pid').write_text(str(child.pid) + '\n')
        while record and process_record(record['pid']) is not None:
            if (root / 'PAUSE').exists() or stop_requested():
                stop_worker(record)
                if child:
                    child.wait()
                write_progress(root, phase, 'paused', output, total)
                return False
            write_progress(root, phase, 'running', output, total,
                           worker_pid=record['pid'], retry=failures)
            time.sleep(5)
        code = child.wait() if child else None
        status = optional_json(output / 'status.json')
        if (code in (None, 0) and status.get('state') == 'complete'
                and count_rows(output) == total):
            return True
        failures += 1
        error = f'Worker stopped: exit={code}, status={status.get("state")}, {status.get("error", "")}'
        if failures > MAX_RETRIES:
            raise RuntimeError(error + '; three retries exhausted')
        write_progress(root, phase, 'retrying', output, total, retry=failures, error=error)
        for _ in range(10):
            if (root / 'PAUSE').exists() or stop_requested():
                break
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    full_manifest = (args.manifest or root / 'full_nine_manifest.json').resolve()
    stopped = [False]
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    for name in (signal.SIGTERM, signal.SIGINT):
        signal.signal(name, lambda *_: stopped.__setitem__(0, True))
    with (root / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        (root / 'supervisor.pid').write_text(str(os.getpid()) + '\n')
        phase, output, total = 'validation_old', root / 'validation_old', 3
        try:
            if (root / 'PAUSE').exists():
                # A prior supervisor may have exited while its independent
                # worker survived.  Honor PAUSE for that verified worker too.
                for stage, manifest, fps, resolution in (
                    ('validation_old', root / 'smoke_manifest.json', 1, 'old'),
                    ('validation_new', root / 'smoke_manifest.json', .5, 'native'),
                    ('full', full_manifest, .5, 'native'),
                ):
                    directory = root / stage
                    record = find_worker(directory, worker_command(directory, manifest, fps, resolution))
                    if record:
                        stop_worker(record)
                        phase, output = stage, directory
                        total = len(json.loads(manifest.read_text()))
                write_progress(root, phase, 'paused', output, total)
                return
            expected = json.loads((root / 'smoke_expected.json').read_text())
            smoke = root / 'smoke_manifest.json'
            validations = {}
            for phase, manifest, fps, resolution in (
                ('validation_old', smoke, 1, 'old'),
                ('validation_new', smoke, .5, 'native'),
                ('full', full_manifest, .5, 'native'),
            ):
                output, total = root / phase, len(json.loads(manifest.read_text()))
                if not run_stage(root, phase, manifest, fps, resolution, lambda: stopped[0]):
                    return
                if phase != 'full':
                    validations[phase] = validate_smoke(output, expected, old=phase == 'validation_old')
                    base.write(root / 'validation_checks.json', validations)
            summary = optional_json(root / 'full/summary.json')
            if summary.get('done') != total or summary.get('total') != total:
                raise ValueError('Full-run summary coverage does not match manifest')
            base.write(root / 'final_summary.json', {**summary, **official_means(summary)})
            write_progress(root, 'full', 'complete', root / 'full', total)
            print(json.dumps({**summary, **official_means(summary)}, ensure_ascii=False), flush=True)
        except BaseException as exc:
            write_progress(root, phase, 'failed', output, total,
                           error=f'{type(exc).__name__}: {exc}')
            raise


if __name__ == '__main__':
    main()
