"""Detached, resumable supervision of the verified tail-only recent control."""

import argparse
from collections import Counter
from datetime import datetime, timedelta
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from sparsemm_native.report_aligned_pseudo_pair import GROUPS, TASKS
from sparsemm_native.run_ovobench import official_correct
from sparsemm_native.supervise_ovobench_dual import (
    atomic_write, now, repair_jsonl_tail, write_json,
)


ROOT = Path(__file__).resolve().parents[1]
MAX_RETRIES = 3


def complete_rows(path):
    """Ignore an in-flight append; malformed newline-terminated data is fatal."""
    if not path.exists():
        return []
    rows = []
    with path.open('rb') as stream:
        for line in stream:
            if line.endswith(b'\n') and line.strip():
                rows.append(json.loads(line))
    return rows


def percent(value):
    return '尚未覆盖全部所需题型' if value is None else f'{100 * value:.2f}%'


def publish(root, manifest, state, attempt, worker_pid=None, exit_code=None):
    rows = complete_rows(root / 'queries.jsonl')
    expected = Counter(query['task'] for query in manifest)
    counts, correct = Counter(), Counter()
    for row in rows:
        counts[row['task']] += 1
        correct[row['task']] += bool(official_correct(row['task'], row['response'], row['target']))
    accuracy = {task: correct[task] / counts[task] for task in TASKS if counts[task]}
    group_means = {group: sum(accuracy[task] for task in tasks) / len(tasks)
                   if all(task in accuracy for task in tasks) else None
                   for group, tasks in GROUPS.items()}
    nine_mean = sum(accuracy.values()) / len(TASKS) if len(accuracy) == len(TASKS) else None
    two_mean = (sum(group_means.values()) / len(GROUPS)
                if all(value is not None for value in group_means.values()) else None)
    duration = sum(float(row['seconds']) for row in rows)
    mean_seconds = duration / len(rows) if rows else None
    remaining = mean_seconds * (len(manifest) - len(rows)) if rows else None
    stamp = now()
    status = dict(state=state, done=len(rows), total=len(manifest), attempt=attempt,
                  worker_pid=worker_pid, supervisor_pid=os.getpid(), updated_at=stamp,
                  exit_code=exit_code, mean_query_seconds=mean_seconds,
                  estimated_remaining_seconds=remaining, log=str(root / 'worker.log'))
    write_json(root / 'supervisor_status.json', status)
    partial = len(rows) != len(manifest)
    scope = '部分样本，不能视为全量结果' if partial else '九类全部完成'
    labels = dict(running='运行中', stopping='正在停止', paused='已暂停',
                  complete='已完成', failed='失败', retrying='等待重试')
    lines = ['# OVO-Bench 末尾帧近期控制版：实时进度', '',
             f"更新时间：{stamp}；状态：{labels.get(state, state)}；第 {attempt} 次运行。", '',
             f"已完成 **{len(rows)} / {len(manifest)} 题**（{scope}）。", '',
             '配置：fps=0.5、K500、KVBench 默认分辨率、无额外像素上限；'
             '56 个一帧头与 56 个四帧头，仅编码末尾所需采样帧，近期独立 prefill，历史 KV 为 0。', '',
             '不含 REC、SSR、CRR。按官方区分大小写的 `target in raw response` 评分；'
             '九类／回溯即时两组均值不代表官方十二类总分。', '',
             '| 任务 | 已完成／总题数 | 答对数 | 已答题正确率 | 完成情况 |',
             '|---|---:|---:|---:|---|']
    for task in TASKS:
        label = '已完成' if counts[task] == expected[task] else '部分样本'
        lines.append(f"| {task} | {counts[task]}/{expected[task]} | {correct[task]} | "
                     f"{percent(accuracy.get(task)) if counts[task] else '尚无结果'} | {label} |")
    weighted = sum(correct.values()) / len(rows) if rows else None
    lines += ['', '| 汇总指标（尚未完成时均为部分样本） | 当前值 |', '|---|---:|',
              f'| 已答题逐题正确率 | {percent(weighted)} |',
              f'| 九任务等权均值 | {percent(nine_mean)} |',
              f"| 回溯三类均值 | {percent(group_means['backward'])} |",
              f"| 即时六类均值 | {percent(group_means['realtime'])} |",
              f'| 回溯／即时两组等权均值 | {percent(two_mean)} |', '']
    if rows:
        mean_tokens = sum(row['budget_audit']['actual_visual_slots'] for row in rows) / (112 * len(rows))
        history_tokens = sum(row['budget_audit']['actual_history_visual_slots'] for row in rows) / (112 * len(rows))
        lines += [f'实际平均视觉预算：**{mean_tokens:.2f} token/头**；历史 {history_tokens:.2f} token/头。', '',
                  '用量按回答前缓存的视觉槽数除以 112（28 层 × 4 KV 头）计算，'
                  '排除文本、问题、生成 token 和临时 prefill。', '',
                  f'已完成题平均推理耗时：{mean_seconds:.2f} 秒/题（不含模型加载和预热）。']
        if partial:
            eta = datetime.fromisoformat(stamp) + timedelta(seconds=remaining)
            lines += [f'按已完成题耗时估计剩余：{remaining / 60:.1f} 分钟；'
                      f'预计完成时间 {eta.isoformat(timespec="minutes")}。此估计会随题目更新。']
    else:
        lines += ['正在准备／加载模型；尚无完成题，暂不能依据实测速率估算剩余时间。']
    atomic_write(root / 'progress.md', '\n'.join(lines) + '\n')
    return len(rows)


def stop_worker(worker):
    worker.terminate()
    try:
        worker.wait(timeout=20)
    except subprocess.TimeoutExpired:
        worker.kill()
        worker.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    root, manifest_path = args.output_root.resolve(), args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    if (not isinstance(manifest, list) or not manifest
            or len({q['query_id'] for q in manifest}) != len(manifest)
            or any(q['task'] not in TASKS for q in manifest)):
        raise ValueError('Expected a unique, nonempty official nine-task manifest')
    root.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-u', '-m', 'sparsemm_native.run_recent_tail_only',
               '--manifest', str(manifest_path), '--output-root', str(root)]
    environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                       TRANSFORMERS_OFFLINE='1', MALLOC_ARENA_MAX='2')
    environment['PYTHONPATH'] = os.pathsep.join((str(ROOT),
        str(ROOT / 'sparsemm_baselines/upstream/lmms-eval'),
        str(ROOT / 'sparsemm_protokv/runtime_deps')))
    environment.pop('SPARSEMM_GPU_LEASE_SHARED', None)
    with (root / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        for attempt in range(1, MAX_RETRIES + 2):
            if (root / 'PAUSE').exists():
                publish(root, manifest, 'paused', attempt)
                return
            repair_jsonl_tail(root / 'queries.jsonl', validate_complete=False)
            with (root / 'worker.log').open('a') as log:
                log.write(f'\n{now()} START attempt={attempt}\n')
                log.flush()
                worker = subprocess.Popen(command, cwd=ROOT, env=environment,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                while worker.poll() is None:
                    pause = (root / 'PAUSE').exists()
                    publish(root, manifest, 'stopping' if pause else 'running', attempt, worker.pid)
                    if pause:
                        stop_worker(worker)
                        break
                    time.sleep(10)
                code = worker.wait()
                log.write(f'{now()} EXIT attempt={attempt} code={code}\n')
                log.flush()
            if (root / 'PAUSE').exists():
                publish(root, manifest, 'paused', attempt, exit_code=code)
                return
            status_path = root / 'status.json'
            worker_status = json.loads(status_path.read_text()) if status_path.exists() else {}
            done = len(complete_rows(root / 'queries.jsonl'))
            if code == 0 and done == len(manifest) and worker_status.get('state') == 'complete':
                publish(root, manifest, 'complete', attempt, exit_code=code)
                return
            if code == 0 or attempt == MAX_RETRIES + 1:
                publish(root, manifest, 'failed', attempt, exit_code=code)
                raise RuntimeError('Tail-only worker did not finish all manifest queries')
            publish(root, manifest, 'retrying', attempt, exit_code=code)
            time.sleep(10)


if __name__ == '__main__':
    main()
