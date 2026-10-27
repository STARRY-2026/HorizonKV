"""Detached native SparseMM fixed-896 smoke validation followed by nine-task OVO.

Create OUTPUT_ROOT/PAUSE to stop the verified worker and preserve predictions.
The process command and Linux start tick are checked before attaching/signalling.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta
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
from zoneinfo import ZoneInfo

from kvbench.fixed_inputs import fixed_input_metadata
from sparsemm_native import supervise_simplestream_k500_input_only as helpers
from sparsemm_native.supervise_ovobench_dual import atomic_write, progress_rows, write_json


ROOT = Path(__file__).resolve().parents[1]
MODULE = 'sparsemm_native.run_ovobench_sparsemm_fixed896'
DEFAULT_OUTPUT = ROOT / 'results/ovobench_sparsemm_native_k500_fps05_fixed896_20261026'
DEFAULT_REFERENCE = ROOT / 'results/ovobench_simplestream_onefour_pseudo_k500_fps05_fixed896_20261026/full'
TASKS = helpers.BACKWARD + helpers.REALTIME
MAX_RETRIES = 3


def worker_command(output, manifest):
    return [sys.executable, '-u', '-m', MODULE, '--output-root', str(output),
            '--manifest', str(manifest), '--concurrent-processes', '1']


def worker_environment():
    env = dict(os.environ, HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               MALLOC_ARENA_MAX='2')
    env['PYTHONPATH'] = os.pathsep.join((str(ROOT),
        str(ROOT / 'sparsemm_baselines/upstream/lmms-eval'), str(ROOT / 'sparsemm_protokv/runtime_deps')))
    # gpu_lease independently validates admission; never inherit a shared-run hint.
    env.pop('KVBENCH_GPU_CONCURRENT_PROCESSES', None)
    env.pop('SPARSEMM_GPU_LEASE_SHARED', None)
    return env


def score(rows, expected):
    counts = Counter(r['task'] for r in rows)
    correct = Counter(r['task'] for r in rows if r['target'] in r['response'])
    tasks = {task: dict(total=counts[task], correct=correct[task],
             accuracy=correct[task]/counts[task] if counts[task] else None,
             expected_total=expected[task]) for task in TASKS if expected[task]}
    def complete_mean(names):
        return (sum(tasks[t]['accuracy'] for t in names) / len(names)
                if all(t in tasks and counts[t] == expected[t] and expected[t] > 0 for t in names) else None)
    back, realtime = complete_mean(helpers.BACKWARD), complete_mean(helpers.REALTIME)
    budgets = [r['budget_audit']['actual_tokens_per_head'] for r in rows]
    return dict(done=len(rows), correct=sum(correct.values()), per_task=tasks,
        question_accuracy=sum(correct.values())/len(rows) if rows else None,
        nine_task_macro=complete_mean(TASKS), backward_macro=back, realtime_macro=realtime,
        official_two_group_macro=(back+realtime)/2 if back is not None and realtime is not None else None,
        actual_tokens_per_head_mean=sum(budgets)/len(budgets) if budgets else None)


def pct(value):
    return '—' if value is None else f'{value:.2%}'


def estimate_remaining(queries, rows, calibration=None):
    """Model seconds = a*new temporal units + b*questions, both nonnegative.

    Only even source prefixes are shared. Each odd question endpoint adds a
    separately encoded padded unit, even when another question has that endpoint.
    Known source lengths cap estimated targets exactly as the native sampler.
    """
    calibration = rows if calibration is None else calibration
    known, completed, cursors = {}, {str(r['query_id']) for r in rows}, {}
    by_id = {str(q['query_id']): q for q in queries}
    for row in calibration:
        audit = row.get('input_audit', {})
        video, frames, fps = audit.get('source_video'), audit.get('source_frames'), audit.get('source_fps')
        if video and frames and fps and round(fps) > 0:
            step = int(round(fps) / .5)
            known[video] = (int(frames)+step-1)//step
    for row in rows:
        query = by_id[str(row['query_id'])]
        video, target = query['video_path'], int(row['frames'])
        cursors[video] = max(cursors.get(video, 0), target-target%2)
    remaining_units, remaining_questions = 0, 0
    for query in queries:
        if str(query['query_id']) in completed:
            continue
        video = query['video_path']
        target = math.ceil(query['clip_end'] * .5)
        target = min(target, known.get(video, target))
        even = target-target%2
        remaining_units += max(0, even-cursors.get(video, 0))//2 + target%2
        cursors[video] = max(cursors.get(video, 0), even)
        remaining_questions += 1
    samples = [((int(r['encoded_frames'])+1)//2, float(r['seconds'])) for r in calibration
               if r.get('encoded_frames') is not None and r.get('seconds') is not None
               and math.isfinite(float(r['seconds'])) and float(r['seconds']) >= 0]
    n, units = len(samples), sum(u for u, _ in samples)
    result = dict(method='calibrating', seconds=None, remaining_temporal_units=remaining_units,
        remaining_questions=remaining_questions, known_source_lengths=len(known),
        calibration_queries=n, calibration_temporal_units=units, provisional=False)
    if not remaining_questions:
        return dict(result, method='complete', seconds=0.)
    if n < 3 or units < 32:
        return result
    su, st = units, sum(t for _, t in samples)
    suu, sut = sum(u*u for u, _ in samples), sum(u*t for u, t in samples)
    determinant = n*suu-su*su
    if determinant <= 1e-12:
        return dict(result, method='query_mean_insufficient_unit_variation',
                    seconds=st/n*remaining_questions, provisional=True)
    # The optimum of the nonnegative two-parameter regression is either the
    # unconstrained positive solution or one of the two boundary solutions.
    candidates = [(0., st/n), (sut/suu, 0.)] if suu else [(0., st/n)]
    a = (n*sut-su*st)/determinant
    b = (st-a*su)/n
    if a >= 0 and b >= 0:
        candidates.append((a, b))
    a, b = min(candidates, key=lambda ab: sum((t-ab[0]*u-ab[1])**2 for u, t in samples))
    return dict(result, method='nonnegative_temporal_units_plus_questions',
                seconds=a*remaining_units+b*remaining_questions,
                seconds_per_temporal_unit=a, seconds_per_question=b)


def write_progress(root, phase, state, output, manifest, reference, **extra):
    queries = json.loads(manifest.read_text())
    expected = Counter(q['task'] for q in queries)
    rows = progress_rows(output / 'queries.jsonl')
    total = sum(expected.values())
    summary = score(rows, expected)
    now = datetime.now(ZoneInfo('Asia/Shanghai'))
    seconds = sum(float(r['seconds']) for r in rows)
    calibration = (progress_rows(root / 'validation_fixed896/queries.jsonl') + rows
                   if phase == 'full' else rows)
    estimate = estimate_remaining(queries, rows, calibration)
    remaining = estimate['seconds']
    eta = (now+timedelta(seconds=remaining)).isoformat(timespec='seconds') if remaining is not None else None
    status = dict(state=state, phase=phase, total=total, **summary,
        updated_at=now.isoformat(timespec='seconds'), supervisor_pid=os.getpid(),
        stage_output=str(output), worker_status=helpers.optional_json(output / 'status.json'),
        measured_query_seconds=seconds, estimated_remaining_seconds=remaining,
        estimated_completion_beijing=eta, eta_model=estimate, **extra)
    lines = ['# SparseMM 固定896×896实验进度', '', f'更新时间（北京时间）：{status["updated_at"]}', '',
        f'状态：**{state}**；阶段：**{phase}**；完成 **{len(rows)}/{total}** 题。', '',
        '配置：Qwen2.5-VL-7B；fps=0.5；896×896等比缩放、灰色补边；'
        '原生视频编码每两真实帧为1024视觉token；外部16帧块为8192。', '',
        'SparseMM原生注意力驱逐和原打分表；平均预算目标K500，protect32，逐头不加上限。'
        '原分配的取整误差、短缓存尚未压缩的实际用量均如实记录。', '',
        '流程：3题固定输入验证 → 九类1468题全量；不运行REC、SSR、CRR，不重跑基线。', '',
        '| 任务 | 正确/已答 | 全量题数 | 已答正确率 |', '|---|---:|---:|---:|']
    for task, value in summary['per_task'].items():
        lines.append(f'| {task} | {value["correct"]}/{value["total"]} | {value["expected_total"]} | {pct(value["accuracy"])} |')
    lines += ['', f'已答逐题正确率：{pct(summary["question_accuracy"])}。',
        f'实际平均视觉缓存：{summary["actual_tokens_per_head_mean"]} token/头。',
        f'官方两组均值：{pct(summary["official_two_group_macro"])}；九类均值：{pct(summary["nine_task_macro"])}。',
        '宏均值只在相关任务全量完成后显示；局部已答成绩不能当作全量结果。']
    if remaining is not None and state not in ('complete', 'paused', 'failed'):
        basis = '样本编码量无差异，暂按逐题均值' if estimate['provisional'] else '按新增视觉时间单元与作答次数拟合'
        lines += ['', f'{basis}估计剩余约{remaining/3600:.2f}小时，预计完成：{eta}。',
            f'剩余计划约{estimate["remaining_temporal_units"]}个视觉时间单元、'
            f'{estimate["remaining_questions"]}次作答；已计入同视频前缀复用和奇数结尾重复编码。'
            '解码差异、当前正在处理的题目和后续视频仍会使估计变化。']
    elif remaining is None and state not in ('complete', 'paused', 'failed'):
        lines += ['', f'用时校准中：已测{estimate["calibration_queries"]}题、'
            f'{estimate["calibration_temporal_units"]}个编码时间单元；至少3题且累计32单元后显示预计用时。']
    if phase == 'full':
        try:
            report_comparison(output, rows, reference, expected)
        except Exception as error:
            write_json(root / 'report_error.json', dict(updated_at=status['updated_at'],
                       done=len(rows), error=f'{type(error).__name__}: {error}'))
            lines += ['', f'同题报告更新失败，推理继续：{type(error).__name__}: {error}']
        lines += ['', f'[与固定896 SimpleStream同题比较]({output / "comparison_report.md"})。']
    if extra.get('error'):
        lines += ['', f'错误：{extra["error"]}']
    lines += ['', f'详细目录：`{output}`', '', '根目录创建`PAUSE`暂停；删除后重启本守护可断点续跑。']
    atomic_write(root / 'progress.md', '\n'.join(lines)+'\n')
    write_json(root / 'supervisor_status.json', status)


def report_comparison(output, rows, reference, expected):
    baselines = {str(r['query_id']): r for r in progress_rows(reference / 'queries.jsonl')}
    current, matched, table = [], [], []
    for row in rows:
        qid = str(row['query_id'])
        if qid not in baselines:
            continue
        previous = baselines[qid]
        if (row['task'], row['target']) != (previous['task'], previous['target']):
            raise ValueError(f'Baseline question identity changed: {qid}')
        if row['input_audit']['fixed_input'] != previous['input_audit']['fixed_input']:
            raise ValueError(f'Fixed input metadata differs: {qid}')
        current.append(row)
        matched.append(previous)
        table.append(dict(query_id=qid, task=row['task'], target=row['target'],
            sparsemm_response=row['response'], simplestream_response=previous['response'],
            sparsemm_correct=row['target'] in row['response'],
            simplestream_correct=previous['target'] in previous['response'],
            sparsemm_tokens_per_head=row['budget_audit']['actual_tokens_per_head'],
            simplestream_tokens_per_head=previous['budget_audit']['actual_tokens_per_head']))
    left, right = score(current, expected), score(matched, expected)
    result = dict(scope='completed-question intersection', matched_questions=len(table),
        sparsemm=left, simplestream=right, baseline_root=str(reference),
        scoring='official raw case-sensitive target-in-response; no answer extraction',
        causal_cache_comparison=False,
        limitation='Same fps, spatial canvas and budget target; sampling, encoding, context, positions and prompts differ.')
    write_json(output / 'comparison_summary.json', result)
    lines = ['# SparseMM与SimpleStream固定896同题比较', '',
        f'共同完成{len(table)}题；官方原始回答按目标字母是否出现评分。', '',
        '| 任务 | 同题数/总数 | SparseMM | SimpleStream一帧／四帧K500 |', '|---|---:|---:|---:|']
    for task in TASKS:
        if task not in left['per_task']:
            continue
        a, b = left['per_task'][task], right['per_task'][task]
        lines.append(f'| {task} | {a["total"]}/{a["expected_total"]} | {pct(a["accuracy"])} | {pct(b["accuracy"])} |')
    for label, key in [('逐题均值', 'question_accuracy'), ('九类等权均值', 'nine_task_macro'),
                       ('官方两组等权均值', 'official_two_group_macro')]:
        lines.append(f'| {label} | {len(table)} | {pct(left[key])} | {pct(right[key])} |')
    lines += ['', f'实际平均视觉token/头：SparseMM {left["actual_tokens_per_head_mean"]}；'
        f'SimpleStream {right["actual_tokens_per_head_mean"]}。', '',
        '此表对齐了fps、空间画布和预算目标。采样位置、逐图与两帧融合编码、读取历史范围、'
        '位置编号和提示词仍沿用各自实现；成绩差异不能单独归因于缓存算法。', '',
        '相关任务全量完成后才报告其宏均值。']
    atomic_write(output / 'comparison_report.md', '\n'.join(lines)+'\n')
    buffer = io.StringIO()
    if table:
        writer = csv.DictWriter(buffer, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    atomic_write(output / 'comparison_per_query.csv', buffer.getvalue())


def validate_smoke(output, manifest):
    rows = progress_rows(output / 'queries.jsonl')
    expected = {str(q['query_id']): q for q in json.loads(manifest.read_text())}
    if len(rows) != len(expected) or {str(r['query_id']) for r in rows} != set(expected):
        raise ValueError('Fixed-input smoke query coverage differs')
    checks = []
    for row in rows:
        audit, budget = row['input_audit'], row['budget_audit']
        slots = [n for layer in budget['per_head_visual_slots'] for n in layer]
        tests = dict(target_exact=row['target'] == expected[str(row['query_id'])]['target'],
            response_present=isinstance(row['response'], str) and bool(row['response'].strip()),
            fixed_input=audit['fixed_input'] == fixed_input_metadata(896),
            fixed_unit_tokens=audit['tokens_per_temporal_unit'] == 1024,
            prefix_token_total=audit['prefix_encoded_visual_tokens'] == audit['video_temporal_units']*1024,
            every_visual_forward_fixed=bool(audit['native_video_forward_audits']) and
                all(a['visual_tokens'] == ((a['raw_frames']+1)//2)*1024
                    for a in audit['native_video_forward_audits']),
            physical_budget_recorded=len(slots) == 112 and min(slots) >= 0 and
                sum(slots) == budget['actual_visual_slots'] and
                abs(sum(slots)/112-budget['actual_tokens_per_head']) < 1e-8,
            no_individual_cap=budget['individual_quota_cap'] is None,
            question_tokens_excluded=budget['actual_includes_question_tokens'] is False)
        if not all(tests.values()):
            raise ValueError(f'Fixed-input smoke failed: {row["query_id"]}: {tests}')
        checks.append(dict(query_id=row['query_id'], checks=tests,
                           actual_tokens_per_head=budget['actual_tokens_per_head']))
    return checks


def run_stage(root, phase, manifest, reference, stopped):
    output = root / phase
    output.mkdir(parents=True, exist_ok=True)
    command = worker_command(output, manifest)
    total = len(json.loads(manifest.read_text()))
    failures = 0
    while True:
        record = helpers.find_worker(output, command)
        if (root / 'PAUSE').exists() or stopped():
            if record:
                helpers.stop_worker(record)
            write_progress(root, phase, 'paused', output, manifest, reference)
            return False
        if len(progress_rows(output / 'queries.jsonl')) == total and helpers.optional_json(output / 'status.json').get('state') == 'complete':
            return True
        child = None
        if record is None:
            helpers.repair_tail(output / 'queries.jsonl')
            with (output / 'worker.log').open('a') as log:
                child = subprocess.Popen(command, cwd=ROOT, env=worker_environment(),
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record = helpers.process_record(child.pid)
            if record:
                write_json(output / 'worker_identity.json', record)
                (output / 'worker.pid').write_text(str(child.pid)+'\n')
        while record and helpers.process_record(record['pid']) is not None:
            if (root / 'PAUSE').exists() or stopped():
                helpers.stop_worker(record)
                if child:
                    child.wait()
                write_progress(root, phase, 'paused', output, manifest, reference)
                return False
            write_progress(root, phase, 'running', output, manifest, reference,
                           worker_pid=record['pid'], retry=failures)
            time.sleep(10)
        code = child.wait() if child else None
        state = helpers.optional_json(output / 'status.json')
        if code in (None, 0) and state.get('state') == 'complete' and len(progress_rows(output / 'queries.jsonl')) == total:
            return True
        failures += 1
        error = f'Worker stopped: exit={code}, state={state.get("state")}, {state.get("error", "")}'
        if failures > MAX_RETRIES:
            raise RuntimeError(error+'; three retries exhausted')
        write_progress(root, phase, 'retrying', output, manifest, reference, retry=failures, error=error)
        for _ in range(10):
            if (root / 'PAUSE').exists() or stopped():
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
    reference = args.reference_root.resolve()
    stages = [('validation_fixed896', root / 'smoke_manifest.json'),
              ('full', (args.manifest or root / 'full_nine_manifest.json').resolve())]
    stopped = [False]
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    for name in (signal.SIGTERM, signal.SIGINT):
        signal.signal(name, lambda *_: stopped.__setitem__(0, True))
    with (root / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        (root / 'supervisor.pid').write_text(str(os.getpid())+'\n')
        phase, manifest = stages[0]
        try:
            if (root / 'PAUSE').exists() or stopped[0]:
                for name, path in stages:
                    output = root / name
                    record = helpers.find_worker(output, worker_command(output, path))
                    if record:
                        helpers.stop_worker(record)
                        phase, manifest = name, path
                write_progress(root, phase, 'paused', root / phase, manifest, reference)
                return
            for phase, manifest in stages:
                if not run_stage(root, phase, manifest, reference, lambda: stopped[0]):
                    return
                if phase == 'validation_fixed896':
                    write_json(root / 'validation_checks.json', dict(passed=True,
                        checks=validate_smoke(root / phase, manifest), updated_at=helpers.base.now()))
            write_json(root / 'final_summary.json', helpers.optional_json(root / 'full/summary.json'))
            write_progress(root, 'full', 'complete', root / 'full', manifest, reference)
        except BaseException as error:
            write_progress(root, phase, 'failed', root / phase, manifest, reference,
                           error=f'{type(error).__name__}: {error}')
            raise


if __name__ == '__main__':
    main()
