"""Detached, resumable HERMES fixed-896 validation and nine-task OVO evaluation.

Create OUTPUT_ROOT/PAUSE to stop only this supervisor's verified detached worker.
The original HERMES cache algorithm and FP16 loader remain in the worker.
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

from kvbench.fixed_inputs import assert_qwen_grid, fixed_input_metadata
from sparsemm_native import supervise_simplestream_k500_input_only as helpers
from sparsemm_native.supervise_ovobench_dual import atomic_write, progress_rows, write_json


ROOT = Path(__file__).resolve().parents[1]
MODULE = 'sparsemm_native.run_ovobench_hermes_fixed896'
DEFAULT_OUTPUT = ROOT / 'results/ovobench_hermes_native_k500_fps05_fixed896_20261027'
DEFAULT_REFERENCES = {
    'sparsemm': ROOT / 'results/ovobench_sparsemm_native_k500_fps05_fixed896_20261026/full',
    'simplestream': ROOT / 'results/ovobench_simplestream_onefour_pseudo_k500_fps05_fixed896_20261026/full',
}
TASKS = helpers.BACKWARD + helpers.REALTIME
MAX_RETRIES = 3
VALIDATION_FIELDS = ('official_prompt_mc_false', 'cache_restored', 'positions_restored',
    'native_backend_unchanged', 'real_qa_history_preserved', 'history_length_increased_by_one')
COMPARISON_LIMITATION = (
    '空间画布896×896、fps=0.5和目标平均K500一致；HERMES使用原生FP16，'
    'SparseMM/SimpleStream使用BF16；HERMES每16真实帧一轮编码与压缩，'
    'SparseMM每两帧一轮，SimpleStream逐图重新prefill。采样、上下文范围、'
    '位置编号及算法细节仍可能不同。按用户指定，HERMES保留上一轮真实问答并据此调整后续压缩，'
    '不是不读取前题的伪查询策略，不能把结果视为只改变缓存算法的严格因果对照。'
)


def worker_command(output, manifest):
    return [sys.executable, '-u', '-m', MODULE, '--output-root', str(output),
            '--manifest', str(manifest), '--concurrent-processes', '1']


def worker_environment():
    env = dict(os.environ, HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MALLOC_ARENA_MAX='2',
        PROTOKV_ENABLE='0', PROTOKV_FULL_ENABLE='0', INFINIPOT_V_PUBLIC='0')
    env['PYTHONPATH'] = os.pathsep.join((str(ROOT),
        str(ROOT / 'sparsemm_baselines/upstream/lmms-eval'), str(ROOT / 'sparsemm_protokv/runtime_deps')))
    env.pop('KVBENCH_GPU_CONCURRENT_PROCESSES', None)
    env.pop('SPARSEMM_GPU_LEASE_SHARED', None)
    return env


def normalized_command(command):
    """Normalize only known filesystem arguments; retain module and flags."""
    command = list(command)
    if command:
        command[0] = os.path.realpath(command[0])
    for flag in ('--output-root', '--manifest'):
        if flag in command:
            index = command.index(flag)+1
            if index >= len(command):
                raise ValueError(f'Missing worker command value for {flag}')
            command[index] = str(Path(command[index]).resolve())
    return command


def process_record(pid):
    """Read immutable Linux identity around cmdline, which may be empty at exec."""
    try:
        directory = Path('/proc')/str(pid)
        before = (directory/'stat').read_text().rsplit(')', 1)[1].split()
        command = [part.decode() for part in (directory/'cmdline').read_bytes().split(b'\0') if part]
        after = (directory/'stat').read_text().rsplit(')', 1)[1].split()
        if after[0] in ('Z', 'X'):
            return None
        # Do not mix an old process's stat with a reused PID's command line.
        identity = (after[19], after[2], after[3])
        if identity != (before[19], before[2], before[3]):
            raise RuntimeError('Linux process identity changed while reading /proc')
        return dict(pid=int(pid), start_ticks=after[19], process_group=int(after[2]),
            session=int(after[3]), command=normalized_command(command))
    except (OSError, ValueError, UnicodeError, IndexError):
        return None


def verified_process(pid, command, saved=None, *, timeout=5.):
    """Wait out empty exec cmdline without accepting PID reuse or another job."""
    expected = normalized_command(command)
    if saved and saved.get('command') and normalized_command(saved['command']) != expected:
        raise RuntimeError('Saved worker command differs from the requested worker')
    deadline = time.monotonic()+timeout
    identity_anchor = saved
    while True:
        current = process_record(pid)
        if current is None:
            return None
        if identity_anchor is None:
            identity_anchor = current
        if (current['pid'] != int(identity_anchor['pid']) or
                str(current['start_ticks']) != str(identity_anchor['start_ticks'])):
            raise RuntimeError('Worker PID was reused; refusing attachment or signals')
        for key in ('session', 'process_group'):
            if (current[key] != current['pid'] or
                    (key in identity_anchor and identity_anchor[key] != current[key])):
                raise RuntimeError('Worker session/process group changed; refusing attachment or signals')
        if current['command']:
            if current['command'] != expected:
                raise RuntimeError('Worker command changed; refusing attachment or signals')
            return current
        if time.monotonic() >= deadline:
            raise RuntimeError('Worker command line remained empty during exec; refusing an unverifiable worker')
        time.sleep(.05)


def find_worker(output, command):
    path = output/'worker_identity.json'
    if not path.exists():
        return None
    saved = json.loads(path.read_text())
    current = verified_process(saved['pid'], command, saved)
    if current is not None and current != saved:
        # Upgrade a legacy pid/start-tick record only after the live command,
        # detached session and process group have all been verified.
        write_json(path, current)
    return current


def stop_worker(record):
    current = verified_process(record['pid'], record['command'], record)
    if current is None:
        return
    os.killpg(record['pid'], signal.SIGTERM)
    deadline = time.monotonic()+15
    while time.monotonic() < deadline:
        if verified_process(record['pid'], record['command'], record) is None:
            return
        time.sleep(.5)
    if verified_process(record['pid'], record['command'], record) is not None:
        os.killpg(record['pid'], signal.SIGKILL)


def score(rows, expected):
    counts = Counter(row['task'] for row in rows)
    correct = Counter(row['task'] for row in rows if row['target'] in row['response'])
    tasks = {task: dict(total=counts[task], correct=correct[task], expected_total=expected[task],
        accuracy=correct[task]/counts[task] if counts[task] else None)
        for task in TASKS if expected[task]}
    complete = all(expected[task] > 0 and counts[task] == expected[task] for task in TASKS)
    def macro(names):
        return sum(tasks[task]['accuracy'] for task in names)/len(names) if complete else None
    backward, realtime = macro(helpers.BACKWARD), macro(helpers.REALTIME)
    budgets = [float(row['budget_audit']['actual_tokens_per_head']) for row in rows]
    return dict(done=len(rows), correct=sum(correct.values()), per_task=tasks,
        complete=complete, question_accuracy=sum(correct.values())/len(rows) if rows else None,
        backward_macro=backward, realtime_macro=realtime, nine_task_macro=macro(TASKS),
        official_two_group_macro=(backward+realtime)/2 if complete else None,
        actual_tokens_per_head_mean=sum(budgets)/len(budgets) if budgets else None,
        actual_tokens_per_head_min=min(budgets) if budgets else None,
        actual_tokens_per_head_max=max(budgets) if budgets else None)


def pct(value):
    return '—' if value is None else f'{value:.2%}'


def new_units(row):
    """Count newly computed units, including padding at question boundaries."""
    if row.get('encoded_temporal_units') is not None:
        return int(row['encoded_temporal_units'])
    if row.get('new_encoded_visual_tokens') is not None:
        count = int(row['new_encoded_visual_tokens'])
        if count % 1024:
            raise ValueError('New encoded visual token count is not a whole fixed-size temporal unit')
        return count // 1024
    raise ValueError('HERMES row must record encoded_temporal_units or new_encoded_visual_tokens')


def source_limits(calibration, source_metadata=None):
    """Use real source frame counts; do not infer video length from question time."""
    records = dict(source_metadata or {})
    for row in calibration:
        audit = row.get('input_audit', {})
        path = audit.get('source_video')
        if path and audit.get('source_frames') and audit.get('source_fps'):
            records[path] = audit
    limits = {}
    for path, record in records.items():
        frames, fps = record.get('source_frames'), record.get('source_fps')
        if frames is None or fps is None or round(float(fps)) <= 0:
            continue
        step = int(round(float(fps))/.5)
        if step < 1:
            raise ValueError('Invalid source FPS for the fixed sampling protocol')
        limits[path] = (int(frames)+step-1)//step
    return limits


def estimate_remaining(queries, rows, calibration=None, source_metadata=None):
    """Fit nonnegative seconds = a*new_units + b*questions.

    Native HERMES commits every question endpoint, including odd padded tails.
    The next question starts from that endpoint rather than from an even or
    sixteen-frame boundary. Question history is retained as requested.
    """
    calibration = rows if calibration is None else calibration
    limits = source_limits(calibration, source_metadata)
    by_id = {str(query['query_id']): query for query in queries}
    completed, cursors = set(), {}
    for row in rows:
        qid = str(row['query_id'])
        query = by_id[qid]
        completed.add(qid)
        video, frames = query['video_path'], int(row['frames'])
        cursors[video] = max(cursors.get(video, 0), frames)
    remaining_units, remaining_questions = 0, 0
    for query in queries:
        if str(query['query_id']) in completed:
            continue
        video = query['video_path']
        target = math.ceil(float(query['clip_end'])*.5)
        target = min(target, limits.get(video, target))
        delta = max(0, target-cursors.get(video, 0))
        remaining_units += (delta//16)*8 + ((delta%16)+1)//2
        cursors[video] = max(cursors.get(video, 0), target)
        remaining_questions += 1
    samples = [(new_units(row), float(row['seconds'])) for row in calibration
        if row.get('seconds') is not None and math.isfinite(float(row['seconds']))
        and float(row['seconds']) >= 0]
    count, units = len(samples), sum(unit for unit, _ in samples)
    result = dict(method='calibrating', seconds=None, remaining_temporal_units=remaining_units,
        remaining_questions=remaining_questions, known_source_lengths=len(limits),
        calibration_queries=count, calibration_temporal_units=units, provisional=False,
        native_chunk_frames=16, incremental_unit_rule='floor(delta_frames/16)*8+ceil((delta_frames%16)/2)',
        replay_note='Restarting inside a video replays prior saved endpoints and QA to restore native history; this startup work is not included.')
    if not remaining_questions:
        return dict(result, method='complete', seconds=0.)
    if count < 3 or units < 32:
        return result
    su, st = units, sum(seconds for _, seconds in samples)
    suu = sum(unit*unit for unit, _ in samples)
    sut = sum(unit*seconds for unit, seconds in samples)
    determinant = count*suu-su*su
    if determinant <= 1e-12:
        return dict(result, method='query_mean_insufficient_unit_variation',
            seconds=st/count*remaining_questions, provisional=True)
    candidates = [(0., st/count), (sut/suu, 0.)] if suu else [(0., st/count)]
    a = (count*sut-su*st)/determinant
    b = (st-a*su)/count
    if a >= 0 and b >= 0:
        candidates.append((a, b))
    a, b = min(candidates, key=lambda ab: sum(
        (seconds-ab[0]*unit-ab[1])**2 for unit, seconds in samples))
    return dict(result, method='nonnegative_temporal_units_plus_questions',
        seconds=a*remaining_units+b*remaining_questions,
        seconds_per_temporal_unit=a, seconds_per_question=b)


def report_comparison(output, rows, references, expected):
    comparisons, csv_rows = {}, []
    lines = ['# HERMES固定896同题对比', '',
        '官方原始回答评分：目标字母是否出现在原始输出中，区分大小写，不提取或改写答案。', '',
        COMPARISON_LIMITATION, '']
    for name, directory in references.items():
        baseline = {str(row['query_id']): row for row in progress_rows(directory/'queries.jsonl')}
        current, matched = [], []
        for row in rows:
            previous = baseline.get(str(row['query_id']))
            if previous is None:
                continue
            if (row['task'], row['target']) != (previous['task'], previous['target']):
                raise ValueError(f'Baseline identity differs for {name}/{row["query_id"]}')
            if row['input_audit']['fixed_input'] != previous['input_audit']['fixed_input']:
                raise ValueError(f'Fixed spatial input metadata differs for {name}/{row["query_id"]}')
            current.append(row)
            matched.append(previous)
            csv_rows.append(dict(baseline=name, query_id=str(row['query_id']), task=row['task'],
                target=row['target'], hermes_response=row['response'], baseline_response=previous['response'],
                hermes_correct=row['target'] in row['response'],
                baseline_correct=previous['target'] in previous['response'],
                hermes_tokens_per_head=row['budget_audit']['actual_tokens_per_head'],
                baseline_tokens_per_head=previous['budget_audit']['actual_tokens_per_head']))
        left, right = score(current, expected), score(matched, expected)
        comparisons[name] = dict(matched_questions=len(current), hermes=left, baseline=right,
            baseline_root=str(directory), causal_cache_comparison=False)
        lines += [f'## 与{name}比较', '', f'双方共同完成 **{len(current)}** 题；以下两列只比较同题交集。', '',
            f'| 任务 | 同题数/全量 | HERMES | {name} |', '|---|---:|---:|---:|']
        for task, value in left['per_task'].items():
            other = right['per_task'][task]
            lines.append(f'| {task} | {value["total"]}/{value["expected_total"]} | '
                f'{pct(value["accuracy"])} | {pct(other["accuracy"])} |')
        for label, key in [('逐题均值', 'question_accuracy'), ('九类等权均值', 'nine_task_macro'),
                ('官方两组等权均值', 'official_two_group_macro')]:
            lines.append(f'| {label} | {len(current)} | {pct(left[key])} | {pct(right[key])} |')
        lines += ['', f'同题实际平均视觉token/头：HERMES {left["actual_tokens_per_head_mean"]}；'
            f'{name} {right["actual_tokens_per_head_mean"]}。', '']
    lines += ['九类全量全部完成后才展示宏均值；部分题目的逐题正确率不能当作全量成绩。']
    write_json(output/'comparison_summary.json', dict(scoring='official raw case-sensitive target-in-response',
        comparisons=comparisons, causal_cache_comparison=False, limitation=COMPARISON_LIMITATION))
    atomic_write(output/'comparison_report.md', '\n'.join(lines)+'\n')
    buffer = io.StringIO()
    if csv_rows:
        writer = csv.DictWriter(buffer, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    atomic_write(output/'comparison_per_query.csv', buffer.getvalue())


def validate_smoke(output, manifest):
    rows = progress_rows(output/'queries.jsonl')
    expected = {str(query['query_id']): query for query in json.loads(manifest.read_text())}
    if len(rows) != len(expected) or {str(row['query_id']) for row in rows} != set(expected):
        raise ValueError('HERMES smoke query coverage differs from the pinned manifest')
    checked = []
    for row in rows:
        audit, budget, validation = row['input_audit'], row['budget_audit'], row['validation']
        layers = budget.get('per_head_visual_slots', budget.get('per_head_visual_tokens', []))
        slots = [value for layer in layers for value in layer]
        forward_checks, raw_frames, actual_prefix_units = [], 0, 0
        for forward in audit['native_video_forward_audits']:
            valid_count = type(forward['raw_frames']) is int and 1 <= forward['raw_frames'] <= 16
            assert_qwen_grid(forward['grid_thw'], forward['visual_tokens'], forward['raw_frames'],
                canvas_size=896, kind='video')
            forward_checks.append(valid_count)
            raw_frames += forward['raw_frames']
            actual_prefix_units += (forward['raw_frames']+1)//2
        history_before = validation.get('history_length_before')
        history_after = validation.get('history_length_after')
        tests = dict(target_exact=row['target'] == expected[str(row['query_id'])]['target'],
            response_present=isinstance(row['response'], str) and bool(row['response'].strip()),
            fixed_input=audit['fixed_input'] == fixed_input_metadata(896),
            fixed_unit_tokens=audit['tokens_per_temporal_unit'] == 1024,
            prefix_token_total=audit['prefix_encoded_visual_tokens'] == actual_prefix_units*1024
                and audit['video_temporal_units'] == actual_prefix_units and raw_frames == row['frames'],
            every_native_forward_fixed=bool(forward_checks) and all(forward_checks),
            physical_budget_recorded=len(layers) == 28 and all(len(layer) == 4 for layer in layers)
                and all(type(value) is int and 0 <= value <= 500 for value in slots)
                and sum(slots) == budget['actual_visual_slots']
                and abs(sum(slots)/112-budget['actual_tokens_per_head']) < 1e-8,
            shared_per_layer_positions=all(len(set(layer)) == 1 for layer in layers),
            question_tokens_excluded=budget['actual_includes_question_tokens'] is False,
            encoded_work_recorded=new_units(row) >= 0,
            native_history_mode=validation.get('history_mode') == 'native_previous_real_qa',
            native_history_count=type(history_before) is int and history_before >= 0
                and history_after == history_before+1,
            **{field: validation.get(field) is True for field in VALIDATION_FIELDS})
        if not all(tests.values()):
            raise ValueError(f'HERMES fixed-input smoke failed for {row["query_id"]}: {tests}')
        checked.append(dict(query_id=row['query_id'], checks=tests,
            actual_tokens_per_head=budget['actual_tokens_per_head']))
    return checked


def write_progress(root, phase, state, output, manifest, references, **extra):
    queries = json.loads(manifest.read_text())
    expected = Counter(query['task'] for query in queries)
    rows = progress_rows(output/'queries.jsonl')
    result = score(rows, expected)
    calibration = (progress_rows(root/'validation_fixed896/queries.jsonl')+rows
        if phase == 'full' else rows)
    source_metadata = helpers.optional_json(root/'source_lengths.json')
    estimate = estimate_remaining(queries, rows, calibration, source_metadata)
    now = datetime.now(ZoneInfo('Asia/Shanghai'))
    remaining = estimate['seconds']
    eta = (now+timedelta(seconds=remaining)).isoformat(timespec='seconds') if remaining is not None else None
    status = dict(state=state, phase=phase, total=len(queries), **result,
        updated_at=now.isoformat(timespec='seconds'), supervisor_pid=os.getpid(), stage_output=str(output),
        worker_status=helpers.optional_json(output/'status.json'),
        measured_query_seconds=sum(float(row['seconds']) for row in rows), eta_model=estimate,
        estimated_remaining_seconds=remaining, estimated_completion_beijing=eta, **extra)
    lines = ['# HERMES 固定896×896实验进度', '', f'更新时间（北京时间）：{status["updated_at"]}', '',
        f'状态：**{state}**；阶段：**{phase}**；完成 **{len(rows)}/{len(queries)}** 题。', '',
        '配置：Qwen2.5-VL-7B，原生HERMES FP16；fps=0.5；896×896等比缩放、gray127补边；'
        '每两真实帧1024视觉token，16帧原生块8192 token。', '',
        '原生HERMES local/global/mixed伪查询压缩；每层共享位置目标K500，初始文本前缀额外；'
        '不添加SparseMM的protect32，不改变近期/长期打分、摘要或位置处理。'
        '保留原生上一轮真实问答，以调整下一次local/global伪查询和压缩；'
        '作答后KV与位置状态回滚，但真实问答历史继续保留。每题新增尾段永久写入视频缓存。', '',
        '流程：固定输入小样本验证 → OVO九类1468题；不跑REC、SSR、CRR，不重跑已有基线。', '',
        '| 任务 | 正确/已答 | 全量题数 | 已答正确率 |', '|---|---:|---:|---:|']
    for task, value in result['per_task'].items():
        lines.append(f'| {task} | {value["correct"]}/{value["total"]} | {value["expected_total"]} | '
            f'{pct(value["accuracy"])} |')
    lines += ['', f'已答逐题正确率：{pct(result["question_accuracy"])}。',
        f'实际平均视觉缓存：{result["actual_tokens_per_head_mean"]} token/头。',
        f'官方两组均值：{pct(result["official_two_group_macro"])}；九类均值：{pct(result["nine_task_macro"])}。',
        '九类全量全部完成后才展示宏均值；部分题目结果不能当作全量成绩。']
    if remaining is not None and state not in ('complete', 'paused', 'failed'):
        basis = '暂按逐题均值' if estimate['provisional'] else '按新增视觉时间单元与作答次数拟合'
        lines += ['', f'{basis}：估计剩余约{remaining/3600:.2f}小时，预计完成：{eta}。',
            f'剩余约{estimate["remaining_temporal_units"]}个视觉单元、{estimate["remaining_questions"]}次作答；'
            '按相邻问题时间增量拆分16帧块和补帧单元，重复时间终点不重新编码，真实源长度限制采样数。'
            '断点从视频中途恢复时，需按已保存问题边界和回答重放前缀以恢复真实历史，此启动开销未计入。'
            '当前处理题的未落盘时间、解码差异和后续视频仍会使估时变化。']
    elif remaining is None and state not in ('complete', 'paused', 'failed'):
        lines += ['', f'用时校准中：已测{estimate["calibration_queries"]}题、'
            f'{estimate["calibration_temporal_units"]}个时间单元；至少3题且累计32单元后显示估时。']
    if phase == 'full':
        try:
            report_comparison(output, rows, references, expected)
        except Exception as error:
            write_json(root/'report_error.json', dict(updated_at=status['updated_at'], done=len(rows),
                error=f'{type(error).__name__}: {error}'))
            lines += ['', f'同题报告更新失败，推理继续：{type(error).__name__}: {error}']
        lines += ['', f'[与已有SparseMM、SimpleStream固定896同题比较]({output/"comparison_report.md"})。',
            '', COMPARISON_LIMITATION]
    if extra.get('error'):
        lines += ['', f'错误：{extra["error"]}']
    lines += ['', f'详细目录：`{output}`', '', '根目录创建`PAUSE`暂停；删除后重启本守护可断点续跑。']
    atomic_write(root/'progress.md', '\n'.join(lines)+'\n')
    write_json(root/'supervisor_status.json', status)


def run_stage(root, phase, manifest, references, stopped):
    output = root/phase
    output.mkdir(parents=True, exist_ok=True)
    command = worker_command(output, manifest)
    total, failures = len(json.loads(manifest.read_text())), 0
    while True:
        record = find_worker(output, command)
        if (root/'PAUSE').exists() or stopped():
            if record:
                stop_worker(record)
            write_progress(root, phase, 'paused', output, manifest, references)
            return False
        if len(progress_rows(output/'queries.jsonl')) == total and helpers.optional_json(output/'status.json').get('state') == 'complete':
            return True
        child = None
        if record is None:
            helpers.repair_tail(output/'queries.jsonl')
            with (output/'worker.log').open('a') as log:
                child = subprocess.Popen(command, cwd=ROOT, env=worker_environment(),
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record = verified_process(child.pid, command)
            if record:
                write_json(output/'worker_identity.json', record)
                (output/'worker.pid').write_text(str(child.pid)+'\n')
        while record:
            current = verified_process(record['pid'], command, record)
            if current is None:
                break
            if (root/'PAUSE').exists() or stopped():
                stop_worker(record)
                if child:
                    child.wait()
                write_progress(root, phase, 'paused', output, manifest, references)
                return False
            write_progress(root, phase, 'running', output, manifest, references,
                worker_pid=record['pid'], retry=failures)
            time.sleep(10)
        code = child.wait() if child else None
        status = helpers.optional_json(output/'status.json')
        if code in (None, 0) and status.get('state') == 'complete' and len(progress_rows(output/'queries.jsonl')) == total:
            return True
        failures += 1
        error = f'Worker stopped: exit={code}, state={status.get("state")}, {status.get("error", "")}'
        if failures > MAX_RETRIES:
            raise RuntimeError(error+'; three retries exhausted')
        write_progress(root, phase, 'retrying', output, manifest, references, retry=failures, error=error)
        for _ in range(10):
            if (root/'PAUSE').exists() or stopped():
                break
            time.sleep(1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--sparsemm-reference-root', type=Path, default=DEFAULT_REFERENCES['sparsemm'])
    parser.add_argument('--simplestream-reference-root', type=Path, default=DEFAULT_REFERENCES['simplestream'])
    args = parser.parse_args(argv)
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    references = dict(sparsemm=args.sparsemm_reference_root.resolve(),
        simplestream=args.simplestream_reference_root.resolve())
    stages = [('validation_fixed896', root/'smoke_manifest.json'),
        ('full', (args.manifest or root/'full_nine_manifest.json').resolve())]
    stopped = [False]
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    for name in (signal.SIGTERM, signal.SIGINT):
        signal.signal(name, lambda *_: stopped.__setitem__(0, True))
    with (root/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        (root/'supervisor.pid').write_text(str(os.getpid())+'\n')
        phase, manifest = stages[0]
        try:
            if (root/'PAUSE').exists() or stopped[0]:
                for name, path in stages:
                    output = root/name
                    record = find_worker(output, worker_command(output, path))
                    if record:
                        stop_worker(record)
                        phase, manifest = name, path
                write_progress(root, phase, 'paused', root/phase, manifest, references)
                return 0
            for phase, manifest in stages:
                if not run_stage(root, phase, manifest, references, lambda: stopped[0]):
                    return 0
                if phase == 'validation_fixed896':
                    write_json(root/'validation_checks.json', dict(passed=True,
                        checks=validate_smoke(root/phase, manifest), updated_at=helpers.base.now()))
            write_json(root/'final_summary.json', helpers.optional_json(root/'full/summary.json'))
            write_progress(root, 'full', 'complete', root/'full', manifest, references)
        except BaseException as error:
            write_progress(root, phase, 'failed', root/phase, manifest, references,
                error=f'{type(error).__name__}: {error}')
            raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
