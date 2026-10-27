"""Resumable OVO-Bench streaming four-frame KV window, with no history bank."""
import argparse
import collections
import copy
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from sparsemm_native import run_ovobench as official
from sparsemm_native.run_ovobench_recent4 import IDENTITY, digest, select_queries, validate_row
from sparsemm_native.supervise_ovobench_dual import worker_environment, write_json

ROOT = official.ROOT
DEFAULT_OUTPUT = ROOT / 'results/ovobench_window_only_recent4_official_full_fps1_p151200_20261021'
PYTHON = Path('/home/yujun/.conda/envs/hermes-qwen/bin/python')
SOURCE_FILES = (
    'sparsemm_native/window_only.py', 'sparsemm_native/run_ovobench_window_only.py',
    'sparsemm_native/run_ovobench.py', 'sparsemm_native/run_ovobench_recent4.py',
    'sparsemm_native/gpu_slots.py', 'sparsemm_native/supervise_ovobench_dual.py',
    'sparsemm_protokv/streaming.py', 'sparsemm_protokv/cache.py', 'sparsemm_protokv/config.py',
    'sparsemm_protokv/qwen2_5.py', 'sparsemm_protokv/monkeypatch.py',
    'sparsemm_protokv/model_loader.py', 'sparsemm_protokv/models/qwen2_5_vl_stream.py',
    'sparsemm_baselines/adapter.py', 'sparsemm_baselines/cache.py',
    'sparsemm_baselines/run_streamingbench.py',
)


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def schedule(queries, grouped=True):
    if not grouped:
        return list(queries)
    groups = collections.defaultdict(list)
    for q in queries:
        groups[q['video_path']].append(q)
    return [q for group in groups.values() for q in sorted(group, key=lambda q: q['clip_end'])]


def batches(queries, grouped=True):
    if not grouped:
        return [[q] for q in queries]
    result = []
    for q in queries:
        if not result or result[-1][0]['video_path'] != q['video_path']:
            result.append([])
        result[-1].append(q)
    return result


def protocol(selected, grouped=True):
    from importlib.metadata import version
    return dict(
        version='official_ovo_streaming_recent4_window_only_v1', method='streaming_window_only',
        model_path=str(official.MODEL.resolve()), source_annotation=str(official.ANNOTATIONS.resolve()),
        fps=1.0, video_max_pixels=151200, dtype='bfloat16', chunk_size=16,
        max_new_tokens=16, greedy=True, repetition_penalty=1.1,
        repetition_penalty_scope='generated_tokens_only',
        prompts='unchanged_official_OVO_templates', score_mode='official_OVOBenchOfflineScore',
        official_aggregation='mean(backward_3_task_mean,realtime_6_task_mean,forward_3_task_mean)',
        recent_temporal_units=2, recent_frames=4,
        recent_frame_semantics='two_Qwen_video_units_usually_four_real_frames_odd_endpoint_three_real_plus_padding',
        visual_encoding='same_Qwen_video_temporal_patch_size2_not_independent_images',
        eviction='drop_outgoing_unit_before_new_visual_attention_at_every_layer_and_head',
        history_tokens_per_head=0, attention_eviction_scoring=False, external_quota_table=None,
        budget=None, compatibility_config_budget_unused=500,
        budget_semantics='actual_tokens_in_recent_units_no_fill_to_500',
        protect32='contained_in_recent_window_no_extra_visual_tokens',
        recent_kv_recomputed=False, earlier_context_may_survive_inside_retained_representations=True,
        text_prefix='same_initial_system_and_user_prefix_preserved_once',
        positions='same_absolute_streaming_video_MRoPE_no_renumbering_on_eviction',
        sampling='same_source_rounded_fps_fixed_step_prefix_ceil_timestamp_now_at_fps1',
        question_state='isolated_question_and_answer_KV_snapshot_restore_no_answer_leak',
        source_data='existing_source_videos_not_downloaded_official_reencoded_clips',
        grouped=grouped, execution_order='whole_video_first_appearance_then_timestamp' if grouped else 'fresh_queries',
        query_count=len(selected), query_ids=[q['query_id'] for q in selected], query_list_sha256=digest(selected),
        code_sha256={str(ROOT / name): official.sha256(ROOT / name) for name in SOURCE_FILES},
        input_sha256={str(official.ANNOTATIONS.resolve()): official.sha256(official.ANNOTATIONS)},
        processor_sha256={str((official.MODEL / name).resolve()): official.sha256(official.MODEL / name)
                          for name in ('config.json', 'preprocessor_config.json', 'tokenizer_config.json',
                                       'tokenizer.json', 'chat_template.json', 'generation_config.json')},
        dependencies={name: version(name) for name in ('torch', 'transformers', 'numpy', 'Pillow', 'decord', 'flash-attn')},
        comparison_note='Existing DP full runs use FPS0.5; original SimpleStream FPS1 also differs in prompts, clip bytes, resize and generation. No strict identical-protocol comparison is implied.',
    )


def pin(output, selected, grouped=True):
    output.mkdir(parents=True, exist_ok=True)
    value = protocol(selected, grouped)
    path = output / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError('Refusing to resume with changed code, inputs, queries or settings')
    if not path.exists():
        write_json(path, value)
        write_json(output / 'query_manifest.json', selected)
        for name in SOURCE_FILES:
            target = output / 'source_snapshot' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
    return value


def check_pins(output):
    value = json.loads((output / 'protocol.json').read_text())
    for section in ('code_sha256', 'input_sha256', 'processor_sha256'):
        for path, expected in value[section].items():
            if official.sha256(path) != expected:
                raise ValueError(f'Pinned window-only input/source changed: {path}')


def read_rows(output, expected):
    path = output / 'queries.jsonl'
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) > len(expected):
        raise ValueError('Too many checkpoint rows')
    for row, q in zip(rows, expected):
        validate_row(row, q)
        audit = row.get('cache_audit', {})
        if (row.get('fps') != 1.0 or audit.get('history_visual_tokens') != 0
                or audit.get('all_heads_same_window') is not True
                or audit.get('no_old_visual_positions') is not True
                or not row.get('qa_restored')):
            raise ValueError('Invalid streaming-window runtime audit')
    return rows


def run_group(bridge, group):
    import torch
    from sparsemm_baselines.run_streamingbench import sampled_indices
    from sparsemm_protokv.models.qwen2_5_vl_stream import BatchScopedVideoReader
    if not group or len({q['video_path'] for q in group}) != 1:
        raise ValueError('A grouped execution needs exactly one video')
    reader = BatchScopedVideoReader(group[0]['video_path'], num_threads=1)
    indices = sampled_indices(len(reader), reader.get_avg_fps(), 1.0)
    targets = [min(len(indices), math.ceil(q['clip_end'])) for q in group]
    if min(targets) < 1 or targets != sorted(targets):
        raise ValueError('Video queries must have increasing sampled prefix lengths')
    bridge._active_video_id = group[0]['query_id']
    bridge.encode_init_prompt()
    base_frames = 0

    def encode(start, stop):
        for offset in range(start, stop, 16):
            end = min(offset + 16, stop)
            with bridge._measure('video_decode_seconds'):
                arrays = reader.get_batch(indices[offset:end]).asnumpy()
                frames = torch.from_numpy(arrays)
            bridge.encode_video_chunk(frames, is_last_chunk=end == stop)

    def answer(q):
        audit = bridge.kv_cache.audit_window()
        length = len(bridge.conv_history)
        try:
            response = str(bridge.question_answering(
                {'question': q['prompt'], 'formatted_question': q['prompt'],
                 'prompt': bridge.get_prompt(q['prompt'], mc=False)}, max_new_tokens=16))
            checks = bridge.validation_checks
            if (checks['failures'] or len(checks['qa_checks']) != 1
                    or checks['qa_checks'][0].get('restored') is not True
                    or checks['qa_checks'][0].get('returned') is not True):
                raise AssertionError('Window-only QA failed to restore persistent state')
            checks['qa_checks'].clear()
            if bridge.kv_cache.audit_window() != audit:
                raise AssertionError('Window changed across an isolated question')
            selected_indices = [int(indices[index]) for unit in bridge.kv_cache.units
                                for index in range(unit.frame_start, unit.frame_start + unit.real_frames)]
            audit['recent_sampled_source_frame_indices'] = selected_indices
            audit['total_seen_visual_tokens'] = bridge._seen_visual_tokens
            audit['recent_positions_sha256'] = digest(torch.cat(
                [unit.positions for unit in bridge.kv_cache.units], dim=1).cpu().tolist())
            return response, audit
        finally:
            del bridge.conv_history[length:]

    for question_index, (q, target) in enumerate(zip(group, targets)):
        bridge._active_question_index = question_index
        bridge._active_question_end_time = q['clip_end']
        before = {k: v for k, v in bridge.timings.items() if isinstance(v, (int, float))}
        started = time.perf_counter()
        even = target - target % 2
        encode(base_frames, even)
        encoded = even - base_frames
        base_frames = even
        if target % 2:
            parent = bridge.kv_cache
            names = ('_next_position', '_seen_visual_tokens', 'total_processed_frames',
                     'last_encoded_frames', '_prefills', '_compressions', 'last_metadata')
            saved = {name: getattr(bridge, name) for name in names}
            bridge.kv_cache = copy.deepcopy(parent)
            try:
                encode(even, target)
                encoded += 1
                response, audit = answer(q)
            finally:
                bridge.kv_cache = parent
                for name, value in saved.items():
                    setattr(bridge, name, value)
        else:
            response, audit = answer(q)
        torch.cuda.synchronize()
        row = {key: q[key] for key in IDENTITY}
        row.update(response=response, correct=official.official_correct(q['task'], response, q['target']),
                   fps=1.0, frames=target, encoded_frames=encoded, seconds=time.perf_counter() - started,
                   prompt_sha256=digest(q['prompt']), cache_audit=audit, qa_restored=True,
                   sampled_source_indices_sha256=digest(indices[:target]),
                   timings={k: bridge.timings[k] - value for k, value in before.items()},
                   execution_mode='video_prefix_reuse_isolated_odd_endpoint')
        validate_row(row, q)
        yield row
    bridge.clear_cache()


def worker(args):
    from sparsemm_native.gpu_slots import gpu_lease
    from sparsemm_native.window_only import load_model
    output = Path(args.output)
    expected = official.queries()
    selected = schedule(select_queries(expected, args.ids), grouped=not args.fresh)
    pin(output, selected, grouped=not args.fresh)
    rows = read_rows(output, selected)
    remaining = selected[len(rows):]
    if not remaining:
        write_json(output / 'done.json', dict(finished_at=now(), queries=len(rows)))
        return
    parent_pause = output.parents[1] / 'PAUSE' if output.parent.name == 'shards' else output / 'PAUSE'
    lease_context = None
    while lease_context is None:
        if parent_pause.exists():
            return
        candidate = gpu_lease(output, concurrent_processes=2)
        try:
            lease = candidate.__enter__()
            lease_context = candidate
        except RuntimeError as exc:
            if not any(term in str(exc) for term in ('busy', 'MiB free')):
                raise
            write_json(output / 'waiting_for_gpu.json', dict(updated_at=now(), reason=str(exc)))
            time.sleep(15)
    try:
        write_json(output / 'worker.json', dict(pid=os.getpid(), session_id=os.getsid(0),
                   started_at=now(), lease=lease, command=sys.argv))
        bridge = load_model(official.MODEL, validate_state=True)
        bridge.timing_enabled = True
        bridge.timings['video_decode_seconds'] = 0.0
        for group in batches(remaining, grouped=not args.fresh):
            for row in run_group(bridge, group):
                validate_row(row, selected[len(rows)])
                with (output / 'queries.jsonl').open('a') as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
                    handle.flush()
                    os.fsync(handle.fileno())
                rows.append(row)
                write_json(output / 'summary.json', official.summary(rows, len(selected)))
                print(f"window_only {len(rows)}/{len(selected)} {row['query_id']} {row['task']} correct={row['correct']} frames={row['frames']} tokens={row['cache_audit']['recent_visual_tokens_per_head']}", flush=True)
                if parent_pause.exists():
                    return
        check_pins(output)
        write_json(output / 'done.json', dict(finished_at=now(), queries=len(rows)))
    finally:
        lease_context.__exit__(*sys.exc_info())


def split_videos(selected):
    groups = batches(selected)
    bins, cost, encoded = [[], []], [0, 0], [0, 0]
    tasks = []
    for index, group in enumerate(groups):
        targets = [math.ceil(q['clip_end']) for q in group]
        frames = max(t - t % 2 for t in targets) + sum(t % 2 for t in targets)
        tasks.append((frames + len(group), index, frames))
    for weight, index, frames in sorted(tasks, key=lambda x: (-x[0], x[1])):
        rank = min(range(2), key=lambda r: (cost[r], r))
        bins[rank].append(index)
        cost[rank] += weight
        encoded[rank] += frames
    jobs = [[q for index in sorted(indices) for q in groups[index]] for indices in bins]
    if ({q['video_path'] for q in jobs[0]} & {q['video_path'] for q in jobs[1]}
            or sum(len(job) for job in jobs) != len(selected)):
        raise AssertionError('Video sharding lost queries or split one video')
    return jobs, encoded


def publish(output, selected, jobs, state):
    available, measurements = {}, {}
    for rank, job in enumerate(jobs):
        rows = read_rows(output / f'shards/shard{rank}', job)
        for row in rows:
            if row['query_id'] in available:
                raise ValueError('Duplicate query across window shards')
            available[row['query_id']] = row
        frames = sum(row['encoded_frames'] for row in rows)
        seconds = sum(row['seconds'] for row in rows)
        expected_frames = state['expected_encoded_frames_by_shard'][rank]
        measurements[str(rank)] = dict(completed_queries=len(rows), total_queries=len(job),
            encoded_frames=frames, expected_encoded_frames=expected_frames, measured_seconds=seconds,
            estimated_remaining_seconds=max(0, expected_frames - frames) * seconds / frames if frames >= 100 else None)
    rows = [available[q['query_id']] for q in selected if q['query_id'] in available]
    result = official.summary(rows, len(selected))
    state.update(updated_at=now(), completed_queries=len(rows), measurements=measurements,
                 correct=sum(row['correct'] for row in rows))
    estimates = [value['estimated_remaining_seconds'] for value in measurements.values()]
    state['estimated_remaining_seconds'] = max(estimates) if all(v is not None for v in estimates) else None
    write_json(output / 'status.json', state)
    write_json(output / 'summary.json', result)
    tokens = [row['cache_audit']['recent_visual_tokens_per_head'] for row in rows]
    write_json(output / 'token_budget_summary.json', dict(queries=len(tokens),
        mean=sum(tokens)/len(tokens) if tokens else None, minimum=min(tokens) if tokens else None,
        maximum=max(tokens) if tokens else None, history_tokens_per_head=0))
    text = (f"更新时间：{state['updated_at']}；状态：{state['state']}。进度 **{len(rows)}/3035**。\n\n"
            "每个头只保留最近两个视频视觉时间单元（通常4帧，奇数结尾3帧加补帧）；"
            "新单元进入注意力前直接丢弃最旧单元；历史KV为0，不打分、不补满500。"
            "保留的近期KV不重算，仍可能携带此前上下文的影响。\n\n"
            "Qwen2.5-VL-7B，fps=1，原视频，像素上限151200，官方提示词和评分，贪心生成16token。"
            "双进程、同视频多题复用，问题KV作答后恢复。\n\n"
            '| 题型 | 完成题数 | 答对 | 正确率 |\n|---|---:|---:|---:|\n')
    for task in sorted({q['task'] for q in selected}):
        values = [row for row in rows if row['task'] == task]
        correct = sum(row['correct'] for row in values)
        accuracy = f'{correct / len(values):.2%}' if values else '—'
        text += f'| {task} | {len(values)} | {correct} | {accuracy} |\n'
    if rows:
        text += f"\n逐题正确率：{result['question_accuracy']:.2%}。实际视觉token平均 {sum(tokens)/len(tokens):.1f}，范围 {min(tokens)}–{max(tokens)}。\n"
    if result.get('official_total_average') is not None:
        text += f"\n官方总分：{result['official_total_average']:.2%}。\n"
    text += '\n此前DP为fps=0.5；原配置SimpleStream还使用不同提示词、分辨率和切片，后续比较需标注协议差异。\n'
    (output / 'live_report.md').write_text(text)
    if len(rows) == len(selected):
        temporary = output / 'queries.jsonl.tmp'
        temporary.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
        os.replace(temporary, output / 'queries.jsonl')
    return rows


def alive_worker(path):
    record = path / 'worker.json'
    if not record.exists():
        return None
    pid = json.loads(record.read_text())['pid']
    try:
        command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        if (b'sparsemm_native.run_ovobench_window_only' in command and b'--worker' in command
                and str(path).encode() in command):
            return pid
    except OSError:
        pass
    return None


def supervise(args):
    output = Path(args.output)
    selected = schedule(official.queries())
    pin(output, selected)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    with (output / '.supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs, expected_frames = split_videos(selected)
        for rank, job in enumerate(jobs):
            pin(output / f'shards/shard{rank}', job)
        state = dict(state='running', started_at=now(), supervisor_pid=os.getpid(),
                     session_id=os.getsid(0), concurrent_processes=2, total_queries=len(selected),
                     fps=1.0, history_tokens_per_head=0,
                     expected_encoded_frames_by_shard=expected_frames)
        children, attempts, next_attempt = {}, collections.Counter(), collections.defaultdict(float)
        while True:
            if (output / 'PAUSE').exists():
                state['state'] = 'pausing' if any(alive_worker(output / f'shards/shard{r}') for r in range(2)) else 'paused'
                publish(output, selected, jobs, state)
                if state['state'] == 'paused':
                    return
                time.sleep(5)
                continue
            for rank, job in enumerate(jobs):
                path = output / f'shards/shard{rank}'
                process = children.get(rank)
                if process is not None and process.poll() is None:
                    continue
                if (path / 'done.json').exists():
                    if len(read_rows(path, job)) != len(job):
                        raise RuntimeError('Worker completion marker disagrees with query checkpoint')
                    continue
                if alive_worker(path) is not None:
                    continue
                if attempts[rank] >= 3:
                    raise RuntimeError(f'Window shard {rank} failed three times; see worker.log')
                if time.monotonic() < next_attempt[rank]:
                    continue
                check_pins(output)
                command = [str(PYTHON), '-u', '-m', 'sparsemm_native.run_ovobench_window_only',
                           '--worker', '--output', str(path), '--ids', ','.join(q['query_id'] for q in job)]
                with (path / 'worker.log').open('ab') as handle:
                    children[rank] = subprocess.Popen(command, cwd=ROOT, env=worker_environment(),
                        stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                attempts[rank] += 1
                next_attempt[rank] = time.monotonic() + 30 * attempts[rank]
            state['worker_pids'] = {str(rank): pid for rank in range(2)
                                    if (pid := alive_worker(output / f'shards/shard{rank}')) is not None}
            state['worker_attempts'] = dict(attempts)
            rows = publish(output, selected, jobs, state)
            if all((output / f'shards/shard{rank}/done.json').exists() for rank in range(2)):
                if len(rows) != 3035:
                    raise RuntimeError('The full dataset is incomplete')
                check_pins(output)
                for rank in range(2):
                    check_pins(output / f'shards/shard{rank}')
                write_json(output / 'final_audit.json', dict(passed=True, checked_at=now(),
                    unique_queries=len({row['query_id'] for row in rows}),
                    all_qa_restored=all(row['qa_restored'] for row in rows),
                    all_history_empty=all(row['cache_audit']['history_visual_tokens'] == 0 for row in rows),
                    actual_per_head_positions_checked=True))
                state.update(state='complete', finished_at=now(), worker_pids={})
                publish(output, selected, jobs, state)
                return
            time.sleep(10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--fresh', action='store_true')
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT))
    parser.add_argument('--ids')
    args = parser.parse_args()
    try:
        worker(args) if args.worker else supervise(args)
    except BaseException as exc:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / 'error.json', dict(failed_at=now(), error=repr(exc)))
        if not args.worker and (output / 'status.json').exists():
            value = json.loads((output / 'status.json').read_text())
            value.update(state='error', error=repr(exc), updated_at=now())
            write_json(output / 'status.json', value)
        raise


if __name__ == '__main__':
    main()
