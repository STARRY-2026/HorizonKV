"""Run unmodified released Qwen2.5 SimpleStream with its original protocol.

The wrapper supplies durable supervision, two independent GPU processes and
measurement only. Model loading, image encoding, prefix building, generation,
prompts and final official scoring are executed from the pinned upstream tree.
"""
import argparse
import collections
import csv
import dataclasses
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import runpy
import signal
import subprocess
import sys
import time

ROOT = Path('/home/yujun/program/kvbench')
UPSTREAM = ROOT.parent / 'SimpleStream'
ENV_PYTHON = ROOT / 'runtime_envs/simplestream-qwen25/bin/python'
BASE_PYTHON = Path('/home/yujun/.conda/envs/hermes-qwen/bin/python')
DATA = ROOT / 'data/ovobench_upstream_chunks_20261021'
ANNOTATION = ROOT.parent / 'hermes/data/ovobench/official_20261014/annotations/ovo_bench_new.json'
MODEL = ROOT.parent / 'hermes/models/Qwen2.5-VL-7B-Instruct'
DEFAULT_OUTPUT = ROOT / 'results/ovobench_simplestream_upstream_recent4_fps1_20261021'
COMMIT = 'be227ee8ce90a258b286a34161688c2a59eeeadb'
UPSTREAM_FILES = ('main_experiments/eval_qwen25vl_ovo.py', 'lib/recent_window_eval.py',
                  'lib/recent_window_eval_qwen25.py', 'lib/qwen_exact_recent_decoder.py',
                  'ovo_constants.py', 'scoring/score_ovo_bench.py',
                  'requirements.txt', 'requirements-qwen25.txt', 'README.md')
MODEL_FILES = ('config.json', 'preprocessor_config.json', 'tokenizer_config.json',
               'tokenizer.json', 'chat_template.json', 'generation_config.json',
               'model.safetensors.index.json')


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path = Path(path)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('w') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def read_jsonl(path):
    if not path.exists():
        return []
    raw = path.read_text()
    lines = raw.splitlines()
    result = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not raw.endswith('\n'):
                # An interrupted final write is never counted as a result.
                break
            raise
    return result


def append(path, value):
    with path.open('a') as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def worker_env():
    env = os.environ.copy()
    for key in ('MIN_PIXELS', 'MAX_PIXELS', 'MODEL_SEQ_LEN', 'WORLD_SIZE', 'RANK',
                'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT',
                'PYTHONPATH', 'QWEN_EXACT_RECENT_DECODE', 'ACCELERATE_USE_CPU',
                'ACCELERATE_MIXED_PRECISION', 'FORCE_QWENVL_VIDEO_READER'):
        env.pop(key, None)
    env.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MALLOC_ARENA_MAX='2',
               TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
    return env


def check_pins(output):
    protocol = json.loads((output / 'protocol.json').read_text())
    pins = {**protocol['code_sha256'], **protocol['upstream_sha256'],
            **protocol['model_files_sha256'], **protocol['input_sha256']}
    for name, expected in pins.items():
        if sha(name) != expected:
            raise RuntimeError(f'Pinned input/source changed: {name}')
    return protocol


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / 'protocol.json'
    if protocol_path.exists():
        return check_pins(output)
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=UPSTREAM, text=True).strip() != COMMIT:
        raise RuntimeError('Upstream commit changed')
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=UPSTREAM, text=True).strip():
        raise RuntimeError('Upstream tracked files were modified')
    annotations = json.loads(ANNOTATION.read_text())
    groups = [[], []]
    costs = [0, 0]
    ordered = list(annotations)
    random.Random(42).shuffle(ordered)
    ordered.sort(key=lambda row: len(row.get('test_info', [None])), reverse=True)
    for row in ordered:
        rank = min(range(2), key=lambda n: costs[n])
        groups[rank].append(row)
        costs[rank] += len(row.get('test_info', [None]))
    inputs = {str(ANNOTATION): sha(ANNOTATION)}
    for rank, rows in enumerate(groups):
        path = output / f'annotations_shard{rank}.json'
        rows.sort(key=lambda a: a['id'])
        write(path, rows)
        inputs[str(path)] = sha(path)
    if len(annotations) != 1640 or sum(costs) != 3035:
        raise ValueError('Unexpected official dataset size')
    for directory, names in ((UPSTREAM, UPSTREAM_FILES), (ROOT, (
            'sparsemm_native/supervise_simplestream_upstream.py',
            'sparsemm_native/download_ovobench_chunks.py', 'sparsemm_native/gpu_slots.py'))):
        for name in names:
            target = output / 'source_snapshot' / directory.name / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((directory / name).read_bytes())
    protocol = dict(
        version='simplestream_released_qwen25_original_settings_v1', created_at=now(),
        upstream_url='https://github.com/EvolvingLMMs-Lab/SimpleStream', upstream_commit=COMMIT,
        model_path=str(MODEL), model='Qwen2.5-VL-7B-Instruct', dtype='bfloat16',
        fps=1.0, chunk_duration=1.0, recent_frames_only=4,
        window_semantics='last_four_nonempty_one_second_chunks_as_released',
        source_videos='official_byte_identical_pre_cut_chunked_videos',
        hf_revision='fec29e3385747b5642d995370143ba92d2819bd2',
        image_processing='qwen_vl_utils0.0.14_fetch_video_dynamic_resize_then_original_AutoProcessor',
        external_min_pixels=None, external_max_pixels=None, fixed_token_budget=None,
        attention_implementation='flash_attention_2', max_new_tokens=256,
        do_sample=False, system_prompt=None, repetition_penalty_override=None,
        prompt_builder='unmodified_lib.recent_window_eval_qwen25.build_qwen25_prompt',
        generation='unmodified_released_Qwen25_continuous_vision_prefix_and_model.generate',
        primary_scorer='unmodified_scoring/score_ovo_bench.py',
        secondary_scorer='released_console_calculate_ovo_scores_separately_labeled',
        decoding_optimization='released_QWEN_EXACT_RECENT_DECODE_only_if_full_plan_last_four_chunks_equal_tail_four_frames; otherwise_default_full_decode',
        lm_kv='fresh_generate_every_question_no_historical_or_cross_question_KV',
        cross_question_vision_reuse=False,
        execution_only_changes=['independent_single_GPU_processes_on_two_balanced_annotation_shards',
                                'detached_supervision_checkpoint_retries_and_observations'],
        dependency_pins={'transformers': '4.52.4', 'accelerate': '1.8.1', 'qwen-vl-utils': '0.0.14'},
        query_count=3035, annotation_count=1640, shard_query_counts=costs,
        upstream_sha256={str(UPSTREAM / n): sha(UPSTREAM / n) for n in UPSTREAM_FILES},
        code_sha256={str(ROOT / n): sha(ROOT / n) for n in (
            'sparsemm_native/supervise_simplestream_upstream.py',
            'sparsemm_native/download_ovobench_chunks.py', 'sparsemm_native/gpu_slots.py')},
        model_files_sha256={str(MODEL / n): sha(MODEL / n) for n in MODEL_FILES},
        input_sha256=inputs,
        comparison_caveat='Original protocol differs from previous FPS0.5 controlled comparison in prompts, source clip bytes, resize, dependencies and generation as well as FPS; changes cannot be attributed to FPS alone.',
    )
    write(protocol_path, protocol)
    return protocol


def full_window_indices(plan, seconds=1.0, count=4):
    buckets = collections.OrderedDict()
    for index in plan.full_indices:
        bucket = int((float(index) / plan.raw_fps) // seconds)
        buckets.setdefault(bucket, []).append(index)
    selected = list(buckets.items())[-count:]
    return [index for _, values in selected for index in values], [key for key, _ in selected]


def install_observer(output):
    sys.path.insert(0, str(UPSTREAM))
    import torch
    from lib import recent_window_eval_qwen25 as upstream
    from lib.qwen_exact_recent_decoder import _build_sampling_plan
    original = upstream.query_recent_window

    def observe(*args, **kwargs):
        bound = __import__('inspect').signature(original).bind(*args, **kwargs)
        bound.apply_defaults()
        request = bound.arguments
        qa = request['qa']
        video = request['video_path']
        plan = _build_sampling_plan({'video': video, 'fps': request['fps']},
                                    last_nframes=request['recent_frames_only'])
        indices, chunks = full_window_indices(plan, request['chunk_duration'], request['recent_frames_only'])
        exact = indices == plan.tail_indices
        os.environ['QWEN_EXACT_RECENT_DECODE'] = '1' if exact else '0'
        trace = {}

        def before(module, values, model_kwargs):
            if 'initial_cache_tokens' in trace:
                return
            cache = model_kwargs.get('past_key_values')
            length = 0 if cache is None else int(cache.get_seq_length())
            trace['initial_cache_tokens'] = length
            embeds = model_kwargs.get('inputs_embeds')
            trace['prefill_input_tokens'] = int(embeds.shape[1]) if embeds is not None else None
            if length:
                raise RuntimeError('Unexpected historical language KV in upstream generation')

        hook = qa.model.register_forward_pre_hook(before, with_kwargs=True)
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        try:
            result, backend = original(*args, **kwargs)
        finally:
            hook.remove()
        if result.final_chunk_ids != chunks or result.num_frames != len(indices):
            raise RuntimeError('Original sampled window differs from audited full sampling plan')
        if trace.get('initial_cache_tokens') != 0:
            raise RuntimeError('Fresh LM prefill was not observed')
        append(output / 'query_observations.jsonl', dict(
            query_id=Path(video).stem, recorded_at=now(),
            source_clip=str(video), prompt_sha256=hashlib.sha256(request['prompt'].encode()).hexdigest(),
            prompt=request['prompt'], full_sampled_frames=plan.sampled_nframes_full,
            raw_fps=plan.raw_fps, raw_total_frames=plan.raw_total_frames,
            selected_raw_frame_indices=indices, selected_chunk_ids=chunks,
            exact_decoder=exact, decoder_backend=backend, result=dataclasses.asdict(result),
            total_query_seconds=time.perf_counter() - started,
            peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
            peak_reserved_bytes=int(torch.cuda.max_memory_reserved()), **trace))
        return result, backend

    upstream.query_recent_window = observe


def worker(args):
    from importlib.metadata import version
    from sparsemm_native.gpu_slots import gpu_lease
    output = Path(args.output)
    if args.check_pins:
        check_pins(output.parent.parent)
    for name, expected in {'transformers': '4.52.4', 'accelerate': '1.8.1', 'qwen-vl-utils': '0.0.14'}.items():
        if version(name) != expected:
            raise RuntimeError(f'Wrong original dependency: {name}={version(name)}')
    output.mkdir(parents=True, exist_ok=True)
    for filename in ('results_incremental.jsonl', 'query_observations.jsonl'):
        path = output / filename
        rows = read_jsonl(path)
        if path.exists() and path.stat().st_size and not path.read_bytes().endswith(b'\n'):
            temp = path.with_suffix('.recovered')
            temp.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
            os.replace(temp, path)
    # A waiting worker holds no model memory. It can wait for another
    # authorized experiment to release the GPU instead of dropping this run.
    lease_context = None
    while lease_context is None:
        candidate = gpu_lease(output, concurrent_processes=2)
        try:
            lease = candidate.__enter__()
            lease_context = candidate
        except RuntimeError as exc:
            if not any(term in str(exc) for term in ('busy', 'MiB free')):
                raise
            write(output / 'waiting_for_gpu.json', dict(updated_at=now(), reason=str(exc)))
            time.sleep(15)
    try:
        write(output / 'worker.json', dict(pid=os.getpid(), session_id=os.getsid(0), started_at=now(),
              lease=lease, dependencies={name: version(name) for name in (
                  'torch', 'torchvision', 'transformers', 'accelerate', 'qwen-vl-utils', 'flash-attn', 'Pillow', 'numpy', 'av', 'decord')}))
        install_observer(output)
        sys.argv = ['eval_qwen25vl_ovo.py', '--model_path', str(MODEL),
                    '--anno_path', args.annotation, '--chunked_dir', args.clips,
                    '--result_dir', str(output), '--recent_frames_only', '4',
                    '--chunk_duration', '1.0', '--fps', '1.0', '--max_qa_tokens', '256']
        runpy.run_path(str(UPSTREAM / 'main_experiments/eval_qwen25vl_ovo.py'), run_name='__main__')
        if args.check_pins:
            check_pins(output.parent.parent)
    finally:
        lease_context.__exit__(*sys.exc_info())


def collect(output):
    results = {'backward': [], 'realtime': [], 'forward': []}
    seen = set()
    for rank in range(2):
        for row in read_jsonl(output / f'shards/shard{rank}/results_incremental.jsonl'):
            key = (row['task'], row['id'])
            if key in seen:
                raise RuntimeError(f'Duplicate annotation result: {key}')
            seen.add(key)
            group = 'forward' if row['task'] in ('REC', 'SSR', 'CRR') else (
                    'backward' if row['task'] in ('EPM', 'ASI', 'HLD') else 'realtime')
            results[group].append(row)
    return results


def report(output, state):
    results = collect(output)
    completed = sum(len(v) for k, v in results.items() if k != 'forward') + sum(
        len(row['test_info']) for row in results['forward'])
    state.update(updated_at=now(), completed_queries=completed,
                 completed_annotations=sum(len(v) for v in results.values()), total_queries=3035)
    write(output / 'status.json', state)
    content = (
        f"状态：{state['state']}；更新时间：{state['updated_at']}。\n\n"
        f"SimpleStream 原配置：Qwen2.5-VL-7B、fps=1、最近4个1秒窗口、256输出token、"
        f"默认动态分辨率、原提示词和仓库评分脚本。\n\n"
        f"全量进度：{completed}/3035 题；{state['completed_annotations']}/1640 条标注。\n\n"
        f"数据下载状态见 {DATA / 'status.json'}。\n")
    if 'download' in state and state['state'] == 'waiting_for_official_clips':
        download = state['download']
        done = download.get('downloaded_bytes') or 0
        total = download.get('total_bytes') or 1
        content += f"\n官方预切片下载：{done / 1e9:.2f}/{total / 1e9:.2f} GB。\n"
    if state['state'] == 'complete':
        summary = state['summary']
        content += (f"\n官方总分：{summary['official_total_average']:.2%}；"
                    f"逐题正确率：{summary['question_accuracy']:.2%}"
                    f"（{summary['correct']}/3035）。\n\n"
                    '| 题型 | 答对/总题数 | 正确率 |\n|---|---:|---:|\n')
        for task, row in sorted(summary['tasks'].items()):
            content += f"| {task} | {row['correct']}/{row['total']} | {row['accuracy']:.2%} |\n"
        content += f"\n实际视觉 token：{summary['visual_tokens']}。\n"
    content += ('\n这次使用原版协议；提示词、切片、分辨率、依赖和生成参数也与之前的 '
                'fps=0.5 对照不同，分数变化不能只归因于 FPS。\n')
    (output / 'progress.md').write_text(content)
    return results, completed


def finish(output, results):
    check_pins(output)
    expected = {row['id']: row for row in json.loads(ANNOTATION.read_text())}
    if sum(len(v) for v in results.values()) != len(expected):
        raise RuntimeError('Incomplete original annotation results')
    spec = importlib.util.spec_from_file_location('upstream_official_ovo_score', UPSTREAM / 'scoring/score_ovo_bench.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scorer = module.OVOBenchOfflineScore(argparse.Namespace(model='Qwen2.5-VL', mode='offline'), results)
    task_scores = {}
    for group in ('backward', 'realtime'):
        _, scores = scorer.calculate_score_backward_realtime(results[group])
        task_scores.update(scores)
    _, scores = scorer.calculate_score_forward(results['forward'])
    task_scores.update(scores)
    group_tasks = {'backward': ('EPM', 'ASI', 'HLD'),
                   'realtime': ('OCR', 'ACR', 'ATR', 'STU', 'FPD', 'OJR'),
                   'forward': ('REC', 'SSR', 'CRR')}
    rows = []
    for values in results.values():
        for row in values:
            annotation = expected[row['id']]
            if row['task'] != annotation['task'] or row['video'] != annotation['video']:
                raise RuntimeError('Result identity does not match source annotation')
            if row['task'] in ('REC', 'SSR', 'CRR'):
                if len(row['test_info']) != len(annotation['test_info']):
                    raise RuntimeError('Forward probe count changed')
                responses = [(f"{row['id']}_{i}", item.get('response')) for i, item in enumerate(row['test_info'])]
            else:
                responses = [(str(row['id']), row.get('response'))]
            if any(not isinstance(response, str) or not response.strip() for _, response in responses):
                raise RuntimeError('Missing/empty model responses must be resolved, not scored as valid queries')
            rows.extend((qid, row['task'], response) for qid, response in responses)
    observations = {}
    for rank in range(2):
        for record in read_jsonl(output / f'shards/shard{rank}/query_observations.jsonl'):
            observations[record['query_id']] = record
    if len(rows) != 3035 or len(observations) != 3035 or set(observations) != {r[0] for r in rows}:
        raise RuntimeError('Incomplete/different per-query observations')
    if any(record['initial_cache_tokens'] != 0 for record in observations.values()):
        raise RuntimeError('Historical language KV detected')
    export = dict(config=json.loads((output / 'protocol.json').read_text()), **results)
    write(output / 'qwen25vl_results_full.json', export)
    with (output / 'official_scoring.log').open('w') as handle:
        subprocess.run([str(ENV_PYTHON), str(UPSTREAM / 'scoring/score_ovo_bench.py'),
                        '--model', 'Qwen2.5-VL', '--result_path', str(output / 'qwen25vl_results_full.json')],
                       cwd=UPSTREAM, env=worker_env(), stdout=handle, stderr=subprocess.STDOUT, check=True)
    group_accuracy = {group: sum(sum(task_scores[task]) / len(task_scores[task]) for task in tasks) / len(tasks)
                      for group, tasks in group_tasks.items()}
    budgets = [record['result']['num_vision_tokens'] for record in observations.values()]
    summary = dict(complete=True, total_queries=3035,
                   correct=sum(sum(v) for v in task_scores.values()),
                   question_accuracy=sum(sum(v) for v in task_scores.values()) / 3035,
                   official_total_average=sum(group_accuracy.values()) / 3,
                   group_accuracy=group_accuracy,
                   tasks={task: dict(correct=sum(vals), total=len(vals), accuracy=sum(vals) / len(vals))
                          for task, vals in task_scores.items()},
                   visual_tokens=dict(mean=sum(budgets) / len(budgets), minimum=min(budgets), maximum=max(budgets)))
    write(output / 'summary.json', summary)
    sys.path.insert(0, str(UPSTREAM))
    from lib.recent_window_eval import calculate_ovo_scores
    write(output / 'released_console_scoring_secondary.json', calculate_ovo_scores(
          results['backward'], results['realtime'], results['forward']))
    with (output / 'per_query.csv').open('w') as handle:
        writer = csv.writer(handle)
        writer.writerow(('query_id', 'task', 'response', 'visual_tokens', 'input_tokens', 'num_frames', 'query_seconds'))
        for qid, task, response in sorted(rows):
            record = observations[qid]
            writer.writerow((qid, task, response, record['result']['num_vision_tokens'],
                             record['prefill_input_tokens'], record['result']['num_frames'], record['total_query_seconds']))
    write(output / 'final_audit.json', dict(passed=True, completed_at=now(), unique_queries=len(observations),
          fresh_prefill_checks=len(observations), official_source_archive_checks=json.loads((DATA / 'completion.json').read_text()),
          original_upstream_tracked_files_unmodified=True))
    return summary


def alive_download():
    path = DATA / 'status.json'
    if not path.exists():
        return False
    state = json.loads(path.read_text())
    pid = state.get('pid')
    try:
        return b'sparsemm_native.download_ovobench_chunks' in Path(f'/proc/{pid}/cmdline').read_bytes()
    except OSError:
        return False


def supervise(args):
    output = Path(args.output)
    protocol = prepare(output)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    with (output / '.supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(state='waiting_for_official_clips', supervisor_pid=os.getpid(),
                     session_id=os.getsid(0), started_at=now(), total_queries=3035,
                     concurrent_processes=2, source_protocol=str(output / 'protocol.json'))
        retries = 0
        while not (DATA / 'completion.json').exists():
            if (output / 'PAUSE').exists():
                state['state'] = 'paused'
                report(output, state)
                return
            if not alive_download():
                if retries >= 5:
                    raise RuntimeError('Official clip download failed five times; see data download.log')
                DATA.mkdir(parents=True, exist_ok=True)
                with (DATA / 'download.log').open('ab') as handle:
                    subprocess.Popen([str(BASE_PYTHON), '-u', '-m', 'sparsemm_native.download_ovobench_chunks'],
                                     cwd=ROOT, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                                     start_new_session=True)
                retries += 1
            if (DATA / 'status.json').exists():
                downloaded = json.loads((DATA / 'status.json').read_text())
                state['download'] = {k: downloaded.get(k) for k in (
                    'status', 'downloaded_bytes', 'total_bytes', 'speed_bytes_per_second', 'updated_at')}
            report(output, state)
            time.sleep(10)
        check_pins(output)
        state.update(state='running', inference_started_at=now())
        children, attempts = {}, collections.Counter()
        while True:
            if (output / 'PAUSE').exists():
                for process in children.values():
                    if process.poll() is None:
                        process.terminate()
                state['state'] = 'paused'
                report(output, state)
                return
            for rank in range(2):
                path = output / f'shards/shard{rank}'
                path.mkdir(parents=True, exist_ok=True)
                if (path / 'done').exists():
                    continue
                current = children.get(rank)
                if current is not None and current.poll() is None:
                    continue
                if current is not None and current.returncode == 0:
                    raise RuntimeError('Upstream worker exited without completion marker')
                if attempts[rank] >= 3:
                    raise RuntimeError(f'Upstream shard {rank} failed three times; see worker.log')
                check_pins(output)
                with (path / 'worker.log').open('ab') as handle:
                    children[rank] = subprocess.Popen([
                        str(ENV_PYTHON), '-u', '-m', 'sparsemm_native.supervise_simplestream_upstream',
                        '--worker', '--check-pins', '--output', str(path),
                        '--annotation', str(output / f'annotations_shard{rank}.json'),
                        '--clips', str(DATA / 'chunked_videos')], cwd=ROOT, env=worker_env(),
                        stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                attempts[rank] += 1
            state['worker_pids'] = {str(rank): p.pid for rank, p in children.items() if p.poll() is None}
            state['worker_attempts'] = dict(attempts)
            results, completed = report(output, state)
            if all((output / f'shards/shard{rank}/done').exists() for rank in range(2)):
                for process in children.values():
                    if process.poll() is None:
                        process.wait(timeout=60)
                    if process.returncode != 0:
                        raise RuntimeError('Upstream worker final source/input validation failed')
                summary = finish(output, results)
                state.update(state='complete', finished_at=now(), summary=summary, worker_pids={})
                report(output, state)
                return
            time.sleep(10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--check-pins', action='store_true')
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT))
    parser.add_argument('--annotation', default=str(ANNOTATION))
    parser.add_argument('--clips', default=str(DATA / 'chunked_videos'))
    args = parser.parse_args()
    try:
        if args.worker:
            worker(args)
        else:
            supervise(args)
    except BaseException as exc:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        path = output / ('worker_error.json' if args.worker else 'error.json')
        write(path, dict(failed_at=now(), error=repr(exc)))
        if not args.worker and (output / 'status.json').exists():
            state = json.loads((output / 'status.json').read_text())
            state.update(state='error', error=repr(exc), updated_at=now())
            write(output / 'status.json', state)
        raise


if __name__ == '__main__':
    main()
