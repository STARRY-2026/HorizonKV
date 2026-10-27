"""Detached, resumable nine-task comparison with identical decoded recent frames.

The mild history arm and its no-history control use the same model, input clips,
fps, image processor, question prompts, M-RoPE timeline, and official scorer.
They run sequentially because the full-history arm can peak above 20 GiB.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from sparsemm_native import supervise_simplestream_upstream as base
from sparsemm_native.finalize_simplestream_pseudo_history250 import summarize
from sparsemm_native.simplestream_recent_share_pseudo_history_mild500 import ALLOCATION_SHA256

ROOT = Path('/home/yujun/program/kvbench')
BASELINE = ROOT / 'results/ovobench_simplestream_pseudo_budget500_20261025/full'
ANNOTATION = ROOT / 'results/ovobench_simplestream_pseudo_history250_20261025/full/annotation.json'
DEFAULT_OUTPUT = ROOT / 'results/ovobench_simplestream_mild500_pair_20261026/full'
SOURCES = ('sparsemm_native/simplestream_recent_share_pseudo_history_mild500.py',
           'sparsemm_native/run_simplestream_recent_pseudo_history_mild500.py',
           'sparsemm_native/supervise_simplestream_mild500_pair.py',
           'sparsemm_native/supervise_simplestream_upstream.py',
           'sparsemm_native/gpu_slots.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temp, path)


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    expected = json.loads(ANNOTATION.read_text())
    if len(expected) != 1468 or len({str(r['id']) for r in expected}) != 1468:
        raise RuntimeError('Nine-task annotation is incomplete')
    if sha(ANNOTATION) != sha(BASELINE / 'annotation.json'):
        raise RuntimeError('Completed baseline uses a different annotation set')
    protocol = dict(version='simplestream_mild_history500_paired_v1',
                    model=str(base.MODEL), fps=1.0, chunk_duration=1.0,
                    input_clips=str(base.DATA / 'chunked_videos'),
                    image_processing='exact_qwen_vl_utils0.0.14_sampling_and_dynamic_resize',
                    recent_window='last_four_sampled_frames',
                    history_window='all_earlier_sampled_frames',
                    same_video_reuse='LRU_of_bit_identical_resized_frame_buckets',
                    max_new_tokens=256, query_count=1468,
                    included_tasks=sorted({r['task'] for r in expected}),
                    prompt='released_SimpleStream_Qwen25_OVO',
                    scorer='released_OVOBench_offline',
                    paired_control='same_timeline_recent_only_500_per_head',
                    intended_history_allocation='mild_per_head_56000_total',
                    concurrent_gpu_processes=1,
                    annotation_sha256=sha(ANNOTATION),
                    allocation_sha256=ALLOCATION_SHA256,
                    model_config_sha256=sha(base.MODEL / 'config.json'),
                    code_sha256={name: sha(ROOT / name) for name in SOURCES})
    path = output / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise RuntimeError('Pinned protocol changed; refuse mixed-code resume')
    if not path.exists():
        atomic_json(path, protocol)
    return protocol


def audit_arm(directory, expected):
    rows = base.read_jsonl(directory / 'results_incremental.jsonl')
    if len(rows) != len(expected) or not (directory / 'done').exists():
        raise RuntimeError(f'Incomplete arm: {directory}')
    by_id = {str(row['id']): row for row in rows}
    if set(by_id) != expected:
        raise RuntimeError('Answer IDs do not match the pinned annotation')
    audits = {row['query_id']: row for row in base.read_jsonl(directory / 'ragged_query_audit.jsonl')}
    observations = {row['query_id']: row for row in base.read_jsonl(directory / 'query_observations.jsonl')}
    if set(audits) != expected or set(observations) != expected:
        raise RuntimeError('Incomplete per-query budget or frame audit')
    for query_id, audit in audits.items():
        if (audit['actual_visual_slots'] > 56000 or
                audit['actual_visual_slots'] != audit['actual_history_visual_slots']
                + audit['actual_recent_visual_slots'] or
                not audit['history_recent_positions_separate'] or
                not audit['pruned_before_question']):
            raise RuntimeError(f'Budget/position audit failed: {query_id}')
    return by_id, audits, observations


def score_pair(output, protocol):
    expected = {str(row['id']): row for row in json.loads(ANNOTATION.read_text())}
    method, method_audit, method_obs = audit_arm(output / 'mild_history', expected)
    control, control_audit, control_obs = audit_arm(output / 'recent_only_control', expected)
    for query_id in expected:
        a, b = method[query_id], control[query_id]
        x, y = method_audit[query_id], control_audit[query_id]
        ox, oy = method_obs[query_id], control_obs[query_id]
        if (a['task'] != b['task'] or a['ground_truth'] != b['ground_truth'] or
                a['question'] != b['question'] or
                x['prompt_sha256'] != y['prompt_sha256'] or
                x['recent_sample_indices'] != y['recent_sample_indices'] or
                x['recent_frames_sha256'] != y['recent_frames_sha256'] or
                ox['selected_raw_frame_indices'] != oy['selected_raw_frame_indices'] or
                a['num_vision_tokens'] != b['num_vision_tokens']):
            raise RuntimeError(f'Paired frame, prompt or question mismatch: {query_id}')
        if x['arm'] != 'mild_history' or y['arm'] != 'recent_only_same_timeline':
            raise RuntimeError(f'Wrong arm audit: {query_id}')
    spec = importlib.util.spec_from_file_location(
        'official_ovo_score', base.UPSTREAM / 'scoring/score_ovo_bench.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scorer = module.OVOBenchOfflineScore(
        argparse.Namespace(model='Qwen2.5-VL', mode='offline'), {})
    result = dict(version='simplestream_mild_history500_paired_v1',
                  completed_at=now(), query_count=len(expected),
                  method=summarize(list(method.values()), scorer),
                  same_timeline_recent_only_control=summarize(list(control.values()), scorer),
                  actual_visual_tokens_per_head_mean=sum(
                      row['actual_visual_slots'] for row in method_audit.values()) / (112 * len(expected)),
                  actual_history_tokens_per_head_mean=sum(
                      row['actual_history_visual_slots'] for row in method_audit.values()) / (112 * len(expected)),
                  actual_recent_tokens_per_head_mean=sum(
                      row['actual_recent_visual_slots'] for row in method_audit.values()) / (112 * len(expected)),
                  cache_hits_method=sum(row['vision_cache_hits'] for row in method_audit.values()),
                  cache_hits_control=sum(row['vision_cache_hits'] for row in control_audit.values()),
                  equality_checks=['question', 'prompt_sha256', 'raw_recent_indices',
                                   'resized_recent_pixel_sha256', 'recent_vision_token_count'],
                  protocol=protocol)
    atomic_json(output / 'summary_official.json', result)
    return result


def run(output):
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    protocol = prepare(output)
    expected = {str(row['id']) for row in json.loads(ANNOTATION.read_text())}
    status = dict(state='running', started_at=now(), supervisor_pid=os.getpid(),
                  total_queries=1468, arms={})
    for arm in ('mild_history', 'recent_only_control'):
        directory = output / arm
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / 'done').exists():
            audit_arm(directory, expected)
            status['arms'][arm] = dict(state='complete', completed_queries=1468)
            atomic_json(output / 'status.json', status)
            continue
        for attempt in range(1, 4):
            prepare(output)
            command = [str(base.ENV_PYTHON), '-u', '-m',
                       'sparsemm_native.run_simplestream_recent_pseudo_history_mild500',
                       '--annotation', str(ANNOTATION), '--output', str(directory)]
            if arm == 'recent_only_control':
                command.append('--recent-only-control')
            with (directory / 'worker.log').open('ab') as log:
                child = subprocess.Popen(command, cwd=ROOT, env=base.worker_env(),
                                         stdin=subprocess.DEVNULL, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
            while child.poll() is None:
                completed = len(base.read_jsonl(directory / 'results_incremental.jsonl'))
                status['arms'][arm] = dict(state='running', completed_queries=completed,
                                           attempt=attempt, worker_pid=child.pid)
                status['updated_at'] = now()
                atomic_json(output / 'status.json', status)
                time.sleep(15)
            if child.returncode == 0 and (directory / 'done').exists():
                audit_arm(directory, expected)
                status['arms'][arm] = dict(state='complete', completed_queries=1468)
                atomic_json(output / 'status.json', status)
                break
            status['arms'][arm] = dict(state='retrying', attempt=attempt,
                                       worker_exit_code=child.returncode,
                                       completed_queries=len(base.read_jsonl(
                                           directory / 'results_incremental.jsonl')))
            atomic_json(output / 'status.json', status)
            time.sleep(5)
        else:
            raise RuntimeError(f'{arm} failed three times; inspect {directory / "worker.log"}')
    result = score_pair(output, protocol)
    status.update(state='complete', completed_at=now(), summary=str(output / 'summary_official.json'),
                  method_accuracy=result['method']['question_accuracy'],
                  control_accuracy=result['same_timeline_recent_only_control']['question_accuracy'])
    atomic_json(output / 'status.json', status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        run(args.output.resolve())
    except BaseException as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output / 'error.json', dict(failed_at=now(), error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
