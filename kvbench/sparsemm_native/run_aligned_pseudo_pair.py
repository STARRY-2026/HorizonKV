"""Pinned small-sample paired OVO experiment; no full run by default."""

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import time

from sparsemm_native import run_ovobench as official
from sparsemm_native.run_ovobench_recent4 import digest, schedule_queries
from sparsemm_native.supervise_ovobench_dual import write_json, now


FPS = 0.5
MAX_NEW_TOKENS = 256
SOURCES = (
    'sparsemm_native/aligned_pseudo_pair.py', 'sparsemm_native/run_aligned_pseudo_pair.py',
    'sparsemm_native/report_aligned_pseudo_pair.py',
    'sparsemm_native/mild_pseudo_history_fps05.py', 'sparsemm_native/decoupled.py',
    'sparsemm_native/simplestream_recent_share_pseudo_budget500.py',
    'sparsemm_native/simplestream_recent_share_pseudo_history_mild500.py',
    'sparsemm_native/run_ovobench.py', 'sparsemm_baselines/adapter.py',
    'sparsemm_baselines/run_streamingbench.py', 'sparsemm_native/video_reader.py',
    'sparsemm_protokv/cache.py', 'sparsemm_protokv/streaming.py',
    'sparsemm_protokv/monkeypatch.py', 'hermes/head_analysis/hermes_kv_head_ragged.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_sample(ids=None):
    candidates = [q for q in official.queries() if q['task'] in official.BACKWARD | official.REALTIME]
    if ids:
        wanted = set(ids.split(','))
        selected = [q for q in candidates if q['query_id'] in wanted]
        if len(selected) != len(wanted):
            raise ValueError('Unknown or duplicate query IDs')
    else:
        tasks = collections.defaultdict(list)
        for query in candidates:
            tasks[query['task']].append(query)
        selected = []
        for task in sorted(tasks):
            ordered = sorted(tasks[task], key=lambda q: (q['clip_end'], int(q['source_id'])))
            selected += [ordered[round((len(ordered) - 1) * fraction)]
                         for fraction in (.25, .5, .75)]
        if len(selected) != 27 or len({q['query_id'] for q in selected}) != 27:
            raise AssertionError('Expected three length-stratified queries per task')
    return schedule_queries(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--ids', help='Optional comma-separated smoke IDs')
    parser.add_argument('--manifest', type=Path,
                        help='Freeze the exact reference query list, including prompts and order')
    parser.add_argument('--video-max-pixels', type=int, default=None,
                        help='Optional extra pixel cap; omit to use the KVBench default processor')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.video_max_pixels is not None and args.video_max_pixels < 3136:
        parser.error('--video-max-pixels must be at least 3136')
    if args.manifest and args.ids:
        parser.error('--manifest and --ids are mutually exclusive')
    if args.manifest:
        selected = json.loads(args.manifest.read_text())
        current = {q['query_id']: q for q in official.queries()}
        if (not selected or len({q['query_id'] for q in selected}) != len(selected)
                or any(q != current.get(q['query_id']) for q in selected)
                or any(q['task'] not in official.BACKWARD | official.REALTIME for q in selected)):
            raise ValueError('Reference manifest identities, prompts or annotations changed')
        if schedule_queries(selected) != selected:
            raise ValueError('Reference query order does not match the grouped schedule')
    else:
        selected = select_sample(args.ids)
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Importing torch is postponed until actual execution; manifest preparation
    # and reporting can run in the standard Python environment.
    allocation_path = (official.ROOT / 'results' /
                       'ovobench_simplestream_uniform_history_budget500_proposal_20261025' /
                       'per_head_allocation.csv')
    pin = dict(version='aligned_video_pseudo_pair_v2_optional_pixel_cap', fps=FPS,
               mean_visual_budget=500, configured_visual_slots=56000,
               video_max_pixels=args.video_max_pixels, model=str(official.MODEL.resolve()),
               model_config_sha256=sha(official.MODEL / 'config.json'),
               preprocessor_config_sha256=sha(official.MODEL / 'preprocessor_config.json'),
               generation_config_sha256=sha(official.MODEL / 'generation_config.json'),
               annotation_sha256=sha(official.ANNOTATIONS),
               allocation_sha256=sha(allocation_path), query_count=len(selected),
               query_ids=[q['query_id'] for q in selected], query_sha256=digest(selected),
               selection=('frozen_reference_manifest' if args.manifest else
                          'within_task_clip_length_quantiles_25_50_75_without_answers'
                          if not args.ids else 'explicit_smoke_ids'),
               reference_manifest_sha256=sha(args.manifest) if args.manifest else None,
               arms={'recent_only': '56 one / 56 four; recent 500, history 0',
                     'mild_history': '56 four recent500/history0; 56 one recent472/history6-96'},
               frontend='unchanged_source_prefix_fixed_step_native_Qwen_two_frame_video',
               chunk_size=16, resize_policy=None,
               position_scheme='shared_absolute_video_MRoPE_for_recent_history_and_question',
               wrapper='old_SimpleStream_user_vision_start_then_vision_end_official_prompt_assistant',
               system_prompt=None, question_text='unchanged_official_MCQA_template',
               selector='shared_fixed_pseudo_query_FP32_GQA_mean_avgpool5_protect32_recent_protect0_history',
               max_new_tokens=MAX_NEW_TOKENS, greedy=True,
               repetition_penalty='model_generation_config_prompt_and_generated_scope',
               eos='model_generation_config_full_EOS_set',
               decoder='released_old_physical_ragged_question_prefill_and_decode_step',
               scoring='official_OVOBenchOfflineScore_raw_response',
               paired_features_and_prefills_shared=True,
               code_sha256={name: sha(official.ROOT / name) for name in SOURCES})
    protocol = root / 'protocol.json'
    if (protocol.exists() and json.loads(protocol.read_text()) != pin
            and (root / 'queries.jsonl').exists()
            and (root / 'queries.jsonl').stat().st_size):
        raise ValueError('Pinned protocol changed; use a new output directory')
    write_json(protocol, pin)
    write_json(root / 'query_manifest.json', selected)
    for name in SOURCES:
        dest = root / 'source_snapshot' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((official.ROOT / name).read_bytes())
    if args.prepare_only:
        print(json.dumps(dict(output=str(root), query_ids=pin['query_ids'], count=len(selected))))
        return
    progress = root / 'queries.jsonl'
    rows = [json.loads(line) for line in progress.read_text().splitlines() if line.strip()] if progress.exists() else []
    if len(rows) > len(selected):
        raise ValueError('Too many existing rows')
    for row, expected in zip(rows, selected):
        if any(row[key] != expected[key] for key in ('query_id', 'task', 'target')):
            raise ValueError('Existing query identity changed')
    from sparsemm_native.aligned_pseudo_pair import load_model
    from sparsemm_native.gpu_slots import gpu_lease
    from sparsemm_native.report_aligned_pseudo_pair import make_report
    from sparsemm_protokv.config import StreamingConfig

    original_answer = official._answer_grouped
    def paired_answer(bridge, query, max_new_tokens, *unused):
        return bridge.answer_pair(query, max_new_tokens)
    official._answer_grouped = paired_answer
    started = time.monotonic()
    try:
        if len(rows) < len(selected):
            with gpu_lease(root) as admission:
                write_json(root / 'gpu_admission.json', admission)
                config = StreamingConfig(method='sparsemm', budget=500, budget_scope='visual',
                                         sparsemm_cache_policy='two_bank',
                                         sparsemm_budget_path=str(official.DP_DUAL_BUDGET),
                                         sparsemm_two_bank_gate='per_head_capacity')
                bridge = load_model(official.MODEL, config, sample_fps=FPS,
                                    device='cuda', dtype='bfloat16',
                                    video_max_pixels=args.video_max_pixels)
                write_json(root / 'runtime_processor.json', dict(
                    video_max_pixels=bridge.video_max_pixels,
                    size_override_passed_to_processor=bridge.video_max_pixels is not None,
                    default_video_size=dict(bridge.processor.video_processor.size),
                    default_image_size=dict(bridge.processor.image_processor.size),
                    patch_size=bridge.model_config.vision_config.patch_size,
                    temporal_patch_size=bridge.model_config.vision_config.temporal_patch_size,
                    spatial_merge_size=bridge.model_config.vision_config.spatial_merge_size))
                write_json(root / 'runtime_decode_config.json', dict(
                    eos_token_id=bridge.model.generation_config.eos_token_id,
                    repetition_penalty=bridge.model.generation_config.repetition_penalty,
                    max_new_tokens=MAX_NEW_TOKENS, greedy=True))
                for group in official.grouped_batches(selected[len(rows):], 100000):
                    answers = official.run_grouped_video(bridge, group, FPS, 16, MAX_NEW_TOKENS,
                                                        resize_policy=None)
                    for query in group:
                        paired, frames, seconds, encoded = answers[query['query_id']]
                        row = {key: query[key] for key in (
                            'query_id', 'source_id', 'probe_id', 'task', 'group',
                            'timestamp', 'clip_end', 'target')}
                        row.update(paired, frames=frames, seconds=seconds, encoded_frames=encoded,
                                   official_prompt_sha256=sha_text(query['prompt']))
                        for result in row['arms'].values():
                            result['correct'] = official.official_correct(
                                row['task'], result['response'], row['target'])
                        with progress.open('a') as handle:
                            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
                            handle.flush()
                            os.fsync(handle.fileno())
                        rows.append(row)
                        write_json(root / 'status.json', dict(state='running', done=len(rows),
                                   total=len(selected), updated_at=now(), pid=os.getpid()))
                        print(f"{len(rows)}/{len(selected)} {row['query_id']} {row['task']} "
                              + ' '.join(f"{arm}={result['response']!r}/{result['correct']} "
                                         f"K={result['budget_audit']['actual_visual_slots']/112:.1f}"
                                         for arm, result in row['arms'].items()), flush=True)
                    make_report(root)
        make_report(root)
        write_json(root / 'status.json', dict(state='complete', done=len(rows), total=len(selected),
                   updated_at=now(), elapsed_this_invocation_seconds=time.monotonic() - started))
    except BaseException as error:
        write_json(root / 'status.json', dict(state='failed', done=len(rows), total=len(selected),
                   error=f'{type(error).__name__}: {error}', updated_at=now()))
        raise
    finally:
        official._answer_grouped = original_answer


def sha_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


if __name__ == '__main__':
    main()
