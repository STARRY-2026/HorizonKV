"""Nine-task native OVO comparison: mild independent history at fps=0.5/K=500."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from sparsemm_native import run_ovobench as official
from sparsemm_native import run_ovobench_sparsemm_fps1 as manifest
from sparsemm_native.mild_pseudo_history_fps05 import (
    ALLOCATION_SHA256, ALLOCATION_TABLE, HISTORY, RECENT, TOTAL,
)
from sparsemm_native.run_ovobench_recent4 import digest, validate_row
from sparsemm_native.supervise_ovobench_dual import write_json

ROOT = official.ROOT
ARM = 'mild_pseudo_history_native_fps05_k500'
FPS = 0.5
PIXELS = 151200
SOURCES = ('sparsemm_native/run_ovobench_mild_pseudo_history_fps05.py',
           'sparsemm_native/mild_pseudo_history_fps05.py',
           'sparsemm_native/decoupled.py', 'sparsemm_native/run_ovobench.py',
           'sparsemm_native/run_ovobench_sparsemm_fps1.py',
           'sparsemm_native/simplestream_recent_share_pseudo_history_mild500.py',
           'sparsemm_native/supervise_ovobench_dual.py',
           'sparsemm_native/video_reader.py',
           'sparsemm_native/gpu_slots.py', 'sparsemm_protokv/streaming.py',
           'sparsemm_protokv/cache.py', 'sparsemm_protokv/monkeypatch.py',
           'sparsemm_protokv/model_loader.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol(selected, expected):
    from sparsemm_native.policy import DEFAULT_SCORE_PATH
    return dict(version='ovo_native_mild_pseudo_independent_fps05_v2', arm=ARM,
                model=str(official.MODEL.resolve()),
                model_config_sha256=sha(official.MODEL / 'config.json'),
                annotation=str(official.ANNOTATIONS.resolve()),
                annotation_sha256=sha(official.ANNOTATIONS),
                query_count=len(selected), full_nine_query_count=len(expected),
                query_ids=[q['query_id'] for q in selected],
                query_sha256=digest(selected), full_nine_query_sha256=digest(expected),
                fps=FPS, mean_visual_budget=500, configured_total_visual_slots=56000,
                allocation_path=str(ALLOCATION_TABLE), allocation_sha256=ALLOCATION_SHA256,
                per_head_total=TOTAL.tolist(), per_head_recent=RECENT.tolist(),
                per_head_history=HISTORY.tolist(),
                one_frame_head_count=56, four_frame_head_count=56,
                historical_input='all_prior_fps05_Qwen_two_frame_temporal_units',
                history_selector='fixed_question_blind_pseudo_QK_FP32_GQA_mean_avgpool5',
                recent_selector='fixed_pseudo_QK_protect32',
                recent_prefill='independent_last_single_sampled_frame_or_two_temporal_units',
                history_shortage='return_unused_history_slots_to_recent_same_head',
                model_input='same_native_Qwen_video_processor_as_existing_sparsemm_dp',
                video_mode='source_prefix_ceil_timestamp',
                video_max_pixels=PIXELS, chunk_size=16,
                max_new_tokens=16, greedy=True, repetition_penalty=1.1,
                score_mode='official_OVOBenchOfflineScore',
                grouped_by_source_video=True,
                same_video_prefix_reuse='even_frame_prefix_with_odd_endpoint_branch',
                sparsemm_score_path=str(DEFAULT_SCORE_PATH.resolve()),
                sparsemm_score_sha256=sha(DEFAULT_SCORE_PATH),
                code_sha256={name: sha(ROOT / name) for name in SOURCES})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--ids', help='Comma-separated official query IDs for a smoke run')
    parser.add_argument('--max-groups', type=int, default=100000)
    args = parser.parse_args(argv)
    expected, selected = manifest.selected_queries(args.ids)
    directory = args.output_root.expanduser().resolve() / ARM
    directory.mkdir(parents=True, exist_ok=True)
    pin = protocol(selected, expected)
    protocol_path = directory / 'protocol.json'
    progress = directory / 'queries.jsonl'
    rows = [json.loads(line) for line in progress.read_text().splitlines()
            if line.strip()] if progress.exists() else []
    if rows and not protocol_path.exists():
        raise ValueError('Existing predictions lack pinned protocol')
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != pin:
        raise ValueError('Pinned source, inputs or parameters changed')
    if len(rows) > len(selected):
        raise ValueError('Excess predictions')
    for row, query in zip(rows, selected):
        validate_row(row, query)
        audit = row.get('budget_audit')
        if audit is None or audit['actual_visual_slots'] > 56000:
            raise RuntimeError('Missing or over-budget per-query audit')
    write_json(protocol_path, pin)
    write_json(directory / 'query_manifest.json', selected)
    for name in SOURCES:
        snapshot = directory / 'source_snapshot' / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes((ROOT / name).read_bytes())
    if len(rows) == len(selected):
        return

    from sparsemm_native.gpu_slots import gpu_lease
    from sparsemm_native.mild_pseudo_history_fps05 import load_model
    from sparsemm_protokv.config import StreamingConfig
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    config = StreamingConfig(method='sparsemm', budget=500, budget_scope='visual',
                             sparsemm_cache_policy='two_bank',
                             sparsemm_budget_path=str(official.DP_DUAL_BUDGET),
                             sparsemm_two_bank_gate='per_head_capacity')
    original_answer = official._answer_grouped

    def tracked_answer(bridge, query, *a, **kw):
        bridge._active_query_id = query['query_id']
        return original_answer(bridge, query, *a, **kw)

    official._answer_grouped = tracked_answer
    try:
        with gpu_lease(directory, concurrent_processes=1) as admission:
            write_json(directory / 'gpu_admission.json', admission)
            bridge = load_model(official.MODEL, config, sample_fps=FPS,
                                trace_path=None, device='cuda', dtype='bfloat16',
                                video_max_pixels=PIXELS)
            if not bridge.budgets.equal(TOTAL) or not bridge.recent_budgets.equal(RECENT):
                raise RuntimeError('Runtime quotas differ from the pinned mild table')
            for group in official.grouped_batches(selected[len(rows):], args.max_groups):
                answers = official.run_grouped_video(
                    bridge, group, FPS, 16, 16, resize_policy=None)
                for query in group:
                    query_id = query['query_id']
                    response, frames, seconds, encoded = answers[query_id]
                    audit = bridge._query_budget_audits[query_id]
                    if (audit['actual_visual_slots'] > 56000 or
                            audit['actual_visual_slots'] != audit['actual_history_visual_slots']
                            + audit['actual_recent_visual_slots'] or
                            audit['history_recent_time_ranges_disjoint'] is not True):
                        raise RuntimeError(f'Per-query budget audit failed: {query_id}')
                    row = {key: query[key] for key in (
                        'query_id', 'source_id', 'probe_id', 'task', 'group',
                        'timestamp', 'clip_end', 'target')}
                    row.update(response=response,
                               correct=official.official_correct(
                                   query['task'], response, query['target']),
                               frames=frames, seconds=seconds, encoded_frames=encoded,
                               prompt_sha256=digest(query['prompt']),
                               budget_audit=audit)
                    validate_row(row, query)
                    with progress.open('a') as handle:
                        handle.write(json.dumps(row, ensure_ascii=False) + '\n')
                        handle.flush()
                        os.fsync(handle.fileno())
                    rows.append(row)
                    write_json(directory / 'summary.json', official.summary(rows, len(selected)))
                    print(f'{ARM} {len(rows)}/{len(selected)} {query_id} {query["task"]} '
                          f'correct={row["correct"]} frames={frames} '
                          f'actualK={audit["actual_visual_slots"] / 112:.1f} '
                          f'elapsed={seconds:.2f}s', flush=True)
    finally:
        official._answer_grouped = original_answer


if __name__ == '__main__':
    main()
