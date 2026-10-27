"""Validate exact sparse-tail control and benchmark against single-control full input."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from sparsemm_native import run_ovobench as official
from sparsemm_native.supervise_ovobench_dual import write_json, now


FPS = .5
MAX_NEW = 256


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    root, reference = args.output_root.resolve(), args.reference_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((reference / 'query_manifest.json').read_text())
    candidates = {q['query_id']: q for q in official.queries()}
    if len(manifest) != 27 or any(q != candidates[q['query_id']] for q in manifest):
        raise ValueError('The original 27-query manifest changed')
    old = {r['query_id']: r for r in
           (json.loads(line) for line in (reference / 'queries.jsonl').read_text().splitlines())}
    if set(old) != {q['query_id'] for q in manifest}:
        raise ValueError('Reference predictions do not cover the original 27 questions')
    from sparsemm_native.run_aligned_pseudo_pair import SOURCES as SHARED_SOURCES
    sources = list(SHARED_SOURCES) + ['sparsemm_native/recent_tail.py',
               'sparsemm_native/run_recent_tail27.py', 'sparsemm_native/report_recent_tail27.py']
    pin = dict(version='recent_tail_exact27_v1', fps=FPS, mean_visual_budget=500,
               video_max_pixels=None, reference_root=str(reference),
               reference_manifest_sha256=sha(reference / 'query_manifest.json'),
               reference_predictions_sha256=sha(reference / 'queries.jsonl'),
               query_ids=[q['query_id'] for q in manifest],
               model=str(official.MODEL.resolve()),
               model_config_sha256=sha(official.MODEL / 'config.json'),
               preprocessor_config_sha256=sha(official.MODEL / 'preprocessor_config.json'),
               max_new_tokens=MAX_NEW, question='same_official_prompt_and_old_wrapper',
               positions='original_source_ordinals_and_native_absolute_video_MRoPE',
               full_prefix_timing_control='one_recent_answer_no_history_language_processing',
               tail_timing='only_original_final_two_units_and_one_independent_last_frame',
               timing_excludes_model_load=True, warmup_queries_excluded=1,
               original_paired_run_is_not_the_speed_baseline=True,
               code_sha256={name: sha(official.ROOT / name) for name in sources})
    progress = root / 'queries.jsonl'
    rows = [json.loads(line) for line in progress.read_text().splitlines() if line.strip()] if progress.exists() else []
    if (root / 'protocol.json').exists() and rows and json.loads((root / 'protocol.json').read_text()) != pin:
        raise ValueError('Pinned benchmark protocol changed')
    if [r['query_id'] for r in rows] != pin['query_ids'][:len(rows)]:
        raise ValueError('Existing query order changed')
    write_json(root / 'protocol.json', pin)
    write_json(root / 'query_manifest.json', manifest)
    for name in sources:
        dest = root / 'source_snapshot' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((official.ROOT / name).read_bytes())
    if args.prepare_only:
        print(json.dumps(dict(queries=27, output=str(root))))
        return
    from sparsemm_native.recent_tail import load_model, run_recent_query, cuda_sync
    from sparsemm_native.gpu_slots import gpu_lease
    from sparsemm_protokv.config import StreamingConfig
    from sparsemm_native.report_recent_tail27 import make_report
    started = time.perf_counter()
    try:
        with gpu_lease(root) as admission:
            write_json(root / 'gpu_admission.json', admission)
            before = time.perf_counter()
            bridge = load_model(official.MODEL, StreamingConfig(method='sparsemm', budget=500,
                                budget_scope='visual', sparsemm_cache_policy='two_bank',
                                sparsemm_budget_path=str(official.DP_DUAL_BUDGET),
                                sparsemm_two_bank_gate='per_head_capacity'),
                                device='cuda', dtype='bfloat16',
                                sample_fps=FPS, video_max_pixels=None)
            cuda_sync(bridge.device)
            write_json(root / 'runtime.json', dict(model_load_seconds=time.perf_counter() - before,
                       default_video_size=dict(bridge.processor.video_processor.size),
                       extra_pixel_cap=None, greedy=True,
                       eos_token_ids=bridge.model.generation_config.eos_token_id,
                       repetition_penalty=bridge.model.generation_config.repetition_penalty))
            # Warm up the actual standalone kernels; exclude this duplicate
            # answer from the 27-query timings and official result counts.
            warmup = run_recent_query(bridge, manifest[0], 'tail_only', FPS, MAX_NEW)
            write_json(root / 'warmup.json', dict(query_id=manifest[0]['query_id'],
                       response=warmup['response'], seconds=warmup['seconds']))
            for query in manifest[len(rows):]:
                qid = query['query_id']
                result = {}
                order = ('tail_only', 'full_prefix') if int(qid) % 2 == 0 else ('full_prefix', 'tail_only')
                for mode in order:
                    write_json(root / 'status.json', dict(state='running', done=len(rows), total=27,
                               active_query=qid, active_mode=mode, updated_at=now(), pid=os.getpid()))
                    result[mode] = run_recent_query(bridge, query, mode, FPS, MAX_NEW)
                    print(f"{qid} {mode} response={result[mode]['response']!r} "
                          f"seconds={result[mode]['seconds']:.3f}", flush=True)
                tail, full = result['tail_only'], result['full_prefix']
                previous = old[qid]['arms']['recent_only']
                ta, fa = tail['tail_audit'], full['tail_audit']
                checks = dict(
                    frame_indices_equal=ta['tail_raw_indices'] == fa['tail_raw_indices'],
                    rgb_pixels_equal=ta['tail_rgb_sha256'] == fa['tail_rgb_sha256'],
                    positions_equal=all(ta[k] == fa[k] for k in
                        ('recent_positions_sha256', 'single_frame_positions_sha256', 'question_position')),
                    features_equal=all(ta[k] == fa[k] for k in
                        ('recent_features_sha256', 'single_frame_features_sha256')),
                    original_question_position_equal=ta['question_position'] == previous['budget_audit']['question_position'],
                    original_selection_equal=tail['budget_audit']['recent_selection_sha256'] == previous['budget_audit']['recent_selection_sha256'],
                    original_budget_equal=tail['budget_audit']['per_head_visual_slots'] == previous['budget_audit']['per_head_visual_slots'],
                    single_full_answer_equal=tail['response'] == full['response'],
                    original_answer_equal=tail['response'] == previous['response'])
                row = {key: query[key] for key in ('query_id', 'source_id', 'probe_id', 'task', 'group',
                       'timestamp', 'clip_end', 'target')}
                row.update(tail, reference_response=previous['response'],
                           reference_budget_audit=previous['budget_audit'],
                           full_prefix_seconds=full['seconds'], full_prefix_response=full['response'],
                           full_prefix_budget_audit=full['budget_audit'],
                           full_prefix_tail_audit=fa, full_prefix_timing=full['timing'],
                           benchmark_order=list(order), validation=checks)
                row['tail_audit'].update(checks)
                row['correct'] = official.official_correct(query['task'], tail['response'], query['target'])
                if not all(checks[k] for k in ('frame_indices_equal', 'rgb_pixels_equal', 'positions_equal',
                                             'original_question_position_equal', 'original_budget_equal')):
                    write_json(root / f'mismatch_{qid}.json', row)
                    raise AssertionError(f'Frame/position/budget mismatch for {qid}: {checks}')
                with progress.open('a') as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
                    handle.flush()
                    os.fsync(handle.fileno())
                rows.append(row)
                make_report(root, reference)
                print(f"{len(rows)}/27 {qid} consistent={checks['original_answer_equal']} "
                      f"tail={tail['seconds']:.2f}s full={full['seconds']:.2f}s", flush=True)
            make_report(root, reference)
        write_json(root / 'status.json', dict(state='complete', done=27, total=27, updated_at=now(),
                   elapsed_seconds=time.perf_counter() - started))
    except BaseException as exc:
        write_json(root / 'status.json', dict(state='failed', done=len(rows), total=27,
                   error=f'{type(exc).__name__}: {exc}', updated_at=now()))
        raise


if __name__ == '__main__':
    main()
