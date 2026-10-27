"""Run only the exact-tail recent control, with pinned/resumable official OVO input.

This entry point never runs the full-prefix benchmark or a historical language
forward. GPU/model imports are postponed until after ``--prepare-only``.
Launch with the same offline model environment as the paired experiment::

    python -m sparsemm_native.run_recent_tail_only --output-root RUN_DIR \
        --manifest REFERENCE_DIR/query_manifest.json \
        --reference-root REFERENCE_DIR
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sparsemm_native import run_ovobench as official  # noqa: E402
from sparsemm_native.run_aligned_pseudo_pair import SOURCES as SHARED_SOURCES  # noqa: E402
from sparsemm_native.supervise_ovobench_dual import write_json, now  # noqa: E402


FPS = .5
BUDGET = 500
MAX_NEW = 256
DEFAULT_REFERENCE = ROOT / 'results/ovobench_aligned_pseudo_pair_default_input_20261026/sample27'
ALLOCATION = (ROOT / 'results/ovobench_simplestream_uniform_history_budget500_proposal_20261025'
              / 'per_head_allocation.csv')
SOURCES = tuple(dict.fromkeys((*SHARED_SOURCES,
    'sparsemm_native/recent_tail.py', 'sparsemm_native/run_recent_tail_only.py',
    'sparsemm_native/report_recent_tail27.py', 'sparsemm_native/gpu_slots.py',
    'sparsemm_protokv/config.py', 'sparsemm_protokv/model_loader.py')))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return ([json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if path.exists() else [])


def validate_manifest(path):
    manifest = json.loads(path.read_text())
    official_queries = {q['query_id']: q for q in official.queries()}
    if (not isinstance(manifest, list) or not manifest
            or len({q['query_id'] for q in manifest}) != len(manifest)
            or any(q != official_queries.get(q['query_id']) for q in manifest)
            or any(q['task'] not in official.BACKWARD | official.REALTIME for q in manifest)):
        raise ValueError('Manifest must contain unique, unchanged official non-forward queries')
    return manifest


def standalone_summary(root, rows):
    """Keep the no-reference invocation useful without inventing a comparison."""
    from sparsemm_native.report_recent_tail27 import normalize_arm, time_stats, distribution
    from sparsemm_native.report_aligned_pseudo_pair import summarize_arm
    normalized = []
    for row in rows:
        normalized.append(dict(query_id=row['query_id'], task=row['task'], group=row['group'],
            target=row['target'], arms={'tail': normalize_arm(row['task'], row['target'],
                row['response'], row['budget_audit'], row['query_id'])}))
    result = dict(questions=len(rows), scope='nine backward/realtime tasks; no REC/SSR/CRR',
                  fps=FPS, configured_mean_visual_budget=BUDGET, video_max_pixels=None,
                  mode='tail_only', historical_language_forwards=False,
                  scoring='official_correct on raw response',
                  budget_unit='post-assembly visual KV slots / 112 layer-KV-heads')
    if rows:
        result.update(accuracy_and_budget=summarize_arm(normalized, 'tail'),
            query_timing_excluding_model_load=time_stats([row['seconds'] for row in rows]),
            single_frame_visual_tokens=distribution([
                row['tail_audit']['single_frame_visual_tokens'] for row in rows]))
    write_json(root / 'standalone_summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_REFERENCE / 'query_manifest.json',
                        help='Frozen official query list; defaults to the original 27 questions')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--reference-root', type=Path,
                        help='Optional paired-run directory for raw-answer and KV comparisons')
    parser.add_argument('--prepare-only', action='store_true',
                        help='Validate/pin sources and manifest without importing torch or using GPU')
    args = parser.parse_args()
    root, manifest_path = args.output_root.resolve(), args.manifest.resolve()
    reference = args.reference_root.resolve() if args.reference_root else None
    if root == reference:
        raise ValueError('Use a new output directory; reference results cannot be overwritten')
    manifest = validate_manifest(manifest_path)
    selected_ids = [q['query_id'] for q in manifest]
    old = {}
    if reference:
        old_rows = read_rows(reference / 'queries.jsonl')
        old = {row['query_id']: row for row in old_rows}
        if len(old) != len(old_rows) or not set(selected_ids) <= set(old):
            raise ValueError('Reference predictions must uniquely cover every selected query')
        for query in manifest:
            previous = old[query['query_id']]
            arm = previous.get('arms', {}).get('recent_only', previous)
            if (any(previous[key] != query[key] for key in ('task', 'target'))
                    or 'response' not in arm or 'budget_audit' not in arm):
                raise ValueError(f"Reference identity or recent-only arm changed: {query['query_id']}")
    pin = dict(version='recent_tail_only_official_v1', mode='tail_only', fps=FPS,
        mean_visual_budget=BUDGET, configured_visual_slots=56000,
        video_max_pixels=None, resolution='KVBench_default_video_processor_no_extra_cap',
        question='unchanged_official_MCQA_prompt_with_aligned_old_wrapper',
        positions='original_source_ordinals_and_native_absolute_video_MRoPE',
        head_groups='frozen_56_one_frame_56_four_frame',
        selector='fixed_pseudo_FP32_GQA_mean_avgpool5_recent_protect32',
        historical_visual_kv=0, historical_language_forwards=False,
        full_prefix_benchmark=False, greedy=True, max_new_tokens=MAX_NEW,
        model=str(official.MODEL.resolve()),
        model_config_sha256=sha(official.MODEL / 'config.json'),
        preprocessor_config_sha256=sha(official.MODEL / 'preprocessor_config.json'),
        generation_config_sha256=sha(official.MODEL / 'generation_config.json'),
        annotation_sha256=sha(official.ANNOTATIONS), allocation_sha256=sha(ALLOCATION),
        sparsemm_two_bank_config_path=str(official.DP_DUAL_BUDGET.resolve()),
        sparsemm_two_bank_config_sha256=sha(official.DP_DUAL_BUDGET),
        manifest_sha256=sha(manifest_path), query_count=len(manifest), query_ids=selected_ids,
        reference_root=str(reference) if reference else None,
        reference_predictions_sha256=sha(reference / 'queries.jsonl') if reference else None,
        timing_excludes_model_load=True, warmup_queries_excluded=1,
        code_sha256={name: sha(ROOT / name) for name in SOURCES})
    root.mkdir(parents=True, exist_ok=True)
    progress, protocol = root / 'queries.jsonl', root / 'protocol.json'
    rows = read_rows(progress)
    if rows and (not protocol.exists() or json.loads(protocol.read_text()) != pin):
        raise ValueError('Pinned standalone protocol changed; use a new output directory')
    if [row['query_id'] for row in rows] != selected_ids[:len(rows)]:
        raise ValueError('Existing query order or number differs from the frozen manifest')
    for row, query in zip(rows, manifest):
        if any(row[key] != query[key] for key in ('task', 'target')):
            raise ValueError('Existing task or target differs from the frozen manifest')
        if ('response' not in row or 'budget_audit' not in row
                or row.get('tail_audit', {}).get('mode') != 'tail_only'):
            raise ValueError('Existing row is not a complete standalone tail-only answer')
    write_json(protocol, pin)
    write_json(root / 'query_manifest.json', manifest)
    for name in SOURCES:
        destination = root / 'source_snapshot' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    if args.prepare_only:
        print(json.dumps(dict(output=str(root), queries=len(manifest), completed=len(rows),
                              fps=FPS, budget=BUDGET, video_max_pixels=None,
                              mode='tail_only', model_loaded=False)))
        return

    # Reporting and completed-run resume do not need a model or GPU lease.
    from sparsemm_native.report_recent_tail27 import make_report
    if len(rows) == len(manifest):
        standalone_summary(root, rows)
        if reference:
            make_report(root, reference)
        write_json(root / 'status.json', dict(state='complete', done=len(rows), total=len(manifest),
                   updated_at=now(), elapsed_this_invocation_seconds=0, resumed_complete=True))
        return
    from sparsemm_native.recent_tail import load_model, run_recent_query, cuda_sync
    from sparsemm_native.gpu_slots import gpu_lease
    from sparsemm_protokv.config import StreamingConfig
    started = time.perf_counter()
    try:
        with gpu_lease(root) as admission:
            write_json(root / 'gpu_admission.json', admission)
            write_json(root / 'status.json', dict(state='loading', done=len(rows), total=len(manifest),
                       pid=os.getpid(), updated_at=now()))
            before = time.perf_counter()
            config = StreamingConfig(method='sparsemm', budget=BUDGET, budget_scope='visual',
                sparsemm_cache_policy='two_bank', sparsemm_budget_path=str(official.DP_DUAL_BUDGET),
                sparsemm_two_bank_gate='per_head_capacity')
            bridge = load_model(official.MODEL, config, device='cuda', dtype='bfloat16',
                                sample_fps=FPS, video_max_pixels=None)
            cuda_sync(bridge.device)
            write_json(root / 'runtime.json', dict(model_load_seconds=time.perf_counter() - before,
                default_video_size=dict(bridge.processor.video_processor.size), extra_pixel_cap=None,
                greedy=True, eos_token_ids=bridge.model.generation_config.eos_token_id,
                repetition_penalty=bridge.model.generation_config.repetition_penalty))
            warmup_query = manifest[len(rows)]
            warmup = run_recent_query(bridge, warmup_query, 'tail_only', FPS, MAX_NEW)
            write_json(root / 'warmup.json', dict(query_id=warmup_query['query_id'],
                response=warmup['response'], seconds=warmup['seconds'], excluded_from_results=True))
            for query in manifest[len(rows):]:
                qid = query['query_id']
                write_json(root / 'status.json', dict(state='running', done=len(rows), total=len(manifest),
                           active_query=qid, active_mode='tail_only', updated_at=now(), pid=os.getpid()))
                result = run_recent_query(bridge, query, 'tail_only', FPS, MAX_NEW)
                audit, ta = result['budget_audit'], result['tail_audit']
                if (audit['actual_history_visual_slots'] != 0
                        or not ta['no_history_language_forwards'] or not ta['no_historical_visual_kv']):
                    raise AssertionError('Standalone recent control used historical KV or language forwards')
                row = {key: query[key] for key in ('query_id', 'source_id', 'probe_id', 'task',
                       'group', 'timestamp', 'clip_end', 'target')}
                row.update(result, correct=official.official_correct(query['task'], result['response'], query['target']))
                if reference:
                    previous_row = old[qid]
                    previous = previous_row.get('arms', {}).get('recent_only', previous_row)
                    previous_audit = previous['budget_audit']
                    checks = dict(original_answer_equal=result['response'] == previous['response'],
                        original_budget_equal=audit['per_head_visual_slots'] == previous_audit['per_head_visual_slots'],
                        original_selection_equal=audit['recent_selection_sha256'] == previous_audit['recent_selection_sha256'],
                        original_question_position_equal=audit['question_position'] == previous_audit['question_position'],
                        original_prompt_equal=audit['official_prompt_sha256'] == previous_audit['official_prompt_sha256'])
                    row.update(reference_response=previous['response'], reference_budget_audit=previous_audit,
                               validation=checks)
                    row['tail_audit'].update(checks)
                    if not all(checks[key] for key in ('original_budget_equal', 'original_question_position_equal',
                                                      'original_prompt_equal')):
                        write_json(root / f'mismatch_{qid}.json', row)
                        raise AssertionError(f'Reference frame-position/budget/prompt mismatch for {qid}')
                with progress.open('a') as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
                    handle.flush()
                    os.fsync(handle.fileno())
                rows.append(row)
                standalone_summary(root, rows)
                if reference:
                    make_report(root, reference)
                print(f"{len(rows)}/{len(manifest)} {qid} response={row['response']!r} "
                      f"correct={row['correct']} tail_seconds={row['seconds']:.3f}", flush=True)
        write_json(root / 'status.json', dict(state='complete', done=len(rows), total=len(manifest),
                   updated_at=now(), elapsed_this_invocation_seconds=time.perf_counter() - started))
    except BaseException as exc:
        write_json(root / 'status.json', dict(state='failed', done=len(rows), total=len(manifest),
                   error=f'{type(exc).__name__}: {exc}', updated_at=now(),
                   elapsed_this_invocation_seconds=time.perf_counter() - started))
        raise


if __name__ == '__main__':
    main()
