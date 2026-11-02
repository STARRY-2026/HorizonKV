"""Resumable fixed-896 OVO pilot of DSCache-style history on SimpleStream K500.

Run one arm per process.  Arms share the official nine-task manifest and the
same sampled RGB images; only the visual KV allocation differs.
"""

import argparse
import csv
import dataclasses
import hashlib
import json
import os
import re
from pathlib import Path
import statistics
import subprocess
import sys
import time

from sparsemm_native import run_ovobench as official
from sparsemm_native import run_simplestream_k500_fixed896 as fixed
from sparsemm_native import run_simplestream_k500_input_only as old
from sparsemm_native import simplestream_dscache_history64 as cache_method
from sparsemm_native import vision_embedding_disk_cache


ROOT = old.ROOT
SOURCES = (
    ROOT / 'sparsemm_native/run_simplestream_dscache_history64.py',
    ROOT / 'sparsemm_native/simplestream_dscache_history64.py',
    ROOT / 'sparsemm_native/dscache_position_agnostic.py',
    ROOT / 'sparsemm_native/dscache_pae_history.py',
    ROOT / 'sparsemm_native/persistent_vision_cache.py',
    ROOT / 'sparsemm_native/vision_embedding_disk_cache.py',
    ROOT / 'sparsemm_native/run_simplestream_k500_fixed896.py',
    ROOT / 'sparsemm_native/simplestream_recent_share_pseudo_budget500.py',
    ROOT / 'sparsemm_native/simplestream_recent_share_attention.py',
    ROOT / 'kvbench/fixed_inputs.py',
    old.RAGGED,
)


def load_baseline(path, manifest):
    """Require the matching completed recent-only run as an explicit input."""
    if not path.is_file():
        raise FileNotFoundError(f'Baseline queries.jsonl is missing: {path}')
    rows = old.read_rows(path)
    baseline = {str(row['query_id']): row for row in rows}
    if len(baseline) != len(rows):
        raise ValueError('Baseline contains duplicate query IDs')
    for query in manifest:
        row = baseline.get(query['query_id'])
        if row is None or not all(key in row.get('input_audit', {}) for key in (
                'selected_raw_frame_indices', 'selected_chunk_ids', 'prompt_sha256')):
            raise ValueError(f"Missing completed fixed-896 baseline row for {query['query_id']}")
    return baseline


def check_upstream_pin():
    if not old.base.UPSTREAM.is_dir():
        raise FileNotFoundError(f'SimpleStream checkout is missing: {old.base.UPSTREAM}')
    commit = subprocess.check_output(
        ['git', '-C', str(old.base.UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != old.base.COMMIT:
        raise ValueError(f'SimpleStream must be checked out at {old.base.COMMIT}; got {commit}')


def snapshot_relative(path):
    """Keep external allocation tables and SimpleStream snapshots under output."""
    try:
        return Path('kvbench') / path.relative_to(ROOT)
    except ValueError:
        try:
            return Path('SimpleStream') / path.relative_to(old.base.UPSTREAM)
        except ValueError:
            return Path('external_allocation') / (old.sha(path)[:12] + '_' + path.name)


def summarize(output, rows, total, arm, allocation_map=None):
    result = old.summarize(output, rows, total, fixed.FPS, 'fixed896')
    configured_mean = sum(row['total_quota'] for row in (
        cache_method.allocation(arm) if allocation_map is None else allocation_map).values()) / 112
    result.update(version='simplestream_dscache_position_agnostic_history_v2', arm=arm,
                  configured_mean_visual_budget=configured_mean,
                  fixed_input=dict(canvas=896, tokens_per_frame=1024),
                  history_token_mean=statistics.mean(
                      row['budget_audit']['actual_history_visual_slots'] / 112 for row in rows)
                  if rows else None,
                  recent_token_mean=statistics.mean(
                      row['budget_audit']['actual_recent_visual_slots'] / 112 for row in rows)
                  if rows else None)
    old.base.write(output / 'summary.json', result)
    old.base.write(output / 'standalone_summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=cache_method.ARMS, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--baseline-queries', type=Path, required=True,
                        help='Completed fixed-896 recent-only queries.jsonl from the same manifest')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--allocation-csv', type=Path,
                        help='112-head offline table with recent_tokens and history_tokens')
    parser.add_argument('--capture-choice-probs', action='store_true',
                        help='Debug only: save conditional A-D first-token probabilities')
    parser.add_argument('--vision-cache-dir', type=Path,
                        help='Optional atomic tensor-only cache for fixed-896 visual embeddings')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.vision_cache_dir and os.environ.get('KV_BENCH_PERSISTENT_VISION_CACHE'):
        raise ValueError('Select either --vision-cache-dir or KV_BENCH_PERSISTENT_VISION_CACHE')
    custom_allocation = None
    allocation_path = args.allocation_csv.resolve() if args.allocation_csv else None
    if allocation_path is not None:
        if args.arm not in ('pae_history_k500', 'pae_compact_control',
                            'pae_online_margin_k500', 'pae_online_e_gate_k500',
                            'pae_online_e_recent_only_k500'):
            raise ValueError('Custom allocation requires a paired PAE history/control arm')
        source_rows = list(csv.DictReader(allocation_path.open(newline='')))
        if len(source_rows) != 112:
            raise ValueError('Expected 112 head rows')
        custom_allocation = {}
        for index, source in enumerate(source_rows):
            layer, head = divmod(index, 4)
            if (int(source['layer']), int(source['kv_head'])) != (layer, head):
                raise ValueError('Allocation head order differs from model head order')
            recent, history = int(source['recent_tokens']), int(source['history_tokens'])
            if not (32 <= recent <= 500 and 0 <= history <= 468 and recent + history == 500):
                raise ValueError('Every head must have 32..500 recent and total K500')
            custom_allocation[layer, head] = dict(
                recent_group=('one_frame' if (layer, head) in cache_method.HEADS
                              else 'four_frames'), recent_quota=recent,
                history_quota=history, total_quota=500)
    output = args.output_root.resolve()
    manifest_path = args.manifest.resolve()
    baseline_path = args.baseline_queries.resolve()
    check_upstream_pin()
    manifest, annotations = old.validate_manifest(manifest_path)
    baseline = load_baseline(baseline_path, manifest)
    output.mkdir(parents=True, exist_ok=True)
    progress = output / 'queries.jsonl'
    rows = old.read_rows(progress, recover=True)
    old.read_rows(output / 'ragged_query_audit.jsonl', recover=True)
    if len(rows) > len(manifest) or [r['query_id'] for r in rows] != [q['query_id'] for q in manifest[:len(rows)]]:
        raise ValueError('Existing results do not match the frozen manifest')
    active_allocation = (cache_method.allocation(args.arm) if custom_allocation is None
                         else custom_allocation)
    expected_total = sum(row['total_quota'] for row in active_allocation.values())
    for row, query in zip(rows, manifest):
        budget = row['budget_audit']
        if (row['task'] != query['task'] or row['target'] != query['target']
                or row['correct'] != official.official_correct(row['task'], row['response'], row['target'])
                or budget['actual_visual_slots'] > expected_total
                or budget['actual_visual_slots'] != budget['actual_recent_visual_slots']
                                                 + budget['actual_history_visual_slots']):
            raise ValueError('Existing identity, score, or budget audit is invalid')
    files = list(dict.fromkeys([*SOURCES, *(ROOT / name for name in old.LOCAL_SOURCES),
                                *(old.base.UPSTREAM / name for name in old.base.UPSTREAM_FILES),
                                *([allocation_path] if allocation_path else [])]))
    pin = dict(version='simplestream_dscache_position_agnostic_history_v2', arm=args.arm,
               fps=fixed.FPS, fixed_input=fixed.fixed_input_metadata(fixed.SIZE, fixed.PAD),
               recent_frames=4, visual_tokens_per_frame=1024,
               configured_visual_slots=expected_total, per_head_allocation={
                   f'{layer}:{head}': row for (layer, head), row in active_allocation.items()},
               history_update='all sampled older images; bounded raw-key one-image updates; pseudo-score prune after each',
               history_question_blind=True, recent_prefill='independent last-one/last-four image passes',
               capture_choice_probs=args.capture_choice_probs,
               persistent_vision_cache=os.environ.get('KV_BENCH_PERSISTENT_VISION_CACHE'),
               answer_policy=('two recent-only A-D/A-E question passes; report E selection and matched control'
                              if args.arm == 'pae_online_e_recent_only_k500' else
                              'two-pass recent-E-triggered A-D history answer gate'
                              if args.arm == 'pae_online_e_gate_k500' else
                              'two-pass same-recent-KV top-two-margin answer gate'
                              if args.arm == 'pae_online_margin_k500' else
                              'first-letter answer with recent-only fallback for out-of-range timeline or non-choice history first token'
                              if args.arm == 'strict_k500_stable' else
                              'first-letter A-D choice' if args.arm in ('pae_compact_control', 'pae_history_k500', 'pae_history_k532_diagnostic')
                              else 'original greedy generation'),
               prompt='released SimpleStream build_qwen25_prompt',
               positions=('bounded active history M-RoPE; common compact answer reindexing'
                          if args.arm in ('pae_compact_control', 'pae_history_k500', 'pae_history_k532_diagnostic', 'pae_online_margin_k500', 'pae_online_e_gate_k500', 'pae_online_e_recent_only_k500') else
                          'recent and question positions directly from upstream recent-only Qwen get_rope_index; historical positions translated before recent'),
               scoring='official OVO raw response', max_new_tokens=old.MAX_NEW,
               manifest_sha256=old.sha(manifest_path), query_count=len(manifest),
               baseline_sha256=old.sha(baseline_path),
               model=str(old.base.MODEL),
               code_sha256={str(path): old.sha(path) for path in files})
    cache_namespace = None
    if args.vision_cache_dir:
        cache_identity = dict(schema=1,
            model_files_sha256={name: old.sha(old.base.MODEL / name)
                                for name in old.base.MODEL_FILES},
            fixed_input=fixed.fixed_input_metadata(fixed.SIZE, fixed.PAD),
            fixed_encoder_sha256=old.sha(ROOT / 'sparsemm_native/run_simplestream_k500_fixed896.py'),
            upstream_encoder_sha256=old.sha(old.base.UPSTREAM / 'lib/recent_window_eval_qwen25.py'))
        cache_namespace = hashlib.sha256(
            json.dumps(cache_identity, sort_keys=True).encode()).hexdigest()
        pin['vision_embedding_cache'] = dict(root=str(args.vision_cache_dir.resolve()),
            namespace=cache_namespace, identity=cache_identity,
            format='torch tensor-only atomic v1')
    if allocation_path is not None:
        pin['allocation_csv'] = str(allocation_path)
        pin['allocation_csv_sha256'] = old.sha(allocation_path)
    protocol = output / 'protocol.json'
    if rows and (not protocol.exists() or json.loads(protocol.read_text()) != pin):
        raise ValueError('Pinned code or protocol changed; use a new output directory')
    old.base.write(protocol, pin)
    old.base.write(output / 'query_manifest.json', manifest)
    for path in files:
        dest = output / 'source_snapshot' / snapshot_relative(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
    summarize(output, rows, len(manifest), args.arm, custom_allocation)
    if args.prepare_only:
        print(json.dumps(dict(arm=args.arm, queries=len(manifest), done=len(rows), output=str(output))))
        return
    if len(rows) == len(manifest):
        old.base.write(output / 'status.json', dict(state='complete', done=len(rows), total=len(manifest)))
        return
    from importlib.metadata import version
    for name, expected in old.REQUIRED_VERSIONS.items():
        if version(name) != expected:
            raise RuntimeError(f'{name}={version(name)}, expected {expected}')
    for key in ('MIN_PIXELS', 'MAX_PIXELS', 'MODEL_SEQ_LEN', 'QWEN_EXACT_RECENT_DECODE'):
        os.environ.pop(key, None)
    sys.path.insert(0, str(old.base.UPSTREAM))
    import torch
    from lib import recent_window_eval_qwen25 as upstream
    from sparsemm_native.gpu_slots import gpu_lease
    cache_method.install(output, arm=args.arm, allocation_map=custom_allocation,
                         capture_choice_probs=args.capture_choice_probs)
    with gpu_lease(output) as admission:
        old.base.write(output / 'gpu_admission.json', admission)
        qa = upstream.RecentWindowQAModel(str(old.base.MODEL), device='cuda',
                                         max_new_tokens=old.MAX_NEW,
                                         attn_implementation='flash_attention_2')
        fixed.install_fixed_encoder(qa)
        if args.vision_cache_dir:
            vision_embedding_disk_cache.install(
                qa, cache_root=args.vision_cache_dir.resolve(), namespace=cache_namespace,
                stats_path=output / 'vision_cache_stats.json')
        old.base.write(output / 'runtime.json', dict(
            dependencies={name: version(name) for name in old.REQUIRED_VERSIONS},
            image_processor=qa.processor.image_processor.to_dict(),
            generation_config=qa.model.generation_config.to_dict()))
        audit_file = output / 'ragged_query_audit.jsonl'
        for query in manifest[len(rows):]:
            if (output / 'PAUSE').exists():
                old.base.write(output / 'status.json', dict(state='paused', done=len(rows), total=len(manifest)))
                return
            old.base.write(output / 'status.json', dict(state='running', done=len(rows),
                total=len(manifest), active_query=query['query_id'], pid=os.getpid()))
            prompt = upstream.build_qwen25_prompt(query['task'], annotations[query['query_id']])
            original_e_option = re.search(r'(?<!\w)E\.\s', prompt) is not None
            abstention_letter = 'F' if original_e_option else 'E'
            original_letters = 'ABCDE' if original_e_option else 'ABCD'
            offset = audit_file.stat().st_size if audit_file.exists() else 0
            clip = old.base.DATA / 'chunked_videos' / f"{query['source_id']}.mp4"
            before = time.perf_counter()
            result, backend = upstream.query_recent_window(
                qa=qa, video_path=str(clip), prompt=prompt,
                chunk_duration=1., fps=fixed.FPS, recent_frames_only=4)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - before
            with audit_file.open('rb') as handle:
                handle.seek(offset)
                audits = [json.loads(line) for line in handle.read().splitlines() if line.strip()]
            if len(audits) != 1:
                raise AssertionError('Expected exactly one physical-cache audit')
            audit = audits[0]
            if args.capture_choice_probs or args.arm in ('pae_online_margin_k500', 'pae_online_e_gate_k500', 'pae_online_e_recent_only_k500'):
                probabilities = audit.get('choice_probabilities')
                expected_letters = ((original_letters + abstention_letter)
                                    if args.arm in ('pae_online_e_gate_k500',
                                                    'pae_online_e_recent_only_k500')
                                    and audit.get('online_gate_source') == 'recent'
                                    else original_letters)
                if (not isinstance(probabilities, dict)
                        or set(probabilities) != set(expected_letters)
                        or abs(sum(probabilities.values()) - 1.0) > 1e-5
                        or audit.get('choice_prediction') != max(
                            probabilities, key=probabilities.get)
                        or abs(audit.get('choice_margin') -
                               (sorted(probabilities.values(), reverse=True)[0]
                                - sorted(probabilities.values(), reverse=True)[1])) > 1e-5):
                    raise AssertionError('Invalid four-choice probability audit')
            if args.arm in ('pae_online_margin_k500', 'pae_online_e_gate_k500',
                            'pae_online_e_recent_only_k500'):
                recent_probs = audit.get('online_recent_probabilities')
                history_probs = audit.get('online_history_probabilities')
                recent_letters = ((original_letters + abstention_letter)
                                  if args.arm in ('pae_online_e_gate_k500',
                                                  'pae_online_e_recent_only_k500')
                                  else original_letters)
                if (not isinstance(recent_probs, dict)
                        or not isinstance(history_probs, dict)
                        or set(recent_probs) != set(recent_letters)
                        or set(history_probs) != set(original_letters)
                        or any(abs(sum(probs.values()) - 1.0) > 1e-5
                               for probs in (recent_probs, history_probs))
                        or audit['online_recent_visual_slots'] !=
                           audit['actual_recent_visual_slots']
                        or audit['online_recent_visual_slots'] +
                           audit['actual_history_visual_slots'] != audit['actual_visual_slots']
                        or audit['online_recent_selection_sha256'] !=
                           audit['online_history_recent_selection_sha256']
                        or not audit['online_same_suffix_position']
                        or audit['online_gate_source'] !=
                           ('recent' if args.arm == 'pae_online_e_recent_only_k500'
                            else 'history' if (audit['online_recent_answer'] == abstention_letter
                                               if args.arm == 'pae_online_e_gate_k500'
                                               else audit['online_history_margin'] >
                                                    audit['online_recent_margin'])
                            else 'recent')
                        or (args.arm in ('pae_online_e_gate_k500',
                                         'pae_online_e_recent_only_k500')
                            and audit['online_recent_prompt_sha256'] == audit['prompt_sha256'])
                        or (args.arm == 'pae_online_e_recent_only_k500'
                            and (audit['actual_history_visual_slots'] != 0
                                 or audit['online_recent_control_answer'] !=
                                    audit['online_history_answer']))
                        or (args.arm in ('pae_online_e_gate_k500',
                                         'pae_online_e_recent_only_k500')
                            and audit['online_recent_abstention_letter'] != abstention_letter)
                        or result.answer != audit[f"online_{audit['online_gate_source']}_answer"]):
                    raise AssertionError('Online two-pass gate, positions or visual budget failed')
            counts = audit['frame_visual_tokens']
            prior = baseline[query['query_id']]
            if (not counts or any(count != 1024 for count in counts)
                    or audit['history_candidate_visual_tokens'] !=
                       1024 * len(audit['history_sample_indices'])
                    or audit['recent_sample_indices'] !=
                       prior['input_audit']['selected_raw_frame_indices']
                    or result.final_chunk_ids != prior['input_audit']['selected_chunk_ids']
                    or audit['prompt_sha256'] != prior['input_audit']['prompt_sha256']
                    or audit['actual_visual_slots'] > expected_total
                    or audit['actual_visual_slots'] != audit['actual_recent_visual_slots']
                                                    + audit['actual_history_visual_slots']
                    or audit['prompt_sha256'] != hashlib.sha256(prompt.encode()).hexdigest()
                    or not audit['history_recent_positions_separate']
                    or audit['recent_image_position'] !=
                       1 + len(qa.processor.tokenizer.encode('user\n', add_special_tokens=False)) + 1
                    or audit['recent_positions_source'] != 'upstream_recent_only_get_rope_index'
                    or audit['recent_probe_position'] !=
                       (1 + len(qa.processor.tokenizer.encode('user\n', add_special_tokens=False)) + 1
                        + sum(counts))
                    or (args.arm in ('strict_k500_stable', 'pae_compact_control', 'pae_history_k500', 'pae_history_k532_diagnostic', 'pae_online_margin_k500', 'pae_online_e_gate_k500')
                        and (result.answer not in original_letters or len(result.answer) != 1))):
                raise AssertionError('Fixed input, position, prompt or budget audit failed')
            budget = dict(audit, actual_tokens_per_head=audit['actual_visual_slots'] / 112,
                          question_text_used_for_selection=False)
            row = {key: query[key] for key in (
                'query_id', 'source_id', 'probe_id', 'task', 'group', 'timestamp',
                'clip_end', 'target')}
            row.update(response=result.answer,
                       choice_probabilities=audit.get('choice_probabilities'),
                       choice_margin=audit.get('choice_margin'),
                       choice_prediction=audit.get('choice_prediction'),
                       recent_answer=audit.get('online_recent_answer'),
                       abstention_letter=audit.get('online_recent_abstention_letter'),
                       recent_control_answer=audit.get('online_recent_control_answer'),
                       history_answer=audit.get('online_history_answer'),
                       recent_choice_probabilities=audit.get('online_recent_probabilities'),
                       history_choice_probabilities=audit.get('online_history_probabilities'),
                       recent_choice_margin=audit.get('online_recent_margin'),
                       history_choice_margin=audit.get('online_history_margin'),
                       gate_source=audit.get('online_gate_source'),
                       correct=official.official_correct(query['task'], result.answer, query['target']),
                       seconds=seconds, budget_audit=budget,
                       input_audit=dict(recent_sample_indices=audit['recent_sample_indices'],
                                        history_sample_indices=audit['history_sample_indices'],
                                        recent_visual_tokens=counts,
                                        fixed_input=fixed.fixed_input_metadata(fixed.SIZE, fixed.PAD),
                                        decode_backend=backend),
                       upstream_result=dataclasses.asdict(result))
            old.base.append(progress, row)
            rows.append(row)
            summarize(output, rows, len(manifest), args.arm, custom_allocation)
            print(f"{len(rows)}/{len(manifest)} {query['query_id']} {query['task']} "
                  f"answer={result.answer!r} correct={row['correct']} "
                  f"K={budget['actual_tokens_per_head']:.2f} sec={seconds:.1f}", flush=True)
    old.base.write(output / 'status.json', dict(state='complete', done=len(rows), total=len(manifest)))


if __name__ == '__main__':
    main()
