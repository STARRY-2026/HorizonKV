"""Controlled EPM probes of the paused recent-tail K500 experiment.

This entry point changes no production code. All high-resolution arms share
RGB, visual features, prefix, head groups, prompt and decoder. The native arm
must reproduce the saved full-run answer, selection and positions exactly.
The deliberately selected diagnostic set is not an unbiased benchmark sample.
"""

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from sparsemm_native import run_ovobench as official
from sparsemm_native.aligned_pseudo_pair import assemble, digest
from sparsemm_native.decoupled import prefill_recent
from sparsemm_native.epm_position_probe import position_variant
from sparsemm_native.gpu_slots import gpu_lease
from sparsemm_native.mild_pseudo_history_fps05 import ROWS
from sparsemm_native.recent_tail import (
    RecentTailQwen, sampling_plan, tensor_digest, cuda_sync,
)
from sparsemm_native.simplestream_recent_share_budget500 import _visual_indices
from sparsemm_native.simplestream_recent_share_pseudo_history_mild500 import (
    _pseudo_queries_and_keys, _ragged_helpers,
)
from sparsemm_native.simplestream_resize import resize_hwc_batch
from sparsemm_native.supervise_ovobench_dual import write_json, now
from sparsemm_native.video_reader import BatchScopedVideoReader
from sparsemm_protokv.config import StreamingConfig


ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / 'results/ovobench_recent_tail_only_default_full_20261026'
DEFAULT_OUTPUT = ROOT / 'results/ovobench_epm_current_diagnosis_20261026'


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def non_attention_cache(history, one, four, one_unit, policy):
    """Keep equal coverage of units, or all candidates, without question data."""
    Layer, Cache, _ = _ragged_helpers()
    prefix = history.prefix_tokens
    layers, indices, per_head = [], [], []
    candidates = total = 0
    for layer_idx in range(history.num_layers):
        keys, values, lengths, counts = [], [], [], []
        for head in range(history.num_kv_heads):
            one_group = ROWS[layer_idx, head]['recent_group'] == 'one_frame'
            branch = one if one_group else four
            units = [one_unit] if one_group else history.units
            rk, rv = branch[layer_idx]
            n = sum(unit.tokens for unit in units)
            counts_per_unit = [unit.tokens for unit in units]
            if policy == 'uniform':
                selected = _visual_indices(counts_per_unit, 500, rk.device) + prefix
            elif policy == 'uniform_protect32' and n > 500:
                counts_per_unit[-1] -= 32
                selected = torch.cat((
                    _visual_indices(counts_per_unit, 468, rk.device) + prefix,
                    torch.arange(prefix + n - 32, prefix + n, device=rk.device)))
            else:
                selected = torch.arange(prefix, prefix + n, device=rk.device)
            keep = torch.cat((torch.arange(prefix, device=rk.device), selected))
            keys.append(rk[0, head].index_select(0, keep))
            values.append(rv[0, head].index_select(0, keep))
            lengths.append(len(keep))
            counts.append(len(selected))
            indices.append(selected.tolist())
            total += len(selected)
            candidates += n
        layers.append(Layer(torch.cat(keys), torch.cat(values),
                      torch.tensor(lengths, device=keys[0].device, dtype=torch.int32)))
        per_head.append(counts)
    return Cache(layers), dict(actual_visual_slots=total, actual_recent_visual_slots=total,
        actual_history_visual_slots=0, per_head_visual_slots=per_head,
        recent_candidate_visual_slots=candidates, recent_selection_sha256=digest(indices),
        configured_visual_slots=56000 if policy in ('uniform', 'uniform_protect32') else None,
        selector=policy, question_text_used_for_selection=False)


def suffix(bridge, prompt):
    tokenizer = bridge.processor.tokenizer
    return ([tokenizer.convert_tokens_to_ids('<|vision_end|>')]
        + tokenizer.encode('\n', add_special_tokens=False)
        + tokenizer.encode(prompt, add_special_tokens=False)
        + [tokenizer.convert_tokens_to_ids('<|im_end|>')]
        + tokenizer.encode('\n', add_special_tokens=False)
        + [tokenizer.convert_tokens_to_ids('<|im_start|>')]
        + tokenizer.encode('assistant\n', add_special_tokens=False))


@torch.inference_mode()
def probe(bridge, query, reference, *, compact=False):
    history = bridge.kv_cache
    assert not bridge._prefills and history.history_units == 0
    one_unit = bridge._last_frame_unit(history)
    original_units = list(history.units)
    if not compact:
        previous = reference['tail_audit']
        assert tensor_digest(one_unit.features) == previous['single_frame_features_sha256']
        assert tensor_digest(torch.cat([u.features for u in original_units], dim=1)) == previous['recent_features_sha256']
        assert tensor_digest(one_unit.positions) == previous['single_frame_positions_sha256']
        assert tensor_digest(torch.cat([u.positions for u in original_units], dim=2)) == previous['recent_positions_sha256']
    frozen = {key: (value, tensor_digest(value)) for key, value in history.state_tensors().items()}
    question_suffix = suffix(bridge, query['prompt'])
    penalty_ids = (bridge.init_prompt_ids[0].tolist()
        + [int(bridge.model_config.image_token_id)] * sum(u.tokens for u in original_units)
        + question_suffix)
    out = {}
    layouts = ['native'] if compact else ['native', 'simplestream_contiguous']
    for layout in layouts:
        one_positioned, units, question_position = position_variant(
            bridge, one_unit, original_units, layout)
        before = time.perf_counter()
        one = prefill_recent(bridge.language_model, history.prefix_dense, [one_positioned])
        four = prefill_recent(bridge.language_model, history.prefix_dense, units)
        cuda_sync(bridge.device)
        prefill_seconds = time.perf_counter() - before
        probe_positions = dict(qpos=question_position)
        if not compact:
            probe_positions['token_end'] = history.prefix_tokens + sum(u.tokens for u in units)
        specs = [('pseudo_' + key, value) for key, value in probe_positions.items()]
        specs += [('uniform', None)]
        if not compact and layout == 'native':
            specs += [('uniform_protect32', None), ('unpruned', None)]
        uniform_selection = None
        for selector, probe_start in specs:
            before = time.perf_counter()
            if probe_start is not None:
                probes = _pseudo_queries_and_keys(bridge.model, bridge.processor.tokenizer, probe_start)
                ragged, audit = assemble(history, one, four, one_positioned, probes, 'recent_only')
                del probes
            else:
                ragged, audit = non_attention_cache(history, one, four, one_positioned, selector)
            name = ('compact_' if compact else '') + layout + '_' + selector
            response, decode_audit = bridge.decode_arm(
                ragged, question_suffix, question_position, penalty_ids, 256)
            del ragged
            cuda_sync(bridge.device)
            audit.update(decode_audit, question_position=question_position,
                pseudo_position=probe_start, selector=selector,
                actual_tokens_per_head=audit['actual_visual_slots'] / 112,
                common_frame_ordinals=[unit.frame_start for unit in original_units],
                prefill_seconds=prefill_seconds,
                one_positions_sha256=tensor_digest(one_positioned.positions),
                four_positions_sha256=tensor_digest(torch.cat([u.positions for u in units], dim=2)))
            out[name] = dict(response=response,
                correct=bool(official.official_correct(query['task'], response, query['target'])),
                seconds=time.perf_counter() - before, budget_audit=audit)
            if selector == 'uniform':
                uniform_selection = audit['recent_selection_sha256']
            if name == 'native_pseudo_qpos':
                expected = reference['budget_audit']
                checks = dict(answer=response == reference['response'],
                    selection=audit['recent_selection_sha256'] == expected['recent_selection_sha256'],
                    budget=audit['per_head_visual_slots'] == expected['per_head_visual_slots'],
                    position=question_position == expected['question_position'])
                out[name]['baseline_checks'] = checks
                if not all(checks.values()):
                    raise AssertionError(f'Baseline failed {query["query_id"]}: {checks}')
            print(query['query_id'], name, repr(response), out[name]['correct'], flush=True)
        del one, four
        if layout == 'native':
            native_uniform = uniform_selection
        elif native_uniform != uniform_selection:
            raise AssertionError('Position-only uniform arms changed selection indices')
    current = history.state_tensors()
    assert current.keys() == frozen.keys()
    assert all(current[key] is original and tensor_digest(current[key]) == before
               for key, (original, before) in frozen.items())
    return out, dict(single_frame_tokens=one_unit.tokens,
        four_branch_tokens=sum(u.tokens for u in original_units),
        one_features_sha256=tensor_digest(one_unit.features),
        four_features_sha256=tensor_digest(torch.cat([u.features for u in original_units], dim=1)),
        history_tokens=0, history_forwards=0, state_unchanged=True)


def summarize(output, rows):
    names = sorted({name for row in rows for name in row['arms']})
    arms = {}
    for name in names:
        selected = [r for r in rows if name in r['arms']]
        values = [r['arms'][name] for r in selected]
        baseline = [r['arms']['native_pseudo_qpos'] for r in selected]
        arms[name] = dict(questions=len(values), correct=sum(a['correct'] for a in values),
            accuracy=sum(a['correct'] for a in values) / len(values),
            fixed_wrong=sum(not b['correct'] and a['correct'] for a,b in zip(values, baseline)),
            broke_correct=sum(b['correct'] and not a['correct'] for a,b in zip(values, baseline)),
            changed_answer=sum(a['response'] != b['response'] for a,b in zip(values,baseline)),
            mean_tokens_per_head=sum(a['budget_audit']['actual_visual_slots']/112 for a in values)/len(values),
            fixed_ids=[r['query_id'] for r,a,b in zip(selected,values,baseline) if not b['correct'] and a['correct']],
            broke_ids=[r['query_id'] for r,a,b in zip(selected,values,baseline) if b['correct'] and not a['correct']])
    result = dict(questions=len(rows), biased_diagnostic_set=True,
        warning='Deliberate regression/control sample; accuracies are not benchmark estimates',
        scoring='official_correct on complete raw response', arms=arms)
    write_json(output/'summary.json',result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    output=args.output_root.resolve()
    assert (FULL/'PAUSE').exists(), 'Full run must remain paused'
    manifest=json.loads((output/'case_manifest.json').read_text())
    meta=json.loads((output/'selection_audit.json').read_text())
    if isinstance(meta.get('per_case'),list):
        metadata={r['query_id']:r for r in meta['per_case']}
    else:
        metadata=meta['per_case']
    references={r['query_id']:r for r in read_rows(FULL/'queries.jsonl')}
    rows=read_rows(output/'answers.jsonl')
    assert [r['query_id'] for r in rows] == [q['query_id'] for q in manifest[:len(rows)]]
    pin=dict(manifest_sha256=hashlib.sha256((output/'case_manifest.json').read_bytes()).hexdigest(),
        fps=.5, visual_budget=500, full_run_paused=True, changed_production_sources=False,
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in ['sparsemm_native/diagnose_recent_tail_epm.py','sparsemm_native/epm_position_probe.py']},
        variants='native/legacy positions x qpos/token_end pseudo probes; uniform; unpruned; compact native')
    protocol=output/'diagnosis_protocol.json'
    if rows:
        assert json.loads(protocol.read_text()) == pin, 'Diagnostic protocol changed'
    write_json(protocol,pin)
    if len(rows)==len(manifest):
        summarize(output,rows)
        return
    from sparsemm_protokv.model_loader import load_pretrained
    from sparsemm_native.assets import asset_metadata
    config=StreamingConfig(method='sparsemm',budget=500,budget_scope='visual',
        sparsemm_cache_policy='two_bank',sparsemm_budget_path=str(official.DP_DUAL_BUDGET),
        sparsemm_two_bank_gate='per_head_capacity')
    started=time.perf_counter()
    try:
        with gpu_lease(output):
            write_json(output/'status.json',dict(state='loading',done=len(rows),total=len(manifest),pid=os.getpid(),updated_at=now()))
            model,processor=load_pretrained(official.MODEL,device='cuda',dtype='bfloat16')
            bridge=RecentTailQwen(model,processor,config,asset_metadata=asset_metadata(),
                sample_fps=.5,video_max_pixels=None)
            for query in manifest[len(rows):]:
                qid=query['query_id']
                write_json(output/'status.json',dict(state='running',done=len(rows),total=len(manifest),active_query=qid,pid=os.getpid(),updated_at=now()))
                reader=BatchScopedVideoReader(query['video_path'],num_threads=1)
                plan=sampling_plan(len(reader),reader.get_avg_fps(),query['clip_end'],.5)
                rgb=torch.from_numpy(reader.get_batch(plan['tail_raw_indices']).asnumpy())
                bridge.encode_init_prompt()
                bridge.total_processed_frames=plan['start_ordinal']
                bridge.encode_video_chunk(rgb,is_last_chunk=True)
                arms,features=probe(bridge,query,references[qid])
                bridge.clear_cache()
                hw=metadata[qid]['old_resize_hw']
                assert len(hw)==2 and min(hw)>0
                compact_rgb=resize_hwc_batch(rgb,*hw)
                bridge.encode_init_prompt()
                bridge.total_processed_frames=plan['start_ordinal']
                bridge.encode_video_chunk(compact_rgb,is_last_chunk=True)
                compact_arms,compact_features=probe(bridge,query,references[qid],compact=True)
                assert compact_features['single_frame_tokens'] == metadata[qid]['old_one_frame_visual_tokens']
                arms.update(compact_arms)
                bridge.clear_cache()
                row=dict(query_id=qid,task=query['task'],target=query['target'],
                    reference_response=references[qid]['response'],old_response=metadata[qid]['old_response'],
                    tail_raw_indices=plan['tail_raw_indices'],tail_ordinals=plan['tail_ordinals'],
                    old_resize_hw=hw,features=features,compact_features=compact_features,arms=arms)
                with (output/'answers.jsonl').open('a') as stream:
                    stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                rows.append(row)
                summarize(output,rows)
                print(f'COMPLETED {len(rows)}/{len(manifest)} query={qid}',flush=True)
        write_json(output/'status.json',dict(state='complete',done=len(rows),total=len(manifest),updated_at=now(),elapsed_seconds=time.perf_counter()-started))
    except BaseException as exc:
        write_json(output/'status.json',dict(state='failed',done=len(rows),total=len(manifest),updated_at=now(),error=f'{type(exc).__name__}: {exc}'))
        raise


if __name__=='__main__':
    main()
