"""Old SimpleStream one/four-frame pseudo-K500 with shared fixed RGB inputs.

Only spatial frame preparation and the already-selected sampling FPS change.
The old image encoder, independent prefills, get_rope_index, question prompt,
pseudo-query selector and physical ragged decoder are reused unchanged.
Use runtime_envs/simplestream-qwen25/bin/python (transformers 4.52.4).
"""

import argparse
import dataclasses
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time

from kvbench.fixed_inputs import assert_qwen_grid, fixed_input_metadata
from sparsemm_native import run_ovobench as official
from sparsemm_native import run_simplestream_k500_input_only as old


ROOT = old.ROOT
FPS = .5
SIZE = 896
PAD = 127
BUDGET = old.BUDGET
MAX_NEW = old.MAX_NEW
CONFIG = ROOT / 'configs/ovobench_fixed896_all_methods.json'
VERSION = 'simplestream_one_four_pseudo_k500_fixed896_v1'


def install_fixed_encoder(qa):
    """Wrap pixels only; assert the actual original image-encoder output grids."""
    import numpy as np
    import torch
    from PIL import Image
    from kvbench.fixed_inputs import frame_rgb_sha256, prepare_fixed_frames

    original = qa.encode_vision
    qa.processor.image_processor.do_resize = False
    state = {'grids': [], 'frames': [], 'calls': []}

    def encode(frames):
        if not frames:
            raise ValueError('Cannot encode an empty image batch')
        fixed, frame_audits = [], []
        # Individual conversion also supports differently shaped source frames;
        # applying the shared pixel transform never reorders or duplicates them.
        for frame in frames:
            source = torch.from_numpy(np.array(frame.convert('RGB'), copy=True)).unsqueeze(0)
            prepared = prepare_fixed_frames(source, canvas_size=SIZE, pad_value=PAD)
            frame_audits.append(dict(source_hw=list(source.shape[1:3]),
                resized_hw=[SIZE, SIZE], rgb_sha256=frame_rgb_sha256(prepared)))
            fixed.append(Image.fromarray(prepared[0].numpy(), mode='RGB'))
        result = original(fixed)
        grids = result[1].detach().cpu().tolist()
        grid_audit = assert_qwen_grid(grids, int(result[0].shape[0]),
            raw_frames=len(frames), canvas_size=SIZE, kind='image')
        state['grids'].extend(grids)
        state['frames'].extend(frame_audits)
        state['calls'].append(grid_audit)
        return result

    qa.encode_vision = encode
    return state


def summarize(output, rows, total, *, configured_slots=56000, version=VERSION):
    result = old.summarize(output, rows, total, FPS, 'fixed896')
    result.update(version=version, configured_mean_visual_budget=configured_slots / 112,
                  fixed_input=fixed_input_metadata(SIZE, PAD),
                  actual_image_tokens_per_real_frame=1024 if rows else None,
                  temporal_encoding='original independent image, no cross-frame temporal merge')
    old.base.write(output / 'summary.json', result)
    old.base.write(output / 'standalone_summary.json', result)
    return result


def snapshot_relative(path):
    try:
        return Path('kvbench') / path.relative_to(ROOT)
    except ValueError:
        try:
            return Path('SimpleStream') / path.relative_to(old.base.UPSTREAM)
        except ValueError:
            return Path('external_budget') / (old.sha(path)[:12] + '_' + path.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--budget-map', type=Path,
                        help='JSON with arm and 28x4 per-head visual budgets (32..500)')
    parser.add_argument('--capture-choice-probs', action='store_true',
                        help='Debug only: save conditional A-D first-token probabilities')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    tiered = args.budget_map is not None
    if tiered:
        budget_path = args.budget_map.resolve()
        budget_config = json.loads(budget_path.read_text())
        budget_rows = budget_config['budgets']
        if (len(budget_rows) != 28 or any(len(row) != 4 for row in budget_rows)
                or any(type(value) is not int or not 32 <= value <= 500
                       for row in budget_rows for value in row)):
            raise ValueError('Expected 28x4 integer head budgets in [32,500]')
        budgets = {(layer, head): budget_rows[layer][head]
                   for layer in range(28) for head in range(4)}
        configured_slots = sum(budgets.values())
        run_version = 'simplestream_one_four_pseudo_tiered_fixed896_v1'
    else:
        budget_path = None
        budgets = {(layer, head): BUDGET for layer in range(28) for head in range(4)}
        configured_slots = 56000
        run_version = VERSION
    output, manifest_path = args.output_root.resolve(), args.manifest.resolve()
    manifest, annotations = old.validate_manifest(manifest_path)
    output.mkdir(parents=True, exist_ok=True)
    progress = output / 'queries.jsonl'
    rows = old.read_rows(progress, recover=True)
    old.read_rows(output / 'ragged_query_audit.jsonl', recover=True)
    if len(rows) > len(manifest) or [r['query_id'] for r in rows] != [q['query_id'] for q in manifest[:len(rows)]]:
        raise ValueError('Existing predictions do not match frozen manifest order')
    for row, query in zip(rows, manifest):
        if (row['task'] != query['task'] or row['target'] != query['target']
                or not isinstance(row.get('response'), str)
                or row.get('correct') != official.official_correct(row['task'], row['response'], row['target'])
                or row.get('budget_audit', {}).get('actual_visual_slots') != configured_slots
                or (tiered and row.get('budget_audit', {}).get('per_head_visual_budgets') != budget_rows)
                or any(count != 1024 for count in row.get('input_audit', {}).get('per_frame_visual_tokens', []))
                or row.get('input_audit', {}).get('fixed_input') != fixed_input_metadata(SIZE, PAD)):
            raise ValueError('Existing prediction identity, fixed input, score or budget is invalid')
    paths = [Path(__file__).resolve(), ROOT / 'kvbench/fixed_inputs.py', CONFIG,
             *(ROOT / name for name in old.LOCAL_SOURCES),
             *(old.base.UPSTREAM / name for name in old.base.UPSTREAM_FILES), old.RAGGED, old.HEAD_TABLE]
    if tiered:
        paths.append(budget_path)
    paths = list(dict.fromkeys(paths))
    sources = {str(path): old.sha(path) for path in paths}
    clips = {str(q['source_id']): dict(path=str(old.base.DATA / 'chunked_videos' / f"{q['source_id']}.mp4"),
                bytes=(old.base.DATA / 'chunked_videos' / f"{q['source_id']}.mp4").stat().st_size)
             for q in manifest}
    pin = dict(version=run_version, fps=FPS, resolution='fixed896', fixed_input=fixed_input_metadata(SIZE, PAD),
        image_tokens_per_real_frame=1024, visual_budget_per_head=BUDGET if not tiered else None,
        per_head_visual_budgets=budget_rows if tiered else None,
        configured_visual_slots=configured_slots,
        recent_bucket_count=4, bucket_seconds=1., max_new_tokens=MAX_NEW, model=str(old.base.MODEL),
        model_files_sha256={str(old.base.MODEL / name): old.sha(old.base.MODEL / name) for name in old.base.MODEL_FILES},
        code_sha256=sources, manifest_sha256=old.sha(manifest_path), query_count=len(manifest),
        annotation_sha256=old.sha(old.base.ANNOTATION), required_dependencies=old.REQUIRED_VERSIONS,
        prompt='unchanged released build_qwen25_prompt', visual_encoder='unchanged per-image get_image_features',
        positions='unchanged released continuous-image prefix get_rope_index',
        selector=('pseudo_budget500.install: probe at visual_end; FP32 GQA mean avgpool5 protect32; per-head cap only'
                  if tiered else
                  'unchanged pseudo_budget500.install: probe at visual_end; FP32 GQA mean avgpool5 protect32'),
        head_groups='unchanged 56 one-frame / 56 four-frame', clip_metadata=clips,
        historical_visual_kv=0, question_used_for_selection=False,
        capture_choice_probs=args.capture_choice_probs,
        source_sampling='original qwen-vl-utils full plan; final four nonempty one-second buckets',
        input_only_changes=['requested sampling FPS=.5', 'shared fixed896 aspect-preserving resize and gray127 padding',
                            'disable image-processor resize after fixed pixel preparation'])
    protocol = output / 'protocol.json'
    if rows and (not protocol.exists() or json.loads(protocol.read_text()) != pin):
        raise ValueError('Pinned inference/input source changed; use a new output directory')
    old.base.write(protocol, pin)
    old.base.write(output / 'query_manifest.json', manifest)
    for path in paths:
        destination = output / 'source_snapshot' / snapshot_relative(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    summarize(output, rows, len(manifest), configured_slots=configured_slots, version=run_version)
    if args.prepare_only:
        print(json.dumps(dict(output=str(output), queries=len(manifest), done=len(rows), fps=FPS,
            fixed_input=fixed_input_metadata(SIZE, PAD), model_loaded=False)))
        return
    if len(rows) == len(manifest):
        old.base.write(output / 'status.json', dict(state='complete', done=len(rows), total=len(manifest), updated_at=old.base.now()))
        return
    for name, expected in old.REQUIRED_VERSIONS.items():
        if version(name) != expected:
            raise RuntimeError(f'Use runtime_envs/simplestream-qwen25/bin/python: {name}={version(name)}, expected {expected}')
    for key in ('MIN_PIXELS', 'MAX_PIXELS', 'MODEL_SEQ_LEN', 'QWEN_EXACT_RECENT_DECODE', 'SPARSEMM_GPU_LEASE_SHARED'):
        os.environ.pop(key, None)
    sys.path.insert(0, str(old.base.UPSTREAM))
    import torch
    from lib import recent_window_eval_qwen25 as upstream
    from sparsemm_native.simplestream_recent_share_pseudo_budget500 import install, HEADS
    from sparsemm_native.gpu_slots import gpu_lease
    install(output, head_budgets=budgets if tiered else None,
            capture_choice_probs=args.capture_choice_probs)
    input_state = old.install_decode_hook(upstream, FPS, 'native')
    started = time.perf_counter()
    try:
        with gpu_lease(output) as admission:
            old.base.write(output / 'gpu_admission.json', admission)
            old.base.write(output / 'status.json', dict(state='loading', done=len(rows), total=len(manifest), pid=os.getpid(), updated_at=old.base.now()))
            qa = upstream.RecentWindowQAModel(str(old.base.MODEL), device='cuda', max_new_tokens=MAX_NEW,
                                            attn_implementation='flash_attention_2')
            fixed_state = install_fixed_encoder(qa)
            old.base.write(output / 'runtime.json', dict(dependencies={name: version(name) for name in (
                'torch', 'torchvision', 'transformers', 'accelerate', 'qwen-vl-utils', 'flash-attn', 'Pillow', 'decord')},
                image_processor=qa.processor.image_processor.to_dict(), fixed_input=fixed_input_metadata(SIZE, PAD),
                generation_config=qa.model.generation_config.to_dict(), gpu_lease=admission, started_at=old.base.now()))
            audit_path = output / 'ragged_query_audit.jsonl'
            for query in manifest[len(rows):]:
                if (output / 'PAUSE').exists():
                    old.base.write(output / 'status.json', dict(state='paused', done=len(rows), total=len(manifest), updated_at=old.base.now()))
                    return
                qid = query['query_id']
                old.base.write(output / 'status.json', dict(state='running', done=len(rows), total=len(manifest),
                    active_query=qid, pid=os.getpid(), updated_at=old.base.now()))
                offset = audit_path.stat().st_size if audit_path.exists() else 0
                prompt = upstream.build_qwen25_prompt(query['task'], annotations[qid])
                clip = old.base.DATA / 'chunked_videos' / f"{query['source_id']}.mp4"
                for values in fixed_state.values():
                    values.clear()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                before = time.perf_counter()
                result, backend = upstream.query_recent_window(qa=qa, video_path=str(clip), prompt=prompt,
                    chunk_duration=1., fps=FPS, recent_frames_only=4)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - before
                with audit_path.open('rb') as handle:
                    handle.seek(offset)
                    appended = [json.loads(line) for line in handle.read().splitlines() if line.strip()]
                if len(appended) != 1:
                    raise AssertionError(f'Expected one physical KV audit, got {len(appended)}')
                audit = appended[0]
                if args.capture_choice_probs:
                    probabilities = audit.get('choice_probabilities')
                    if (not isinstance(probabilities, dict)
                            or set(probabilities) != set('ABCD')
                            or abs(sum(probabilities.values()) - 1.0) > 1e-5
                            or audit.get('choice_prediction') != max(
                                probabilities, key=probabilities.get)
                            or abs(audit.get('choice_margin') -
                                   (sorted(probabilities.values(), reverse=True)[0]
                                    - sorted(probabilities.values(), reverse=True)[1])) > 1e-5):
                        raise AssertionError('Invalid four-choice probability audit')
                counts, grids = audit['frame_visual_tokens'], fixed_state['grids']
                if (not counts or any(count != 1024 for count in counts)
                        or [int(t*h*w)//4 for t,h,w in grids] != counts
                        or len(fixed_state['frames']) != len(input_state['selected_raw_frame_indices'])):
                    raise AssertionError('Fixed image grid/token count differs from actual frame audit')
                per_head = [[min(counts[-1] if (layer, head) in HEADS else sum(counts), budgets[layer, head])
                             for head in range(4)] for layer in range(28)]
                total_slots = sum(map(sum, per_head))
                if (audit['actual_visual_slots'] != total_slots or total_slots != configured_slots
                        or audit['visual_budget_per_head'] != (BUDGET if not tiered else None)
                        or (tiered and audit['per_head_visual_budgets'] != budget_rows)
                        or audit['selector'] != 'fixed_mixed_pseudo_qk_fp32_gqa_mean_avgpool5_topk_protect32'
                        or audit['prompt_sha256'] != hashlib.sha256(prompt.encode()).hexdigest()
                        or result.final_chunk_ids != input_state['selected_chunk_ids']
                        or result.num_frames != len(input_state['selected_raw_frame_indices'])):
                    raise AssertionError('Original inference input/budget audit failed')
                input_audit = dict(input_state, resolution='fixed896', fixed_input=fixed_input_metadata(SIZE, PAD),
                    resize_hw=[SIZE, SIZE], per_frame_visual_tokens=counts, image_grid_thw=list(grids),
                    processor_resize_hw_per_frame=[[int(h)*14, int(w)*14] for _,h,w in grids],
                    fixed_frame_audits=list(fixed_state['frames']), fixed_encoder_call_audits=list(fixed_state['calls']),
                    processor_do_resize=bool(qa.processor.image_processor.do_resize),
                    prompt=prompt, prompt_sha256=audit['prompt_sha256'], visual_encoder='original_per_image',
                    position_layout='original_continuous_image_get_rope_index', decode_backend=backend)
                budget_audit = dict(audit, actual_tokens_per_head=total_slots / 112,
                    per_head_visual_slots=per_head, actual_history_visual_slots=0,
                    actual_recent_visual_slots=total_slots, recent_candidate_visual_slots=56 * (counts[-1] + sum(counts)),
                    configured_visual_slots=configured_slots, question_text_used_for_selection=False)
                row = {key: query[key] for key in ('query_id', 'source_id', 'probe_id', 'task', 'group', 'timestamp', 'clip_end', 'target')}
                row.update(response=result.answer,
                    choice_probabilities=audit.get('choice_probabilities'),
                    choice_margin=audit.get('choice_margin'),
                    choice_prediction=audit.get('choice_prediction'),
                    correct=official.official_correct(query['task'], result.answer, query['target']),
                    seconds=seconds, input_audit=input_audit, budget_audit=budget_audit,
                    cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/1048576,
                    cuda_peak_reserved_mib=torch.cuda.max_memory_reserved()/1048576,
                    upstream_result=dataclasses.asdict(result))
                old.base.append(progress, row)
                rows.append(row)
                summarize(output, rows, len(manifest), configured_slots=configured_slots, version=run_version)
                print(f"{len(rows)}/{len(manifest)} {qid} {query['task']} response={result.answer!r} "
                      f"correct={row['correct']} tokens/frame=1024 tokens/head={total_slots/112:.2f} seconds={seconds:.3f}", flush=True)
        old.base.write(output / 'status.json', dict(state='complete', done=len(rows), total=len(manifest), updated_at=old.base.now(),
            elapsed_this_invocation_seconds=time.perf_counter()-started))
    except BaseException as exc:
        old.base.write(output / 'status.json', dict(state='failed', done=len(rows), total=len(manifest), updated_at=old.base.now(),
            error=f'{type(exc).__name__}: {exc}', elapsed_this_invocation_seconds=time.perf_counter()-started))
        raise


if __name__ == '__main__':
    main()
