"""Old SimpleStream pseudo-K500 inference with only FPS/input resize changed.

The released model wrapper, per-image encoder, continuous-image get_rope_index,
prompt builder and the completed pseudo-K500 plugin are called unchanged. Only
the decoder hook selects the exact final four nonempty one-second buckets from
the full qwen-vl-utils sampling plan, then optionally omits its video resize.
Use runtime_envs/simplestream-qwen25/bin/python (transformers 4.52.4).
"""

import argparse
import collections
import dataclasses
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

from sparsemm_native import run_ovobench as official
from sparsemm_native import supervise_simplestream_upstream as base


ROOT = Path(__file__).resolve().parents[1]
BUDGET = 500
MAX_NEW = 256
REQUIRED_VERSIONS = {'transformers': '4.52.4', 'accelerate': '1.8.1', 'qwen-vl-utils': '0.0.14'}
HEAD_TABLE = ROOT.parent / 'hermes/results/myobservations/lukv_samecalib_two_bank_k1000_20261022/per_head_allocations.csv'
RAGGED = ROOT.parent / 'hermes/head_analysis/hermes_kv_head_ragged.py'
LOCAL_SOURCES = (
    'sparsemm_native/run_simplestream_k500_input_only.py',
    'sparsemm_native/simplestream_recent_share_pseudo_budget500.py',
    'sparsemm_native/simplestream_recent_share_attention.py',
    'sparsemm_native/simplestream_headmix_attention.py',
    'sparsemm_native/supervise_simplestream_upstream.py',
    'sparsemm_native/gpu_slots.py', 'sparsemm_native/run_ovobench.py',
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path, *, recover=False):
    if not path.exists():
        return []
    raw = path.read_bytes()
    lines, rows, offset = raw.splitlines(keepends=True), [], 0
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if index != len(lines) - 1 or line.endswith(b'\n'):
                raise
            if recover:
                with path.open('r+b') as handle:
                    handle.truncate(offset)
                    handle.flush()
                    os.fsync(handle.fileno())
            break
        if not line.endswith(b'\n'):
            # A fully written JSON object still needs its record delimiter.
            if recover:
                with path.open('ab') as handle:
                    handle.write(b'\n')
                    handle.flush()
                    os.fsync(handle.fileno())
        rows.append(row)
        offset += len(line)
    return rows


def validate_manifest(path):
    manifest = json.loads(path.read_text())
    expected = {q['query_id']: q for q in official.queries()}
    if (not isinstance(manifest, list) or not manifest
            or len({q['query_id'] for q in manifest}) != len(manifest)
            or any(q != expected.get(q['query_id']) for q in manifest)
            or any(q['task'] not in official.BACKWARD | official.REALTIME for q in manifest)):
        raise ValueError('Expected unique unchanged official nine-task query dictionaries')
    annotations = {str(a['id']): a for a in json.loads(base.ANNOTATION.read_text())}
    for query in manifest:
        annotation = annotations[query['query_id']]
        if annotation['task'] != query['task'] or chr(65 + annotation['gt']) != query['target']:
            raise ValueError('Manifest and original SimpleStream annotation labels differ')
        clip = base.DATA / 'chunked_videos' / f"{query['source_id']}.mp4"
        if not clip.is_file() or clip.stat().st_size == 0:
            raise FileNotFoundError(clip)
    return manifest, annotations


def summarize(output, rows, total, fps, resolution):
    per_task = {}
    for row in rows:
        item = per_task.setdefault(row['task'], dict(correct=0, total=0))
        item['correct'] += int(row['correct'])
        item['total'] += 1
    for item in per_task.values():
        item['accuracy'] = item['correct'] / item['total']
    correct = sum(item['correct'] for item in per_task.values())
    backward = [item['accuracy'] for task, item in per_task.items() if task in official.BACKWARD]
    realtime = [item['accuracy'] for task, item in per_task.items() if task in official.REALTIME]
    seconds = [row['seconds'] for row in rows]
    result = dict(version='simplestream_k500_input_only_v1', done=len(rows), questions=len(rows),
        total=total, complete=len(rows) == total, fps=fps, resolution=resolution,
        configured_mean_visual_budget=BUDGET, per_task=per_task, correct=correct,
        question_accuracy=correct / len(rows) if rows else None,
        nine_task_macro_average=statistics.mean(item['accuracy'] for item in per_task.values())
                                if len(per_task) == 9 else None,
        backward_macro_average=statistics.mean(backward) if len(backward) == 3 else None,
        realtime_macro_average=statistics.mean(realtime) if len(realtime) == 6 else None,
        official_two_group_macro_average=(statistics.mean([statistics.mean(backward), statistics.mean(realtime)])
                                          if len(backward) == 3 and len(realtime) == 6 else None),
        scoring='released official raw-response target-in-response; no answer extraction',
        actual_tokens_per_head_mean=statistics.mean(row['budget_audit']['actual_tokens_per_head'] for row in rows) if rows else None,
        query_seconds_mean=statistics.mean(seconds) if seconds else None,
        query_seconds_total=sum(seconds), updated_at=base.now())
    base.write(output / 'summary.json', result)
    base.write(output / 'standalone_summary.json', result)
    return result


def install_decode_hook(upstream, fps, resolution):
    """Replace input decoding only; preserve the old query/model pipeline."""
    import torch
    from PIL import Image
    from lib.recent_window_eval import EvalChunk
    from lib.qwen_exact_recent_decoder import (
        _build_sampling_plan, _decode_indices, _resize_like_fetch_video,
    )
    last_input = {}

    def decode(video_path, chunk_duration, fps, recent_frames_only=None,
               video_start=None, video_end=None):
        if (recent_frames_only != 4 or float(chunk_duration) != 1.0
                or float(fps) <= 0 or not math.isfinite(float(fps))):
            raise ValueError('This input hook requires four recent one-second buckets and positive FPS')
        request = dict(video=str(video_path), fps=float(fps))
        if video_start is not None:
            request['video_start'] = max(0., float(video_start))
        if video_end is not None:
            request['video_end'] = max(0., float(video_end))
        plan = _build_sampling_plan(request, last_nframes=4)
        indices, bucket_ids = base.full_window_indices(plan, seconds=chunk_duration, count=4)
        video = _decode_indices(str(video_path), plan.backend, indices)
        source_hw = list(map(int, video.shape[-2:]))
        if resolution == 'old':
            video = _resize_like_fetch_video(video, request, plan.sampled_nframes_full, image_patch_size=14)
        decoded_hw = list(map(int, video.shape[-2:]))
        timestamps = [index / plan.raw_fps for index in indices]
        if len(indices) > 1:
            sampled_fps = (len(indices) - 1) / max(timestamps[-1] - timestamps[0], 1e-6)
            frame_dt = max(timestamps[-1] - timestamps[-2], 1. / plan.raw_fps)
        else:
            sampled_fps, frame_dt = float(fps), 1. / plan.raw_fps
        max_valid_end = max(timestamps, default=0.) + frame_dt
        buckets = collections.OrderedDict()
        for frame, timestamp in zip(video, timestamps):
            # This is exactly the released decoder's conversion/clamp order.
            pixels = frame.clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            bucket = int(timestamp // chunk_duration)
            buckets.setdefault(bucket, []).append((Image.fromarray(pixels), timestamp))
        chunks = [EvalChunk(frames=[frame for frame, _ in frames],
                  frame_timestamps=[stamp for _, stamp in frames],
                  start_time=bucket * chunk_duration,
                  end_time=min((bucket + 1) * chunk_duration, max_valid_end),
                  chunk_index=bucket, fps=sampled_fps)
                  for bucket, frames in buckets.items()]
        if [chunk.chunk_index for chunk in chunks] != bucket_ids or not chunks:
            raise AssertionError('Decoded buckets differ from the original full sampling plan')
        last_input.clear()
        last_input.update(source_clip=str(video_path), requested_fps=float(fps),
            effective_plan_fps=plan.sample_fps, raw_fps=plan.raw_fps,
            raw_total_frames=plan.raw_total_frames, full_sampled_frames=plan.sampled_nframes_full,
            selected_raw_frame_indices=indices, selected_chunk_ids=bucket_ids,
            sampled_timestamps=timestamps, source_hw=source_hw, resize_hw=decoded_hw,
            resolution=resolution, decode_backend=plan.backend,
            external_video_resize=resolution == 'old', selected_frames=len(indices))
        return chunks, f'{plan.backend}_exact_original_full_buckets_{resolution}'

    upstream.decode_video_to_chunks_qwen = decode
    return last_input


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fps', type=float, default=.5)
    parser.add_argument('--resolution', choices=('native', 'old'), default='native')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.fps <= 0 or not math.isfinite(args.fps):
        raise ValueError('FPS must be finite and positive')
    output, manifest_path = args.output_root.resolve(), args.manifest.resolve()
    manifest, annotations = validate_manifest(manifest_path)
    output.mkdir(parents=True, exist_ok=True)
    progress = output / 'queries.jsonl'
    rows = read_rows(progress, recover=True)
    # A process can die during the plugin's separate fsynced audit append.
    # Completed orphan audits are harmless: each query reads from its own new
    # byte offset. Only an incomplete final record needs removal on resume.
    read_rows(output / 'ragged_query_audit.jsonl', recover=True)
    if [r['query_id'] for r in rows] != [q['query_id'] for q in manifest[:len(rows)]] or len(rows) > len(manifest):
        raise ValueError('Existing predictions do not match the frozen manifest order')
    for row, query in zip(rows, manifest):
        if (row['task'] != query['task'] or row['target'] != query['target']
                or not isinstance(row.get('response'), str)
                or row.get('correct') != official.official_correct(row['task'], row['response'], row['target'])
                or row.get('budget_audit', {}).get('actual_visual_slots', 56001) > 56000):
            raise ValueError('Existing prediction identity, score or budget is invalid')
    paths = [*(ROOT / name for name in LOCAL_SOURCES),
             *(base.UPSTREAM / name for name in base.UPSTREAM_FILES), RAGGED, HEAD_TABLE]
    input_sources = {str(path): sha(path) for path in paths}
    clips = {str(q['source_id']): dict(path=str(base.DATA / 'chunked_videos' / f"{q['source_id']}.mp4"),
              bytes=(base.DATA / 'chunked_videos' / f"{q['source_id']}.mp4").stat().st_size)
             for q in manifest}
    pin = dict(version='simplestream_k500_input_only_v1', fps=args.fps, resolution=args.resolution,
        visual_budget_per_head=BUDGET, configured_visual_slots=56000, recent_bucket_count=4,
        bucket_seconds=1., max_new_tokens=MAX_NEW, model=str(base.MODEL),
        model_files_sha256={str(base.MODEL / name): sha(base.MODEL / name) for name in base.MODEL_FILES},
        code_sha256=input_sources, manifest_sha256=sha(manifest_path), query_count=len(manifest),
        annotation_sha256=sha(base.ANNOTATION), required_dependencies=REQUIRED_VERSIONS,
        prompt='unchanged released build_qwen25_prompt', visual_encoder='unchanged per-image get_image_features',
        positions='unchanged released continuous-image prefix get_rope_index',
        selector='unchanged pseudo_budget500.install: probe at visual_end; FP32 GQA mean avgpool5 protect32',
        head_groups='unchanged 56 one-frame / 56 four-frame',
        clip_metadata=clips, historical_visual_kv=0, question_used_for_selection=False,
        source_sampling='original qwen-vl-utils full plan; final four nonempty one-second buckets',
        input_only_changes=['requested sampling FPS', 'omit fetch_video resize when resolution=native'])
    protocol = output / 'protocol.json'
    if rows and (not protocol.exists() or json.loads(protocol.read_text()) != pin):
        raise ValueError('Pinned inference/input source changed; use a new output directory')
    base.write(protocol, pin)
    base.write(output / 'query_manifest.json', manifest)
    for path in paths:
        destination = output / 'source_snapshot' / path.relative_to(ROOT.parent)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    summarize(output, rows, len(manifest), args.fps, args.resolution)
    if args.prepare_only:
        print(json.dumps(dict(output=str(output), queries=len(manifest), done=len(rows),
            fps=args.fps, resolution=args.resolution, model_loaded=False)))
        return
    if len(rows) == len(manifest):
        base.write(output / 'status.json', dict(state='complete', done=len(rows), total=len(manifest), updated_at=base.now()))
        return
    for name, expected in REQUIRED_VERSIONS.items():
        if version(name) != expected:
            raise RuntimeError(f'Use runtime_envs/simplestream-qwen25/bin/python: {name}={version(name)}, expected {expected}')
    for key in ('MIN_PIXELS', 'MAX_PIXELS', 'MODEL_SEQ_LEN', 'QWEN_EXACT_RECENT_DECODE',
                'SPARSEMM_GPU_LEASE_SHARED'):
        os.environ.pop(key, None)
    sys.path.insert(0, str(base.UPSTREAM))
    import torch
    from lib import recent_window_eval_qwen25 as upstream
    from sparsemm_native.simplestream_recent_share_pseudo_budget500 import install, HEADS
    from sparsemm_native.gpu_slots import gpu_lease
    install(output)
    input_state = install_decode_hook(upstream, args.fps, args.resolution)
    started = time.perf_counter()
    try:
        with gpu_lease(output) as admission:
            base.write(output / 'gpu_admission.json', admission)
            base.write(output / 'status.json', dict(state='loading', done=len(rows), total=len(manifest), pid=os.getpid(), updated_at=base.now()))
            qa = upstream.RecentWindowQAModel(str(base.MODEL), device='cuda', max_new_tokens=MAX_NEW,
                                            attn_implementation='flash_attention_2')
            base.write(output / 'runtime.json', dict(dependencies={name: version(name) for name in (
                'torch', 'torchvision', 'transformers', 'accelerate', 'qwen-vl-utils', 'flash-attn', 'Pillow', 'decord')},
                image_processor=qa.processor.image_processor.to_dict(), generation_config=qa.model.generation_config.to_dict(),
                gpu_lease=admission, started_at=base.now()))
            # Observe the original encoder's returned grids; no inputs or
            # outputs of the released vision forward are changed.
            encoded_grids = []
            original_encode = qa.encode_vision
            def measured_encode(frames):
                result = original_encode(frames)
                encoded_grids.extend(result[1].detach().cpu().tolist())
                return result
            qa.encode_vision = measured_encode
            audit_path = output / 'ragged_query_audit.jsonl'
            for query in manifest[len(rows):]:
                if (output / 'PAUSE').exists():
                    base.write(output / 'status.json', dict(state='paused', done=len(rows), total=len(manifest), updated_at=base.now()))
                    return
                qid = query['query_id']
                base.write(output / 'status.json', dict(state='running', done=len(rows), total=len(manifest), active_query=qid,
                           pid=os.getpid(), updated_at=base.now()))
                offset = audit_path.stat().st_size if audit_path.exists() else 0
                prompt = upstream.build_qwen25_prompt(query['task'], annotations[qid])
                clip = base.DATA / 'chunked_videos' / f"{query['source_id']}.mp4"
                encoded_grids.clear()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                before = time.perf_counter()
                result, backend = upstream.query_recent_window(qa=qa, video_path=str(clip), prompt=prompt,
                    chunk_duration=1., fps=args.fps, recent_frames_only=4)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - before
                with audit_path.open('rb') as handle:
                    handle.seek(offset)
                    appended = [json.loads(line) for line in handle.read().splitlines() if line.strip()]
                if len(appended) != 1:
                    raise AssertionError(f'Expected one new physical KV audit, received {len(appended)}')
                audit = appended[0]
                counts = audit['frame_visual_tokens']
                if [int(t*h*w)//4 for t,h,w in encoded_grids] != counts:
                    raise AssertionError('Observed original vision grids differ from candidate token audit')
                per_head = [[min(counts[-1] if (layer, head) in HEADS else sum(counts), BUDGET)
                             for head in range(4)] for layer in range(28)]
                total_slots = sum(map(sum, per_head))
                if (audit['actual_visual_slots'] != total_slots or total_slots > 56000
                        or audit['visual_budget_per_head'] != BUDGET
                        or audit['selector'] != 'fixed_mixed_pseudo_qk_fp32_gqa_mean_avgpool5_topk_protect32'
                        or audit['prompt_sha256'] != hashlib.sha256(prompt.encode()).hexdigest()
                        or result.final_chunk_ids != input_state['selected_chunk_ids']
                        or result.num_frames != len(input_state['selected_raw_frame_indices'])):
                    raise AssertionError('Original inference input/budget audit failed')
                input_audit = dict(input_state, per_frame_visual_tokens=counts,
                    image_grid_thw=list(encoded_grids),
                    processor_resize_hw_per_frame=[[int(h)*14, int(w)*14] for _,h,w in encoded_grids],
                    prompt=prompt, prompt_sha256=audit['prompt_sha256'], visual_encoder='original_per_image',
                    position_layout='original_continuous_image_get_rope_index', decode_backend=backend)
                budget_audit = dict(audit, actual_tokens_per_head=total_slots / 112,
                    per_head_visual_slots=per_head, actual_history_visual_slots=0,
                    actual_recent_visual_slots=total_slots,
                    recent_candidate_visual_slots=56 * (counts[-1] + sum(counts)),
                    configured_visual_slots=56000, question_text_used_for_selection=False)
                row = {key: query[key] for key in ('query_id', 'source_id', 'probe_id', 'task', 'group', 'timestamp', 'clip_end', 'target')}
                row.update(response=result.answer, correct=official.official_correct(query['task'], result.answer, query['target']),
                    seconds=seconds, input_audit=input_audit, budget_audit=budget_audit,
                    cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/1048576,
                    cuda_peak_reserved_mib=torch.cuda.max_memory_reserved()/1048576,
                    upstream_result=dataclasses.asdict(result))
                base.append(progress, row)
                rows.append(row)
                summarize(output, rows, len(manifest), args.fps, args.resolution)
                print(f"{len(rows)}/{len(manifest)} {qid} {query['task']} response={result.answer!r} "
                      f"correct={row['correct']} tokens/head={total_slots / 112:.2f} seconds={seconds:.3f}", flush=True)
        base.write(output / 'status.json', dict(state='complete', done=len(rows), total=len(manifest), updated_at=base.now(),
                   elapsed_this_invocation_seconds=time.perf_counter() - started))
    except BaseException as exc:
        base.write(output / 'status.json', dict(state='failed', done=len(rows), total=len(manifest), updated_at=base.now(),
                   error=f'{type(exc).__name__}: {exc}', elapsed_this_invocation_seconds=time.perf_counter() - started))
        raise


if __name__ == '__main__':
    main()
