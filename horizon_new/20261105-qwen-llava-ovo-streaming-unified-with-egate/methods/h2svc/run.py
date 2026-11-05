"""Sharded H2SVC runner using the unified Qwen/LLaVA input protocol."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from methods.registry import PROJECT_ROOT, SHARED_KVBENCH, LLAVA_BACKBONE, LUKV_PROFILE
from methods.horizonkv_egate.data import validate_annotations, save_exact
from .routing import read_head_map

WAREHOUSE = Path(os.environ.get('HORIZON_WAREHOUSE', '/mnt/aifordata2026-p/common/aifordata/541707'))
RELEASES = WAREHOUSE / 'code/horizonkv/releases'
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', required=True, choices=['streamingbench', 'ovobench'])
    parser.add_argument('--model', required=True, choices=['qwen25vl', 'llavaov'])
    parser.add_argument('--method', required=True, choices=['h2svc'])
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--annotation', type=Path)
    parser.add_argument('--max-videos', type=int)
    parser.add_argument('--cohort', default='heldout', choices=['heldout', 'full'])
    parser.add_argument('--allow-calibration-overlap', action='store_true')
    parser.add_argument('--accept-equivalent-choices', action='store_true')
    parser.add_argument('--visual-input', choices=['dscache448-video'])
    parser.add_argument('--sample-fps', type=float, default=1)
    parser.add_argument('--chunk-frames', type=int, default=2)
    parser.add_argument('--input-profile', choices=['dscache'])
    parser.add_argument('--head-map', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.sample_fps != 1 or args.chunk_frames != 2:
        parser.error('H2SVC uses unified 1 fps input and Qwen two-frame merge')
    head_map = (args.head_map or HERE / 'assets' / f'{args.model}_head_map.json').resolve()
    routes, map_data = read_head_map(head_map)
    if map_data['model'] != args.model:
        raise ValueError('Head map belongs to a different model')
    stream = args.benchmark == 'streamingbench'
    default = ('data/streamingbench/streamingbench_realtime_local.json' if stream else
               'data/ovobench/ovobench_realtime_backward_warehouse.json')
    source = (args.annotation or RELEASES / '20261022-ovobench-adapter-v1' / default).resolve()
    raw = json.loads(source.read_text())
    calibration = json.loads(LUKV_PROFILE.read_text())['calibration']['selected_videos']
    def name(item):
        return Path(item if isinstance(item, str) else item.get('video_path', item.get('path', ''))).name
    excluded_names = {name(item) for item in calibration}
    if len(excluded_names) != 50:
        raise ValueError('Invalid held-out video cohort')
    profile_names = {name(item) for item in map_data['calibration_videos']}
    if stream and args.cohort == 'heldout' and profile_names != excluded_names:
        raise ValueError('H2SVC heldout comparison requires the same 50 excluded calibration videos')
    data, excluded_questions = [], []
    for original in raw:
        if stream and args.cohort == 'heldout' and name(original['video_path']) in excluded_names:
            continue
        video = dict(original)
        video['conversations'] = []
        for original_q in original['conversations']:
            question = dict(original_q)
            if stream and question['choices'].count(question['answer']) > 1 and not args.accept_equivalent_choices:
                excluded_questions.append(dict(video_id=video['video_id'], question=question['question']))
                continue
            if not question.get('correct_choice') and question['answer'] in question['choices']:
                question['correct_choice'] = 'ABCDEFGH'[question['choices'].index(question['answer'])]
            video['conversations'].append(question)
        if not video['conversations']:
            continue
        media = Path(video['video_path'])
        if not media.is_absolute():
            media = source.parent / media
        if not media.is_file() and stream:
            media = WAREHOUSE / 'dataset/StreamingBench/videos' / media.name
        video['video_path'] = str(media.resolve())
        data.append(video)
    if args.max_videos is not None:
        if not 1 <= args.max_videos <= len(data):
            parser.error('Invalid max-videos')
        data = data[:args.max_videos]
    validate_annotations(data)
    overlap = sum(name(video['video_path']) in excluded_names for video in data) if stream else 0
    if stream and overlap and not args.allow_calibration_overlap:
        parser.error('Full cohort overlaps frozen calibration videos')
    gpus = args.gpus.split(',')
    if not gpus or any(not gpu.isdigit() for gpu in gpus) or len(set(map(int, gpus))) != len(gpus):
        parser.error('Invalid GPU IDs')
    gpus = gpus[:min(len(gpus), len(data))]
    config = dict(schema_version=1, backend='h2svc', benchmark=args.benchmark, model=args.model,
                  method='h2svc', source=str(source), source_sha256=sha(source),
                  cohort=args.cohort if stream else None, gpus=gpus, videos=len(data),
                  questions=sum(len(video['conversations']) for video in data), fps=1,
                  ordinary_chunk_frames=2 if args.model == 'qwen25vl' else 1,
                  qwen_eviction_unit='indivisible merged two-frame KV unit',
                  ordinary_pixels=448**2 if args.model == 'qwen25vl' else [384, 384],
                  latest_frame_replay=False,
                  llava_visual_tokens=196 if args.model == 'llavaov' else None,
                  head_routes={kind:sum(route == kind for route in routes.values()) for kind in
                               ('instantaneous', 'short_term', 'episodic')},
                  frame_budgets={'instantaneous':4, 'short_term':16, 'episodic':64},
                  head_map=str(head_map), head_map_sha256=sha(head_map),
                  calibration_source_sha256=map_data['calibration_source_sha256'],
                  reference_heldout_overlap_videos=overlap,
                  excluded_ambiguous_questions=excluded_questions,
                  accept_equivalent_choices=args.accept_equivalent_choices,
                  dtype='float16', detailed_cache_trace=False,
                  code_sha256={str(file.relative_to(PROJECT_ROOT)):sha(file)
                               for file in sorted(HERE.rglob('*.py'))},
                  shared_cache_sha256=sha(SHARED_KVBENCH / 'sparsemm_baselines/cache.py'))
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return 0
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not (output / 'manifest.json').is_file():
        raise ValueError('Nonempty output without manifest')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'annotations').mkdir(exist_ok=True)
    save_exact(output / 'manifest.json', config)
    environment = os.environ.copy()
    environment.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                       PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', DECORD_EOF_RETRY_MAX='40960')
    paths = ([Path('/tmp/llava-ovo-transformers440')] if args.model == 'llavaov' else
             [WAREHOUSE / 'code/horizonkv/runtime/qwen-py310-deps-20261016'])
    paths += [PROJECT_ROOT, PROJECT_ROOT / 'qwen', LLAVA_BACKBONE, SHARED_KVBENCH,
              WAREHOUSE / 'code/horizonkv/runtime/ppu-py310-overlay-20261015']
    environment['PYTHONPATH'] = os.pathsep.join(map(str, paths))
    processes = []
    try:
        for index, gpu in enumerate(gpus):
            shard = data[index*len(data)//len(gpus):(index+1)*len(data)//len(gpus)]
            annotation = output / 'annotations' / f'shard_{index:02}.json'
            save_exact(annotation, shard)
            destination = output / f'shard_{index:02}'
            destination.mkdir(exist_ok=True)
            command = [sys.executable, '-m', 'methods.h2svc.worker', '--model', args.model,
                       '--annotation', str(annotation), '--output', str(destination),
                       '--head-map', str(head_map), '--gpu', gpu]
            if args.accept_equivalent_choices:
                command.append('--accept-equivalent-choices')
            child = environment.copy()
            child.update(CUDA_VISIBLE_DEVICES=gpu, SPARSEMM_GPU_INDEX=gpu)
            with (destination / 'run.log').open('a') as log:
                process = subprocess.Popen(command, env=child, stdout=log,
                                           stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            processes.append(process)
            print('START', index, 'GPU', gpu, 'PID', process.pid, flush=True)
        codes = [process.wait() for process in processes]
    except BaseException:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        raise
    print('EXIT CODES', codes, flush=True)
    return int(any(codes))


if __name__ == '__main__':
    sys.exit(main())
