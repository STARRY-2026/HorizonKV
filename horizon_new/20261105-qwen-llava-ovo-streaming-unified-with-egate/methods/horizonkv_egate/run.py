"""HorizonKV-EGate sharded runner; called through the unified root launcher."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from methods.registry import PROJECT_ROOT, SHARED_KVBENCH, LLAVA_BACKBONE, LUKV_PROFILE
from .data import save_exact, validate_annotations
from .protocol import load_allocation

WAREHOUSE = Path('/mnt/aifordata2026-p/common/aifordata/541707')
RELEASES = WAREHOUSE / 'code/horizonkv/releases'
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark', required=True, choices=['streamingbench', 'ovobench'])
    p.add_argument('--model', required=True, choices=['qwen25vl', 'llavaov'])
    p.add_argument('--method', required=True, choices=['horizonkv-egate'])
    p.add_argument('--gpus', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--annotation', type=Path)
    p.add_argument('--max-videos', type=int)
    p.add_argument('--cohort', default='heldout', choices=['heldout', 'full'])
    p.add_argument('--allow-calibration-overlap', action='store_true')
    p.add_argument('--accept-equivalent-choices', action='store_true')
    p.add_argument('--visual-input', choices=['dscache448-video'])
    p.add_argument('--sample-fps', type=float, default=1)
    p.add_argument('--chunk-frames', type=int, default=2)
    p.add_argument('--input-profile', choices=['dscache'])
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.sample_fps != 1 or a.chunk_frames != 2:
        p.error('Unified EGate requires 1 fps with <=2 Qwen input frames')
    stream = a.benchmark == 'streamingbench'
    default = ('data/streamingbench/streamingbench_realtime_local.json' if stream else
               'data/ovobench/ovobench_realtime_backward_warehouse.json')
    source = (a.annotation or RELEASES / '20261022-ovobench-adapter-v1' / default).resolve()
    raw = json.loads(source.read_text())
    calibration = json.loads(LUKV_PROFILE.read_text())['calibration']['selected_videos']
    def name(x):
        return Path(x if isinstance(x, str) else x.get('video_path', x.get('path', ''))).name
    cal_names = {name(x) for x in calibration}
    if len(cal_names) != 50 or '' in cal_names:
        raise ValueError('Invalid reference held-out cohort')
    excluded = []
    data = []
    for original in raw:
        if stream and a.cohort == 'heldout' and name(original['video_path']) in cal_names:
            continue
        v = dict(original)
        v['conversations'] = []
        for original_q in original['conversations']:
            q = dict(original_q)
            if stream and q['choices'].count(q['answer']) > 1 and not a.accept_equivalent_choices:
                excluded.append(dict(video_id=v['video_id'], question=q['question']))
                continue
            if not q.get('correct_choice') and q.get('answer') in q.get('choices', []):
                q['correct_choice'] = 'ABCDEFGH'[q['choices'].index(q['answer'])]
            v['conversations'].append(q)
        if not v['conversations']:
            continue
        media = Path(v['video_path'])
        if not media.is_absolute():
            media = source.parent / media
        if not media.is_file() and stream:
            media = WAREHOUSE / 'dataset/StreamingBench/videos' / media.name
        v['video_path'] = str(media.resolve())
        data.append(v)
    if a.max_videos is not None:
        if not 1 <= a.max_videos <= len(data):
            p.error('Invalid max-videos')
        data = data[:a.max_videos]
    validate_annotations(data)
    overlap = sum(name(v['video_path']) in cal_names for v in data) if stream else 0
    if stream and overlap and not a.allow_calibration_overlap:
        p.error('Frozen allocation calibration is not repeated here; full cohort overlap requires --allow-calibration-overlap')
    allocation, digest = load_allocation()
    gpus = a.gpus.split(',')[:min(len(a.gpus.split(',')), len(data))]
    if any(not g.isdigit() for g in gpus) or len(set(map(int, gpus))) != len(gpus):
        p.error('Invalid GPU IDs')
    config = dict(schema_version=1, backend='egate', benchmark=a.benchmark, model=a.model, method=a.method,
                  experiment_arm='pae_online_e_gate_k500', source=str(source), source_sha256=sha(source),
                  cohort=a.cohort if stream else None, gpus=gpus, videos=len(data),
                  questions=sum(len(v['conversations']) for v in data), fps=1,
                  ordinary_chunk_frames=2 if a.model == 'qwen25vl' else 1, memory_batch_frames=4,
                  visual_budget_per_head=500, prefix_outside_budget=True, dtype='float16',
                  ordinary_pixels=448**2 if a.model == 'qwen25vl' else [384, 384],
                  latest_pixels=896**2 if a.model == 'qwen25vl' else [384, 384],
                  llava_visual_tokens=[196, 729] if a.model == 'llavaov' else None,
                  allocation_sha256=digest, one_frame_heads=sum(r['recent_group']=='one_frame' for r in allocation.values()),
                  calibration_note='Frozen original Qwen fixed896 allocation transferred to current inputs; not recalibrated. LLaVA allocation is cross-backbone transfer.',
                  reference_heldout_overlap_videos=overlap, excluded_ambiguous_questions=excluded,
                  accept_equivalent_choices=a.accept_equivalent_choices,
                  detailed_cache_trace=False, cpu_video_feature_archive=False,
                  question_policy='Always execute recent-only + original-options history pass; gate selects source; constrained first-token letter decoding.',
                  input_note='Reuse unified visual sizes and latest-frame replay. EGate keeps its own independent branches, compact RoPE and four-frame memory boundaries. Qwen paired units may split at a memory boundary; DSCache temporal parity is not claimed.',
                  code_sha256={str(f.relative_to(PROJECT_ROOT)): sha(f) for f in sorted(HERE.rglob('*.py'))},
                  shared_cache_sha256=sha(SHARED_KVBENCH / 'sparsemm_baselines/cache.py'))
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)
    if a.dry_run:
        return 0
    output = a.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not (output / 'manifest.json').exists():
        raise ValueError('Nonempty output without manifest')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'annotations').mkdir(exist_ok=True)
    save_exact(output / 'manifest.json', config)
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
               PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', DECORD_EOF_RETRY_MAX='40960')
    paths = [Path('/tmp/llava-ovo-transformers440')] if a.model == 'llavaov' else [WAREHOUSE / 'code/horizonkv/runtime/qwen-py310-deps-20261016']
    paths += [PROJECT_ROOT, PROJECT_ROOT / 'qwen', LLAVA_BACKBONE, SHARED_KVBENCH,
              WAREHOUSE / 'code/horizonkv/runtime/ppu-py310-overlay-20261015']
    env['PYTHONPATH'] = os.pathsep.join(map(str, paths))
    processes = []
    try:
        for i, gpu in enumerate(gpus):
            shard = data[i * len(data) // len(gpus):(i + 1) * len(data) // len(gpus)]
            annotation = output / 'annotations' / f'shard_{i:02}.json'
            save_exact(annotation, shard)
            dest = output / f'shard_{i:02}'
            dest.mkdir(exist_ok=True)
            command = [sys.executable, '-m', 'methods.horizonkv_egate.worker', '--model', a.model,
                       '--annotation', str(annotation), '--output', str(dest), '--gpu', gpu]
            if a.accept_equivalent_choices:
                command.append('--accept-equivalent-choices')
            child_env = env.copy()
            child_env.update(CUDA_VISIBLE_DEVICES=gpu, SPARSEMM_GPU_INDEX=gpu)
            with (dest / 'run.log').open('a') as log:
                process = subprocess.Popen(command, env=child_env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            processes.append(process)
            print('START', i, 'GPU', gpu, 'PID', process.pid, flush=True)
        codes = [proc.wait() for proc in processes]
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
