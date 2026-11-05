"""Pure planning and explicit result provenance for the unified CLI."""
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

from .registry import BACKENDS, BENCHMARKS, INPUT_PROFILES, LLAVA_EQUIVALENT_CHOICES, METHODS, MODELS


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def make_plan(args, root):
    if args.model not in MODELS or args.method not in METHODS or args.benchmark not in BENCHMARKS:
        raise ValueError('Unsupported model, method or benchmark')
    gpu_ids = args.gpus.split(',')
    if any(not gpu.isdigit() for gpu in gpu_ids) or len(set(map(int, gpu_ids))) != len(gpu_ids):
        raise ValueError('gpus must contain unique nonnegative integer IDs, e.g. 0,1')
    if args.max_videos is not None and args.max_videos < 1:
        raise ValueError('max-videos must be positive')
    stream = args.benchmark == 'streamingbench'
    if not stream and (args.cohort != 'heldout' or args.allow_calibration_overlap or args.accept_equivalent_choices):
        raise ValueError('cohort/calibration/choice options apply to StreamingBench only')
    if stream and args.cohort == 'full' and args.method in ('lukv', 'horizonkv-egate', 'h2svc') and not args.allow_calibration_overlap:
        raise ValueError('Calibrated method full cohort requires explicit --allow-calibration-overlap')
    if stream and args.model == 'qwen25vl' and args.cohort == 'full' and not args.accept_equivalent_choices:
        raise ValueError('Full Qwen StreamingBench requires --accept-equivalent-choices')
    if stream and args.model == 'llavaov' and args.accept_equivalent_choices and args.method not in LLAVA_EQUIVALENT_CHOICES:
        raise ValueError('LLaVA AdaKV/PyramidKV do not support equivalent-choice scoring in the pinned adapters; this option would silently exclude the duplicate-answer question')
    backend = ('qwen_full2495' if stream and args.accept_equivalent_choices else 'qwen') if args.model == 'qwen25vl' else 'llava'
    if args.method in ('horizonkv-egate', 'h2svc'):
        backend = 'egate' if args.method == 'horizonkv-egate' else 'h2svc'
    head_map = getattr(args, 'head_map', None)
    if head_map and args.method != 'h2svc':
        raise ValueError('--head-map applies to h2svc only')
    output = str(Path(args.output_dir).resolve())
    config = dict(benchmark=args.benchmark, model=args.model, method=args.method,
                  gpus=','.join(str(int(g)) for g in gpu_ids), annotation=str(Path(args.annotation).resolve()) if args.annotation else None,
                  max_videos=args.max_videos, cohort=args.cohort if stream else None,
                  allow_calibration_overlap=args.allow_calibration_overlap,
                  accept_equivalent_choices=args.accept_equivalent_choices,
                  input_profile=INPUT_PROFILES[args.model], visual_budget=None if args.method in ('fullkv', 'h2svc') else 500,
                  dtype='float16', detailed_cache_trace=False)
    if args.method == 'horizonkv-egate':
        method_root = Path(root)/'methods'/'horizonkv_egate'
        config['memory_batch_frames'] = 4
        config['allocation_recalibrated'] = False
        if method_root.exists():
            config['egate_sources'] = {str(f.relative_to(method_root)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(method_root.rglob('*')) if f.is_file() and (f.suffix in ('.py', '.csv'))}
    if args.method == 'h2svc':
        method_root = Path(root)/'methods'/'h2svc'
        frozen = Path(head_map).resolve() if head_map else method_root/'assets'/f'{args.model}_head_map.json'
        config['head_map'] = str(frozen)
        config['head_map_sha256'] = hashlib.sha256(frozen.read_bytes()).hexdigest() if frozen.is_file() else None
        config['frame_budgets'] = dict(instantaneous=4, short_term=16, episodic=64)
        config['qwen_eviction_unit'] = 'indivisible merged two-frame KV unit'
        config['h2svc_sources'] = {str(f.relative_to(method_root)): hashlib.sha256(f.read_bytes()).hexdigest()
                                   for f in sorted(method_root.rglob('*.py'))}
    source_manifest = Path(root)/'methods'/'SOURCE_MANIFEST.json'
    if source_manifest.is_file():
        config['method_sources_manifest_sha256'] = hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    command = [sys.executable, str(Path(root)/backend/'run.py'), '--benchmark', args.benchmark,
               '--model', args.model, '--method', args.method, '--gpus', config['gpus'], '--output-dir', output]
    if args.model == 'qwen25vl':
        command += ['--visual-input', 'dscache448-video', '--sample-fps', '1', '--chunk-frames', '2']
    else:
        command += ['--input-profile', 'dscache']
    if stream:
        command += ['--cohort', args.cohort]
        if args.allow_calibration_overlap:
            command += ['--allow-calibration-overlap']
        if (args.model == 'llavaov' or backend in ('egate', 'h2svc')) and args.accept_equivalent_choices:
            command += ['--accept-equivalent-choices']
    if config['annotation']:
        command += ['--annotation', config['annotation']]
    if args.method == 'h2svc':
        command += ['--head-map', config['head_map']]
    if args.max_videos is not None:
        command += ['--max-videos', str(args.max_videos)]
    return dict(schema_version=1, backend=backend, config=config, config_sha256=canonical_hash(config),
                output_dir=output, command=command)


def check_saved_plan(plan):
    saved = Path(plan['output_dir'])/'launcher_plan.json'
    if saved.exists():
        old = json.loads(saved.read_text())
        if old.get('backend') != plan['backend'] or old.get('config_sha256') != plan['config_sha256']:
            raise ValueError('Output directory was recorded with a different launcher configuration; use a new output directory')


def save_completed_plan(plan, root):
    """Save only after the backend returns successfully; never prepopulate its output."""
    output = Path(plan['output_dir'])
    if not any((output/name).is_file() for name in ('manifest.json', 'unified_manifest.json')):
        raise RuntimeError('Backend returned success without an output manifest; launcher provenance was not saved')
    record = dict(plan)
    record['code_sha256'] = {name: hashlib.sha256((Path(root)/name).read_bytes()).hexdigest()
                              for name in ('run.py', 'core/dispatch.py', 'core/registry.py', plan['backend']+'/run.py')}
    fd, temporary = tempfile.mkstemp(prefix='.launcher_plan-', suffix='.json', dir=output)
    try:
        with os.fdopen(fd, 'w') as file:
            json.dump(record, file, ensure_ascii=False, indent=2)
            file.write('\n')
        os.replace(temporary, output/'launcher_plan.json')
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def summary_backend(output, explicit=None):
    output = Path(output)
    saved = output/'launcher_plan.json'
    if saved.exists():
        backend = json.loads(saved.read_text())['backend']
        if backend not in BACKENDS:
            raise ValueError('Unknown backend in launcher_plan.json')
        if explicit and explicit != backend:
            raise ValueError('--backend conflicts with recorded launcher provenance')
        return backend
    if explicit:
        return explicit
    manifest = next((output/n for n in ('manifest.json', 'unified_manifest.json') if (output/n).exists()), None)
    if manifest is None:
        raise ValueError('No result manifest found')
    data = json.loads(manifest.read_text())
    if data.get('backend') in ('egate', 'h2svc'):
        return data['backend']
    if data.get('model') == 'llavaov':
        return 'llava'
    if data.get('model') == 'qwen25vl':
        if data.get('benchmark') == 'ovobench':
            return 'qwen'
        # A partial run may not have 2495 questions even when it uses this protocol.
        raise ValueError('Legacy Qwen StreamingBench output has no explicit scoring provenance; pass --backend qwen or --backend qwen_full2495 after checking its original command')
    raise ValueError('Unknown model in result manifest')
