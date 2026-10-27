"""Bounded, evaluation-disjoint native attention calibration for LU budgets."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

WORKSPACE = Path(__file__).resolve().parents[1]
PACKAGE = WORKSPACE / 'sparsemm_lukv'
DEFAULT_OUTPUT = PACKAGE / 'results/calibration/native_attention10v_20261014'
DEFAULT_ASSETS = PACKAGE / 'assets/attention_native10v_20261014'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def model_fingerprints(model_path):
    names = ('config.json', 'generation_config.json', 'preprocessor_config.json',
             'tokenizer_config.json', 'tokenizer.json', 'model.safetensors.index.json')
    return {name: sha256(model_path / name) for name in names if (model_path / name).is_file()}


def collection_sources():
    names = ['sparsemm_lukv/calibration.py', 'sparsemm_lukv/calibration_data.py',
             'sparsemm_lukv/profile.py', 'sparsemm_lukv/run_calibration.py',
             'sparsemm_protokv/cache.py', 'sparsemm_protokv/config.py',
             'sparsemm_protokv/streaming.py', 'sparsemm_protokv/qwen2_5.py',
             'sparsemm_protokv/qwen2.py', 'sparsemm_protokv/model_loader.py',
             'sparsemm_protokv/monkeypatch.py', 'sparsemm_protokv/models/qwen2_5_vl_stream.py',
             'sparsemm_baselines/core.py', 'sparsemm_baselines/cache.py',
             'sparsemm_baselines/adapter.py', 'sparsemm_baselines/run_streamingbench.py']
    return {name: sha256(WORKSPACE / name) for name in names}


def main(argv=None):
    from sparsemm_lukv.calibration_data import DEFAULT_SOURCE, DEFAULT_EXCLUDE, select_calibration
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--exclude', type=Path, default=DEFAULT_EXCLUDE)
    parser.add_argument('--num-videos', type=int, default=10)
    parser.add_argument('--max-time', type=float, default=120)
    parser.add_argument('--model-path', type=Path, default=WORKSPACE / 'hermes/models/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--assets-dir', type=Path, default=DEFAULT_ASSETS)
    parser.add_argument('--run-calibration', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if not (args.run_calibration or args.dry_run):
        parser.error('Choose --run-calibration for the bounded authorized calibration or --dry-run')
    selected, selection = select_calibration(args.source, args.exclude, args.num_videos, args.max_time)
    if not selected:
        raise ValueError('No disjoint calibration observations')
    out, assets, model_path = args.output_dir.resolve(), args.assets_dir.resolve(), args.model_path.resolve()
    from sparsemm_protokv.model_loader import model_type
    is_llava = model_type(model_path) == 'llava_onevision'
    if is_llava:
        selection['token_window_protocol'] = dict(selection['token_window_protocol'],
            model_family='LLaVA-OneVision', video_resolution='resize_and_pad_384x384',
            temporal_patch_frames=1)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        raise FileExistsError('Calibration never overwrites existing evidence; choose a new output directory')
    if not args.dry_run and any(assets.glob('*.json')):
        raise FileExistsError('Calibration never overwrites existing profiles; choose a new assets directory')
    protocol = {**selection['token_window_protocol'], 'window_size': 32, 'kernel_size': 5,
                'pooling': 'avgpool', 'gqa_reduce': 'mean', 'score_precision': 'float32',
                'native_resolution': not is_llava, 'utility_mode': 'projected',
                'value_norm_normalization': 'layer_sum_over_query_heads_and_persistent_context',
                'encode_chunk_size': 16, 'max_new_tokens': 16, 'temperature': 0,
                'repetition_penalty': 1.1, 'initial_text_tokens': 'all_protected',
                'history_profile_domain': 'visual_history_excluding_recent_window',
                'per_head_minimum': 'protected_recent_window',
                'paper_1percent_floor': False,
                'paper_1percent_floor_reason': 'fixed global video budgets can be below 1percent of growing uncompressed T'}
    selected_sha = hashlib.sha256(json.dumps(selected, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    manifest = {'status': 'prepared', 'started_utc': datetime.now(timezone.utc).isoformat(),
                'selector': 'sparsemm_attention', 'utility_mode': 'projected',
                'selected_videos': selected, 'selected_sha256': selected_sha,
                'selection': selection, 'token_window_protocol': protocol,
                'source_sha256': selection['source_hashes'], 'model_path': str(model_path),
                'model_files_sha256': model_fingerprints(model_path),
                'observations': [], 'videos': [], 'full_benchmarks_paused': True}
    write_json(out / 'annotations.json', selected)
    write_json(out / 'selection.json', selection)
    write_json(out / 'protocol.json', protocol)
    if args.dry_run:
        write_json(out / 'preflight.json', manifest)
        print(json.dumps({'videos':len(selected),'questions':selection['selected_question_count'],
                          'selected_sha256':selected_sha,'selection':selection}, indent=2), flush=True)
        return 0
    for path in (WORKSPACE / 'sparsemm_protokv/runtime_deps', WORKSPACE / 'sparsemm_baselines/upstream/lmms-eval'):
        sys.path.insert(0, str(path))
    os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MALLOC_ARENA_MAX='2',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', DECORD_EOF_RETRY_MAX='40960')
    import numpy as np
    import torch
    import transformers
    from sparsemm_baselines.run_streamingbench import analyze_video, gpu_lease, write_results
    from sparsemm_protokv.models.qwen2_5_vl_stream import BatchScopedVideoReader
    from sparsemm_lukv.calibration import load_calibration_model
    from sparsemm_lukv.profile import build_profile, allocate_profile
    torch.set_num_threads(1)
    torch.manual_seed(20261014)
    torch.backends.cuda.matmul.allow_tf32 = False
    manifest['code_sha256'] = collection_sources()
    manifest['runtime'] = {'torch': torch.__version__, 'transformers':transformers.__version__,
                           'python': sys.version, 'dtype':'bfloat16'}
    write_json(out / 'manifest.json', manifest)
    (out / 'run.status').write_text('running\n')
    started = time.perf_counter()
    rows = []
    try:
        with gpu_lease(out):
            model = load_calibration_model(str(model_path), utility_mode='projected', validate_state=False)
            model.timing_enabled = True
            model_load = time.perf_counter() - started
            original_qa = model.question_answering
            def observe_qa(input_text, *positional, **keyword):
                video = current_video[0]
                question = video['conversations'][model._active_question_index]
                model.begin_observation()
                try:
                    answer = original_qa(input_text, *positional, **keyword)
                    observation = model.end_observation()
                except BaseException:
                    model.kv_cache.abort_observation()
                    raise
                name = f"v{video['video_id']}_q{question['source_question_index']}"
                path = out / f'{name}.npz'
                arrays = {key: observation[key].numpy() for key in
                          ('ranking','score','utility_attention','utility_projected')}
                if (not all(np.isfinite(value).all() for value in arrays.values())
                        or (arrays['utility_projected'] < 0).any() or arrays['utility_projected'].sum() <= 0):
                    raise AssertionError('Invalid calibrated attention/oracle arrays')
                np.savez_compressed(path, **arrays)
                metadata = {**observation['metadata'], 'video_id':video['video_id'],
                            'question':question, 'pred_answer':answer,
                            'source_question_index':question['source_question_index']}
                write_json(out / f'{name}.json', metadata)
                manifest['observations'].append({
                    'video_id':video['video_id'], 'source_question_index':question['source_question_index'],
                    'end_time':question['end_time'], 'path':str(path), 'sha256':sha256(path),
                    'metadata_path':str(out/f'{name}.json'), 'metadata_sha256':sha256(out/f'{name}.json'),
                    'visual_tokens':arrays['utility_projected'].shape[1],
                    'qa_restore_passed':metadata['query_restored'],
                    'additional_scoring_forwards':metadata['additional_scoring_forwards']})
                write_json(out/'manifest.json', manifest)
                print(f"CALIBRATION_OBSERVATION {name} tokens={arrays['utility_projected'].shape[1]} restored={metadata['query_restored']}", flush=True)
                return answer
            model.question_answering = observe_qa
            current_video = [None]
            for video in selected:
                current_video[0] = video
                wall = time.perf_counter()
                print(f"CALIBRATION_VIDEO_START id={video['video_id']}",flush=True)
                video_rows = analyze_video(model,video,.5,16,torch,reader_factory=BatchScopedVideoReader)
                rows.extend(video_rows)
                manifest['videos'].append({'video_id':video['video_id'],'questions':len(video_rows),
                                           'wall_seconds':time.perf_counter()-wall})
                write_results(out/'calibration_full_cache.csv',rows)
                write_json(out/'manifest.json',manifest)
                print(f"CALIBRATION_VIDEO_DONE id={video['video_id']} seconds={manifest['videos'][-1]['wall_seconds']:.3f}",flush=True)
            manifest['runtime_checks'] = model.validation_checks
            manifest['timing'] = {'model_load_seconds':model_load,'components':model.timings,
                                  'gpu_collection_seconds':time.perf_counter()-started}
            manifest['peak_gpu_allocated_gib'] = torch.cuda.max_memory_allocated()/2**30
            del model
            torch.cuda.empty_cache()
        if collection_sources()!=manifest['code_sha256']:
            raise RuntimeError('Calibration source changed during collection')
        expected = selection['selected_question_count']
        if len(manifest['observations'])!=expected or len(manifest['videos'])!=len(selected):
            raise RuntimeError('Calibration did not complete the exact selected observations')
        if manifest['runtime_checks']['failures'] or not all(v['qa_restore_passed'] for v in manifest['observations']):
            raise RuntimeError('Calibration state restoration failed')
        manifest.update(status='complete',finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(out/'manifest.json',manifest)
        assets.mkdir(parents=True,exist_ok=True)
        calibration = {'selector':'sparsemm_attention','utility_mode':'projected',
                       'token_window_protocol':protocol,'source_sha256':selection['source_hashes'],
                       'exclude_annotation':str(args.exclude.resolve()),'selected_videos':selected,
                       'selected_sha256':selected_sha,'manifest_path':str(out/'manifest.json'),
                       'manifest_sha256':sha256(out/'manifest.json'),'observations':manifest['observations'],
                       'model_path':str(model_path),'model_files_sha256':manifest['model_files_sha256'],
                       'code_sha256':manifest['code_sha256']}
        observations=[]
        for record in manifest['observations']:
            with np.load(record['path']) as saved:
                observations.append({'video_id':record['video_id'],'ranking':saved['ranking'],
                                     'utility':saved['utility_projected']})
        profile=build_profile(observations,28,4,window_size=32)
        profile['calibration']=calibration
        profile['adaptations']=['physical GQA mean aggregation','native video calibration',
                                'history-only lookup with protected 32-token window',
                                'all initial text protected outside visual capacity',
                                'no 1percent minimum to support fixed-budget long video streams',
                                '16 answer-prediction positions maximum']
        write_json(assets/'profile.json',profile)
        reference_T = int(np.median([r['visual_tokens'] for r in manifest['observations']]))
        budgets, allocation=allocate_profile(profile,reference_T,1000,32)
        with (assets/f'budget_k1000_at_T{reference_T}.csv').open('w',newline='') as handle:
            writer=csv.writer(handle)
            writer.writerow(['layer','kv_head','budget'])
            for layer in range(28):
                for head in range(4): writer.writerow([layer,head,int(budgets[layer,head])])
        write_json(assets/'reference_allocation.json',allocation)
        (assets/'profile.sha256').write_text(sha256(assets/'profile.json')+'  profile.json\n')
        snapshot=out/'source_snapshot'
        for name in manifest['code_sha256']:
            destination=snapshot/name
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(WORKSPACE/name,destination)
        summary={'status':'complete','videos':len(selected),'observations':expected,
                 'utility_mode':'projected','selector':'sparsemm_attention',
                 'profile':str(assets/'profile.json'),'profile_sha256':sha256(assets/'profile.json'),
                 'reference_T':reference_T,'reference_head_min':int(budgets.min()),
                 'reference_head_max':int(budgets.max()),'reference_total_slots':int(budgets.sum()),
                 'total_seconds':time.perf_counter()-started,
                 'peak_gpu_allocated_gib':manifest['peak_gpu_allocated_gib'],
                 'full_benchmarks_paused':True}
        write_json(out/'summary.json',summary)
        (out/'run.status').write_text('complete\n')
        print(json.dumps(summary,indent=2),flush=True)
    except BaseException as error:
        (out/'run.status').write_text(f'failed {type(error).__name__}: {error}\n')
        raise
    return 0


if __name__=='__main__':
    raise SystemExit(main())
