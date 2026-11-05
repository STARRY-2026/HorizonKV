#!/usr/bin/env python3
"""Unified OVO-Bench / StreamingBench launcher using pinned model adapters."""
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from methods.registry import (PROJECT_ROOT, METHODS_ROOT, SHARED_KVBENCH, LLAVA_BACKBONE, SPARSEMM_SCORES, LUKV_PROFILE, llava_impl, qwen_impl)
ROOT=Path('/mnt/aifordata2026-p/common/aifordata/541707')
REL=ROOT/'code/horizonkv/releases'
OLD=REL/'20261102-ovobench-multimodel-unified-v1'
SOURCE=REL/'20261022-ovobench-adapter-v1/data/streamingbench/streamingbench_realtime_local.json'
PROFILE=LUKV_PROFILE
VIDEO=ROOT/'dataset/StreamingBench/videos'
METHODS=('adakv','pyramidkv','sparsemm','lukv','hermes','fullkv','horizonkv-egate')
LLAVA={name:str(llava_impl(name)) for name in ('adakv','pyramidkv','sparsemm','lukv','hermes','fullkv')}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_exact(p,s):
    if p.exists() and p.read_text()!=s: raise RuntimeError('Resume configuration changed: '+str(p))
    if not p.exists(): p.write_text(s)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark',required=True,choices=['ovobench','streamingbench'])
    p.add_argument('--model',required=True,choices=['qwen25vl','llavaov'])
    p.add_argument('--method',required=True,choices=METHODS)
    p.add_argument('--gpus',default='0'); p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--annotation',type=Path); p.add_argument('--max-videos',type=int)
    p.add_argument('--cohort',choices=['heldout','full'],default='heldout')
    p.add_argument('--allow-calibration-overlap',action='store_true')
    p.add_argument('--video-root',type=Path,default=VIDEO); p.add_argument('--dry-run',action='store_true')
    p.add_argument('--input-profile',choices=['original','dscache'],default='original')
    p.add_argument('--accept-equivalent-choices',action='store_true',help='Include questions with duplicate answer text and score all equivalent options as correct')
    a=p.parse_args(); gs=a.gpus.split(',')
    if any(not x.isdigit() for x in gs) or len(gs)!=len(set(gs)): p.error('gpus must be unique IDs, e.g. 0,1')
    if a.max_videos is not None and a.max_videos<1: p.error('max-videos must be positive')
    out=a.output_dir.resolve()
    if a.benchmark=='ovobench':
        if a.input_profile=='dscache' and (a.model!='llavaov' or a.method=='horizonkv-egate'):
            p.error('OVO DSCache input currently supports LLaVA baseline methods only')
        cmd=[sys.executable,str(Path(__file__).with_name('ovo_run.py')),'--model',a.model,
             '--method',a.method,'--gpus',a.gpus,'--output-dir',str(out),'--input-profile',a.input_profile]
        if a.annotation: cmd+=['--annotation',str(a.annotation.resolve())]
        if a.max_videos is not None: cmd+=['--max-videos',str(a.max_videos)]
        if a.dry_run: cmd+=['--dry-run']
        print('OVO delegate:',json.dumps(cmd),flush=True)
        return subprocess.call(cmd)
    if a.model=='llavaov' and a.method in ('sparsemm','lukv','hermes','fullkv') and a.input_profile!='dscache':
        p.error('additional baseline adapters require --input-profile dscache; use the original release for old settings')
    if a.input_profile=='dscache' and not (a.benchmark=='streamingbench' and a.model=='llavaov' and a.method in ('adakv','pyramidkv','sparsemm','lukv','hermes','fullkv')):
        p.error('dscache input profile currently supports StreamingBench LLaVA baselines only; Qwen variable-token support is not implemented')
    if a.benchmark != 'streamingbench' or not ((a.model=='qwen25vl' and a.method=='horizonkv-egate') or (a.model=='llavaov' and a.method in ('adakv','pyramidkv','sparsemm','lukv','hermes','fullkv'))):
        p.error('1fps pilot supports StreamingBench Qwen EGate and LLaVA baselines only')
    if a.method=='horizonkv-egate' and (a.benchmark!='streamingbench' or a.model!='qwen25vl'):
        p.error('horizonkv-egate currently supports only streamingbench + qwen25vl; OVO and LLaVA are not validated')
    if a.benchmark=='ovobench':
        cmd=[sys.executable,str(OLD/'run.py'),'--model',a.model,'--method',a.method,'--gpus',a.gpus,'--output-dir',str(out)]
        if a.annotation: cmd+=['--annotation',str(a.annotation.resolve())]
        if a.max_videos: cmd+=['--max-videos',str(a.max_videos)]
        if a.dry_run: cmd+=['--dry-run']
        print('OVO delegate:',json.dumps(cmd),flush=True); return subprocess.call(cmd)
    anno=(a.annotation or SOURCE).resolve(); raw=json.loads(anno.read_text()); profile=json.loads(PROFILE.read_text())
    cal=profile['calibration']['selected_videos']
    def name(x): return Path(x if isinstance(x,str) else x.get('video_path',x.get('path',''))).name
    calibration={name(x) for x in cal}
    if len(calibration)!=50 or '' in calibration: raise RuntimeError('Unexpected calibration media list')
    if len({str(x['video_id']) for x in raw})!=len(raw): raise ValueError('Duplicate video IDs')
    overlap=[name(x['video_path']) for x in raw if name(x['video_path']) in calibration]
    if a.cohort=='heldout': data=[x for x in raw if name(x['video_path']) not in calibration]
    else:
        data=raw
        if a.method=='lukv' and overlap and not a.allow_calibration_overlap: raise ValueError('LU-KV full cohort overlaps calibration; use heldout or explicitly pass --allow-calibration-overlap')
    # Ambiguous identical correct options cannot be uniquely scored by the frozen evaluator.
    excluded=[]; cleaned=[]
    for v in data:
        item=dict(v); questions=[]
        for original in v['conversations']:
            q=dict(original); choices=q['choices']; answer=q['answer']
            if choices.count(answer)>1:
                if a.accept_equivalent_choices and a.model=='llavaov' and a.method in ('sparsemm','lukv','hermes','fullkv') and a.input_profile=='dscache':
                    q['correct_choice']='ABCDEFGH'[choices.index(answer)]
                    questions.append(q)
                    continue
                excluded.append({'video_id':v['video_id'],'question':q['question'],'reason':'correct answer appears in multiple options'})
                continue
            if answer not in choices: raise ValueError('Answer missing from choices')
            if len(set(choices))!=len(choices) and not q.get('correct_choice'):
                q['correct_choice']='ABCDEFGH'[choices.index(answer)]
            questions.append(q)
        if questions:
            item['conversations']=questions; cleaned.append(item)
    data=cleaned
    if a.max_videos is not None:
        if a.max_videos>len(data): raise ValueError('max-videos exceeds cohort')
        data=data[:a.max_videos]
    if not data: raise ValueError('No videos selected')
    for video in data:
        orig=Path(video['video_path']); new=a.video_root/orig.name
        if not orig.is_file() and not new.is_file(): raise FileNotFoundError(f'Media missing: {orig} or {new}')
        video['video_path']=str(orig if orig.is_file() else new)
        if not video.get('conversations'): raise ValueError('Video without questions')
    gs=gs[:min(len(gs),len(data))]
    config=dict(input_profile=a.input_profile,accept_equivalent_choices=a.accept_equivalent_choices,version=1,benchmark=a.benchmark,model=a.model,method=a.method,source=str(anno),source_sha256=sha(anno),cohort=a.cohort,excluded_ambiguous_questions=excluded,profile_sha256=sha(PROFILE),calibration_overlap_selected=sum(name(x['video_path']) in calibration for x in data),videos=len(data),questions=sum(len(x['conversations']) for x in data),gpus=gs,fps=1.0,chunk_frames=(4 if a.model=='qwen25vl' else 1),visual_budget=None if a.method=='fullkv' else 500,dtype='float16',detailed_cache_trace=False,calibration='Qwen2.5-VL cross-model transfer' if a.model=='llavaov' and a.method in ('sparsemm','lukv') else None,code_sha256={x:sha(Path(__file__).with_name(x)) for x in ('run.py','worker.py')})
    if a.method=='horizonkv-egate':
        egate_dir=Path(__file__).with_name('egate')
        config.update(chunk_frames=4, frame_encoding='fixed896 per-image; concatenate available frames into language model prefill blocks, up to 4', history='direct per-head recent KV transfer; visual-query SnapKV-style selection', recent_window='current non-overlapping batch; one or four frames per head', egate_allocation_sha256=sha(egate_dir/'assets/per_head_allocation.csv'), egate_code_sha256={x:sha(egate_dir/x) for x in ('worker.py','pae_variable_recent.py','egate_protocol.py','snap_stream.py')}, limitations='Experimental non-overlapping four-frame batch streaming changes cache semantics; not parity-verified; 1fps only: DSCache frame-resolution and cache-construction settings NOT implemented; not parity verified')
    if a.model=='llavaov' and a.method in ('adakv','pyramidkv'):
        config['llava_evaluator_sha256']=sha(Path(__file__).with_name('llava_1fps_evaluate.py'))
    if a.model=='llavaov' and a.input_profile=='dscache' and a.method in ('sparsemm','lukv','hermes','fullkv'):
        tag='sparsemm' if a.method in ('sparsemm','lukv') else a.method
        config['dscache_adapter_sha256']=sha(Path(__file__).with_name('llava_dscache_extra.py'))
        config['llava_evaluator_sha256']=sha(Path(__file__).with_name('llava_dscache_'+tag+'_evaluate.py'))
    print(json.dumps(config,indent=2,ensure_ascii=False),flush=True)
    if a.dry_run: return 0
    if out.exists() and not (out/'manifest.json').exists() and any(out.iterdir()): raise RuntimeError('Nonempty output directory without manifest')
    out.mkdir(parents=True,exist_ok=True); (out/'annotations').mkdir(exist_ok=True)
    write_exact(out/'manifest.json',json.dumps(config,indent=2,ensure_ascii=False)+'\n')
    env=os.environ.copy(); env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',DECORD_EOF_RETRY_MAX='40960')
    paths=[Path('/tmp/llava-ovo-transformers440')] if a.model=='llavaov' else ([Path('/tmp/egate-transformers4524')] if a.method=='horizonkv-egate' else [])
    if a.method=='horizonkv-egate' and not (paths[0]/'transformers/__init__.py').is_file(): raise RuntimeError('Missing pinned /tmp/egate-transformers4524 runtime; see README')
    paths += [LLAVA_BACKBONE,SHARED_KVBENCH,METHODS_ROOT/'lukv',METHODS_ROOT/'sparsemm',qwen_impl('hermes'),ROOT/'code/horizonkv/runtime/ppu-py310-overlay-20261015']
    env['PYTHONPATH']=os.pathsep.join(map(str,[PROJECT_ROOT]+paths)); procs=[]
    for i,gpu in enumerate(gs):
        shard=data[i*len(data)//len(gs):(i+1)*len(data)//len(gs)]
        annot=out/'annotations'/f'shard_{i:02}.json'; write_exact(annot,json.dumps(shard,ensure_ascii=False))
        dest=out/f'shard_{i:02}'; dest.mkdir(exist_ok=True)
        if a.method=='horizonkv-egate':
            cmd=[sys.executable,str(Path(__file__).with_name('egate')/'worker.py'),'--annotation',str(annot),'--output',str(dest)]
        else:
            cmd=[sys.executable,str(Path(__file__).with_name('worker.py')),'--model',a.model,'--method',a.method,'--annotation',str(annot),'--output',str(dest),'--gpu',gpu]
        childenv=env.copy(); childenv.update(CUDA_VISIBLE_DEVICES=gpu,SPARSEMM_GPU_INDEX=gpu,HORIZON_INPUT_PROFILE=a.input_profile)
        with (dest/'run.log').open('a') as log: proc=subprocess.Popen(cmd,env=childenv,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        procs.append(proc); print('START',i,'GPU',gpu,'PID',proc.pid,'VIDEOS',len(shard),flush=True)
    try:
        codes=[proc.wait() for proc in procs]
    except BaseException:
        for proc in procs:
            if proc.poll() is None: proc.terminate()
        for proc in procs: proc.wait()
        raise
    print('EXIT CODES',codes,flush=True); return int(any(codes))
if __name__=='__main__': sys.exit(main())
