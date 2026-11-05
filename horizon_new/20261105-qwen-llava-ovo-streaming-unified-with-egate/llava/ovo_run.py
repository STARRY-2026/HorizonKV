#!/usr/bin/env python3
"""Launch pinned OVO-Bench baselines by model and method; resumable per video."""
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from methods.registry import (PROJECT_ROOT, METHODS_ROOT, SHARED_KVBENCH, LLAVA_BACKBONE, SPARSEMM_SCORES, LUKV_PROFILE, llava_impl, qwen_impl)
ROOT=Path('/mnt/aifordata2026-p/common/aifordata/541707')
REL=ROOT/'code/horizonkv/releases'
ANNO=REL/'20261022-ovobench-adapter-v1/data/ovobench/ovobench_realtime_backward_warehouse.json'
QWEN_RUN=REL/'20261101-ovobench-unified-baselines-v1/run.py'
QWEN_METHODS=('adakv','pyramidkv','sparsemm','lukv','hermes','fullkv')
LLAVA_IMPL={'adakv':'20261101-llava-adakv-pyramidkv-prototype-v1','pyramidkv':'20261101-llava-adakv-pyramidkv-prototype-v1','sparsemm':'20261102-llava-sparsemm-lukv-transfer-v1','lukv':'20261102-llava-sparsemm-lukv-transfer-v1','hermes':'20261102-llava-hermes-ovobench-v1','fullkv':'20261102-llava-fullkv-ovobench-v1'}

def gpu_list(value):
    items=value.split(',')
    if not items or any(not x.isdigit() for x in items) or len(set(items))!=len(items):
        raise argparse.ArgumentTypeError('Use distinct GPU IDs, e.g. 0,1')
    return items

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save_exact(path,text):
    if path.exists() and path.read_text()!=text: raise RuntimeError(f'Existing file differs: {path}')
    if not path.exists(): path.write_text(text)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=('qwen25vl','llavaov'),required=True)
    p.add_argument('--method',choices=QWEN_METHODS,required=True)
    p.add_argument('--gpus',type=gpu_list,default=['0'])
    p.add_argument('--annotation',type=Path,default=ANNO)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--max-videos',type=int)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--input-profile',choices=['original','dscache'],default='original')
    args=p.parse_args()
    anno=args.annotation.resolve(); out=args.output_dir.resolve()
    if args.model=='qwen25vl' and args.input_profile=='dscache': p.error('Qwen DSCache input is not implemented')
    if args.model=='qwen25vl':
        cmd=[sys.executable,str(QWEN_RUN),'--method',args.method,'--annotation',str(anno),'--output-dir',str(out),'--gpus',','.join(args.gpus)]
        if args.max_videos is not None: cmd+=['--max-videos',str(args.max_videos)]
        if args.dry_run: cmd.append('--dry-run')
        print('Delegating to frozen Qwen OVO runner:',json.dumps(cmd),flush=True)
        return subprocess.call(cmd)
    if args.method not in LLAVA_IMPL: p.error('Unsupported LLaVA method')
    data=json.loads(anno.read_text())
    if not isinstance(data,list) or not data: p.error('annotation must be a nonempty list')
    count=len(data) if args.max_videos is None else args.max_videos
    if count<1 or count>len(data): p.error('max-videos must be within annotation length')
    data=data[:count]
    if len({str(v['video_id']) for v in data})!=count: p.error('duplicate video IDs')
    gpus=args.gpus[:min(len(args.gpus),count)]; impl=llava_impl(args.method)
    env=os.environ.copy()
    env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',DECORD_EOF_RETRY_MAX='40960')
    paths=[Path('/tmp/llava-ovo-transformers440'),LLAVA_BACKBONE,SHARED_KVBENCH,METHODS_ROOT/'lukv',METHODS_ROOT/'sparsemm',qwen_impl('hermes'),ROOT/'code/horizonkv/runtime/ppu-py310-overlay-20261015']
    env['PYTHONPATH']=os.pathsep.join(map(str,[PROJECT_ROOT,impl]+paths))
    env['HORIZON_INPUT_PROFILE']=args.input_profile
    names={'adakv':'llava_1fps_evaluate.py','pyramidkv':'llava_1fps_evaluate.py',
           'sparsemm':'llava_dscache_sparsemm_evaluate.py','lukv':'llava_dscache_sparsemm_evaluate.py',
           'hermes':'llava_dscache_hermes_evaluate.py','fullkv':'llava_dscache_fullkv_evaluate.py'}
    evaluator=(Path(__file__).with_name(names[args.method]) if args.input_profile=='dscache' else impl/'evaluate.py')
    sources=[evaluator,impl/'smoke.py']
    if args.input_profile=='dscache' and args.method in ('sparsemm','lukv','hermes','fullkv'):
        sources.append(Path(__file__).with_name('llava_dscache_extra.py'))
    if args.method=='hermes': sources += [impl/'hermes_adapter.py',qwen_impl('hermes')/'inference/llavaov_hermes.py']
    config={'version':1,'model':args.model,'method':args.method,'annotation':str(anno),'annotation_sha256':digest(anno),'max_videos':count,'gpus':gpus,'fps':0.5,'chunk_frames':16,'budget_visual_mean':None if args.method=='fullkv' else 500,'dtype':'float16','trace':False,'implementation':str(impl),'implementation_sha256':{str(x):digest(x) for x in sources},'calibration':'Qwen2.5-VL cross-model transfer' if args.method in ('sparsemm','lukv') else None,'note':'LLaVA streaming feature assembly is not validated as equivalent to the original full-video template'}
    config.update(input_profile=args.input_profile)
    if args.input_profile=='dscache':
        config.update(fps=1.0,chunk_frames=1,version=2,
                      visual_input='384x384; ordinary stride 2 (196 tokens); latest stride 1 (729 tokens)',
                      note='Experimental DSCache-style input; latest-frame replay uses each baseline cache, not DSCache cache construction; GPU validation pending.')
    print(json.dumps(config,indent=2,ensure_ascii=False),flush=True)
    if args.dry_run: return 0
    for path in (paths[0]/'transformers/__init__.py',Path('/tmp/siglip-so400m-patch14-384/config.json')):
        if not path.is_file(): raise FileNotFoundError(f'Missing ephemeral LLaVA runtime: {path}; restore it before launching')
    if out.exists() and not (out/'unified_manifest.json').exists() and any(out.iterdir()): raise RuntimeError('Output directory is nonempty and has no unified manifest')
    out.mkdir(parents=True,exist_ok=True); (out/'annotations').mkdir(exist_ok=True)
    save_exact(out/'unified_manifest.json',json.dumps(config,indent=2,ensure_ascii=False)+'\n')
    procs=[]
    for i,gpu in enumerate(gpus):
        shard=data[i*count//len(gpus):(i+1)*count//len(gpus)]
        annotation=out/'annotations'/f'shard_{i:02}.json'; save_exact(annotation,json.dumps(shard,ensure_ascii=False))
        dest=out/f'shard_{i:02}'; dest.mkdir(exist_ok=True)
        gpu_env=env.copy(); gpu_env['CUDA_VISIBLE_DEVICES']=gpu
        cmd=[sys.executable,str(evaluator),'--method',args.method,'--annotation',str(annotation),'--output',str(dest)]
        with (dest/'unified.log').open('a') as log:
            proc=subprocess.Popen(cmd,env=gpu_env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        procs.append((i,proc)); print('START',i,'GPU',gpu,'PID',proc.pid,'VIDEOS',len(shard),flush=True)
    failures=[]
    try:
        for i,proc in procs:
            rc=proc.wait(); print('EXIT',i,rc,flush=True)
            if rc: failures.append((i,rc))
    except BaseException:
        for _,proc in procs:
            if proc.poll() is None: proc.terminate()
        for _,proc in procs: proc.wait()
        raise
    if failures: print('Failed shards:',failures,file=sys.stderr); return 1
    return 0

if __name__=='__main__': sys.exit(main())
