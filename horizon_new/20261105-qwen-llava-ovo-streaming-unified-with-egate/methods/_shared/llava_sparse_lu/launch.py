import hashlib, json, os, subprocess, sys
from pathlib import Path

root=Path('/mnt/aifordata2026-p/common/aifordata/541707')
release=root/'code/horizonkv/releases/20261102-llava-sparsemm-lukv-transfer-v1'
annotation=root/'code/horizonkv/releases/20261022-ovobench-adapter-v1/data/ovobench/ovobench_realtime_backward_warehouse.json'
output=root/'experiments/horizonkv/llava-ovobench-sparsemm-lukv-qwenbudget-transfer-k500-20261102'
source=annotation.read_bytes(); data=json.loads(source); assert len(data)==1468
output.mkdir(parents=True,exist_ok=True); (output/'annotations').mkdir(exist_ok=True)
manifest={'annotation':str(annotation),'sha256':hashlib.sha256(source).hexdigest(),'total':len(data),'method_gpus':{'sparsemm':[0,1,2,3],'lukv':[4,5,6,7]},'fps':0.5,'chunk_frames':16,'budget_visual_per_head_average':500,'model':'llava-onevision-qwen2-7b-ov','calibration':'Qwen2.5-VL cross-model transfer; not LLaVA-calibrated'}
mp=output/'manifest.json'
if mp.exists() and json.loads(mp.read_text())!=manifest: raise RuntimeError('experiment manifest differs')
mp.write_text(json.dumps(manifest,indent=2)+'\n')
envbase=os.environ.copy(); envbase.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',DECORD_EOF_RETRY_MAX='40960')
envbase['PYTHONPATH']=':'.join(['/tmp/llava-ovo-transformers440',str(root/'code/MM-ShiftKV-main/LLaVA-NeXT'),str(root/'code/horizonkv/releases/20261022-ovobench-baselines-v1/kvbench'),str(root/'code/horizonkv/releases/20261026-lukv-cal50-ovobench-v1/kvbench'),str(root/'code/horizonkv/releases/20261019-sparsemm-fp16-chunk16-v2/kvbench'),str(root/'code/horizonkv/runtime/ppu-py310-overlay-20261015')])
for i in range(4):
    shard=data[i*len(data)//4:(i+1)*len(data)//4]; path=output/'annotations'/f'shard_{i:02}.json'
    payload=json.dumps(shard,ensure_ascii=False)
    if path.exists() and path.read_text()!=payload: raise RuntimeError('annotation shard differs')
    if not path.exists(): path.write_text(payload)
    for method,gpus in manifest['method_gpus'].items():
        target=output/method/f'shard_{i:02}'; target.mkdir(parents=True,exist_ok=True)
        pidfile=target/'pid'
        if pidfile.exists():
            try: os.kill(int(pidfile.read_text()),0)
            except ProcessLookupError: pass
            else: raise RuntimeError(f'process already running: {pidfile.read_text()}')
        env=envbase.copy(); env['CUDA_VISIBLE_DEVICES']=str(gpus[i])
        with (target/'run.log').open('a') as log:
            proc=subprocess.Popen([sys.executable,str(release/'evaluate.py'),'--method',method,'--annotation',str(path),'--output',str(target)],env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        pidfile.write_text(str(proc.pid)+'\n'); print(method,'shard',i,'gpu',gpus[i],'pid',proc.pid,'questions',len(shard),flush=True)
