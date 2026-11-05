#!/usr/bin/env python3
"""Unified launcher for 1 fps Qwen and LLaVA OVO/StreamingBench baselines."""
import argparse, subprocess, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--benchmark',required=True,choices=['ovobench','streamingbench'])
p.add_argument('--model',required=True,choices=['qwen25vl','llavaov'])
p.add_argument('--method',required=True,choices=['adakv','pyramidkv','sparsemm','lukv','hermes','fullkv'])
p.add_argument('--gpus',default='0');p.add_argument('--output-dir',required=True,type=Path)
p.add_argument('--annotation',type=Path);p.add_argument('--max-videos',type=int)
p.add_argument('--cohort',choices=['heldout','full'],default='heldout')
p.add_argument('--allow-calibration-overlap',action='store_true')
p.add_argument('--accept-equivalent-choices',action='store_true')
p.add_argument('--dry-run',action='store_true')
a=p.parse_args()
if a.model=='qwen25vl':
 if a.benchmark=='ovobench' and (a.cohort!='heldout' or a.allow_calibration_overlap or a.accept_equivalent_choices):p.error('cohort/choice options apply to StreamingBench only')
 if a.benchmark=='streamingbench' and a.cohort=='full' and not a.accept_equivalent_choices:p.error('Full StreamingBench includes duplicate correct options; specify --accept-equivalent-choices explicitly')
 backend=HERE/('qwen_full2495' if a.benchmark=='streamingbench' and a.accept_equivalent_choices else 'qwen')
 cmd=[sys.executable,str(backend/'run.py'),'--benchmark',a.benchmark,'--model',a.model,'--method',a.method,'--gpus',a.gpus,'--output-dir',str(a.output_dir),'--visual-input','dscache448-video','--sample-fps','1','--chunk-frames','2']
 if a.benchmark=='streamingbench':
  cmd+=['--cohort',a.cohort]
  if a.allow_calibration_overlap:cmd+=['--allow-calibration-overlap']
else:
 backend=HERE/'llava'
 cmd=[sys.executable,str(backend/'run.py'),'--benchmark',a.benchmark,'--model',a.model,'--method',a.method,'--gpus',a.gpus,'--output-dir',str(a.output_dir),'--input-profile','dscache']
 if a.benchmark=='streamingbench':
  cmd+=['--cohort',a.cohort]
  if a.allow_calibration_overlap:cmd+=['--allow-calibration-overlap']
  if a.accept_equivalent_choices:cmd+=['--accept-equivalent-choices']
 elif a.cohort!='heldout' or a.allow_calibration_overlap or a.accept_equivalent_choices:p.error('cohort/choice options apply to StreamingBench only')
if a.annotation:cmd+=['--annotation',str(a.annotation)]
if a.max_videos is not None:cmd+=['--max-videos',str(a.max_videos)]
if a.dry_run:cmd+=['--dry-run']
print('BACKEND',str(backend),flush=True);print('COMMAND',repr(cmd),flush=True)
sys.exit(subprocess.call(cmd))
