#!/usr/bin/env python3
"""Validate and summarize a complete unified evaluation."""
import argparse, csv, json, subprocess, sys
from collections import Counter
from pathlib import Path
from run import OLD
p=argparse.ArgumentParser(); p.add_argument('output_dir',type=Path); a=p.parse_args(); root=a.output_dir.resolve()
if not (root/'manifest.json').exists():
    sys.exit(subprocess.call([sys.executable,str(OLD/'summarize.py'),str(root)]))
m=json.loads((root/'manifest.json').read_text()); rows=[]; tasks={}; abilities={}
for i,_ in enumerate(m['gpus']):
    anno=json.loads((root/'annotations'/f'shard_{i:02}.json').read_text()); out=root/f'shard_{i:02}'
    path=out/('1_0.csv' if m['model']=='qwen25vl' and m['method'] in ('adakv','pyramidkv') and m.get('visual_input','native')=='native' else 'results.csv')
    with path.open(newline='',encoding='utf-8-sig') as f: actual=list(csv.DictReader(f))
    expected=[(str(v['video_id']),q) for v in anno for q in v['conversations']]
    if len(actual)!=len(expected): raise RuntimeError(f'Incomplete shard {i}: {len(actual)}/{len(expected)} questions')
    for row,(video,q) in zip(actual,expected):
        correct='ABCDEFGH'[q['choices'].index(q['answer'])]
        if row['video_id']!=video or row['question']!=q['question'] or row['correct_choice']!=correct or float(row['qa_acc'])!=100*(row['pred_choice']==correct): raise RuntimeError(f'Inconsistent row: {video}, {q["question"]}')
        for dest,key in ((tasks,q.get('task','')),(abilities,q.get('required_ability',''))):
            entry=dest.setdefault(key,{'count':0,'correct':0}); entry['count']+=1; entry['correct']+=row['pred_choice']==correct
        rows.append(row)
for group in (tasks,abilities):
    for entry in group.values(): entry['accuracy']=100*entry['correct']/entry['count']
if len(rows)!=m['questions']: raise RuntimeError('Total question count differs from manifest')
columns=[('OP','Object Recognition'),('CR','Causal Reasoning'),('CS','Clips Summarize'),('ATP','Attribute Recognition'),('EU','Event Understanding'),('TR','Text-Rich Understanding'),('PR','Prospective Reasoning'),('SU','Spatial Understanding'),('ACP','Action Recognition'),('CT','Counting')]
result=dict(benchmark=m['benchmark'],model=m['model'],method=m['method'],cohort=m['cohort'],videos=m['videos'],questions=len(rows),correct=sum(float(x['qa_acc'])==100 for x in rows),micro_accuracy=sum(float(x['qa_acc']) for x in rows)/len(rows),tasks=tasks,required_ability=abilities)
with (root/'results.csv.tmp').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
(root/'results.csv.tmp').replace(root/'results.csv')
result['table_columns']={short:tasks[name]['accuracy'] if name in tasks else None for short,name in columns}
result['table_columns']['Overall']=result['micro_accuracy']
(root/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
labels=['Model','Method','#Frames']+[short for short,_ in columns]+['Overall']
values=[m['model'],m['method'],str(m['fps'])+' fps']+[(f"{result['table_columns'][short]:.2f}" if result['table_columns'][short] is not None else '—') for short,_ in columns]+[f"{result['micro_accuracy']:.2f}"]
table='| '+' | '.join(labels)+' |\n| '+' | '.join(['---']*len(labels))+' |\n| '+' | '.join(values)+' |\n'
(root/'table.md').write_text(table,encoding='utf-8')
print(json.dumps(result,indent=2,ensure_ascii=False)); print(table)
