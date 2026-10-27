"""Detached K3000 last32 calibration handoff and allocation only.

The independently detached calibration is already running. Wait for its 68
checkpoints, then allocate. QA inference is held for user review.
"""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from sparsemm_native import run_ovobench_last32_dp_k3000 as runner
from sparsemm_native.supervise_ovobench_recent4_only import split_videos

ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT.parent
HERMES_PYTHON = Path('/home/yujun/.conda/envs/hermes-qwen/bin/python')
OUTPUT = ROOT/'results/ovobench_last32_dp_k3000_fps05_nine_20261022'
CALIBRATION = PROGRAM/'hermes/results/myobservations/last32_visual_attention_k3000_calibration_20261022'
ALLOCATION = PROGRAM/'hermes/results/myobservations/last32_two_bank_k3000_floor2250_cap4500_20261022'
ARM = 'dp_last32_k3000'


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    os.replace(temporary,path)


def count(path):
    if not path.exists():return 0
    with path.open() as f:return sum(1 for line in f if line.strip())


def command(argv,log_path,cwd=PROGRAM):
    with log_path.open('ab',buffering=0) as log:
        proc=subprocess.Popen([str(v) for v in argv],cwd=cwd,stdin=subprocess.DEVNULL,
                              stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while proc.poll() is None:
            update(stage_pid=proc.pid)
            time.sleep(15)
        if proc.returncode:
            raise RuntimeError(f'Stage exited {proc.returncode}: {log_path}')


state={'state':'initializing','pid':os.getpid(),'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}


def update(**fields):
    state.update(fields,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    save(OUTPUT/'status.json',state)


def calibration_pid():
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:cmdline=(entry/'cmdline').read_bytes()
        except OSError:continue
        arguments=cmdline.split(b'\0')
        if (arguments and arguments[0].endswith(b'python')
                and any(arg.endswith(b'collect_last32_visual_attention_k3000.py') for arg in arguments[1:])
                and b'--max-observations' not in arguments):
            return int(entry.name)
    return None


def run_calibration():
    allocation=PROGRAM/'hermes/scripts/allocate_last32_two_bank_k3000.py'
    while True:
        checkpoint=json.loads((CALIBRATION/'status.json').read_text())
        update(state='calibration_full',completed_observations=checkpoint.get('completed',0),
               calibration_pid=calibration_pid())
        if checkpoint.get('state')=='complete' and checkpoint.get('completed')==68:break
        if calibration_pid() is None:
            raise RuntimeError('K3000 calibration process stopped before 68 observations completed')
        time.sleep(30)
    update(state='exact_dp')
    command([HERMES_PYTHON,'-u',allocation],OUTPUT/'allocation.log')
    if json.loads((ALLOCATION/'report.json').read_text()).get('status')!='complete':
        raise RuntimeError('Exact K3000 dual-cache DP did not complete')
    update(state='plotting_allocation')
    command([HERMES_PYTHON,'-u',PROGRAM/'hermes/scripts/plot_last32_two_bank_k3000.py'],
            OUTPUT/'plot.log')
    if not (ALLOCATION/'allocation_heatmap.png').is_file():
        raise RuntimeError('K3000 allocation heatmap was not created')


def worker_args(name, ids, root):
    return [HERMES_PYTHON,'-u','-m','sparsemm_native.run_ovobench_last32_dp_k3000',
            '--arm',ARM,'--output-root',root,'--ids',','.join(ids),
            '--budget','3000','--fps','0.5','--chunk-size','16',
            '--max-new-tokens','16','--video-max-pixels','151200',
            '--grouped','--concurrent-processes','2','--max-new','100000']


def run_pilot():
    expected=runner.legacy.queries()
    chosen=[];counts={}
    for query in expected:
        if query['task'] in runner.legacy.FORWARD:continue
        if counts.get(query['task'],0)<2:
            chosen.append(query['query_id'])
            counts[query['task']]=counts.get(query['task'],0)+1
    if len(chosen)!=18:raise AssertionError('Pilot lacks two official queries per retained task')
    update(state='official_pilot',pilot_query_ids=chosen)
    args=worker_args('pilot',chosen,OUTPUT/'pilot')
    args[args.index('--concurrent-processes')+1]='1'
    command(args,OUTPUT/'pilot.log',cwd=ROOT)
    if count(OUTPUT/'pilot'/ARM/'queries.jsonl')!=18:
        raise RuntimeError('Official pilot did not finish 18 queries')


def run_full():
    expected=runner.legacy.queries()
    selected=runner.schedule_queries([q for q in expected if q['task'] not in runner.legacy.FORWARD])
    jobs=split_videos(selected)
    if sum(len(job['query_ids']) for job in jobs.values())!=1468:
        raise AssertionError('Nine-task official query sharding changed')
    update(state='official_full',full_shards={name:len(job['query_ids']) for name,job in jobs.items()})
    attempts={name:0 for name in jobs};active={}
    while True:
        completed={name:count(OUTPUT/'full/shards'/name/ARM/'queries.jsonl') for name in jobs}
        update(state='official_full',completed_queries=sum(completed.values()),shard_queries=completed,
               worker_pids={name:proc.pid for name,proc in active.items() if proc.poll() is None})
        if all(completed[name]==len(jobs[name]['query_ids']) for name in jobs):break
        for name,job in jobs.items():
            if completed[name]==len(job['query_ids']):continue
            proc=active.get(name)
            if proc is not None and proc.poll() is None:continue
            if attempts[name]>=3:raise RuntimeError(f'{name} failed three times')
            log_path=OUTPUT/(name+'.log')
            with log_path.open('ab',buffering=0) as log:
                proc=subprocess.Popen([str(v) for v in worker_args(name,job['query_ids'],OUTPUT/'full/shards'/name)],
                    cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active[name]=proc;attempts[name]+=1
        time.sleep(20)
    for proc in active.values():
        if proc.poll() is None:proc.wait(timeout=60)
    rows={}
    for name in jobs:
        for line in (OUTPUT/'full/shards'/name/ARM/'queries.jsonl').open():
            row=json.loads(line)
            if row['query_id'] in rows:raise RuntimeError('Duplicate official query')
            rows[row['query_id']]=row
    if set(rows)!={q['query_id'] for q in selected}:
        raise RuntimeError('Nine-task official query set incomplete')
    ordered=[rows[q['query_id']] for q in selected]
    for row,query in zip(ordered,selected):runner.validate_row(row,query)
    destination=OUTPUT/'full'/ARM;destination.mkdir(parents=True,exist_ok=True)
    (destination/'queries.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in ordered))
    summary=runner.legacy.summary(ordered,1468)
    save(destination/'summary.json',summary)
    update(state='complete',completed_queries=1468,
           nine_task_group_average=summary['hermes_category_average'],
           question_accuracy=summary['question_accuracy'],worker_pids={})


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    try:
        run_calibration()
        report=json.loads((ALLOCATION/'report.json').read_text())
        update(state='allocation_complete_waiting_for_review',
               allocation_report=str(ALLOCATION/'report.json'),
               allocation_heatmap=str(ALLOCATION/'allocation_heatmap.png'),
               recent_total=report['global_dp']['recent_total'],
               history_total=report['global_dp']['history_total'],
               pilot_started=False,full_inference_started=False)
    except BaseException as error:
        update(state='failed',error=f'{type(error).__name__}: {error}')
        raise


if __name__=='__main__':main()
