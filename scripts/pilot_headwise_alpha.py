#!/usr/bin/env python3
"""LOOV static proxy pilot for head-specific HERMES recency weight alpha."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT=Path("results/myobservations/lu_kv_offline_50v_20260913")
K250=Path("results/myobservations/two_bank_grid_dp_k250_floor188_20260923/per_head_allocation.csv")
K1000=Path("results/myobservations/two_bank_grid_dp_floor750_20260923/per_head_allocation.csv")
DEFAULT_OUT=Path("results/myobservations/headwise_alpha")
HEADS=112
PROTECTED=32
DELTAS=np.array([-.75,-.50,-.30,-.15,0.,.15,.30])
BASELINE=int(np.where(DELTAS==0)[0][0])
SHORT_THRESHOLD=2
LONG_THRESHOLD=19
ROW=np.arange(HEADS)


def layer_policy(layer):
    if layer<SHORT_THRESHOLD:
        return 1.,20.
    if layer>=LONG_THRESHOLD:
        return 0.,0.
    progress=(layer-SHORT_THRESHOLD)/(LONG_THRESHOLD-SHORT_THRESHOLD)
    return .75-.6*progress,20.-12.*progress


def recency(k,distance):
    raw=np.exp(-float(k)*distance)
    return (raw-raw.min())/(raw.max()-raw.min()+1e-6)


def load_scores(path,budgets):
    with np.load(path) as z:
        utility=z["utility"].astype(np.float64)
        combined=z["hermes_score"].astype(np.float64)
        ranking=z["ranking"].astype(np.int64)
        times=z["source_times"].astype(np.float64)
    unique=np.unique(times)
    if len(unique)<2:
        return None
    start=int(np.searchsorted(times,unique[-2],side="left"))
    if len(times)-start<512 or start<1472:
        return None
    n=utility.shape[1]
    distance=(n-1-np.arange(n,dtype=np.float64))/max(n-1,1)
    values=np.zeros((HEADS,len(DELTAS)))
    recent_counts=np.zeros_like(values)
    for h in range(HEADS):
        budget=int(budgets[h]); q=budget-PROTECTED
        baseline_idx=ranking[h,:q]
        baseline_value=float(utility[h,-PROTECTED:].sum()+utility[h,baseline_idx].sum())/float(utility.sum())
        values[h,BASELINE]=baseline_value
        recent_counts[h,BASELINE]=PROTECTED+int(np.sum(baseline_idx>=start))
        alpha0,k=layer_policy(h//4)
        if alpha0>=1.-1e-12:
            values[h,:]=baseline_value
            recent_counts[h,:]=recent_counts[h,BASELINE]
            continue
        r=recency(k,distance)
        attention=(combined[h]-alpha0*r)/(1.-alpha0)
        for j,delta in enumerate(DELTAS):
            if j==BASELINE:
                continue
            alpha=float(np.clip(alpha0+delta,0.,1.))
            adjusted=attention[:-PROTECTED]*(1.-alpha)+alpha*r[:-PROTECTED]
            chosen=np.argpartition(-adjusted,q-1)[:q]
            values[h,j]=(utility[h,-PROTECTED:].sum()+utility[h,chosen].sum())/float(utility.sum())
            recent_counts[h,j]=PROTECTED+int(np.sum(chosen>=start))
    return values,recent_counts


def paired(d):
    rng=np.random.default_rng(20260923)
    boot=rng.choice(d,size=(10000,len(d)),replace=True).mean(axis=1)
    return {"mean_pp":float(100*d.mean()),
            "bootstrap95_pp":(100*np.quantile(boot,[.025,.975])).tolist(),
            "positive_videos":int(np.sum(d>1e-12)),"negative_videos":int(np.sum(d< -1e-12))}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--average-budget",type=int,choices=(250,1000),default=250)
    parser.add_argument("--output-dir",type=Path,default=None)
    parser.add_argument("--uniform-budgets",action="store_true",
                        help="Use equal per-head capacity to remove leakage from prefit head allocations.")
    args=parser.parse_args()
    budget_path=K250 if args.average_budget==250 else K1000
    suffix="_uniform" if args.uniform_budgets else ""
    out=args.output_dir or Path(f"{DEFAULT_OUT}_k{args.average_budget}{suffix}_20260923")
    if args.uniform_budgets:
        budgets=np.full(HEADS,args.average_budget,dtype=np.int64)
    else:
        rows=list(csv.DictReader(budget_path.open()))
        col="capacity_n" if "capacity_n" in rows[0] else "floor750_n"
        budgets=np.array([int(row[col]) for row in rows])
    by_video=defaultdict(list)
    for path in sorted(ROOT.glob("v*_q*.npz")):
        prof=load_scores(path,budgets)
        if prof is not None:
            by_video[path.stem.split("_")[0]].append(prof)
    videos=sorted(by_video)
    values=np.stack([np.mean([x[0] for x in by_video[v]],axis=0) for v in videos])
    recent=np.stack([np.mean([x[1] for x in by_video[v]],axis=0) for v in videos])
    if len(videos)!=48 or sum(map(len,by_video.values()))!=68:
        raise AssertionError("Cohort mismatch")

    reports=[]
    head_counts=np.zeros(len(DELTAS),dtype=int)
    global_counts=np.zeros(len(DELTAS),dtype=int)
    for i,v in enumerate(videos):
        train=(values.sum(axis=0)-values[i])/(len(videos)-1)
        gc=int(np.argmax(train.sum(axis=0)))
        hc=np.argmax(train,axis=1)
        hc[:SHORT_THRESHOLD*4]=BASELINE # Attention component is not identifiable at alpha=1.
        head_counts+=np.bincount(hc,minlength=len(DELTAS))
        global_counts[gc]+=1
        reports.append({"video":v,"observations":len(by_video[v]),
                        "baseline":float(values[i,:,BASELINE].sum()),
                        "global_alpha":float(values[i,:,gc].sum()),
                        "headwise_alpha":float(values[i,ROW,hc].sum()),
                        "baseline_recent_tokens":float(recent[i,:,BASELINE].sum()),
                        "global_recent_tokens":float(recent[i,:,gc].sum()),
                        "headwise_recent_tokens":float(recent[i,ROW,hc].sum()),
                        "global_delta":float(DELTAS[gc])})
    full=values.mean(axis=0)
    selected=np.argmax(full,axis=1)
    selected[:SHORT_THRESHOLD*4]=BASELINE
    out.mkdir(parents=True,exist_ok=True)
    with (out/"per_video.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(reports[0]));w.writeheader();w.writerows(reports)
    with (out/"per_head_alpha.csv").open("w",newline="") as f:
        fields=["head_id","layer","kv_head","budget","base_alpha","selected_delta","selected_alpha"]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for h in range(HEADS):
            a,_=layer_policy(h//4)
            w.writerow({"head_id":h,"layer":h//4,"kv_head":h%4,"budget":int(budgets[h]),
                        "base_alpha":a,"selected_delta":float(DELTAS[selected[h]]),
                        "selected_alpha":float(np.clip(a+DELTAS[selected[h]],0,1))})
    fig,ax=plt.subplots(figsize=(13,4.4),layout="constrained")
    chosen_alpha=np.array([np.clip(layer_policy(h//4)[0]+DELTAS[selected[h]],0,1) for h in range(HEADS)])
    ax.scatter(ROW,chosen_alpha,s=12,c=np.arange(HEADS)//4,cmap="turbo",vmin=0,vmax=27)
    base=np.array([layer_policy(h//4)[0] for h in range(HEADS)])
    ax.plot(ROW,base,color="#444444",linewidth=.65,alpha=.7,label="Default HERMES α")
    ax.set(xlim=(-1,HEADS),ylim=(-.03,1.03),xlabel="Physical KV head (layer × 4 + head)",
           ylabel="Selected α",title=f"Full-cohort selected head-wise α at K={args.average_budget} (static proxy)")
    ax.set_xticks(np.arange(0,HEADS,8));ax.legend(frameon=False)
    fig.savefig(out/"headwise_alpha.png",dpi=250);fig.savefig(out/"headwise_alpha.pdf");plt.close(fig)

    baseline=np.array([x["baseline"] for x in reports])
    global_=np.array([x["global_alpha"] for x in reports])
    headwise=np.array([x["headwise_alpha"] for x in reports])
    report={"protocol":{"videos":len(videos),"observations":sum(map(len,by_video.values())),
                         "average_per_head":args.average_budget,"total_budget":int(budgets.sum()),
                         "budget_scheme":"uniform" if args.uniform_budgets else "fixed full-cohort two-bank allocation",
                         "protected_recent":PROTECTED,"deltas":DELTAS.tolist(),
                         "k":"unchanged layer-default HERMES k",
                         "calibration":"leave-one-video-out, equal video weight",
                         "caveat":"Static question-time full-cache utility proxy, not sequential cache replay or compressed QA; alpha=1 layers held fixed because raw attention component is unavailable"},
            "loov_mean_video_proxy":{"baseline":float(baseline.mean()),"global_alpha":float(global_.mean()),"headwise_alpha":float(headwise.mean())},
            "loov_mean_recent_tokens":{"baseline":float(np.mean([x["baseline_recent_tokens"] for x in reports])),
                                       "global_alpha":float(np.mean([x["global_recent_tokens"] for x in reports])),
                                       "headwise_alpha":float(np.mean([x["headwise_recent_tokens"] for x in reports]))},
            "paired":{"global_minus_baseline":paired(global_-baseline),
                      "headwise_minus_baseline":paired(headwise-baseline),
                      "headwise_minus_global":paired(headwise-global_)},
            "global_delta_counts":{str(float(d)):int(n) for d,n in zip(DELTAS,global_counts)},
            "headwise_delta_counts_across_folds":{str(float(d)):int(n) for d,n in zip(DELTAS,head_counts)},
            "full_cohort_alpha_counts":{str(float(a)):int(np.sum(np.isclose(chosen_alpha,a))) for a in np.unique(chosen_alpha)}}
    (out/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
