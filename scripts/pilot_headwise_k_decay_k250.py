#!/usr/bin/env python3
"""LOOV static proxy pilot for head-specific HERMES recency decay at K=250.

The original question-time full-cache HERMES score is held fixed except for
the exact change in its normalized recency component when k is rescaled.
This is a ranking pilot, not sequential replay or compressed-cache QA.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("results/myobservations/lu_kv_offline_50v_20260913")
DEFAULT_BUDGETS = Path("results/myobservations/two_bank_grid_dp_k250_floor188_20260923/per_head_allocation.csv")
DEFAULT_OUT = Path("results/myobservations/headwise_k_decay_k250_20260923")
HEADS = 112
PROTECTED = 32
MULTIPLIERS = np.array([0.25, 0.5, 1.0, 1.5, 2.0])
BASELINE = 2
SHORT_THRESHOLD = 2
LONG_THRESHOLD = 19
ROW = np.arange(HEADS)


def layer_policy(layer):
    if layer < SHORT_THRESHOLD:
        return 1.0, 20.0
    if layer >= LONG_THRESHOLD:
        return 0.0, 0.0
    progress = (layer-SHORT_THRESHOLD)/(LONG_THRESHOLD-SHORT_THRESHOLD)
    return 0.75-0.6*progress, 20.0-12.0*progress


def recency(k, distance):
    raw = np.exp(-float(k)*distance)
    return (raw-raw.min())/(raw.max()-raw.min()+1e-6)


def retained(utility, ranking, start, budget):
    chosen = ranking[:budget-PROTECTED]
    return float(utility[-PROTECTED:].sum()+utility[chosen].sum()), \
           int(PROTECTED+np.sum(chosen>=start))


def trace_profiles(path, budgets):
    with np.load(path) as z:
        utility=z["utility"].astype(np.float64)
        score=z["hermes_score"].astype(np.float64)
        ranking=z["ranking"].astype(np.int64)
        times=z["source_times"].astype(np.float64)
    unique=np.unique(times)
    if len(unique)<2:
        return None
    start=int(np.searchsorted(times,unique[-2],side="left"))
    if len(times)-start<512 or start<1472:
        return None
    full=float(utility.sum())
    n=utility.shape[1]
    distance=(n-1-np.arange(n,dtype=np.float64))/max(n-1,1)
    values=np.zeros((HEADS,len(MULTIPLIERS)))
    recent_counts=np.zeros_like(values)
    for h in range(HEADS):
        b=int(budgets[h]); q=b-PROTECTED
        baseline_value,baseline_count=retained(utility[h],ranking[h],start,b)
        values[h,BASELINE]=baseline_value/full
        recent_counts[h,BASELINE]=baseline_count
        alpha,k_base=layer_policy(h//4)
        if alpha==0:
            values[h,:]=baseline_value/full
            recent_counts[h,:]=baseline_count
            continue
        base_r=recency(k_base,distance)[:-PROTECTED]
        for j,mult in enumerate(MULTIPLIERS):
            if j==BASELINE:
                continue
            adjusted=score[h,:-PROTECTED]+alpha*(
                recency(k_base*mult,distance)[:-PROTECTED]-base_r)
            chosen=np.argpartition(-adjusted,q-1)[:q]
            values[h,j]=(utility[h,-PROTECTED:].sum()+utility[h,chosen].sum())/full
            recent_counts[h,j]=PROTECTED+np.sum(chosen>=start)
    return values,recent_counts


def paired(diffs):
    rng=np.random.default_rng(20260923)
    boot=rng.choice(diffs,size=(10000,len(diffs)),replace=True).mean(axis=1)
    return {"mean_pp":float(100*np.mean(diffs)),
            "bootstrap95_pp":(100*np.quantile(boot,[.025,.975])).tolist(),
            "positive_videos":int(np.sum(diffs>1e-12)),
            "negative_videos":int(np.sum(diffs< -1e-12))}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--budget-csv",type=Path,default=DEFAULT_BUDGETS)
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUT)
    parser.add_argument("--average-budget",type=int,default=250)
    args=parser.parse_args()
    rows=list(csv.DictReader(args.budget_csv.open()))
    budget_column="capacity_n" if "capacity_n" in rows[0] else "floor750_n"
    budgets=np.array([int(row[budget_column]) for row in rows])
    if len(budgets)!=HEADS or budgets.sum()!=HEADS*args.average_budget or budgets.min()<PROTECTED:
        raise AssertionError("Unexpected per-head budget table")
    by_video=defaultdict(list)
    for path in sorted(ROOT.glob("v*_q*.npz")):
        profile=trace_profiles(path,budgets)
        if profile is not None:
            by_video[path.stem.split("_")[0]].append(profile)
    videos=sorted(by_video)
    values=np.stack([np.mean([x[0] for x in by_video[v]],axis=0) for v in videos])
    recent=np.stack([np.mean([x[1] for x in by_video[v]],axis=0) for v in videos])
    if len(videos)!=48 or sum(map(len,by_video.values()))!=68:
        raise AssertionError("Cohort differs from the K=250 two-bank study")

    reports=[]
    head_choice_counts=np.zeros(len(MULTIPLIERS),dtype=int)
    global_choice_counts=np.zeros(len(MULTIPLIERS),dtype=int)
    for i,video in enumerate(videos):
        train=(values.sum(axis=0)-values[i])/(len(videos)-1)
        global_choice=int(np.argmax(train.sum(axis=0)))
        head_choice=np.argmax(train,axis=1)
        head_choice[19*4:]=BASELINE  # k has no effect when HERMES alpha=0.
        head_choice_counts+=np.bincount(head_choice,minlength=len(MULTIPLIERS))
        global_choice_counts[global_choice]+=1
        reports.append({"video":video,"observations":len(by_video[video]),
                        "baseline_proxy":float(values[i,:,BASELINE].sum()),
                        "global_k_proxy":float(values[i,:,global_choice].sum()),
                        "headwise_k_proxy":float(values[i,ROW,head_choice].sum()),
                        "baseline_recent_slots":float(recent[i,:,BASELINE].sum()),
                        "global_k_recent_slots":float(recent[i,:,global_choice].sum()),
                        "headwise_k_recent_slots":float(recent[i,ROW,head_choice].sum()),
                        "global_k_multiplier":float(MULTIPLIERS[global_choice])})
    full_train=values.mean(axis=0)
    selected=np.argmax(full_train,axis=1)
    selected[19*4:]=BASELINE

    args.output_dir.mkdir(parents=True,exist_ok=True)
    with (args.output_dir/"per_video.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(reports[0]));writer.writeheader();writer.writerows(reports)
    with (args.output_dir/"per_head_parameter.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["head_id","layer","kv_head","budget","base_alpha","base_k","selected_multiplier","selected_k"])
        writer.writeheader()
        for h in range(HEADS):
            alpha,k=layer_policy(h//4)
            writer.writerow({"head_id":h,"layer":h//4,"kv_head":h%4,"budget":int(budgets[h]),
                             "base_alpha":alpha,"base_k":k,
                             "selected_multiplier":float(MULTIPLIERS[selected[h]]),
                             "selected_k":float(k*MULTIPLIERS[selected[h]])})

    fig,ax=plt.subplots(figsize=(13,4.4),layout="constrained")
    ax.scatter(ROW,MULTIPLIERS[selected],s=12,c=np.arange(HEADS)//4,cmap="turbo",vmin=0,vmax=27)
    ax.axhline(1.0,color="#555555",linewidth=.8,linestyle="--",label="Default HERMES k")
    ax.set(xlim=(-1,HEADS),ylim=(.1,2.15),xlabel="Physical KV head (layer × 4 + head)",
           ylabel="Chosen k multiplier",title="Full-cohort selected head-wise k multiplier (static proxy)")
    ax.set_xticks(np.arange(0,HEADS,8))
    ax.legend(frameon=False)
    fig.savefig(args.output_dir/"headwise_k_multiplier.png",dpi=250)
    fig.savefig(args.output_dir/"headwise_k_multiplier.pdf")
    plt.close(fig)

    baseline=np.array([r["baseline_proxy"] for r in reports])
    global_=np.array([r["global_k_proxy"] for r in reports])
    headwise=np.array([r["headwise_k_proxy"] for r in reports])
    report={"protocol":{"videos":len(videos),"observations":sum(map(len,by_video.values())),
                         "total_budget":int(budgets.sum()),"head_budgets":str(args.budget_csv),"protected_recent":PROTECTED,
                         "k_multipliers":MULTIPLIERS.tolist(),
                         "k_policy":"Original HERMES position-normalized recency term with k scaled; alpha and attention unchanged",
                         "calibration":"leave-one-video-out, equal video weight",
                         "caveat":"Question-time full-cache projected utility; not sequential eviction or compressed QA"},
            "loov_mean_video_proxy":{"baseline":float(baseline.mean()),
                                     "global_k":float(global_.mean()),
                                     "headwise_k":float(headwise.mean())},
            "loov_mean_recent_slots":{"baseline":float(np.mean([r["baseline_recent_slots"] for r in reports])),
                                      "global_k":float(np.mean([r["global_k_recent_slots"] for r in reports])),
                                      "headwise_k":float(np.mean([r["headwise_k_recent_slots"] for r in reports]))},
            "paired":{"global_minus_baseline":paired(global_-baseline),
                      "headwise_minus_baseline":paired(headwise-baseline),
                      "headwise_minus_global":paired(headwise-global_)},
            "global_choice_counts":{str(float(m)):int(n) for m,n in zip(MULTIPLIERS,global_choice_counts)},
            "headwise_choice_counts_across_folds":{str(float(m)):int(n) for m,n in zip(MULTIPLIERS,head_choice_counts)},
            "full_cohort_selected_counts":{str(float(m)):int(np.sum(MULTIPLIERS[selected]==m)) for m in MULTIPLIERS}}
    (args.output_dir/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
