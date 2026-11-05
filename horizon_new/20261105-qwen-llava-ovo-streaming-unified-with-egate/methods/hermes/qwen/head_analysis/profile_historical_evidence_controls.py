"""Profile matched, question-mismatched, and temporally shuffled history controls."""

import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from head_analysis.profile_effective_memory_readout import (
    build_future_prompt,
    compute_readout_observation,
    load_video_tensor,
    pool_query_attention,
    query_heads_for_kv,
    tokenize,
)
from head_analysis.obs_prev_current_chunk_attention_eager import build_observed_model
from video_qa.base import BaseVQA


HEAD_KEYS = ["layer", "kv_head"]
PAIR_KEYS = ["video_idx", "question_idx", "layer", "kv_head"]
METRIC = "internal_top100_mean_attention"


def build_mismatch_samples(anno):
    mismatches = {}
    for video_idx, video in enumerate(anno):
        other = anno[(video_idx + 1) % len(anno)]
        other_conversations = other.get("conversations", [])
        if not other_conversations:
            raise ValueError("Question-mismatch control requires conversations in every selected video.")
        for question_idx, _ in enumerate(video.get("conversations", [])):
            mismatches[(video_idx, question_idx)] = other_conversations[question_idx % len(other_conversations)]
    return mismatches


def reset_stream(model):
    model.clear_cache()
    model.encode_init_prompt()


def compress_stream(model):
    if model.compress_mode == "streamingvlm":
        model._sliding_window_compress()
    else:
        local_q, global_q = model.predict_next_question()
        model.pseudo_forward(local_q, global_q)


def observe_condition(model, processor, analyzer, sample, condition, meta, pre_lens, post_lens, args):
    prompt = build_future_prompt(analyzer, model, sample)
    attentions = model.compute_eager_attentions(tokenize(processor, prompt, model.device))
    visual_start = int(model.visual_start_idx)
    pos_cache = getattr(model, "_position_ids_cache", [])
    rows = []
    for layer_idx, attention in enumerate(attentions):
        pooled = pool_query_attention(attention, mode=args.query_pool, last_n=args.last_n)
        if pooled is None:
            continue
        if layer_idx < len(pos_cache) and pos_cache[layer_idx] is not None:
            cached_kv_len = int(pos_cache[layer_idx].shape[1])
        else:
            cached_kv_len = int(pooled.shape[-1])
        for kv_head in range(args.num_kv_heads):
            row = compute_readout_observation(
                future_attn=pooled,
                layer_idx=layer_idx,
                kv_head=kv_head,
                q_heads=query_heads_for_kv(kv_head, args.num_query_heads, args.num_kv_heads),
                visual_start=visual_start,
                pre_len=pre_lens[layer_idx],
                post_len=post_lens[layer_idx],
                cached_kv_len=cached_kv_len,
                boundary_window=args.boundary_window,
                current_boundary_window=args.current_boundary_window,
                top_k=args.top_k,
            )
            if row is not None:
                row.update(meta)
                row["condition"] = condition
                rows.append(row)
    return rows


def encode_prefix_and_observe(
    model,
    processor,
    analyzer,
    chunks,
    sample,
    mismatch_sample,
    meta,
    args,
    shuffled,
):
    reset_stream(model)
    historical_order = list(range(max(0, len(chunks) - 1)))
    if shuffled and len(historical_order) > 1:
        local_rng = random.Random(args.seed + meta["video_idx"] * 100003 + meta["question_idx"])
        local_rng.shuffle(historical_order)
        if historical_order == list(range(len(historical_order))):
            historical_order = historical_order[1:] + historical_order[:1]
    order = historical_order + [len(chunks) - 1]

    rows = []
    for order_idx, chunk_idx in enumerate(order):
        pre_lens = model._get_cache_seq_len_per_layer()
        model.encode_video_chunk(chunks[chunk_idx])
        post_lens = model._get_cache_seq_len_per_layer()
        is_final = order_idx == len(order) - 1
        if is_final:
            condition = "temporal_shuffled" if shuffled else "matched"
            rows.extend(
                observe_condition(
                    model, processor, analyzer, sample, condition, meta, pre_lens, post_lens, args
                )
            )
            if not shuffled:
                rows.extend(
                    observe_condition(
                        model,
                        processor,
                        analyzer,
                        mismatch_sample,
                        "question_mismatched",
                        meta,
                        pre_lens,
                        post_lens,
                        args,
                    )
                )
        compress_stream(model)
    return rows, order


def bh_fdr(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output


def sign_flip_pvalue(values, rng, permutations):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return 1.0
    observed = float(values.mean())
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(permutations, len(values)))
    null = (signs * values[None, :]).mean(axis=1)
    return float((1 + np.count_nonzero(null >= observed)) / (permutations + 1))


def spearman(left, right):
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3:
        return None
    left_rank = pd.Series(left[valid]).rank(method="average").to_numpy(float)
    right_rank = pd.Series(right[valid]).rank(method="average").to_numpy(float)
    value = np.corrcoef(left_rank, right_rank)[0, 1]
    return float(value) if np.isfinite(value) else None


def aggregate_controls(raw, args):
    score = raw.groupby(HEAD_KEYS + ["condition"], as_index=False)[METRIC].mean()
    pivot = raw.pivot_table(index=PAIR_KEYS, columns="condition", values=METRIC, aggfunc="mean").reset_index()
    required = ["matched", "question_mismatched", "temporal_shuffled"]
    for condition in required:
        if condition not in pivot:
            pivot[condition] = np.nan
    pivot["delta_question"] = pivot["matched"] - pivot["question_mismatched"]
    pivot["delta_temporal"] = pivot["matched"] - pivot["temporal_shuffled"]
    pivot["delta_combined"] = pivot["matched"] - 0.5 * (
        pivot["question_mismatched"] + pivot["temporal_shuffled"]
    )

    per_video = pivot.groupby(["video_idx"] + HEAD_KEYS, as_index=False)[
        ["delta_question", "delta_temporal", "delta_combined"]
    ].mean()
    rng = np.random.default_rng(args.seed)
    rows = []
    for (layer, kv_head), part in per_video.groupby(HEAD_KEYS):
        row = {"layer": int(layer), "kv_head": int(kv_head), "num_videos": int(len(part))}
        for delta in ["delta_question", "delta_temporal", "delta_combined"]:
            values = part[delta].dropna().to_numpy(float)
            row[delta] = float(values.mean()) if len(values) else None
            row[f"{delta}_p"] = sign_flip_pvalue(values, rng, args.permutations)
        rows.append(row)
    contrast = pd.DataFrame(rows)
    for delta in ["delta_question", "delta_temporal", "delta_combined"]:
        contrast[f"{delta}_q"] = bh_fdr(contrast[f"{delta}_p"].to_numpy(float))

    matched = score[score.condition == "matched"][HEAD_KEYS + [METRIC]]
    merged = matched.merge(contrast, on=HEAD_KEYS)
    correlations = {}
    for delta in ["delta_question", "delta_temporal", "delta_combined"]:
        valid = merged[[METRIC, delta]].dropna()
        correlations[f"matched_score_vs_{delta}_spearman"] = spearman(
            valid[METRIC].to_numpy(float), valid[delta].to_numpy(float)
        )

    summary = {
        "metric": METRIC,
        "num_raw_rows": int(len(raw)),
        "num_question_pairs": int(len(pivot.groupby(["video_idx", "question_idx"]))),
        "num_videos": int(raw.video_idx.nunique()),
        "num_heads": int(len(contrast)),
        "condition_means": {
            key: float(value)
            for key, value in raw.groupby("condition")[METRIC].mean().to_dict().items()
        },
        "positive_head_counts": {
            delta: int((contrast[delta] > 0).sum())
            for delta in ["delta_question", "delta_temporal", "delta_combined"]
        },
        "fdr_0.05_positive_head_counts": {
            delta: int(((contrast[delta] > 0) & (contrast[f"{delta}_q"] < 0.05)).sum())
            for delta in ["delta_question", "delta_temporal", "delta_combined"]
        },
        "correlations": correlations,
        "inference_note": "FDR results are exploratory when the number of videos is small.",
    }
    return score, pivot, contrast, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5_vl_7b")
    parser.add_argument("--anno_path", default="data/streamingbench/streamingbench_realtime.json")
    parser.add_argument("--sample_fps", type=float, default=0.5)
    parser.add_argument("--kv_size", type=int, default=6000)
    parser.add_argument("--compress_mode", default="hermes")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num_videos", type=int, default=2)
    parser.add_argument("--max_questions", type=int, default=1)
    parser.add_argument(
        "--question_indices",
        type=int,
        nargs="+",
        default=None,
        help="Optional zero-based question indices to profile within each video.",
    )
    parser.add_argument(
        "--skip_temporal_shuffle",
        action="store_true",
        help="Run only the matched and cross-video question-mismatch controls.",
    )
    parser.add_argument(
        "--load_question_prefix_only",
        action="store_true",
        help="Decode only through each selected question time (best for one question per video).",
    )
    parser.add_argument("--encode_chunk_size", type=int, default=16)
    parser.add_argument("--num_query_heads", type=int, default=28)
    parser.add_argument("--num_kv_heads", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--boundary_window", type=int, default=64)
    parser.add_argument("--current_boundary_window", type=int, default=64)
    parser.add_argument("--query_pool", choices=["mean", "last", "last_n"], default="mean")
    parser.add_argument("--last_n", type=int, default=4)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--save_dir", default="results/myobservations/historical_evidence_controls_smoke"
    )
    args = parser.parse_args()

    with open(args.anno_path) as f:
        full_anno = json.load(f)
    if args.num_videos < 2:
        raise ValueError("At least two videos are required for cross-video question mismatch.")
    anno = full_anno[: args.num_videos]
    mismatches = build_mismatch_samples(anno)

    model_name = "Qwen2.5-VL-7B-Instruct" if args.model == "qwen2.5_vl_7b" else args.model
    model_path = f"models/{model_name}"
    print(f"Loading model on cuda:{args.device}: {model_path}")
    model, processor = build_observed_model(args, model_path)

    class TempBase(BaseVQA):
        pass

    analyzer = TempBase(
        anno=anno,
        save_dir="/tmp/historical_evidence_controls_tmp",
        sample_fps=args.sample_fps,
        qa_model=model,
        qa_processor=processor,
        num_chunks=None,
        chunk_idx=None,
    )

    rows = []
    shuffle_orders = []
    for video_idx, video_sample in enumerate(tqdm(anno, desc="Control videos")):
        shared_video_tensor = None
        if not args.load_question_prefix_only:
            shared_video_tensor = load_video_tensor(analyzer, video_sample)
        all_conversations = video_sample.get("conversations", [])
        selected_indices = (
            args.question_indices
            if args.question_indices is not None
            else list(range(min(args.max_questions, len(all_conversations))))
        )
        for question_idx in selected_indices:
            if question_idx < 0 or question_idx >= len(all_conversations):
                continue
            sample = all_conversations[question_idx]
            if args.load_question_prefix_only and not video_sample.get("clip"):
                prefix_sample = dict(video_sample)
                prefix_sample["clip"] = [0.0, float(sample.get("end_time", 0.0))]
                video_tensor = load_video_tensor(analyzer, prefix_sample)
            else:
                video_tensor = (
                    shared_video_tensor
                    if shared_video_tensor is not None
                    else load_video_tensor(analyzer, video_sample)
                )
            end_frame = min(
                len(video_tensor), math.ceil(sample.get("end_time", len(video_tensor)) * args.sample_fps)
            )
            chunks = [
                video_tensor[start : min(start + args.encode_chunk_size, end_frame)]
                for start in range(0, end_frame, args.encode_chunk_size)
            ]
            if len(chunks) < 2:
                print(f"Skipping video {video_idx} question {question_idx}: no historical chunk")
                continue
            meta = {
                "video_idx": video_idx,
                "video_id": video_sample.get("video_id", video_idx),
                "question_idx": question_idx,
                "task": sample.get("task", "Unknown"),
                "num_chunks": len(chunks),
            }
            normal_rows, normal_order = encode_prefix_and_observe(
                model,
                processor,
                analyzer,
                chunks,
                sample,
                mismatches[(video_idx, question_idx)],
                meta,
                args,
                shuffled=False,
            )
            shuffled_rows = []
            shuffled_order = None
            if not args.skip_temporal_shuffle:
                shuffled_rows, shuffled_order = encode_prefix_and_observe(
                    model,
                    processor,
                    analyzer,
                    chunks,
                    sample,
                    mismatches[(video_idx, question_idx)],
                    meta,
                    args,
                    shuffled=True,
                )
            rows.extend(normal_rows)
            rows.extend(shuffled_rows)
            shuffle_orders.append({**meta, "normal_order": normal_order, "shuffled_order": shuffled_order})
            torch.cuda.empty_cache()

    if not rows:
        raise RuntimeError("No valid control observations were produced.")
    os.makedirs(args.save_dir, exist_ok=True)
    raw = pd.DataFrame(rows)
    raw_path = os.path.join(args.save_dir, "raw_historical_evidence_controls.csv")
    raw.to_csv(raw_path, index=False, quoting=csv.QUOTE_MINIMAL)
    score, paired, contrast, summary = aggregate_controls(raw, args)
    score.to_csv(os.path.join(args.save_dir, "condition_head_scores.csv"), index=False)
    paired.to_csv(os.path.join(args.save_dir, "paired_head_observations.csv"), index=False)
    contrast.to_csv(os.path.join(args.save_dir, "head_control_contrasts.csv"), index=False)
    summary.update(
        {
            "annotation": os.path.abspath(args.anno_path),
            "top_k": args.top_k,
            "boundary_window": args.boundary_window,
            "current_boundary_window": args.current_boundary_window,
            "shuffle_orders": shuffle_orders,
            "raw_csv": raw_path,
        }
    )
    summary_path = os.path.join(args.save_dir, "historical_evidence_controls_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
