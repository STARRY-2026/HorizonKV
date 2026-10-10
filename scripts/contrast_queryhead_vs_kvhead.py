"""Contrast 28 query-head profiles against the GQA-averaged 4 KV-head profiles.

The production profile (``profile_horizonkv_pilot.py``) maps every physical KV
head to its 7 query heads and averages their attention maps before building the
eight-bin temporal profile. This script keeps the per-query-head attention and
quantifies how much head diversity is discarded by that averaging:

  * R2 = between-group variance / total variance of the 28 query-head profiles
    (a per-layer ANOVA-style decomposition; R2 close to 0 means the four group
    means explain almost none of the 28-head diversity).
  * mean pairwise cosine among the 28 query heads vs among the 4 KV-head means,
    both in raw and within-bin z-scored space.
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from head_analysis.profile_horizonkv_pilot import (
    AGE_BINS_SEC,
    EPS,
    age_bin_label,
    build_future_prompt,
    build_pilot_model,
    load_video_tensor,
    pool_query_attention,
    query_heads_for_kv,
    select_pilot_subset,
    token_times_for_chunk,
    tokenize,
)
from video_qa.base import BaseVQA


def profile_vectors_for_layer(attn, model, token_times, question_time, num_query_heads, num_kv_heads):
    """Return (query_head_matrix, kv_head_matrix) of 8-bin reuse_score vectors."""
    pooled = pool_query_attention(attn, mode="last_n", last_n=4)
    if pooled is None:
        return None, None
    kv_len = int(pooled.shape[-1])
    visual_start = int(model.visual_start_idx)
    visual_count = len(token_times)
    v_end = min(visual_start + visual_count, kv_len)
    if v_end <= visual_start:
        return None, None
    usable = v_end - visual_start
    local_times = token_times[:usable]
    age_labels = [age_bin_label(float(question_time) - float(t)) for t in local_times]
    age_counts = defaultdict(int)
    for label in age_labels:
        age_counts[label] += 1

    query_vectors = []
    for qh in range(num_query_heads):
        qattn = pooled[qh].float()
        visual_attn = torch.clamp(qattn[visual_start:v_end], min=0.0)
        visual_mass = float(visual_attn.sum().item())
        if visual_mass <= EPS:
            query_vectors.append(np.zeros(len(AGE_BINS_SEC), dtype=np.float64))
            continue
        age_mass = defaultdict(float)
        for token_idx, value in enumerate(visual_attn.detach().cpu().tolist()):
            age_mass[age_labels[token_idx]] += float(value)
        vector = np.zeros(len(AGE_BINS_SEC), dtype=np.float64)
        for idx, (label, _, _) in enumerate(AGE_BINS_SEC):
            count = int(age_counts.get(label, 0))
            mass = float(age_mass.get(label, 0.0))
            token_prior = count / float(usable) if usable else 0.0
            mass_share = mass / (visual_mass + EPS)
            vector[idx] = math.log1p(mass_share / (token_prior + EPS)) if count > 0 else 0.0
        query_vectors.append(vector)

    query_matrix = np.stack(query_vectors)  # [num_query_heads, 8]
    group_size = num_query_heads // max(num_kv_heads, 1)
    kv_matrix = np.stack(
        [query_matrix[i * group_size : (i + 1) * group_size].mean(axis=0) for i in range(num_kv_heads)]
    )
    return query_matrix, kv_matrix


def variance_decomposition(query_matrix, num_kv_heads):
    group_size = query_matrix.shape[0] // num_kv_heads
    total_mean = query_matrix.mean(axis=0, keepdims=True)
    sst = float(((query_matrix - total_mean) ** 2).sum())
    group_means = np.stack(
        [query_matrix[i * group_size : (i + 1) * group_size].mean(axis=0) for i in range(num_kv_heads)]
    )
    ssb = float((((group_means - total_mean) ** 2).sum()) * group_size)
    r2 = ssb / sst if sst > EPS else float("nan")
    return r2


def cosine_stats(matrix):
    values = np.nan_to_num(matrix, nan=0.0)
    values = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), EPS)
    sim = values @ values.T
    off = sim[~np.eye(sim.shape[0], dtype=bool)]
    return float(off.mean()) if off.size else float("nan")


def column_zscore(matrix):
    mean = np.nanmean(matrix, axis=0, keepdims=True)
    std = np.nanstd(matrix, axis=0, keepdims=True)
    return np.nan_to_num((matrix - mean) / np.maximum(std, EPS), nan=0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5_vl_7b")
    parser.add_argument("--model_path", default="models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--anno_path", default="data/streamingbench/streamingbench_realtime.json")
    parser.add_argument("--video_root", default="/home/yujun/program/streaming_data/streamingbench/videos")
    parser.add_argument("--sample_fps", type=float, default=0.5)
    parser.add_argument("--kv_size", type=int, default=6000)
    parser.add_argument("--compress_mode", default="hermes")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num_videos", type=int, default=3)
    parser.add_argument("--max_questions", type=int, default=2)
    parser.add_argument("--max_last_end", type=float, default=90.0)
    parser.add_argument("--encode_chunk_size", type=int, default=16)
    parser.add_argument("--fixed_frame_canvas_size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_dir", default="results/myobservations/queryhead_vs_kvhead_contrast")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    with open(args.anno_path) as f:
        full_anno = json.load(f)
    filtered_anno, manifest = select_pilot_subset(
        full_anno, args.num_videos, args.max_last_end, args.seed
    )
    for sample in filtered_anno:
        sample["video_path"] = os.path.join(args.video_root, os.path.basename(sample["video_path"]))

    model, processor = build_pilot_model(args, args.model_path)
    text_config = model.language_model.config
    num_layers = int(text_config.num_hidden_layers)
    num_query_heads = int(text_config.num_attention_heads)
    num_kv_heads = int(getattr(text_config, "num_key_value_heads", num_query_heads))

    class TempBase(BaseVQA):
        pass

    analyzer = TempBase(
        anno=filtered_anno,
        save_dir="/tmp/queryhead_vs_kvhead_tmp",
        sample_fps=args.sample_fps,
        qa_model=model,
        qa_processor=processor,
        num_chunks=None,
        chunk_idx=None,
    )

    # per (layer, query_head) -> list of 8-dim vectors
    accum = defaultdict(list)
    for video_idx, video_sample in enumerate(tqdm(filtered_anno, desc="Videos")):
        video_tensor = load_video_tensor(analyzer, video_sample)
        model.clear_cache()
        model.encode_init_prompt()
        token_times = []
        token_chunks = []
        current_frame_idx = 0
        chunk_idx = 0
        conversations = video_sample.get("conversations", [])
        if args.max_questions is not None:
            conversations = conversations[: args.max_questions]

        for question_idx, sample in enumerate(conversations):
            question_time = float(sample.get("end_time", len(video_tensor) / args.sample_fps))
            end_frame_idx = min(len(video_tensor), int(math.ceil(question_time * args.sample_fps)))
            while current_frame_idx < end_frame_idx:
                next_encode_end = min(current_frame_idx + args.encode_chunk_size, end_frame_idx)
                video_chunk = video_tensor[current_frame_idx:next_encode_end]
                pre_encode_len = int(model.kv_cache[0][0].shape[2]) if model.kv_cache is not None else 0
                model.encode_video_chunk(video_chunk)
                post_encode_len = int(model.kv_cache[0][0].shape[2])
                new_times, new_chunks, _ = token_times_for_chunk(
                    model,
                    processor,
                    video_chunk,
                    current_frame_idx,
                    chunk_idx,
                    args.sample_fps,
                    new_visual_tokens=post_encode_len - pre_encode_len,
                )
                token_times.extend(new_times)
                token_chunks.extend(new_chunks)
                current_frame_idx = next_encode_end
                chunk_idx += 1
                torch.cuda.empty_cache()

            prompt = build_future_prompt(analyzer, model, sample)
            attn = model.compute_eager_attentions(tokenize(processor, prompt, model.device))
            for layer_idx in range(num_layers):
                query_matrix, kv_matrix = profile_vectors_for_layer(
                    attn[layer_idx],
                    model,
                    token_times,
                    question_time,
                    num_query_heads,
                    num_kv_heads,
                )
                if query_matrix is None:
                    continue
                for qh in range(num_query_heads):
                    accum[(layer_idx, qh)].append(query_matrix[qh])
            del attn
            torch.cuda.empty_cache()

    per_layer = {}
    for layer_idx in range(num_layers):
        rows = []
        for qh in range(num_query_heads):
            vecs = accum.get((layer_idx, qh), [])
            rows.append(np.mean(vecs, axis=0) if vecs else np.zeros(len(AGE_BINS_SEC)))
        query_matrix = np.stack(rows)
        group_size = num_query_heads // max(num_kv_heads, 1)
        kv_matrix = np.stack(
            [query_matrix[i * group_size : (i + 1) * group_size].mean(axis=0) for i in range(num_kv_heads)]
        )
        q_z = column_zscore(query_matrix)
        kv_z = column_zscore(query_matrix)[
            np.repeat(np.arange(num_kv_heads), group_size)
        ].reshape(num_kv_heads, group_size, -1).mean(axis=1)
        per_layer[layer_idx] = {
            "r2_raw": variance_decomposition(query_matrix, num_kv_heads),
            "r2_z": variance_decomposition(q_z, num_kv_heads),
            "query_head_cos_raw": cosine_stats(query_matrix),
            "kv_head_cos_raw": cosine_stats(kv_matrix),
            "query_head_cos_z": cosine_stats(q_z),
            "kv_head_cos_z": cosine_stats(kv_z),
        }

    def mean(key):
        values = [per_layer[l][key] for l in per_layer if not math.isnan(per_layer[l][key])]
        return float(np.mean(values)) if values else float("nan")

    summary = {
        "num_videos": args.num_videos,
        "num_layers": num_layers,
        "num_query_heads": num_query_heads,
        "num_kv_heads": num_kv_heads,
        "mean_r2_raw": mean("r2_raw"),
        "mean_r2_z": mean("r2_z"),
        "mean_query_head_cos_raw": mean("query_head_cos_raw"),
        "mean_kv_head_cos_raw": mean("kv_head_cos_raw"),
        "mean_query_head_cos_z": mean("query_head_cos_z"),
        "mean_kv_head_cos_z": mean("kv_head_cos_z"),
        "per_layer": {str(k): v for k, v in per_layer.items()},
    }
    out_path = os.path.join(args.save_dir, "queryhead_vs_kvhead_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
