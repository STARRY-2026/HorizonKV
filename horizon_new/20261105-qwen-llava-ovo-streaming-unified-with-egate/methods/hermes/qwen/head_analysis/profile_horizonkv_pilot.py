"""Pilot H/N profiling for Qwen2.5-VL on StreamingBench.

This script is intentionally offline and full-prefix for provenance quality:
video tokens are never compressed, so every visual token can be mapped back to
an approximate sampled-frame time and chunk id. It is meant for a small pilot
subset, not full-benchmark profiling.
"""

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
from collections import defaultdict

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from head_analysis.obs_prev_current_chunk_attention_eager import EagerObservedQwenVL
from head_analysis.profile_effective_memory_readout import (
    build_future_prompt,
    load_video_tensor,
    pool_query_attention,
    query_heads_for_kv,
    tokenize,
)
from inference.abstract_hermes import Abstract_Hermes
from inference.reindex_3d import _get_mrope_section
from video_qa.base import BaseVQA


EPS = 1e-12

PSEUDO_PROBES = {
    "neutral": "Summarize the important visual information observed in the video so far.",
    "history": (
        "What information introduced much earlier in the video is still important "
        "for understanding the video now?"
    ),
    "latest": "What new visual information appeared or changed in the most recent segment?",
}

AGE_BINS_SEC = [
    ("0-4s", 0.0, 4.0),
    ("4-8s", 4.0, 8.0),
    ("8-16s", 8.0, 16.0),
    ("16-32s", 16.0, 32.0),
    ("32-64s", 32.0, 64.0),
    ("64-128s", 64.0, 128.0),
    ("128-256s", 128.0, 256.0),
    ("256s+", 256.0, float("inf")),
]


def build_pilot_model(args, model_path):
    if args.model == "llava_ov_7b":
        from head_analysis.obs_llavaov_attention_eager import (
            build_observed_llava_model,
        )

        return build_observed_llava_model(args, model_path)

    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

    device = f"cuda:{args.device}" if args.device >= 0 else "cuda"
    if int(args.fixed_frame_canvas_size or 0) > 0:
        fixed_px = int(args.fixed_frame_canvas_size) * int(args.fixed_frame_canvas_size)
        processor = Qwen2_5_VLProcessor.from_pretrained(
            model_path,
            min_pixels=fixed_px,
            max_pixels=fixed_px,
        )
    else:
        processor = Qwen2_5_VLProcessor.from_pretrained(model_path)

    system_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
    init_prompt_ids = processor.tokenizer(system_prompt, return_tensors="pt").input_ids.to(device)

    raw_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.bfloat16,
    )

    model = EagerObservedQwenVL.__new__(EagerObservedQwenVL)
    model.__dict__ = raw_model.__dict__.copy()

    Abstract_Hermes.__init__(model, processor, init_prompt_ids.tolist(), args.kv_size)
    model.streaming = True
    model.sample_fps = args.sample_fps
    model.compress_mode = args.compress_mode
    model.fixed_frame_canvas_size = int(args.fixed_frame_canvas_size or 0)
    model.fixed_frame_pad_mode = "letterbox"
    model.fixed_frame_pad_value = 0.0

    num_layers = raw_model.model.config.num_hidden_layers
    model.num_layers = num_layers
    model._position_ids_cache = [None for _ in range(num_layers)]
    model.short_term_ratio = 0.1
    model.long_term_ratio = 0.3
    model.short_term_threshold = int(model.num_layers * model.short_term_ratio)
    model.long_term_threshold = int(model.num_layers * (1 - model.long_term_ratio))
    model.total_processed_frames = 0
    model._mrope_section = _get_mrope_section(raw_model.model)
    model._layer_position_ids = {}
    model._hook_handles = []
    model._register_forward_hooks()
    model._temporal_profile_family = "qwen2_5_vl"
    model.eval()
    return model, processor


def finite_mean(values):
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else None


def finite_median(values):
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(statistics.median(vals)) if vals else None


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def last_end_time(sample):
    ends = [
        float(conv["end_time"])
        for conv in sample.get("conversations", [])
        if "end_time" in conv
    ]
    return max(ends) if ends else 0.0


def task_set(sample):
    return sorted(
        {
            str(conv.get("task", "Unknown"))
            for conv in sample.get("conversations", [])
        }
    )


def select_pilot_subset(anno, num_videos, max_last_end, seed):
    candidates = [
        (idx, sample, last_end_time(sample), task_set(sample))
        for idx, sample in enumerate(anno)
        if last_end_time(sample) > 0 and last_end_time(sample) <= max_last_end
    ]
    if len(candidates) < num_videos:
        raise ValueError(
            f"Only {len(candidates)} videos have last_end <= {max_last_end}; "
            f"need {num_videos}."
        )

    candidates = sorted(candidates, key=lambda item: item[2])
    n_short = max(1, num_videos * 3 // 10)
    n_long = max(1, num_videos * 3 // 10)
    n_mid = num_videos - n_short - n_long

    short_pool = candidates[: max(n_short * 3, n_short)]
    long_pool = candidates[-max(n_long * 3, n_long) :]
    mid_start = max(0, len(candidates) // 2 - max(n_mid * 2, n_mid) // 2)
    mid_pool = candidates[mid_start : mid_start + max(n_mid * 2, n_mid)]

    rng = random.Random(seed)
    selected = []
    used = set()
    for pool, count in [(short_pool, n_short), (mid_pool, n_mid), (long_pool, n_long)]:
        pool = [item for item in pool if item[0] not in used]
        rng.shuffle(pool)
        for item in pool[:count]:
            selected.append(item)
            used.add(item[0])

    if len(selected) < num_videos:
        for item in candidates:
            if item[0] not in used:
                selected.append(item)
                used.add(item[0])
                if len(selected) >= num_videos:
                    break

    selected = sorted(selected, key=lambda item: item[0])[:num_videos]
    filtered = []
    manifest = []
    for idx, sample, end_time, tasks in selected:
        copied = dict(sample)
        copied["_source_index"] = idx
        filtered.append(copied)
        manifest.append(
            {
                "source_index": idx,
                "video_id": sample.get("video_id", idx),
                "video_path": sample.get("video_path"),
                "last_end_time": end_time,
                "num_questions": len(sample.get("conversations", [])),
                "tasks": tasks,
            }
        )
    return filtered, manifest


def token_times_for_chunk(
    model,
    processor,
    video_chunk,
    start_frame_idx,
    chunk_idx,
    sample_fps,
    new_visual_tokens=None,
):
    if getattr(model, "_temporal_profile_family", None) == "llava_ov":
        frame_count = int(video_chunk.shape[0])
        if not new_visual_tokens or int(new_visual_tokens) % frame_count != 0:
            raise RuntimeError(
                "LLaVA visual-token count must be positive and divisible by the "
                f"number of frames: tokens={new_visual_tokens}, frames={frame_count}"
            )
        tokens_per_frame = int(new_visual_tokens) // frame_count
        times = []
        chunks = []
        for frame_offset in range(frame_count):
            frame_time = (start_frame_idx + frame_offset) / float(sample_fps)
            times.extend([frame_time] * tokens_per_frame)
            chunks.extend([int(chunk_idx)] * tokens_per_frame)
        return times, chunks, [frame_count, tokens_per_frame]

    processor_chunk = video_chunk
    if (
        int(getattr(model, "fixed_frame_canvas_size", 0) or 0) > 0
        and hasattr(model, "_apply_fixed_frame_canvas")
    ):
        processor_chunk = model._apply_fixed_frame_canvas(processor_chunk)
    if len(processor_chunk.shape) == 4 and processor_chunk.shape[-1] == 3:
        processor_chunk = processor_chunk.permute(0, 3, 1, 2)
    video_input = processor(text=[""], videos=processor_chunk, return_tensors="pt")
    grid_t, grid_h, grid_w = [int(v) for v in video_input["video_grid_thw"][0].tolist()]
    spatial_merge_size = int(getattr(model.config.vision_config, "spatial_merge_size", 2))
    tokens_per_t = (grid_h // spatial_merge_size) * (grid_w // spatial_merge_size)
    frame_count = int(video_chunk.shape[0])
    times = []
    chunks = []
    for t_idx in range(grid_t):
        local_frame = min(frame_count - 1, t_idx * 2)
        frame_time = (start_frame_idx + local_frame) / float(sample_fps)
        times.extend([frame_time] * tokens_per_t)
        chunks.extend([int(chunk_idx)] * tokens_per_t)
    return times, chunks, video_input["video_grid_thw"][0].tolist()


def age_bin_label(age_sec):
    for label, lo, hi in AGE_BINS_SEC:
        if age_sec >= lo and age_sec < hi:
            return label
    return AGE_BINS_SEC[-1][0]


def delta_bin_label(delta):
    delta = int(delta)
    if delta <= 0:
        return "0"
    if delta == 1:
        return "1"
    if delta == 2:
        return "2"
    if delta == 3:
        return "3"
    return "4+"


def topk_bin_statistics(values, topk_values):
    """Return strong-read statistics without retaining per-token attention."""
    if values.numel() == 0:
        return {
            f"top{k}_{metric}": None
            for k in topk_values
            for metric in ("mean_attention", "mass", "concentration")
        }

    total = float(values.sum().item())
    out = {}
    for k in topk_values:
        k_eff = min(int(k), int(values.numel()))
        top = torch.topk(values, k=k_eff, sorted=False).values
        top_mass = float(top.sum().item())
        out[f"top{k}_mean_attention"] = float(top.mean().item())
        out[f"top{k}_mass"] = top_mass
        out[f"top{k}_concentration"] = top_mass / (total + EPS)
    return out


def unit_balanced_topk_statistics(values, source_times, topk_values):
    """Average per-time-unit top-K statistics to avoid wider-bin candidate bias."""
    if values.numel() == 0:
        return {
            **{
                f"unit_top{k}_{metric}": None
                for k in topk_values
                for metric in ("mean_attention", "concentration")
            },
            "num_temporal_units": 0,
        }

    grouped_indices = defaultdict(list)
    for token_idx, source_time in enumerate(source_times):
        grouped_indices[float(source_time)].append(token_idx)

    units = []
    for indices in grouped_indices.values():
        index_tensor = torch.as_tensor(indices, device=values.device, dtype=torch.long)
        units.append(values.index_select(0, index_tensor))

    unit_size = min(int(unit.numel()) for unit in units)
    unit_matrix = torch.stack([unit[:unit_size] for unit in units], dim=0)
    max_k = min(max(topk_values), unit_size)
    top_values = torch.topk(unit_matrix, k=max_k, dim=1, sorted=True).values
    cumulative_mass = top_values.cumsum(dim=1)
    unit_mass = unit_matrix.sum(dim=1)

    packed = []
    for k in topk_values:
        k_eff = min(int(k), unit_size)
        top_mass = cumulative_mass[:, k_eff - 1]
        packed.extend(
            [
                top_mass.mean() / float(k_eff),
                (top_mass / (unit_mass + EPS)).mean(),
            ]
        )
    packed_values = torch.stack(packed).detach().cpu().tolist()

    out = {"num_temporal_units": len(grouped_indices)}
    for offset, k in enumerate(topk_values):
        out[f"unit_top{k}_mean_attention"] = float(packed_values[2 * offset])
        out[f"unit_top{k}_concentration"] = float(packed_values[2 * offset + 1])
    return out


def attention_bin_rows(
    model,
    attns,
    query_source,
    query_name,
    meta,
    token_times,
    token_chunks,
    current_chunk_idx,
    question_time,
    args,
):
    rows = []
    visual_start = int(model.visual_start_idx)
    visual_count = len(token_times)
    if visual_count <= 0:
        return rows

    age_labels = [age_bin_label(float(question_time) - float(t)) for t in token_times]
    delta_labels = [delta_bin_label(int(current_chunk_idx) - int(c)) for c in token_chunks]
    age_counts = defaultdict(int)
    delta_counts = defaultdict(int)
    for label in age_labels:
        age_counts[label] += 1
    for label in delta_labels:
        delta_counts[label] += 1

    for layer_idx, attn in enumerate(attns):
        pooled = pool_query_attention(attn, mode=args.query_pool, last_n=args.last_n)
        if pooled is None:
            continue
        kv_len = int(pooled.shape[-1])
        v_start = min(visual_start, kv_len)
        v_end = min(visual_start + visual_count, kv_len)
        if v_end <= v_start:
            continue
        usable_count = v_end - v_start
        local_age_labels = age_labels[:usable_count]
        local_delta_labels = delta_labels[:usable_count]
        local_token_times = token_times[:usable_count]

        for kv_head in range(int(args.num_kv_heads)):
            q_heads = query_heads_for_kv(kv_head, args.num_query_heads, args.num_kv_heads)
            q_idx = torch.as_tensor(q_heads, device=pooled.device, dtype=torch.long)
            group_attn = pooled.index_select(0, q_idx).mean(dim=0).float()
            visual_attn = torch.clamp(group_attn[v_start:v_end], min=0.0)
            visual_mass = float(visual_attn.sum().item())
            if visual_mass <= EPS:
                continue

            age_mass = defaultdict(float)
            delta_mass = defaultdict(float)
            age_indices = defaultdict(list)
            vals = visual_attn.detach().cpu().tolist()
            for token_idx, (value, age_label, delta_label) in enumerate(
                zip(vals, local_age_labels, local_delta_labels)
            ):
                age_mass[age_label] += float(value)
                delta_mass[delta_label] += float(value)
                age_indices[age_label].append(token_idx)

            for label, _, _ in AGE_BINS_SEC:
                count = int(age_counts.get(label, 0))
                mass = float(age_mass.get(label, 0.0))
                token_prior = count / float(visual_count) if visual_count else 0.0
                mass_share = mass / (visual_mass + EPS)
                indices = age_indices.get(label, [])
                if indices:
                    index_tensor = torch.as_tensor(
                        indices, device=visual_attn.device, dtype=torch.long
                    )
                    bin_attention = visual_attn.index_select(0, index_tensor)
                    bin_source_times = [local_token_times[i] for i in indices]
                else:
                    bin_attention = visual_attn[:0]
                    bin_source_times = []
                rows.append(
                    {
                        **meta,
                        "query_source": query_source,
                        "query_name": query_name,
                        "bin_type": "age_sec",
                        "bin_label": label,
                        "layer": layer_idx,
                        "kv_head": kv_head,
                        "token_count": count,
                        "visual_token_count": visual_count,
                        "attention_mass": mass,
                        "visual_attention_mass": visual_mass,
                        "mass_share": mass_share,
                        "token_prior": token_prior,
                        "reuse_score": math.log1p(mass_share / (token_prior + EPS))
                        if count > 0
                        else 0.0,
                        **topk_bin_statistics(bin_attention, args.topk_values),
                        **unit_balanced_topk_statistics(
                            bin_attention, bin_source_times, args.topk_values
                        ),
                    }
                )

            for label in ["0", "1", "2", "3", "4+"]:
                count = int(delta_counts.get(label, 0))
                mass = float(delta_mass.get(label, 0.0))
                token_prior = count / float(visual_count) if visual_count else 0.0
                mass_share = mass / (visual_mass + EPS)
                rows.append(
                    {
                        **meta,
                        "query_source": query_source,
                        "query_name": query_name,
                        "bin_type": "delta_chunk",
                        "bin_label": label,
                        "layer": layer_idx,
                        "kv_head": kv_head,
                        "token_count": count,
                        "visual_token_count": visual_count,
                        "attention_mass": mass,
                        "visual_attention_mass": visual_mass,
                        "mass_share": mass_share,
                        "token_prior": token_prior,
                        "reuse_score": math.log1p(mass_share / (token_prior + EPS))
                        if count > 0
                        else 0.0,
                    }
                )
    return rows


def aggregate_scores(rows, query_source):
    grouped = defaultdict(list)
    for row in rows:
        if row["query_source"] == query_source and row["bin_type"] == "delta_chunk":
            grouped[(int(row["layer"]), int(row["kv_head"]), row["bin_label"])].append(row)

    by_head = defaultdict(dict)
    for (layer, kv_head, label), part in grouped.items():
        by_head[(layer, kv_head)][label] = finite_mean([r["reuse_score"] for r in part])

    out = []
    for (layer, kv_head), values in sorted(by_head.items()):
        n_values = [values.get("1"), values.get("2")]
        h_values = [values.get("4+")]
        n_score = finite_mean(n_values)
        h_score = finite_mean(h_values)
        out.append(
            {
                "layer": layer,
                "kv_head": kv_head,
                "N_delta1_2": n_score,
                "H_delta4plus": h_score,
                "HN_diff": (h_score - n_score)
                if h_score is not None and n_score is not None
                else None,
                "delta0_diagnostic": values.get("0"),
                "num_bins_observed": len(values),
            }
        )
    return out


def rank_corr(rows_a, rows_b, col):
    a = {(int(r["layer"]), int(r["kv_head"])): r.get(col) for r in rows_a}
    b = {(int(r["layer"]), int(r["kv_head"])): r.get(col) for r in rows_b}
    keys = [k for k in a if k in b and a[k] is not None and b[k] is not None]
    if len(keys) < 3:
        return None
    xs = [float(a[k]) for k in keys]
    ys = [float(b[k]) for k in keys]
    rx = {idx: rank for rank, idx in enumerate(sorted(range(len(xs)), key=lambda i: xs[i]))}
    ry = {idx: rank for rank, idx in enumerate(sorted(range(len(ys)), key=lambda i: ys[i]))}
    mx = sum(rx.values()) / len(rx)
    my = sum(ry.values()) / len(ry)
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(len(xs)))
    den_x = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(len(xs))))
    den_y = math.sqrt(sum((ry[i] - my) ** 2 for i in range(len(ys))))
    return float(num / (den_x * den_y + EPS))


def save_outputs(rows, filtered_anno, manifest, args):
    os.makedirs(args.save_dir, exist_ok=True)
    anno_path = os.path.join(args.save_dir, "filtered_anno_10v.json")
    with open(anno_path, "w") as f:
        json.dump(filtered_anno, f, indent=2)
    manifest_path = os.path.join(args.save_dir, "selection_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    raw_fields = [
        "video_idx",
        "source_index",
        "video_id",
        "question_idx",
        "task",
        "question_time",
        "current_chunk_idx",
        "query_source",
        "query_name",
        "bin_type",
        "bin_label",
        "layer",
        "kv_head",
        "token_count",
        "visual_token_count",
        "attention_mass",
        "visual_attention_mass",
        "mass_share",
        "token_prior",
        "reuse_score",
    ]
    for k in args.topk_values:
        raw_fields.extend(
            [
                f"top{k}_mean_attention",
                f"top{k}_mass",
                f"top{k}_concentration",
                f"unit_top{k}_mean_attention",
                f"unit_top{k}_concentration",
            ]
        )
    raw_fields.append("num_temporal_units")
    raw_path = os.path.join(args.save_dir, "raw_age_bin_attention.csv")
    write_csv(raw_path, rows, raw_fields)
    write_csv(
        os.path.join(args.save_dir, "raw_real_question_age_bins.csv"),
        [r for r in rows if r["query_source"] == "real"],
        raw_fields,
    )
    write_csv(
        os.path.join(args.save_dir, "raw_pseudo_query_age_bins.csv"),
        [r for r in rows if r["query_source"] == "pseudo"],
        raw_fields,
    )

    real_scores = aggregate_scores(rows, "real")
    pseudo_scores = aggregate_scores(rows, "pseudo")
    score_fields = [
        "layer",
        "kv_head",
        "N_delta1_2",
        "H_delta4plus",
        "HN_diff",
        "delta0_diagnostic",
        "num_bins_observed",
    ]
    real_path = os.path.join(args.save_dir, "head_scores_real.csv")
    pseudo_path = os.path.join(args.save_dir, "head_scores_pseudo.csv")
    write_csv(real_path, real_scores, score_fields)
    write_csv(pseudo_path, pseudo_scores, score_fields)

    pseudo_by_head = {(r["layer"], r["kv_head"]): r for r in pseudo_scores}
    compare = []
    for r in real_scores:
        p = pseudo_by_head.get((r["layer"], r["kv_head"]))
        if not p:
            continue
        compare.append(
            {
                "layer": r["layer"],
                "kv_head": r["kv_head"],
                "real_N": r["N_delta1_2"],
                "real_H": r["H_delta4plus"],
                "real_HN_diff": r["HN_diff"],
                "pseudo_N": p["N_delta1_2"],
                "pseudo_H": p["H_delta4plus"],
                "pseudo_HN_diff": p["HN_diff"],
                "HN_diff_gap": (
                    float(r["HN_diff"]) - float(p["HN_diff"])
                    if r["HN_diff"] is not None and p["HN_diff"] is not None
                    else None
                ),
            }
        )
    compare_path = os.path.join(args.save_dir, "real_vs_pseudo_compare.csv")
    write_csv(
        compare_path,
        compare,
        [
            "layer",
            "kv_head",
            "real_N",
            "real_H",
            "real_HN_diff",
            "pseudo_N",
            "pseudo_H",
            "pseudo_HN_diff",
            "HN_diff_gap",
        ],
    )

    summary = {
        "mode": "full_prefix_no_compression_pilot",
        "fixed_frame_canvas_size": int(args.fixed_frame_canvas_size or 0),
        "save_dir": args.save_dir,
        "filtered_anno": anno_path,
        "selection_manifest": manifest_path,
        "raw_csv": raw_path,
        "real_scores_csv": real_path,
        "pseudo_scores_csv": pseudo_path,
        "compare_csv": compare_path,
        "num_videos": len(filtered_anno),
        "num_questions_in_filtered_anno": sum(len(v.get("conversations", [])) for v in filtered_anno),
        "num_profiled_questions": len(
            {
                (r["video_idx"], r["question_idx"])
                for r in rows
            }
        ),
        "num_raw_rows": len(rows),
        "num_layer_kv_heads": int(args.num_layers) * int(args.num_kv_heads),
        "query_pool": args.query_pool,
        "last_n": args.last_n,
        "topk_values": args.topk_values,
        "age_bins_sec": [b[0] for b in AGE_BINS_SEC],
        "delta_bins": ["0", "1", "2", "3", "4+"],
        "real_pseudo_spearman_N": rank_corr(real_scores, pseudo_scores, "N_delta1_2"),
        "real_pseudo_spearman_H": rank_corr(real_scores, pseudo_scores, "H_delta4plus"),
        "real_pseudo_spearman_HN_diff": rank_corr(real_scores, pseudo_scores, "HN_diff"),
        "mean_real_N": finite_mean([r["N_delta1_2"] for r in real_scores]),
        "mean_real_H": finite_mean([r["H_delta4plus"] for r in real_scores]),
        "mean_pseudo_N": finite_mean([r["N_delta1_2"] for r in pseudo_scores]),
        "mean_pseudo_H": finite_mean([r["H_delta4plus"] for r in pseudo_scores]),
        "max_last_end_time": max(last_end_time(v) for v in filtered_anno) if filtered_anno else None,
    }
    summary_path = os.path.join(args.save_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def run(args):
    os.makedirs(args.save_dir, exist_ok=True)
    with open(args.anno_path) as f:
        full_anno = json.load(f)
    filtered_anno, manifest = select_pilot_subset(
        full_anno,
        args.num_videos,
        args.max_last_end,
        args.seed,
    )
    if args.video_root:
        missing = []
        for sample in filtered_anno:
            video_path = os.path.join(
                args.video_root, os.path.basename(sample.get("video_path", ""))
            )
            sample["video_path"] = video_path
            if not os.path.isfile(video_path):
                missing.append(video_path)
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} selected videos are missing under --video_root; "
                f"first missing path: {missing[0]}"
            )

    model_names = {
        "qwen2.5_vl_7b": "Qwen2.5-VL-7B-Instruct",
        "llava_ov_7b": "llava-onevision-qwen2-7b-ov-hf",
    }
    model_name = model_names.get(args.model, args.model)
    model_path = f"models/{model_name}"
    print(f"Loading model on cuda:{args.device}: {model_path}")
    model, processor = build_pilot_model(args, model_path)
    text_config = model.language_model.config
    args.num_layers = int(text_config.num_hidden_layers)
    args.num_query_heads = int(text_config.num_attention_heads)
    args.num_kv_heads = int(
        getattr(text_config, "num_key_value_heads", args.num_query_heads)
    )
    print(
        "Profile structure: "
        f"layers={args.num_layers}, query_heads={args.num_query_heads}, "
        f"kv_heads={args.num_kv_heads}"
    )

    class TempBase(BaseVQA):
        pass

    analyzer = TempBase(
        anno=filtered_anno,
        save_dir="/tmp/horizonkv_pilot_tmp",
        sample_fps=args.sample_fps,
        qa_model=model,
        qa_processor=processor,
        num_chunks=None,
        chunk_idx=None,
    )

    rows = []
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
            end_frame_idx = min(
                len(video_tensor),
                int(math.ceil(question_time * args.sample_fps)),
            )
            while current_frame_idx < end_frame_idx:
                next_encode_end = min(current_frame_idx + args.encode_chunk_size, end_frame_idx)
                video_chunk = video_tensor[current_frame_idx:next_encode_end]
                pre_encode_len = (
                    int(model.kv_cache[0][0].shape[2]) if model.kv_cache is not None else 0
                )
                model.encode_video_chunk(video_chunk)
                post_encode_len = int(model.kv_cache[0][0].shape[2])
                new_times, new_chunks, grid = token_times_for_chunk(
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

            current_chunk_idx = max(0, chunk_idx - 1)
            meta = {
                "video_idx": video_idx,
                "source_index": video_sample.get("_source_index", video_idx),
                "video_id": video_sample.get("video_id", video_idx),
                "question_idx": question_idx,
                "task": sample.get("task", "Unknown"),
                "question_time": question_time,
                "current_chunk_idx": current_chunk_idx,
            }

            real_prompt = build_future_prompt(analyzer, model, sample)
            real_attn = model.compute_eager_attentions(tokenize(processor, real_prompt, model.device))
            rows.extend(
                attention_bin_rows(
                    model,
                    real_attn,
                    "real",
                    "streamingbench_question",
                    meta,
                    token_times,
                    token_chunks,
                    current_chunk_idx,
                    question_time,
                    args,
                )
            )
            del real_attn
            torch.cuda.empty_cache()

            for probe_name, probe_text in PSEUDO_PROBES.items():
                prompt = model.get_prompt(probe_text)
                attn = model.compute_eager_attentions(tokenize(processor, prompt, model.device))
                rows.extend(
                    attention_bin_rows(
                        model,
                        attn,
                        "pseudo",
                        probe_name,
                        meta,
                        token_times,
                        token_chunks,
                        current_chunk_idx,
                        question_time,
                        args,
                    )
                )
                del attn
                torch.cuda.empty_cache()

    summary = save_outputs(rows, filtered_anno, manifest, args)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5_vl_7b")
    parser.add_argument("--anno_path", default="data/streamingbench/streamingbench_realtime.json")
    parser.add_argument(
        "--video_root",
        default=None,
        help="Optional local video directory overriding paths stored in the annotation.",
    )
    parser.add_argument("--sample_fps", type=float, default=0.5)
    parser.add_argument("--kv_size", type=int, default=6000)
    parser.add_argument("--compress_mode", default="hermes")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num_videos", type=int, default=10)
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument("--max_last_end", type=float, default=260.0)
    parser.add_argument("--encode_chunk_size", type=int, default=16)
    parser.add_argument("--fixed_frame_canvas_size", type=int, default=0)
    parser.add_argument("--num_layers", type=int, default=28)
    parser.add_argument("--num_query_heads", type=int, default=28)
    parser.add_argument("--num_kv_heads", type=int, default=4)
    parser.add_argument("--query_pool", choices=["mean", "last", "last_n"], default="last_n")
    parser.add_argument("--last_n", type=int, default=4)
    parser.add_argument(
        "--topk_values",
        default="1,4,16,64,100",
        help="Comma-separated K values for within-age-bin strong-read statistics.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save_dir",
        default="results/myobservations/horizonkv_pilot_10v",
    )
    args = parser.parse_args()
    args.topk_values = sorted(
        {int(value) for value in args.topk_values.split(",") if int(value) > 0}
    )
    if not args.topk_values:
        raise ValueError("--topk_values must contain at least one positive integer")
    run(args)


if __name__ == "__main__":
    main()
