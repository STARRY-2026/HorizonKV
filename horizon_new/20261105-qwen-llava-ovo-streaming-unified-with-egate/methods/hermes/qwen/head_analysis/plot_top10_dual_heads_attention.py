"""Plot actual attention curves for top-10 dual-score KV heads.

Captures per-token attention distribution (over visual KV cache) for the
highest-scoring heads from ours_dual_top100.csv, using a single video + question.
"""

import argparse
import json
import math
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from head_analysis.obs_prev_current_chunk_attention_eager import build_observed_model
from head_analysis.profile_effective_memory_readout import (
    pool_query_attention,
    query_heads_for_kv,
    load_video_tensor,
    build_future_prompt,
    tokenize,
)
from video_qa.base import BaseVQA

TOP10_HEADS = [
    (0, 2),    # L0-K2: 1.01
    (22, 0),   # L22-K0: 1.01
    (19, 1),   # L19-K1: 0.870
    (27, 1),   # L27-K1: 0.870
    (0, 3),    # L0-K3: 0.857
    (18, 2),   # L18-K2: 0.831
    (19, 2),   # L19-K2: 0.832
    (21, 2),   # L21-K2: 0.803
    (14, 2),   # L14-K2: 0.795
    (2, 0),    # L2-K0: 0.785
]

NUM_QUERY_HEADS = 28
NUM_KV_HEADS = 4


def capture_attention_curves(model, processor, analyzer, video_tensor, sample, args):
    """Encode video chunk-by-chunk, capture attention after final chunk using real prompt."""
    model.clear_cache()
    model.encode_init_prompt()

    # Encode several chunks to build up history
    current_frame_idx = 0
    encode_chunk_size = args.encode_chunk_size
    num_chunks_before = args.num_chunks_before

    for _ in range(num_chunks_before):
        if current_frame_idx + encode_chunk_size > len(video_tensor):
            break
        next_end = min(current_frame_idx + encode_chunk_size, len(video_tensor))
        video_chunk = video_tensor[current_frame_idx:next_end]
        pre_lens = model._get_cache_seq_len_per_layer()
        model.encode_video_chunk(video_chunk)
        post_lens = model._get_cache_seq_len_per_layer()
        current_frame_idx = next_end

        # Compress
        if model.compress_mode == "streamingvlm":
            model._sliding_window_compress()
        else:
            local_q, global_q = model.predict_next_question()
            model.pseudo_forward(local_q, global_q)

    # Encode one more chunk and capture attention
    if current_frame_idx + encode_chunk_size > len(video_tensor):
        end_idx = len(video_tensor)
    else:
        end_idx = current_frame_idx + encode_chunk_size
    if end_idx <= current_frame_idx:
        print("No more frames to encode")
        return None

    video_chunk = video_tensor[current_frame_idx:end_idx]
    pre_lens = model._get_cache_seq_len_per_layer()
    model.encode_video_chunk(video_chunk)
    post_lens = model._get_cache_seq_len_per_layer()

    # Capture attention with real question prompt
    future_prompt = build_future_prompt(analyzer, model, sample)
    attn_future = model.compute_eager_attentions(tokenize(processor, future_prompt, model.device))

    visual_start = int(model.visual_start_idx)
    results = {}
    for layer_idx in range(len(attn_future)):
        af = attn_future[layer_idx]
        if af is None:
            continue
        future = pool_query_attention(af, mode="mean")
        if future is None:
            continue

        kv_len = int(future.shape[-1])
        if layer_idx < len(pre_lens):
            pre_len = pre_lens[layer_idx]
            post_len = post_lens[layer_idx]
        else:
            pre_len = kv_len
            post_len = kv_len

        for kv_head in range(NUM_KV_HEADS):
            key = (layer_idx, kv_head)
            if key not in TOP10_HEADS:
                continue
            q_heads = query_heads_for_kv(kv_head, NUM_QUERY_HEADS, NUM_KV_HEADS)
            q_idx = torch.as_tensor(q_heads, device=future.device, dtype=torch.long)
            group_attn = future.index_select(0, q_idx).mean(dim=0)[:kv_len]
            group_attn = torch.clamp(group_attn, min=0.0)

            # Full visual token attention curve
            vis_start = min(int(visual_start), kv_len)
            vis_end = min(int(post_len), kv_len)
            if vis_end > vis_start:
                curve = group_attn[vis_start:vis_end].detach().cpu().numpy()
                results[(layer_idx, kv_head)] = {
                    "curve": curve,
                    "pre_len": int(pre_len),
                    "post_len": int(post_len),
                    "visual_start": vis_start,
                }

    return results


def plot_results(results, save_path):
    """Plot attention curves in a 5x2 grid."""
    fig, axes = plt.subplots(5, 2, figsize=(18, 24))
    axes = axes.flatten()

    for idx, (layer, kv_head) in enumerate(TOP10_HEADS):
        ax = axes[idx]
        key = (layer, kv_head)
        if key not in results:
            ax.set_title(f"L{layer}-K{kv_head} (no data)")
            continue

        data = results[key]
        curve = data["curve"]
        pre_len = data["pre_len"]
        post_len = data["post_len"]
        vis_start = data["visual_start"]

        # Compute absolute positions
        positions = np.arange(len(curve)) + vis_start

        # Mark memory_core and current zones
        boundary_window = 64
        mem_start = vis_start
        mem_end = min(pre_len, positions[-1] + 1)
        mem_core_start = min(mem_end, mem_start + boundary_window)
        mem_core_end = max(mem_core_start, mem_end - boundary_window)

        ax.plot(positions, curve, linewidth=0.6, color="steelblue", alpha=0.8)

        # Mark memory core region
        if mem_core_end > mem_core_start:
            ax.axvspan(mem_core_start, mem_core_end, alpha=0.12, color="green",
                       label="memory core (internal)")

        # Mark current chunk region
        if post_len > pre_len:
            ax.axvspan(pre_len, post_len, alpha=0.12, color="orange",
                       label="current chunk")

        # Mark pre/post boundary
        if pre_len >= vis_start:
            ax.axvline(x=pre_len, color="red", linestyle="--", linewidth=0.8,
                       label="chunk boundary")

        # Compute zone stats for annotation
        if mem_core_end > mem_core_start:
            internal_mass = curve[mem_core_start - vis_start:mem_core_end - vis_start].sum()
        else:
            internal_mass = 0.0
        current_start = max(pre_len - vis_start, 0)
        current_end = min(post_len - vis_start, len(curve))
        if current_end > current_start:
            current_mass = curve[current_start:current_end].sum()
        else:
            current_mass = 0.0
        total_mass = curve.sum()

        internal_share = internal_mass / (total_mass + 1e-12)
        current_share = current_mass / (total_mass + 1e-12)

        ax.set_title(
            f"L{layer}-K{kv_head}  "
            f"long={internal_share:.3f}  short={current_share:.3f}",
            fontsize=10,
        )
        ax.set_xlabel("visual token position")
        ax.set_ylabel("attention")
        if idx in (0, 1):
            ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        "Top-10 Dual-Score Heads: Real Attention Distribution over Visual KV Cache",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {save_path}")


def run(args):
    model_name = "Qwen2.5-VL-7B-Instruct" if args.model == "qwen2.5_vl_7b" else args.model
    model_path = f"models/{model_name}"
    print(f"Loading model on cuda:{args.device}: {model_path}")
    model, processor = build_observed_model(args, model_path)

    with open(args.anno_path) as f:
        anno = json.load(f)
    video_sample = anno[args.video_idx]

    class TempBase(BaseVQA):
        pass
    analyzer = TempBase(
        anno=anno,
        save_dir="/tmp/plot_top10_dual_tmp",
        sample_fps=args.sample_fps,
        qa_model=model,
        qa_processor=processor,
        num_chunks=None,
        chunk_idx=None,
    )
    video_tensor = load_video_tensor(analyzer, video_sample)

    conversations = video_sample.get("conversations", [])
    question_idx = min(args.question_idx, len(conversations) - 1)
    sample = conversations[question_idx]
    print(f"Video idx={args.video_idx}, question_idx={question_idx}, task={sample.get('task', '?')}")

    results = capture_attention_curves(model, processor, analyzer, video_tensor, sample, args)
    if results:
        print(f"Captured {len(results)} head attention curves")
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(args.save_dir, "top10_dual_heads_attention.png")
        plot_results(results, save_path)

        # Also save numeric data
        npz_path = os.path.join(args.save_dir, "top10_dual_heads_attention.npz")
        np.savez(npz_path, **{f"L{l}_K{k}": results[(l, k)]["curve"]
                              for (l, k) in results})
        print(f"Saved numeric data: {npz_path}")
    else:
        print("No attention curves captured - video may be too short.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5_vl_7b")
    parser.add_argument("--anno_path",
                        default="data/streamingbench/streamingbench_realtime_half.json")
    parser.add_argument("--sample_fps", type=float, default=0.5)
    parser.add_argument("--kv_size", type=int, default=6000)
    parser.add_argument("--compress_mode", default="hermes")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--video_idx", type=int, default=0,
                        help="Which video to use (0-based)")
    parser.add_argument("--question_idx", type=int, default=0,
                        help="Which question to use for the prompt")
    parser.add_argument("--encode_chunk_size", type=int, default=16)
    parser.add_argument("--num_chunks_before", type=int, default=4,
                        help="How many chunks to encode before capturing attention")
    parser.add_argument("--save_dir",
                        default="results/observations/top10_dual_heads_attention")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
