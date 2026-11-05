"""
Task-Specific Attention Head Profiling.

Profiles which KV-head attention scores are triggered by task-specific
pseudo-questions, producing per-task score CSVs usable for task-adaptive
budget allocation.

Usage:
  python head_analysis/profile_task_specific_heads.py \
      --model qwen2.5_vl_7b --kv_size 6000 --sample_fps 0.5 \
      --anno data/streamingbench/streamingbench_realtime_201_250.json \
      --save_dir results/myobservations/task_specific_profiles

This script does NOT modify any existing code — it imports and calls
existing functions in read-only fashion.
"""

import argparse
import gc
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_qa.base import BaseVQA, MODELS


# ─── Task-specific pseudo-questions ───
# local: short-term / immediate perception
# global: long-term / cross-video integration

TASK_QUESTIONS = {
    "Counting": {
        "local": "How many objects or people are visible in the current frame?",
        "global": "How many distinct objects appear across the entire video?",
    },
    "Attribute Recognition": {
        "local": "What color, texture, or material properties are visible right now?",
        "global": "What visual attributes (color, material, shape) persist throughout the video?",
    },
    "Object Recognition": {
        "local": "What objects are present in the current scene?",
        "global": "What objects appear and are tracked across the entire video?",
    },
    "Action Recognition": {
        "local": "What action or motion is happening at this moment?",
        "global": "What sequence of actions takes place over the course of the video?",
    },
    "Causal Reasoning": {
        "local": "What just happened in the immediate moment?",
        "global": "What caused the chain of events to unfold across this video?",
    },
    "Event Understanding": {
        "local": "What event is occurring in the current segment?",
        "global": "What are the main events that structure the entire video?",
    },
    "Spatial Understanding": {
        "local": "Where are objects positioned relative to each other right now?",
        "global": "How do spatial relationships between objects change over the video?",
    },
    "Text-Rich Understanding": {
        "local": "What text or written content is visible in the current frame?",
        "global": "What textual information appears and is relevant across the video?",
    },
    "Prospective Reasoning": {
        "local": "What is the current state or situation?",
        "global": "Based on the video progression, what is most likely to happen next?",
    },
    "Clips Summarize": {
        "local": "What is the key information in this moment?",
        "global": "What is the overall summary or topic of this entire video?",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Task-specific attention head profiling"
    )
    parser.add_argument("--model", type=str, default="qwen2.5_vl_7b")
    parser.add_argument("--kv_size", type=int, default=6000)
    parser.add_argument("--sample_fps", type=float, default=0.5)
    parser.add_argument("--compress_mode", type=str, default="hermes")
    parser.add_argument("--anno", type=str,
                        default="data/streamingbench/streamingbench_realtime_201_250.json")
    parser.add_argument("--save_dir", type=str,
                        default="results/myobservations/task_specific_profiles")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max_videos", type=int, default=0,
                        help="Limit number of videos (0 = all)")
    parser.add_argument("--chunk_size", type=int, default=16,
                        help="Frames per video chunk")
    return parser.parse_args()


def load_model_with_ragged(args):
    """Load model with ragged prefill, matching existing experiment setup."""
    model_path = MODELS[args.model]["model_path"]
    load_func = MODELS[args.model]["load_func"]

    print(f"Loading model: {model_path}")
    videoqa_model, videoqa_processor = load_func(
        model_path=model_path,
        kv_size=args.kv_size,
        streaming=True,
        sample_fps=args.sample_fps,
        compress_mode=args.compress_mode,
    )

    # Install ragged prefill (same as --kv_head_ragged_prefill flag)
    from head_analysis.hermes_kv_head_ragged_prefill import (
        apply_kv_head_ragged_prefill,
        _ragged_query_scores,
    )

    videoqa_model = apply_kv_head_ragged_prefill(videoqa_model)

    # Store _ragged_query_scores for direct access
    videoqa_model._task_ragged_query_scores = _ragged_query_scores

    # Ensure score collection is available
    if not hasattr(videoqa_model, "num_layers"):
        videoqa_model.num_layers = len(videoqa_model.kv_cache.layers)

    return videoqa_model, videoqa_processor


def encode_video(model, video_tensor, chunk_size=16, compress_every=8):
    """Encode video chunks into the ragged cache, with periodic compression.

    compress_every: trigger predict_and_compress every N chunks to prevent
                    unbounded cache growth (matches real streaming pipeline).

    Returns True if at least one chunk was successfully encoded.
    """
    total_frames = len(video_tensor)
    frame_idx = 0
    chunk_count = 0
    any_ok = False
    while frame_idx < total_frames:
        next_end = min(frame_idx + chunk_size, total_frames)
        if next_end > frame_idx:
            try:
                model.encode_video_chunk(video_tensor[frame_idx:next_end])
                frame_idx = next_end
                chunk_count += 1
                any_ok = True
                # Periodic compression to cap cache size at kv_size
                if chunk_count % compress_every == 0:
                    current_len = max(model.kv_cache.max_lens_per_layer())
                    if current_len > model.kv_size:
                        model.predict_and_compress()
            except torch.cuda.OutOfMemoryError:
                print(f"  Warning: OOM at chunk [{frame_idx}:{next_end}], emptying cache")
                torch.cuda.empty_cache()
                frame_idx = next_end
            except Exception as e:
                print(f"  Warning: encode chunk [{frame_idx}:{next_end}] failed: {e}")
                frame_idx = next_end
        else:
            break
    return any_ok


def collect_task_scores(model, task_questions):
    """Collect per-KV-head attention scores for each task's pseudo-questions.

    Returns:
        dict: task -> {"local": [L, H_kv], "global": [L, H_kv], "mixed": [L, H_kv]}
    """
    from head_analysis.hermes_kv_head_ragged_prefill import (
        _ragged_query_scores,
        _attention_shape,
    )

    device = model.device
    num_query_heads, num_kv_heads, _ = _attention_shape(model)
    num_layers = model.num_layers

    offsets = model._get_next_global_offset_per_layer()

    all_task_scores = {}

    for task, questions in task_questions.items():
        local_q = questions["local"]
        global_q = questions["global"]
        mixed_q = local_q + "; " + global_q

        try:
            # Local scores
            local_ids = model.processor.tokenizer(local_q).input_ids
            local_ids = torch.as_tensor([local_ids], device=device, dtype=torch.long)
            local_scores = _ragged_query_scores(model, local_ids, offsets)
            local_kv = _scores_to_kv_matrix(local_scores, num_layers, num_kv_heads, num_query_heads)
            del local_scores

            # Global scores
            global_ids = model.processor.tokenizer(global_q).input_ids
            global_ids = torch.as_tensor([global_ids], device=device, dtype=torch.long)
            global_scores = _ragged_query_scores(model, global_ids, offsets)
            global_kv = _scores_to_kv_matrix(global_scores, num_layers, num_kv_heads, num_query_heads)
            del global_scores

            # Mixed scores
            mixed_ids = model.processor.tokenizer(mixed_q).input_ids
            mixed_ids = torch.as_tensor([mixed_ids], device=device, dtype=torch.long)
            mixed_scores = _ragged_query_scores(model, mixed_ids, offsets)
            mixed_kv = _scores_to_kv_matrix(mixed_scores, num_layers, num_kv_heads, num_query_heads)
            del mixed_scores

            all_task_scores[task] = {
                "local": local_kv,
                "global": global_kv,
                "mixed": mixed_kv,
            }

        except Exception as e:
            print(f"  Warning: failed to collect scores for task '{task}': {e}")
            all_task_scores[task] = None

    return all_task_scores


def _scores_to_kv_matrix(scores_list, num_layers, num_kv_heads, num_query_heads):
    """Convert per-layer score tensors to [L, H_kv] numpy matrix.

    scores_list: list of length num_layers, each entry is a list of H_kv tensors
                 of shape [group_size, q_len, hist_len] or similar.
    We take the mean attention over all dimensions to get a scalar per KV head.
    """
    group_size = num_query_heads // num_kv_heads
    kv_mat = np.zeros((num_layers, num_kv_heads), dtype=np.float64)

    for l, layer_scores in enumerate(scores_list):
        if layer_scores is None or len(layer_scores) == 0:
            continue
        for kv in range(min(num_kv_heads, len(layer_scores))):
            s = layer_scores[kv]
            if s is not None and s.numel() > 0:
                kv_mat[l, kv] = float(s.detach().cpu().mean().item())

    return kv_mat


def main():
    args = parse_args()

    # --- Set device ---
    if args.device >= 0:
        torch.cuda.set_device(args.device)
        print(f"Using GPU {args.device}")

    # --- Load model ---
    model, processor = load_model_with_ragged(args)

    # --- Load annotations ---
    with open(args.anno) as f:
        anno = json.load(f)

    if args.max_videos and args.max_videos > 0:
        anno = anno[:args.max_videos]

    print(f"Processing {len(anno)} videos")

    # --- Create analyzer for video loading ---
    class TempVQA(BaseVQA):
        pass

    analyzer = TempVQA(
        anno=anno, save_dir="/tmp/task_prof_tmp", sample_fps=args.sample_fps,
        qa_model=model, qa_processor=processor,
        num_chunks=None, chunk_idx=None,
    )

    # --- Accumulators: task -> [local, global, mixed] each [L, H_kv] ---
    num_layers = model.num_layers
    num_kv_heads = 4  # Qwen2.5-VL-7B

    task_accum = defaultdict(lambda: {
        "local": np.zeros((num_layers, num_kv_heads), dtype=np.float64),
        "global": np.zeros((num_layers, num_kv_heads), dtype=np.float64),
        "mixed": np.zeros((num_layers, num_kv_heads), dtype=np.float64),
    })
    task_counts = defaultdict(int)

    # --- Profiling loop ---
    for video_sample in tqdm(anno, desc="Profiling videos"):
        video_id = video_sample.get("video_id", "?")
        video_path = video_sample.get("video_path", "")

        # Load video
        try:
            if video_path.endswith(".npy"):
                video = analyzer.load_video(video_path, clip=video_sample.get("clip", None))
                video_tensor = torch.from_numpy(video)
            elif os.path.isdir(video_path):
                vfps = video_sample.get("fps", None) or 30
                video = analyzer.load_video_frames(video_path, vfps, clip=video_sample.get("clip", None))
                video_tensor = torch.from_numpy(video)
            else:
                video = analyzer.load_video(video_path, clip=video_sample.get("clip", None))
                video_tensor = torch.from_numpy(video)
        except Exception as e:
            print(f"  Skipping video {video_id}: failed to load ({e})")
            continue

        # Encode all frames into ragged cache
        model.clear_cache()
        model.encode_init_prompt()

        try:
            ok = encode_video(model, video_tensor, chunk_size=args.chunk_size)
            if not ok:
                print(f"  Warning: video {video_id} had no successful chunks, skipping")
                torch.cuda.empty_cache()
                continue
        except torch.cuda.OutOfMemoryError:
            print(f"  Warning: OOM during encode for video {video_id}")
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"  Warning: encode failed for video {video_id} ({e})")
            torch.cuda.empty_cache()
            continue

        # Final compression pass to cap cache at kv_size before score collection
        current_len = max(model.kv_cache.max_lens_per_layer())
        if current_len > model.kv_size:
            try:
                model.predict_and_compress()
            except Exception as e:
                print(f"  Warning: final compression failed for video {video_id}: {e}")

        # Collect task scores
        try:
            video_task_scores = collect_task_scores(model, TASK_QUESTIONS)
        except torch.cuda.OutOfMemoryError:
            print(f"  Warning: OOM during score collection for video {video_id}")
            video_task_scores = {}
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  Warning: score collection failed for video {video_id} ({e})")
            video_task_scores = {}
            torch.cuda.empty_cache()

        # Find which tasks this video has from annotations
        video_tasks = set()
        for conv in video_sample.get("conversations", []):
            task = conv.get("task", "")
            if task:
                video_tasks.add(task)

        # Accumulate scores for tasks present in this video
        for task in video_tasks:
            if task in video_task_scores and video_task_scores[task] is not None:
                ts = video_task_scores[task]
                for key in ("local", "global", "mixed"):
                    task_accum[task][key] += ts[key]
                task_counts[task] += 1

        # Free GPU memory aggressively
        del video_tensor
        torch.cuda.empty_cache()
        gc.collect()

    # --- Average and save ---
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("Task-specific profiling results")
    print(f"{'='*60}")
    print(f"{'Task':<28} {'Videos':>8} {'local_max':>12} {'global_max':>12}")
    print("-" * 60)

    all_task_avg = {}  # task -> averaged mixed score [L, H_kv]

    for task in sorted(task_accum.keys()):
        n = task_counts[task]
        if n == 0:
            continue
        avg_local = task_accum[task]["local"] / n
        avg_global = task_accum[task]["global"] / n
        avg_mixed = task_accum[task]["mixed"] / n

        all_task_avg[task] = {
            "local": avg_local,
            "global": avg_global,
            "mixed": avg_mixed,
        }

        print(f"{task:<28} {n:>8} {avg_local.max():>12.6f} {avg_global.max():>12.6f}")

        # Save per-task CSV
        for key, mat in [("local", avg_local), ("global", avg_global), ("mixed", avg_mixed)]:
            rows = []
            for l in range(num_layers):
                for kv in range(num_kv_heads):
                    rows.append({
                        "layer": l,
                        "kv_head": kv,
                        "score": mat[l, kv],
                    })
            import pandas as pd
            df = pd.DataFrame(rows)
            safe_task = task.replace(" ", "_").replace("-", "_")
            csv_path = os.path.join(args.save_dir, f"{safe_task}_{key}.csv")
            df.to_csv(csv_path, index=False)

    print(f"\nSaved {len(all_task_avg)} task profiles to {args.save_dir}/")

    # --- Cross-task analysis ---
    print(f"\n{'='*60}")
    print("Cross-task head specialization")
    print(f"{'='*60}")

    if len(all_task_avg) >= 2:
        # For each head, compute which task it specializes in
        tasks = sorted(all_task_avg.keys())
        head_task_scores = np.zeros((num_layers, num_kv_heads, len(tasks)), dtype=np.float64)

        for ti, task in enumerate(tasks):
            head_task_scores[:, :, ti] = all_task_avg[task]["mixed"]

        # Normalize across tasks per head
        head_task_norm = np.zeros_like(head_task_scores)
        for l in range(num_layers):
            for kv in range(num_kv_heads):
                row = head_task_scores[l, kv, :]
                rmax = row.max()
                if rmax > 0:
                    head_task_norm[l, kv, :] = row / rmax

        # Compute specialization: max / second_max ratio
        specialization = np.zeros((num_layers, num_kv_heads))
        top_task = np.zeros((num_layers, num_kv_heads), dtype=int)
        for l in range(num_layers):
            for kv in range(num_kv_heads):
                row = head_task_scores[l, kv, :]
                order = np.argsort(-row)
                top_task[l, kv] = order[0]
                second = row[order[1]] if len(order) > 1 and row[order[1]] > 0 else 1e-12
                specialization[l, kv] = row[order[0]] / max(second, 1e-12)

        # Print most specialized heads per task
        print("\nTop-5 specialized heads per task:")
        for ti, task in enumerate(tasks):
            is_this_task = top_task == ti
            spec_scores = np.where(is_this_task, specialization, 0)
            top_heads = np.argsort(-spec_scores.reshape(-1))[:5]
            print(f"\n  {task}:")
            for rank, idx in enumerate(top_heads, 1):
                l, kv = idx // num_kv_heads, idx % num_kv_heads
                spec = specialization[l, kv]
                score = head_task_scores[l, kv, ti]
                print(f"    {rank}. L{l:02d}-KV{kv}  spec_ratio={spec:.2f}x  score={score:.6f}")

        # Save specialization matrix
        spec_rows = []
        for l in range(num_layers):
            for kv in range(num_kv_heads):
                spec_rows.append({
                    "layer": l,
                    "kv_head": kv,
                    "top_task": tasks[top_task[l, kv]],
                    "specialization_ratio": specialization[l, kv],
                })
        import pandas as pd
        spec_df = pd.DataFrame(spec_rows)
        spec_df.to_csv(os.path.join(args.save_dir, "head_specialization.csv"), index=False)
        print(f"\nSaved head specialization to {args.save_dir}/head_specialization.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
