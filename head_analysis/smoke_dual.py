"""Single-video smoke test for dual memory budget.

Runs one StreamingBench video end-to-end with dual budget scores and prints QA results.
"""
import argparse
import json
import logging
import math
import os
import random
import sys
import warnings

from logzero import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_qa.base import MODELS
from video_qa.hermes_vqa import HermesVQA


def run_one(args):
    model_key = args.model
    model_cfg = MODELS[model_key]
    model_path = model_cfg["model_path"]
    load_func = model_cfg["load_func"]
    print(f"Model: {model_path}")

    with open(args.anno_path) as f:
        anno = json.load(f)
    if args.video_idx >= len(anno):
        raise ValueError(f"video_idx={args.video_idx} >= {len(anno)} videos")
    anno = [anno[args.video_idx]]

    # Reproduce exactly the flow from base.py work()
    random.seed(2024)
    videoqa_model, videoqa_processor = load_func(
        model_path=model_path,
        kv_size=args.kv_size,
        streaming=True,
        sample_fps=args.sample_fps,
        compress_mode=args.compress_mode,
    )

    # Apply KV head budget
    from head_analysis.hermes_kv_head_budget import apply_kv_head_budget
    language_model = getattr(videoqa_model, "language_model", None)
    language_config = getattr(language_model, "config", None)
    num_heads = getattr(language_config, "num_attention_heads", None)
    num_kv_heads = getattr(language_config, "num_key_value_heads", None)
    if num_heads is None and hasattr(language_model, "model"):
        num_heads = getattr(language_model.model.config, "num_attention_heads", None)
    if num_kv_heads is None and hasattr(language_model, "model"):
        num_kv_heads = getattr(language_model.model.config, "num_key_value_heads", None)
    if num_heads is None:
        num_heads = 28
    if num_kv_heads is None:
        num_kv_heads = 4

    videoqa_model = apply_kv_head_budget(
        videoqa_model,
        kv_head_scores_path=args.kv_head_budget_scores,
        num_layers=videoqa_model.num_layers,
        num_query_heads=num_heads,
        num_kv_heads=num_kv_heads,
        budget_scheme=args.kv_head_budget_scheme,
        sparsemm_ratio=args.kv_head_budget_sparsemm_ratio,
        sparsemm_window_size=args.kv_head_budget_sparsemm_window_size,
        union_cap_ratio=args.kv_head_budget_union_cap_ratio,
        max_mask_q_len=args.kv_head_budget_max_mask_q_len,
    )

    # Apply ragged prefill
    from head_analysis.hermes_kv_head_ragged_prefill import apply_kv_head_ragged_prefill
    videoqa_model = apply_kv_head_ragged_prefill(videoqa_model)

    # Limit questions for smoke test
    video_sample = anno[0]
    max_q = min(args.max_questions, len(video_sample.get("conversations", [])))
    video_sample["conversations"] = video_sample["conversations"][:max_q]

    print(f"Video: {video_sample.get('video_id', '?')}, questions: {max_q}")

    # Ensure save_dir exists
    os.makedirs(args.save_dir, exist_ok=True)

    # Build analyzer and run
    analyzer = HermesVQA(
        anno=anno,
        save_dir=args.save_dir,
        sample_fps=args.sample_fps,
        qa_model=videoqa_model,
        qa_processor=videoqa_processor,
    )

    analyzer.analyze(debug=args.debug == "true")

    # Collect results
    results = analyzer.record
    if results:
        correct = sum(1 for r in results if r.get("pred_choice") == r.get("correct_choice"))
        total = len(results)
        print(f"\nSmoke test results: {correct}/{total} = {100 * correct / total:.1f}%")
        for r in results:
            status = "OK" if r.get("pred_choice") == r.get("correct_choice") else "FAIL"
            print(f"  [{status}] Q: {r.get('question','?')[:80]}...")
            print(f"        pred={r.get('pred_choice','?')}, gold={r.get('correct_choice','?')}")
    else:
        print("No results collected")

    # Budget stats
    last_stats = getattr(videoqa_model, "_kv_head_budget_last_stats", None)
    if last_stats:
        print(f"\nBudget stats: union=[{last_stats.get('union_min')}, {last_stats.get('union_max')}], "
              f"visible=[{last_stats.get('visible_min')}, {last_stats.get('visible_max')}], "
              f"budget=[{last_stats.get('budget_min')}, {last_stats.get('budget_max')}]")

    budget_table = getattr(videoqa_model, "_kv_head_budget_values", None)
    if budget_table is not None:
        print("\nPer-head budget table (first 4 layers):")
        for layer in range(min(4, budget_table.shape[0])):
            print(f"  L{layer}: {[int(budget_table[layer, k]) for k in range(budget_table.shape[1])]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5_vl_7b")
    parser.add_argument("--anno_path", default="data/streamingbench/streamingbench_realtime_half.json")
    parser.add_argument("--sample_fps", type=float, default=0.5)
    parser.add_argument("--kv_size", type=int, default=6000)
    parser.add_argument("--compress_mode", default="hermes")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--video_idx", type=int, default=0, help="Which video (0=first in anno)")
    parser.add_argument("--max_questions", type=int, default=3, help="Max questions to test")
    parser.add_argument("--debug", type=str, default="true")
    parser.add_argument("--kv_head_budget_scores",
                        default="results/observations/memory_core_budget_scores_dual/ours_dual_top100.csv")
    parser.add_argument("--kv_head_budget_scheme", default="sparsemm")
    parser.add_argument("--kv_head_budget_sparsemm_ratio", type=float, default=0.2)
    parser.add_argument("--kv_head_budget_sparsemm_window_size", type=int, default=32)
    parser.add_argument("--kv_head_budget_union_cap_ratio", type=float, default=1.0)
    parser.add_argument("--kv_head_budget_max_mask_q_len", type=int, default=128)
    parser.add_argument("--save_dir", default="/tmp/smoke_dual")
    args = parser.parse_args()
    run_one(args)


if __name__ == "__main__":
    main()
