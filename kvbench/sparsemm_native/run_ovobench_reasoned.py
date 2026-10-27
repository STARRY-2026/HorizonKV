"""Diagnostic one-pass CRR rationale and decision on the native SparseMM bridge.

This changes the question prompt and generation length, so its answers are not
official OVO-Bench scores. It records the rationale and final decision generated
in the same call, using the same video prefix and cache allocation as the run.
"""

import argparse
import json
from pathlib import Path
import re

from sparsemm_native.run_ovobench import DP_BUDGET, MODEL, forward_queries, run_query


def reasoned_prompt(question):
    return (
        "你正在观看截至当前提问时间的视频画面。请判断现有画面是否已经足以回答下面的问题。\n"
        f"问题：{question}\n"
        "请在同一次回答中依次写出：\n"
        "可见证据：列出最多三条能从已给画面核查的事实。\n"
        "具体答案：如果画面已显示问题所问的事件，写出该事件；否则写‘无法确定’。\n"
        "判断依据：用一句话说明证据是否足够。\n"
        "最后一行严格写 Final: Yes 或 Final: No。Yes 表示现在的画面已足以回答具体问题，"
        "No 表示仍不足以回答。不要使用音频、未提供的未来画面或问题文字暗示来补全事件。"
    )


def variant_prompt(question, variant):
    if variant == "concrete_zh":
        return reasoned_prompt(question)
    if variant == "binary_en":
        return ("Based only on the video frames shown so far, is there enough visual evidence "
                "to answer the following question?\n"
                f"Question: {question}\nAnswer only Yes or No.")
    base = (
        "你负责根据视频画面回答问题。以下问题与视频末尾附近的最新画面有关。\n"
        f"问题：{question}\n"
        "判断现有视觉内容，尤其末尾附近的画面，是否已经提供足够信息回答这个问题。"
    )
    if variant == "binary_zh":
        return base + "\n只回答 Yes 或 No，不要加入解释或其他文字。"
    if variant == "evidence_zh":
        return (base + "\n请先用中文列出最多三条可从画面核查的事实，"
                "再用一句话解释是否有足够证据。最后一行严格写 Final: Yes 或 Final: No。")
    raise ValueError(f"Unknown diagnostic prompt variant: {variant}")


def final_decision(response):
    match = re.search(r"(?:^|\n)Final:\s*(Yes|No)\s*$", response.strip(), re.I)
    if match:
        return match.group(1).capitalize()
    match = re.fullmatch(r"\s*(Yes|No)\.?\s*", response, re.I)
    return match.group(1).capitalize() if match else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("dp", "native"), required=True)
    parser.add_argument("--query-id", action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--variant", choices=("concrete_zh", "binary_zh", "evidence_zh",
                                              "binary_en"),
                        default="concrete_zh")
    args = parser.parse_args(argv)
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")

    from sparsemm_baselines.run_streamingbench import gpu_lease
    from sparsemm_native.streaming import load_model
    from sparsemm_protokv.config import StreamingConfig

    requested = set(args.query_id)
    selected = [query for query in forward_queries()
                if query["query_id"] in requested and query["task"] == "CRR"]
    if len(selected) != len(requested):
        parser.error("Every --query-id must name a distinct CRR probe in forward3")
    output = args.output_root.resolve() / args.arm
    output.mkdir(parents=True, exist_ok=True)
    progress = output / "reasoned_answers.jsonl"
    completed = {json.loads(line)["query_id"] for line in progress.read_text().splitlines() if line.strip()} if progress.exists() else set()
    config = StreamingConfig(method="sparsemm", budget=500, budget_scope="visual",
                             sparsemm_budget_path=str(DP_BUDGET) if args.arm == "dp" else "")
    with gpu_lease(output):
        bridge = load_model(MODEL, config, sample_fps=0.5,
                            trace_path=output / "cache_trace.jsonl", device="cuda",
                            dtype="bfloat16", video_max_pixels=151200)
        for original in selected:
            if original["query_id"] in completed:
                continue
            match = re.search(r"the end of the video\.\n(.*?)\nDecide", original["prompt"], re.S)
            if match is None:
                raise ValueError(f"Could not extract CRR question: {original['query_id']}")
            content_question = match.group(1).strip()
            query = dict(original, prompt=variant_prompt(content_question, args.variant))
            response, frames = run_query(bridge, query, 0.5, 16, args.max_new_tokens)
            row = {"query_id": original["query_id"], "task": "CRR", "arm": args.arm,
                   "video_path": original["video_path"], "timestamp": original["timestamp"],
                   "content_question": content_question, "official_target": original["target"],
                   "frames": frames, "prompt_variant": args.variant,
                   "rationale_and_answer": response,
                   "final_decision": final_decision(response),
                   "protocol": "one_pass_reasoned_prompt_not_official_score"}
            with progress.open("a") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(args.arm, original["query_id"], "Final", row["final_decision"],
                  "frames", frames, flush=True)


if __name__ == "__main__":
    main()
