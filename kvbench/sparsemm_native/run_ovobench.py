"""Resumable full OVO-Bench evaluation on the native SparseMM streaming bridge.

Each official query is a fresh prefix of its source video, as in the official
offline evaluator. Frames are sampled at the requested FPS directly from the
source (the official re-encoded chunks are not required). Both allocation arms
use exactly the same prompts, model, frame indices, and scoring.
"""

import argparse
import collections
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "hermes/data/ovobench/official_20261014"
ANNOTATIONS = DATA / "annotations/ovo_bench_new.json"
HERMES_ANNOTATIONS = ROOT / "hermes/data/ovobench/ovobench_realtime_backeward.json"
DP_BUDGET = ROOT / "hermes/results/myobservations/two_bank_grid_dp_k500_floor375_20260925/per_head_allocation.csv"
DP_DUAL_BUDGET = ROOT / "hermes/results/myobservations/two_bank_grid_dp_k500_floor375_recent32_20261020/per_head_allocation_fixed_capacity.csv"
MODEL = ROOT / "hermes/models/Qwen2.5-VL-7B-Instruct"
BACKWARD = {"EPM", "ASI", "HLD"}
REALTIME = {"STU", "OJR", "ATR", "ACR", "OCR", "FPD"}
FORWARD = {"REC", "SSR", "CRR"}

# Verbatim official prompt templates, OVO-Bench main/constant.py.
MCQA_PROMPT = """
Question: {}
Options:
{}

Respond only with the letter corresponding to your chosen option (e.g., A, B, C).
Do not include any additional text or explanation in your response.
"""
REC_PROMPT = """
You're watching a video in which people may perform a certain type of action repetively.
The person performing this kind of action are referred to as 'they' in the following statement.
You're task is to count how many times have different people in the video perform this kind of action in total.
One complete motion counts as one.
Now, answer the following question: {}
Provide your answer as a single number (e.g., 0, 1, 2, 3…) indicating the total count.
Do not include any additional text or explanation in your response.
"""
SSR_PROMPT = """
You're watching a tutorial video which contain a sequential of steps.
The following is one step from the whole procedures:
{}
Your task is to determine if the man or woman in the video is currently performing this step.
Answer only with “Yes” or “No”.
Do not include any additional text or explanation in your response.
"""
CRR_PROMPT = """
You're responsible of answering questions based on the video content.
The following question are relevant to the latest frames, i.e. the end of the video.
{}
Decide whether existing visual content, especially latest frames, i.e. frames that near the end of the video, provide enough information for answering the question.
Answer only with “Yes” or “No”.
Do not include any additional text or explanation in your response.
"""


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def queries(annotation_path=ANNOTATIONS, video_root=DATA / "src_videos"):
    annotations = json.loads(Path(annotation_path).read_text())
    if len(annotations) != 1640:
        raise ValueError(f"Expected 1640 official records, received {len(annotations)}")
    result = []
    for item in annotations:
        task, source_id = item["task"], item["id"]
        if task not in BACKWARD | REALTIME | FORWARD:
            raise ValueError(f"Unexpected official task: {task}")
        video_path = video_root / item["video"]
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        if task in BACKWARD | REALTIME:
            options = '; '.join(f'{chr(65 + i)}. {option}' for i, option in enumerate(item['options'])) + ';'
            prompt = MCQA_PROMPT.format(item["question"], options)
            entries = [(None, item["realtime"], prompt, chr(65 + item["gt"]))]
        else:
            entries = []
            for probe_id, probe in enumerate(item["test_info"]):
                if task == "REC":
                    prompt = REC_PROMPT.format("How many times did they " + item["activity"] + "?")
                    target = str(probe["count"])
                elif task == "SSR":
                    prompt = SSR_PROMPT.format(probe["step"])
                    target = "Yes" if probe["type"] == 1 else "No"
                else:
                    prompt = CRR_PROMPT.format(item["question"])
                    target = "Yes" if probe["type"] == 1 else "No"
                entries.append((probe_id, probe["realtime"], prompt, target))
        for probe_id, timestamp, prompt, target in entries:
            if not math.isfinite(float(timestamp)) or float(timestamp) <= 0:
                raise ValueError(f"Bad timestamp for {source_id}/{probe_id}")
            result.append(dict(query_id=str(source_id) if probe_id is None else f"{source_id}_{probe_id}",
                               source_id=source_id, probe_id=probe_id, task=task,
                               group="backward" if task in BACKWARD else "realtime" if task in REALTIME else "forward",
                               video_path=str(video_path), timestamp=float(timestamp),
                               clip_end=float(math.ceil(timestamp)), prompt=prompt, target=target))
    if len(result) != 3035 or len({q["query_id"] for q in result}) != len(result):
        raise ValueError("Official query expansion count or IDs changed")
    return result


def hermes_queries(annotation_path=HERMES_ANNOTATIONS, official_path=ANNOTATIONS,
                   video_root=DATA / "src_videos"):
    """The nine-task, 1,468-question selection and MC labels shipped by HERMES."""
    selected = json.loads(Path(annotation_path).read_text())
    official = json.loads(Path(official_path).read_text())
    if len(selected) != 1468:
        raise ValueError(f"Expected 1468 HERMES questions, received {len(selected)}")
    result = []
    for record in selected:
        source_id = record["video_id"]
        source = official[source_id]
        probes = record["conversations"]
        if (source["id"] != source_id or record["task"] not in BACKWARD | REALTIME
                or source["task"] != record["task"] or len(probes) != 1):
            raise ValueError(f"HERMES/official mapping changed at {source_id}")
        probe = probes[0]
        if (probe["question"] != source["question"]
                or probe["choices"] != source["options"]
                or float(probe["end_time"]) != float(source["realtime"])
                or not record["video_path"].endswith("/" + source["video"])):
            raise ValueError(f"HERMES content changed at {source_id}")
        video_path = video_root / source["video"]
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        target = chr(65 + probe["choices"].index(probe["answer"]))
        options = "\n".join(f"({chr(65 + i)}) {choice}"
                            for i, choice in enumerate(probe["choices"]))
        prompt = f"Question: {probe['question']}\nOptions:\n{options}\nOnly give the best option."
        timestamp = float(probe["end_time"])
        result.append(dict(query_id=str(source_id), source_id=source_id, probe_id=None,
                           task=record["task"],
                           group="backward" if record["task"] in BACKWARD else "realtime",
                           video_path=str(video_path), timestamp=timestamp,
                           clip_end=float(math.ceil(timestamp)), question=probe["question"],
                           prompt=prompt, target=target, mc=True))
    if len({q["query_id"] for q in result}) != 1468:
        raise ValueError("Duplicate HERMES question IDs")
    return result


def forward_queries():
    """All official REC, SSR, and CRR probes, retaining official order."""
    result = [query for query in queries() if query["task"] in FORWARD]
    counts = collections.Counter(query["task"] for query in result)
    if counts != {"REC": 698, "SSR": 629, "CRR": 240}:
        raise ValueError(f"Forward-task query counts changed: {counts}")
    return result


def hermes_correct(response, target):
    """Match HERMES BaseVQA.extract_characters_regex and exact-letter accuracy."""
    response = str(response).strip()
    if ")" in response:
        index = response.index(")")
        predicted = response[index - 1:index]
    else:
        predicted = response[:1]
    return predicted == target


def official_correct(task, response, target):
    """Match OVO-Bench utils/OVOBenchScore.py exactly."""
    if response is None:
        return False
    if task == "REC":
        return "".join(re.findall(r"\d+", response)) == str(target)
    if task in ("SSR", "CRR") and ((response == "N" and target == "No") or (response == "Y" and target == "Yes")):
        return True
    return target in response


def summary(rows, total=3035):
    buckets = collections.defaultdict(list)
    groups = collections.defaultdict(list)
    for row in rows:
        buckets[row["task"]].append(int(row["correct"]))
    task_accuracy = {task: sum(values) / len(values) for task, values in sorted(buckets.items())}
    for task, accuracy in task_accuracy.items():
        group = "backward" if task in BACKWARD else "realtime" if task in REALTIME else "forward"
        groups[group].append(accuracy)
    group_accuracy = {group: sum(values) / len(values) for group, values in groups.items()}
    return {"complete": len(rows) == total, "completed_queries": len(rows), "total_queries": total,
            "question_accuracy": sum(int(row["correct"]) for row in rows) / len(rows) if rows else None,
            "task_accuracy": task_accuracy, "group_accuracy": group_accuracy,
            "hermes_category_average": sum(group_accuracy.values()) / 2
            if len(group_accuracy) == 2 and len(task_accuracy) == 9 else None,
            "official_total_average": sum(group_accuracy.values()) / 3
            if len(rows) == total and len(group_accuracy) == 3 else None}


def read_progress(path, expected):
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for index, row in enumerate(rows):
        if index >= len(expected) or row["query_id"] != expected[index]["query_id"]:
            raise ValueError("Progress file is not a prefix of the pinned official query list")
    return rows


def debug_explanation(bridge, query, max_new_tokens=96):
    """Ask for a separate visual-evidence summary; never score this text."""
    prompt = query["prompt"]
    if query["task"] == "REC":
        match = re.search(r"Now, answer the following question: (.*?)\n", prompt)
    elif query["task"] == "SSR":
        match = re.search(r"following is one step from the whole procedures:\n(.*?)\n", prompt)
    elif query["task"] == "CRR":
        match = re.search(r"the end of the video\.\n(.*?)\nDecide", prompt, re.S)
    else:
        match = None
    question = query.get("question") or (match.group(1).strip() if match else prompt)
    diagnostic = (
        "This is a separate diagnostic request after an answer has already been recorded.\n"
        f"Task: {query['task']}\nQuestion or step: {question}\n"
        "Only sampled video frames were provided; there is no audio. "
        "In 1-3 sentences, describe only visible evidence relevant to the question. "
        "Mention uncertainty or missing evidence. Do not invent sound or events between "
        "sampled frames, do not give a new final answer, and do not claim to reveal "
        "hidden reasoning."
    )
    history_length = len(bridge.conv_history)
    try:
        return str(bridge.question_answering(
            {"question": diagnostic, "prompt": bridge.get_prompt(diagnostic)},
            max_new_tokens=max_new_tokens))
    finally:
        del bridge.conv_history[history_length:]


def run_query(bridge, query, fps, chunk_size, max_new_tokens,
              debug_explain=False, debug_max_new_tokens=96):
    import torch
    from sparsemm_baselines.run_streamingbench import sampled_indices
    from sparsemm_protokv.models.qwen2_5_vl_stream import BatchScopedVideoReader

    reader = BatchScopedVideoReader(query["video_path"], num_threads=1)
    # Official chunks end at ceil(realtime); use source frames at identical times.
    indices = sampled_indices(len(reader), reader.get_avg_fps(), fps)
    target = min(len(indices), math.ceil(query["clip_end"] * fps))
    if target < 1:
        raise ValueError(f"No frames for {query['query_id']}")
    bridge._active_video_id = query["query_id"]
    bridge._active_question_index = 0
    bridge._active_question = query.get("question", query["prompt"])
    bridge._active_question_end_time = query["clip_end"]
    bridge.encode_init_prompt()
    for start in range(0, target, chunk_size):
        end = min(start + chunk_size, target)
        frames = torch.from_numpy(reader.get_batch(indices[start:end]).asnumpy())
        bridge.encode_video_chunk(frames, is_last_chunk=end == target)
        bridge.predict_and_compress()
    prompt = {"question": query.get("question", query["prompt"]),
              "formatted_question": query["prompt"],
              "prompt": bridge.get_prompt(query["prompt"], mc=query.get("mc", False))}
    response = str(bridge.question_answering(prompt, max_new_tokens=max_new_tokens))
    explanation = (debug_explanation(bridge, query, debug_max_new_tokens)
                   if debug_explain else None)
    checks = bridge.validation_checks
    selectors = checks["selector_checks"]
    qa = checks["qa_checks"]
    if (len(qa) != (2 if debug_explain else 1)
            or any(item.get("returned") is not True or item.get("restored") is not True for item in qa)
            or checks["failures"] or not selectors
            or any(item.get("validation_passed") is not True
                   or item.get("head_budgets") != bridge.budgets[item["layer_idx"]].tolist()
                   for item in selectors)):
        raise RuntimeError(f"Native SparseMM runtime audit failed for {query['query_id']}")
    # Keep the long full-benchmark process bounded; the trace remains on disk.
    checks["selector_checks"].clear()
    checks["qa_checks"].clear()
    bridge.clear_cache()
    return (response, target, explanation) if debug_explain else (response, target)


def grouped_batches(remaining, minimum_groups):
    """Keep complete adjacent video runs together while preserving result order."""
    selected = []
    cursor = 0
    while cursor < len(remaining) and len(selected) < minimum_groups:
        end = cursor + 1
        while (end < len(remaining)
               and remaining[end]["video_path"] == remaining[cursor]["video_path"]):
            end += 1
        selected.append(remaining[cursor:end])
        cursor = end
    return selected


def run_grouped_video(bridge, group, fps, chunk_size, max_new_tokens,
                      debug_explain=False, debug_max_new_tokens=96):
    """Reuse even-frame prefixes; branch odd endpoints to preserve Qwen pairs."""
    import torch
    from sparsemm_baselines.run_streamingbench import sampled_indices
    from sparsemm_protokv.models.qwen2_5_vl_stream import BatchScopedVideoReader

    if not group or len({q["video_path"] for q in group}) != 1:
        raise ValueError("A grouped run requires one source video")
    reader = BatchScopedVideoReader(group[0]["video_path"], num_threads=1)
    indices = sampled_indices(len(reader), reader.get_avg_fps(), fps)
    targets = [min(len(indices), math.ceil(q["clip_end"] * fps)) for q in group]
    if any(target < 1 for target in targets):
        raise ValueError("Grouped video contains a question with no frames")

    bridge._active_video_id = group[0]["query_id"]
    bridge.encode_init_prompt()
    base_frames = 0
    answers = {}

    def encode_range(start, end):
        for offset in range(start, end, chunk_size):
            stop = min(offset + chunk_size, end)
            frames = torch.from_numpy(reader.get_batch(indices[offset:stop]).asnumpy())
            bridge.encode_video_chunk(frames, is_last_chunk=stop == end)
            bridge.predict_and_compress()

    for question_index in sorted(range(len(group)), key=lambda i: (targets[i], i)):
        query, target = group[question_index], targets[question_index]
        started = time.monotonic()
        even_target = target - target % 2
        if even_target < base_frames:
            raise AssertionError("Grouped frame cursor moved backward")
        bridge._active_question_index = question_index
        bridge._active_question = query.get("question", query["prompt"])
        bridge._active_question_end_time = query["clip_end"]
        encode_range(base_frames, even_target)
        newly_encoded = even_target - base_frames
        base_frames = even_target

        # The fresh-query path encodes an odd final frame alone. Continuing
        # from that state would pair the next frame differently, so use a
        # temporary cache branch and retain the even-frame base state.
        if target % 2:
            base_cache = bridge.kv_cache
            cursors = (bridge._next_position, bridge._seen_visual_tokens,
                       bridge.total_processed_frames, bridge.last_encoded_frames,
                       bridge._prefills, bridge._compressions, bridge.last_metadata)
            history_length = len(bridge.conv_history)
            bridge.kv_cache = copy.deepcopy(base_cache)
            try:
                encode_range(even_target, target)
                newly_encoded += 1
                result = _answer_grouped(bridge, query, max_new_tokens,
                                         newly_encoded > 0, debug_explain,
                                         debug_max_new_tokens)
            finally:
                bridge.kv_cache = base_cache
                (bridge._next_position, bridge._seen_visual_tokens,
                 bridge.total_processed_frames, bridge.last_encoded_frames,
                 bridge._prefills, bridge._compressions, bridge.last_metadata) = cursors
                del bridge.conv_history[history_length:]
        else:
            result = _answer_grouped(bridge, query, max_new_tokens,
                                     newly_encoded > 0, debug_explain,
                                     debug_max_new_tokens)
        answers[query["query_id"]] = ((result, target, time.monotonic() - started,
                                        newly_encoded) if not debug_explain else
                                       (result[0], target, time.monotonic() - started,
                                        newly_encoded, result[1]))

    bridge.clear_cache()
    return answers


def _answer_grouped(bridge, query, max_new_tokens, encoded_frames,
                    debug_explain=False, debug_max_new_tokens=96):
    prompt = {"question": query.get("question", query["prompt"]),
              "formatted_question": query["prompt"],
              "prompt": bridge.get_prompt(query["prompt"], mc=query.get("mc", False))}
    response = str(bridge.question_answering(prompt, max_new_tokens=max_new_tokens))
    explanation = (debug_explanation(bridge, query, debug_max_new_tokens)
                   if debug_explain else None)
    checks = bridge.validation_checks
    selectors, qa = checks["selector_checks"], checks["qa_checks"]
    if (len(qa) != (2 if debug_explain else 1)
            or any(item.get("returned") is not True or item.get("restored") is not True for item in qa)
            or checks["failures"]
            or (encoded_frames and not selectors)
            or any(item.get("validation_passed") is not True
                   or item.get("head_budgets") != bridge.budgets[item["layer_idx"]].tolist()
                   for item in selectors)):
        raise RuntimeError(f"Grouped native SparseMM audit failed for {query['query_id']}")
    selectors.clear()
    qa.clear()
    return (response, explanation) if debug_explain else response


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("dp", "native", "dp_dual"), required=True)
    parser.add_argument("--dataset", choices=("official_full", "hermes9", "forward3"), default="official_full")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--fps", type=float, default=0.5)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--max-new", type=int, default=25,
                        help="Queries per invocation, or video groups with --grouped")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--video-max-pixels", type=int, default=360 * 420)
    parser.add_argument("--grouped", action="store_true",
                        help="Share video prefixes across adjacent questions")
    parser.add_argument("--debug-explain", action="store_true",
                        help="After scoring the unchanged answer, save a separate evidence summary")
    parser.add_argument("--debug-max-new-tokens", type=int, default=96)
    parser.add_argument("--query-id", action="append",
                        help="Run only this query ID (repeatable); use a separate output root")
    args = parser.parse_args(argv)
    if (args.budget != 500 or args.fps != 0.5 or args.chunk_size < 1 or args.max_new < 1
            or args.video_max_pixels < 3136):
        parser.error("This paired protocol is pinned to K=500 and FPS=0.5")
    if args.grouped and args.chunk_size % 2:
        parser.error("Grouped Qwen video inference requires an even chunk size")
    if args.debug_max_new_tokens < 1:
        parser.error("--debug-max-new-tokens must be positive")
    from sparsemm_baselines.run_streamingbench import gpu_lease, write_json
    from sparsemm_native.streaming import load_model
    from sparsemm_protokv.config import StreamingConfig
    from sparsemm_native.policy import DEFAULT_SCORE_PATH, load_fixed_head_budgets
    from sparsemm_native.two_bank import load_two_bank_budgets

    all_queries = (hermes_queries() if args.dataset == "hermes9" else
                   forward_queries() if args.dataset == "forward3" else queries())
    full_query_hash = hashlib.sha256(json.dumps(all_queries, sort_keys=True).encode()).hexdigest()
    if args.query_id:
        requested = set(args.query_id)
        available = {query["query_id"] for query in all_queries}
        missing = requested - available
        if missing:
            parser.error(f"Unknown query IDs for {args.dataset}: {', '.join(sorted(missing))}")
        all_queries = [query for query in all_queries if query["query_id"] in requested]
    output = args.output_root.resolve() / args.arm
    output.mkdir(parents=True, exist_ok=True)
    progress = output / "queries.jsonl"
    rows = read_progress(progress, all_queries)
    if len(rows) == len(all_queries) and args.arm != "dp_dual":
        return
    budget_path = DP_DUAL_BUDGET if args.arm == "dp_dual" else DP_BUDGET
    config = StreamingConfig(method="sparsemm", budget=500, budget_scope="visual",
                             sparsemm_budget_path=str(budget_path) if args.arm != "native" else "",
                             sparsemm_cache_policy="two_bank" if args.arm == "dp_dual" else "capacity",
                             sparsemm_two_bank_gate="legacy_global" if args.arm == "dp_dual" else "per_head_capacity")
    if args.arm == "dp_dual":
        table, _, allocation = load_two_bank_budgets(
            budget_path, 28, 4, 500, config.window_size,
            short_cache_gate=config.sparsemm_two_bank_gate)
        old_table, _ = load_fixed_head_budgets(DP_BUDGET, 28, 4, 500, config.window_size)
        if not table.equal(old_table):
            raise ValueError("Primary dual-bank arm must preserve every previous DP head capacity")
    elif args.arm == "dp":
        table, allocation = load_fixed_head_budgets(budget_path, 28, 4, 500, config.window_size)
        assert int(table.sum()) == 56000
    else:
        allocation = None
    source_annotation = HERMES_ANNOTATIONS if args.dataset == "hermes9" else ANNOTATIONS
    protocol = {"source_annotation": str(source_annotation), "source_sha256": sha256(source_annotation),
                "model_path": str(MODEL), "model_config_sha256": sha256(MODEL / "config.json"),
                "dp_budget_path": str(budget_path), "dp_budget_sha256": sha256(budget_path),
                "arm": args.arm, "budget": 500, "fps": 0.5, "dtype": "bfloat16",
                "chunk_size": args.chunk_size, "max_new_tokens": args.max_new_tokens,
                "video_max_pixels": args.video_max_pixels,
                "score_mode": "HERMES_exact_MC_letter" if args.dataset == "hermes9"
                              else "official_OVOBenchOfflineScore",
                "video_mode": "source_prefix_ceil_timestamp",
                "query_count": len(all_queries), "dp_allocation": allocation,
                "query_list_sha256": hashlib.sha256(json.dumps(all_queries, sort_keys=True).encode()).hexdigest()}
    if args.dataset != "official_full":
        protocol["dataset"] = args.dataset
    if args.arm == "dp_dual":
        source_files = ["sparsemm_native/two_bank.py", "sparsemm_native/policy.py",
                        "sparsemm_native/cache.py", "sparsemm_native/streaming.py",
                        "sparsemm_native/run_ovobench.py", "sparsemm_protokv/config.py",
                        "sparsemm_native/assets.py", "sparsemm_protokv/cache.py",
                        "sparsemm_protokv/streaming.py", "sparsemm_protokv/monkeypatch.py",
                        "sparsemm_protokv/model_loader.py", "sparsemm_protokv/models/qwen2_5_vl_stream.py",
                        "sparsemm_baselines/cache.py", "sparsemm_baselines/core.py",
                        "sparsemm_baselines/adapter.py"]
        from importlib.metadata import version
        protocol.update(
            cache_policy="two_bank", protect_window=32, attention_observation_window=32,
            short_cache_gate="legacy_global",
            protect_window_included_in_recent=True,
            grouped=args.grouped, debug_explain=args.debug_explain,
            query_ids=[query["query_id"] for query in all_queries],
            full_query_list_sha256=full_query_hash,
            decoding="greedy", repetition_penalty=1.1,
            allocation_comparison="same capacity_n per head as prior DP; optimize split with recent>=32",
            reference_dp_budget_sha256=sha256(DP_BUDGET),
            sparsemm_score_sha256=sha256(DEFAULT_SCORE_PATH),
            dependencies={name: version(name) for name in ("torch", "transformers", "numpy", "decord", "Pillow")},
            processor_files_sha256={name: sha256(MODEL / name) for name in (
                "config.json", "preprocessor_config.json", "tokenizer_config.json",
                "tokenizer.json", "chat_template.json", "generation_config.json")},
            code_sha256={name: sha256(ROOT / name) for name in source_files})
    protocol_path = output / "protocol.json"
    if args.arm == "dp_dual" and rows and not protocol_path.exists():
        raise ValueError("Dual-bank progress has no protocol; refusing to adopt unknown results")
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Protocol changed; refusing to resume")
    write_json(protocol_path, protocol)
    if args.arm == "dp_dual":
        identity_fields = ("query_id", "source_id", "probe_id", "task", "group",
                           "timestamp", "clip_end", "target")
        for row, query in zip(rows, all_queries):
            if any(row.get(field) != query[field] for field in identity_fields):
                raise ValueError("Dual-bank resume query identity or target changed")
        write_json(output / "query_manifest.json", all_queries)
        snapshot = output / "source_snapshot"
        for name in source_files:
            destination = snapshot / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((ROOT / name).read_bytes())
    if len(rows) == len(all_queries):
        return
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    with gpu_lease(output):
        bridge = load_model(MODEL, config, sample_fps=.5,
                            trace_path=output / "cache_trace.jsonl", device="cuda", dtype="bfloat16",
                            video_max_pixels=args.video_max_pixels)
        remaining = all_queries[len(rows):]
        batches = (grouped_batches(remaining, args.max_new) if args.grouped else
                   [[query] for query in remaining[:args.max_new]])
        if args.grouped:
            with (output / "execution_changes.jsonl").open("a") as stream:
                stream.write(json.dumps({"mode": "contiguous_video_reuse_even_prefix_odd_branch",
                                         "start_index": len(rows), "epoch": time.time()}) + "\n")
        for group in batches:
            if args.grouped:
                answers = run_grouped_video(bridge, group, .5, args.chunk_size,
                                            args.max_new_tokens, args.debug_explain,
                                            args.debug_max_new_tokens)
            for query in group:
                if args.grouped:
                    answer = answers[query["query_id"]]
                    response, frames, seconds, encoded_frames = answer[:4]
                    explanation = answer[4] if args.debug_explain else None
                else:
                    started = time.monotonic()
                    answer = run_query(bridge, query, .5, args.chunk_size,
                                       args.max_new_tokens, args.debug_explain,
                                       args.debug_max_new_tokens)
                    response, frames = answer[:2]
                    explanation = answer[2] if args.debug_explain else None
                    seconds = time.monotonic() - started
                    encoded_frames = frames
                row = {key: query[key] for key in ("query_id", "source_id", "probe_id", "task", "group", "timestamp", "clip_end", "target")}
                correct = (hermes_correct(response, query["target"]) if args.dataset == "hermes9"
                           else official_correct(query["task"], response, query["target"]))
                row.update(response=response, correct=correct, frames=frames,
                           seconds=seconds)
                if args.grouped:
                    row.update(encoded_frames=encoded_frames, execution_mode="grouped")
                with progress.open("a") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                rows.append(row)
                if args.debug_explain:
                    with (output / "debug_explanations.jsonl").open("a") as stream:
                        stream.write(json.dumps({"query_id": query["query_id"],
                                                 "task": query["task"],
                                                 "timestamp": query["timestamp"],
                                                 "standard_response": response,
                                                 "posthoc_evidence_summary": explanation},
                                                ensure_ascii=False) + "\n")
                write_json(output / "summary.json", summary(rows, len(all_queries)))
                print(f"{args.arm}: {len(rows)}/{len(all_queries)} {query['query_id']} {query['task']} "
                      f"correct={row['correct']} frames={frames} encoded={encoded_frames} "
                      f"seconds={seconds:.1f}", flush=True)


if __name__ == "__main__":
    main()
