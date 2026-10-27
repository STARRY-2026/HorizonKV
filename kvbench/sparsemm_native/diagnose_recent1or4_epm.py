"""Five-query EPM ablation without changing the resumable full-run sources.

The saved full-run resize is reused verbatim. Each video is encoded once; four
answers then share identical image features, recent prefill and stream state.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from sparsemm_native import capped_history, recent1or4_history
from sparsemm_native import run_ovobench as ovo
from sparsemm_native.run_ovobench_sparsemm_fps1 import MAX_PIXELS
from sparsemm_native.simplestream_resize import resize_hwc_batch
from sparsemm_baselines.run_streamingbench import sampled_indices
from sparsemm_protokv.cache import StreamingCache
from sparsemm_protokv.config import StreamingConfig
from sparsemm_protokv.models.qwen2_5_vl_stream import BatchScopedVideoReader


DEFAULT_RUN = Path("/home/yujun/program/kvbench/results/ovobench_recent1or4_history_cap4000_simplestream_video_constant_fps1_20261023")
OLD_RUN = Path("/home/yujun/program/kvbench/results/ovobench_simplestream_recent_share_independent_last1_nine_20261022")
IDS = ("32", "10", "288", "241", "192")


def recent_only_assemble(history, one, four, logical_tokens):
    """Use exactly the same independently rebuilt recent KV, omitting history."""
    fused = StreamingCache(history.config, history.num_layers, history.num_kv_heads,
                           history.head_dim, history.device, history.dtype)
    p = history.prefix_tokens
    actual_recent = 0
    for layer_idx, layer in enumerate(history.layers):
        keys, values, positions, lengths = [], [], [], []
        for head in range(history.num_kv_heads):
            r = int(history.recent_budgets[layer_idx, head])
            branch = one if history.rows[layer_idx * 4 + head]["recent_group"] == "one_frame" else four
            rk, rv = branch[layer_idx]
            if rk.shape[2] != p + r:
                raise AssertionError("Recent branch length changed")
            hk, hv = layer.get_segment(head)
            hp = layer.get_position_segment(head)
            units = history.units[-1:] if branch is one else history.units
            recent_positions = torch.cat([unit.positions[:, 0] for unit in units], dim=1)
            keys.append(torch.cat((hk[:p], rk[0, head, p:])))
            values.append(torch.cat((hv[:p], rv[0, head, p:])))
            positions.append(torch.cat((hp[:, :p], recent_positions), dim=1))
            lengths.append(p + r)
            actual_recent += r
        target = fused.layers[layer_idx]
        target.flat_k, target.flat_v = torch.cat(keys), torch.cat(values)
        target.flat_pos = torch.cat(positions, dim=1)
        target.head_lens = target.head_lens.new_tensor(lengths)
        target._refresh()
    fused.prefix_tokens, fused.logical_tokens = p, int(logical_tokens)
    fused._initialized, fused._context_done = True, True
    fused.phase = "video"
    return fused, dict(actual_history_visual_slots=0,
                       actual_recent_visual_slots=actual_recent,
                       actual_visual_slots=actual_recent,
                       configured_history_visual_slots=0,
                       history_head_cap=0)


def read_cases(run_root, ids):
    saved, manifests = {}, {}
    for shard in ("shard0", "shard1"):
        arm = run_root / "shards" / shard / "recent1or4_history_cap4000"
        for line in (arm / "queries.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row["query_id"] in ids:
                saved[row["query_id"]] = row
        for row in json.loads((arm / "query_manifest.json").read_text()):
            if row["query_id"] in ids:
                manifests[row["query_id"]] = row
    old = {row["query_id"]: row for row in csv.DictReader(
        (OLD_RUN / "per_query.csv").open()) if row["query_id"] in ids}
    annotations = {str(item["id"]): item for item in json.loads(ovo.ANNOTATIONS.read_text())
                   if str(item["id"]) in ids}
    if any(set(table) != set(ids) for table in (saved, manifests, old, annotations)):
        raise ValueError("A selected query is missing from a pinned source")
    for query_id in ids:
        row = saved[query_id]
        if row["task"] != "EPM" or row["correct"] or row["target"] not in old[query_id]["response"]:
            raise ValueError(f"Query {query_id} is not old-only-correct EPM")
    return saved, manifests, old, annotations


def old_prompt(annotation):
    options = "; ".join(f"{chr(65 + i)}. {option}" for i, option in enumerate(annotation["options"])) + ";"
    return f"{annotation['question']}\nOptions: {options}\nOnly give the best option's letter directly."


@torch.inference_mode()
def encode_exact_saved_prefix(bridge, query, saved):
    reader = BatchScopedVideoReader(query["video_path"], num_threads=1)
    indices = sampled_indices(len(reader), reader.get_avg_fps(), 1.0)
    target = min(len(indices), math.ceil(query["clip_end"]))
    if target != saved["frames"]:
        raise AssertionError(f"Sampled frame count changed: {target} vs {saved['frames']}")
    bridge.encode_init_prompt()
    height, width = saved["resize_hw"]
    visual_before = bridge.timings["visual_tokens"]
    for offset in range(0, target, 16):
        stop = min(offset + 16, target)
        frames = torch.from_numpy(reader.get_batch(indices[offset:stop]).asnumpy())
        frames = resize_hwc_batch(frames, height, width)
        bridge.encode_video_chunk(frames, is_last_chunk=stop == target)
        bridge.predict_and_compress()
    if bridge.total_processed_frames != target:
        raise AssertionError("Streaming frame count changed")
    encoded = bridge.timings["visual_tokens"] - visual_before
    # In the full run, encoded_visual_tokens is incremental across questions;
    # the diagnostic starts each video from frame zero, so compare with size.
    expected_per_frame = height * width // 784
    if encoded != target * expected_per_frame:
        raise AssertionError(f"Encoded token count changed: {encoded} vs {target * expected_per_frame}")
    return encoded


def answer(bridge, prompt, *, with_history):
    original = recent1or4_history.assemble
    if not with_history:
        recent1or4_history.assemble = recent_only_assemble
    try:
        response = bridge.question_answering({"prompt": bridge.get_prompt(prompt)}, max_new_tokens=16)
        return response, dict(bridge.last_question_metrics)
    finally:
        recent1or4_history.assemble = original


def run(output, ids, run_root, variants=("current", "no_history", "old_prompt_with_history", "old_prompt_no_history")):
    saved, manifests, old, annotations = read_cases(run_root, ids)
    output.mkdir(parents=True, exist_ok=True)
    config = StreamingConfig(method="sparsemm", budget=2576, budget_scope="visual",
                             sparsemm_budget_path=str(capped_history.SOURCE),
                             sparsemm_cache_policy="two_bank",
                             sparsemm_two_bank_gate="per_head_capacity")
    bridge = recent1or4_history.load_model(ovo.MODEL, config, sample_fps=1.0,
                                            trace_path=None, device="cuda", dtype="bfloat16",
                                            video_max_pixels=MAX_PIXELS)
    rows = []
    for query_id in ids:
        baseline, query, annotation = saved[query_id], manifests[query_id], annotations[query_id]
        encoded = encode_exact_saved_prefix(bridge, query, baseline)
        bridge._active_video_id = query_id
        bridge._active_question_index = 0
        available = dict(current=(query["prompt"], True),
                         no_history=(query["prompt"], False),
                         old_prompt_with_history=(old_prompt(annotation), True),
                         old_prompt_no_history=(old_prompt(annotation), False))
        for name in variants:
            prompt, with_history = available[name]
            response, audit = answer(bridge, prompt, with_history=with_history)
            row = dict(query_id=query_id, variant=name, target=baseline["target"],
                       response=response, correct=ovo.official_correct("EPM", response, baseline["target"]),
                       saved_new_response=baseline["response"], old_response=old[query_id]["response"],
                       baseline_reproduced=(response == baseline["response"]) if name == "current" else None,
                       frames=baseline["frames"], resize_hw=baseline["resize_hw"],
                       encoded_visual_tokens=encoded, cache_budget_audit=audit)
            rows.append(row)
            with (output / "answers.jsonl").open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{query_id} {name} {response!r} target={baseline['target']} correct={row['correct']}", flush=True)
        bridge.clear_cache()
    if set(variants) != set(available):
        return
    with (output / "summary.md").open("w") as handle:
        handle.write("# EPM 五题逐项诊断\n\n")
        handle.write("旧一帧／四帧版五题均正确；全量实验保持暂停。本诊断每题使用新版已记录的尺寸和完整帧前缀。\n\n")
        handle.write("| ID | 目标 | 旧版 | 新版原记录 | 重现 | 去历史 | 旧提示词＋历史 | 旧提示词＋去历史 |\n")
        handle.write("|---|---|---|---|---|---|---|---|\n")
        for query_id in ids:
            group = {row["variant"]: row for row in rows if row["query_id"] == query_id}
            handle.write("| " + " | ".join((query_id, saved[query_id]["target"], old[query_id]["response"],
                        saved[query_id]["response"], group["current"]["response"],
                        group["no_history"]["response"], group["old_prompt_with_history"]["response"],
                        group["old_prompt_no_history"]["response"])) + " |\n")
        handle.write("\n重现列若不同于新版原记录，须先调查复现差异，再解释该题的消融。\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ids", nargs="+", default=IDS)
    parser.add_argument("--variants", nargs="+", choices=("current", "no_history",
                        "old_prompt_with_history", "old_prompt_no_history"))
    args = parser.parse_args()
    run(args.output, tuple(args.ids), args.run_root,
        tuple(args.variants) if args.variants else
        ("current", "no_history", "old_prompt_with_history", "old_prompt_no_history"))
