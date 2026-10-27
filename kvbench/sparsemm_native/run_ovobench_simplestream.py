"""Four independent recent images, a fresh Qwen2.5 LM prefill for every query.

The image-feature path and continuous vision-prefix layout are ported from
SimpleStream be227ee8, lib/recent_window_eval_qwen25.py. Official OVO prompts,
source-prefix sampling, pixel limit and generation settings match our completed
DP recent-four run. No persistent language KV, attention eviction or K500 cap
is applied. Only independently encoded image features may be reused.
"""

import argparse
import collections
import json
import math
import os
from pathlib import Path
import time

from sparsemm_native import run_ovobench as official
from sparsemm_native.run_ovobench_recent4 import (
    IDENTITY, digest, read_progress, schedule_queries, select_queries, validate_row,
)
from sparsemm_native.supervise_ovobench_dual import write_json


ROOT = official.ROOT
ARM = "simplestream_recent4_refill"
UPSTREAM = ROOT.parent / "SimpleStream"
UPSTREAM_COMMIT = "be227ee8ce90a258b286a34161688c2a59eeeadb"
REFERENCE = ROOT / "results/ovobench_recent4_official_full_k500_fps05_p151200_20261021/dp_recent4"
SOURCE_FILES = (
    "sparsemm_native/run_ovobench_simplestream.py",
    "sparsemm_native/run_ovobench.py", "sparsemm_native/run_ovobench_recent4.py",
    "sparsemm_native/gpu_slots.py", "sparsemm_native/supervise_ovobench_dual.py",
    "sparsemm_protokv/model_loader.py", "sparsemm_protokv/models/qwen2_5_vl_stream.py",
    "sparsemm_baselines/run_streamingbench.py",
)
PROCESSOR_FILES = (
    "config.json", "preprocessor_config.json", "tokenizer_config.json", "tokenizer.json",
    "chat_template.json", "generation_config.json",
)


def window_plan(query, indices, fps=0.5):
    """The exact existing DP prefix, followed by its last four *real* frames."""
    count = min(len(indices), math.ceil(query["clip_end"] * fps))
    if count < 1:
        raise ValueError("No sampled frames in official query")
    ordinals = list(range(max(0, count - 4), count))
    return count, ordinals, [int(indices[i]) for i in ordinals]


def prefix_ids(tokenizer, visual_tokens, prompt):
    """Released continuous vision block, with the same fixed DP system prompt."""
    def special(text):
        value = tokenizer.convert_tokens_to_ids(text)
        if value is None or value == tokenizer.unk_token_id:
            raise ValueError(f"Unknown Qwen special token {text}")
        return value

    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    return (
        encode("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n")
        + [special("<|im_start|>")] + encode("user\n")
        + [special("<|vision_start|>")]
        + [special("<|image_pad|>")] * visual_tokens
        + [special("<|vision_end|>")] + encode("\n") + encode(prompt)
        + [special("<|im_end|>")] + encode("\n")
        + [special("<|im_start|>")] + encode("assistant\n")
    )


def build_protocol(args, selected, expected):
    from importlib.metadata import version
    return dict(
        protocol_version="official_ovo_simplestream_recent4_refill_v1",
        dataset="official_full", arm=ARM, model_path=str(official.MODEL.resolve()),
        model_config_sha256=official.sha256(official.MODEL / "config.json"),
        source_annotation=str(official.ANNOTATIONS.resolve()),
        source_sha256=official.sha256(official.ANNOTATIONS),
        upstream_url="https://github.com/EvolvingLMMs-Lab/SimpleStream",
        upstream_commit=UPSTREAM_COMMIT,
        upstream_sha256={name: official.sha256(UPSTREAM / name) for name in (
            "lib/recent_window_eval_qwen25.py", "lib/recent_window_eval.py")},
        fps=0.5, dtype="bfloat16", max_new_tokens=16, video_max_pixels=151200,
        image_min_pixels=3136, decoding="greedy", repetition_penalty=1.1,
        repetition_penalty_scope="generated_tokens_only_matching_DP",
        system_prompt="You are a helpful assistant.",
        attention_implementation="sdpa", score_mode="official_OVOBenchOfflineScore",
        video_mode="source_prefix_ceil_timestamp", budget=None, reference_dp_budget=500,
        recent_frames=4, recent_frame_semantics="last_four_real_sampled_frames_or_all_if_shorter",
        vision_encoding="independent_images_not_temporally_paired_video",
        vision_prefix_layout="released_qwen25_single_continuous_image_block",
        mrope="native_HF_get_rope_index_on_released_single_block_then_native_decode_positions",
        lm_cache="new_empty_cache_per_question_full_prefill_no_cross_question_KV",
        image_feature_reuse=args.reuse_vision,
        sampling="existing_DP_rounded_source_fps_fixed_step_exact_prefix_indices",
        controlled_changes_from_upstream=[
            "official_prompts_and_scorer_instead_of_SimpleStream_custom_templates_and_console_scoring",
            "same_DP_source_sampling_at_fps0.5_instead_of_full_clip_linspace_sampling",
            "same_pixel_cap151200_greedy_max_new16_repetition_penalty1.1",
            "same_DP_system_message_and_generated_only_repetition_penalty_scope",
            "decode_only_needed_window_frames_and_reuse_image_features_without_reusing_LM_KV",
            "transformers4.57_native_generation_API_for_current_environment",
        ],
        query_count=len(selected), full_query_count=len(expected),
        query_ids=[q["query_id"] for q in selected], query_list_sha256=digest(selected),
        full_query_list_sha256=digest(expected), grouped=True,
        execution_schedule="global_video_first_appearance_then_official_relative_order",
        concurrent_processes=args.concurrent_processes,
        official_aggregation="mean(backward_3_task_mean,realtime_6_task_mean,forward_3_task_mean)",
        debug_explain=False,
        code_sha256={name: official.sha256(ROOT / name) for name in SOURCE_FILES},
        processor_files_sha256={name: official.sha256(official.MODEL / name) for name in PROCESSOR_FILES},
        dependencies={name: version(name) for name in ("torch", "transformers", "numpy", "decord", "Pillow")},
    )


def cache_measurement(cache):
    layers = []
    for index, (key, value) in enumerate(cache):
        if key.ndim != 4 or key.shape != value.shape:
            raise RuntimeError("Unexpected physical KV tensor layout")
        layers.append(dict(layer=index, shape=list(key.shape), tokens_per_head=int(key.shape[-2]),
                           physical_slots=int(key.shape[0] * key.shape[1] * key.shape[2]),
                           bytes=int(key.numel() * key.element_size() + value.numel() * value.element_size())))
    return dict(layers=layers, physical_slots=sum(l["physical_slots"] for l in layers),
                bytes=sum(l["bytes"] for l in layers),
                tokens_per_head=layers[0]["tokens_per_head"] if layers else 0)


class GeneratedOnlyRepetitionPenalty:
    """Match the DP decoder: penalize only tokens generated for this answer."""
    def __init__(self, prompt_length, dtype, penalty=1.1):
        self.prompt_length, self.dtype, self.penalty = prompt_length, dtype, penalty

    def __call__(self, input_ids, scores):
        scores = scores.to(self.dtype)
        for token in set(input_ids[0, self.prompt_length:].tolist()):
            scores[0, token] = (scores[0, token] * self.penalty if scores[0, token] < 0
                                else scores[0, token] / self.penalty)
        return scores


class RecentFourRefill:
    def __init__(self):
        import torch
        from sparsemm_protokv.model_loader import load_pretrained
        self.torch = torch
        self.model, self.processor = load_pretrained(official.MODEL, dtype="bfloat16")
        self.image_token_id = self.model.config.image_token_id
        self.merge_size = self.model.config.vision_config.spatial_merge_size

    def encode_images(self, arrays):
        """Image ViT is independent of question, prefix and other cached frames."""
        from PIL import Image
        torch = self.torch
        with torch.inference_mode():
            images = [Image.fromarray(array).convert("RGB") for array in arrays]
            inputs = self.processor(
                text=[""], images=images, return_tensors="pt", min_pixels=3136, max_pixels=151200)
            grid = inputs["image_grid_thw"].to("cuda")
            pixels = inputs["pixel_values"].to("cuda", dtype=self.model.dtype)
            result = self.model.model.get_image_features(pixels, grid)
            if isinstance(result, (tuple, list)):
                result = torch.cat(result, dim=0)
            counts = (grid.prod(dim=-1) // self.merge_size ** 2).tolist()
            if result.shape[0] != sum(counts) or len(counts) != len(arrays):
                raise RuntimeError("Independent image feature count mismatch")
            features = list(result.split([int(n) for n in counts], dim=0))
            return [(features[i].clone(), grid[i:i + 1].clone()) for i in range(len(counts))]

    def answer(self, features, prompt):
        torch = self.torch
        torch.cuda.synchronize()
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            visual = torch.cat([f[0] for f in features], dim=0)
            grid = torch.cat([f[1] for f in features], dim=0)
            counts = [int(f[0].shape[0]) for f in features]
            tokens = sum(counts)
            ids = torch.tensor([prefix_ids(self.processor.tokenizer, tokens, prompt)], device="cuda")
            mask = torch.ones_like(ids)
            embeddings = self.model.get_input_embeddings()(ids)
            embeddings = embeddings.masked_scatter(
                (ids == self.image_token_id).unsqueeze(-1).expand_as(embeddings), visual)
            expected_positions, _ = self.model.model.get_rope_index(
                ids, image_grid_thw=grid, video_grid_thw=None, attention_mask=mask)
            # HF 4.57 constructs decode positions from these native deltas. The
            # released single-block prefix and its prefill MROPE are unchanged.
            self.model.model.rope_deltas = None
            trace = {"calls": 0}

            def before(module, args, kwargs):
                if trace["calls"] == 0:
                    past = kwargs.get("past_key_values")
                    length = 0 if past is None else int(past.get_seq_length())
                    trace["initial_cache_tokens"] = length
                    if length != 0:
                        raise RuntimeError("Language KV was reused across questions")
                    positions = kwargs.get("position_ids")
                    vision_positions = positions[-3:]
                    if not torch.equal(vision_positions, expected_positions):
                        raise RuntimeError("Prefill MROPE differs from released SimpleStream builder")
                    trace["prefill_position_sha256"] = digest(vision_positions.cpu().tolist())
                    trace["prefill_started"] = time.perf_counter()
                    # Let native generation construct MROPE from the real
                    # placeholder IDs first, then inject the pre-encoded image
                    # features. Passing inputs_embeds directly to generate in
                    # HF4.57 drops those IDs before multimodal MROPE creation.
                    kwargs["input_ids"] = None
                    kwargs["inputs_embeds"] = embeddings
                    return args, kwargs

            def after(module, args, kwargs, result):
                cache = result.past_key_values
                if trace["calls"] == 0:
                    measured = cache_measurement(cache)
                    if (len(measured["layers"]) != self.model.config.text_config.num_hidden_layers
                            or any(layer["tokens_per_head"] != ids.shape[1] for layer in measured["layers"])):
                        raise RuntimeError("The full prompt was not freshly prefilled on every layer")
                    trace["prefill_kv"] = measured
                    torch.cuda.synchronize()
                    trace["prefill_seconds"] = time.perf_counter() - trace["prefill_started"]
                trace["calls"] += 1

            pre = self.model.register_forward_pre_hook(before, with_kwargs=True)
            post = self.model.register_forward_hook(after, with_kwargs=True)
            try:
                generated = self.model.generate(
                    input_ids=ids, attention_mask=mask,
                    image_grid_thw=grid, max_new_tokens=16, do_sample=False,
                    repetition_penalty=1.0,
                    logits_processor=[GeneratedOnlyRepetitionPenalty(ids.shape[1], self.model.dtype)],
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True, return_dict_in_generate=True,
                )
            finally:
                pre.remove()
                post.remove()
            new_ids = generated.sequences[0, ids.shape[1]:]
            if new_ids.numel() < 1 or trace["calls"] < 1:
                raise RuntimeError("Empty fresh-prefill generation")
            response = self.processor.tokenizer.decode(new_ids, skip_special_tokens=True,
                                                        clean_up_tokenization_spaces=False).strip()
            final_kv = cache_measurement(generated.past_key_values)
            torch.cuda.synchronize()
            metadata = dict(
                image_grid_thw=grid.cpu().tolist(), visual_tokens_per_frame=counts,
                visual_tokens=tokens, input_tokens=int(ids.shape[1]),
                non_visual_input_tokens=int(ids.shape[1]) - tokens,
                official_prompt_text_tokens=len(self.processor.tokenizer.encode(prompt, add_special_tokens=False)),
                visual_slots_all_kv_heads=tokens * sum(l["shape"][1] for l in trace["prefill_kv"]["layers"]),
                initial_cache_tokens=trace["initial_cache_tokens"], fresh_lm_prefill=True,
                prefill_position_sha256=trace["prefill_position_sha256"],
                prefill_kv=trace["prefill_kv"], final_kv=final_kv,
                generated_tokens=int(new_ids.numel()), generated_token_ids=new_ids.cpu().tolist(),
                prefill_seconds=trace["prefill_seconds"],
                generation_seconds=time.perf_counter() - started,
                peak_process_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                peak_process_reserved_bytes=int(torch.cuda.max_memory_reserved()),
            )
            # No generation output/cache is stored on the adapter or reused by
            # the next query. Only feature tensors survive in the caller.
            return response, metadata


def validate_result(row, query):
    validate_row(row, query)
    if row.get("fresh_lm_prefill") is not True or row.get("initial_cache_tokens") != 0:
        raise ValueError("Result lacks proof of a fresh empty LM cache")
    expected_count = min(4, row["frames"])
    ordinals = list(range(row["frames"] - expected_count, row["frames"]))
    if (row.get("window_sample_ordinals") != ordinals or row.get("window_real_frames") != expected_count
            or len(row.get("window_source_frame_indices", [])) != expected_count
            or sum(row.get("visual_tokens_per_frame", [])) != row.get("visual_tokens")
            or row.get("input_tokens", 0) <= row.get("visual_tokens", 0)
            or row.get("prefill_kv", {}).get("tokens_per_head") != row.get("input_tokens")):
        raise ValueError("Invalid window/token-budget measurements")


def run_group(adapter, group, reuse_vision=True):
    from sparsemm_baselines.run_streamingbench import sampled_indices
    from sparsemm_protokv.models.qwen2_5_vl_stream import BatchScopedVideoReader
    if len({q["video_path"] for q in group}) != 1:
        raise ValueError("Feature reuse must stay within one video")
    started = time.perf_counter()
    reader = BatchScopedVideoReader(group[0]["video_path"], num_threads=1)
    indices = sampled_indices(len(reader), reader.get_avg_fps(), 0.5)
    reader_seconds = time.perf_counter() - started
    cache = {}
    for ordinal, query in enumerate(group):
        started = time.perf_counter()
        frames, sample_ordinals, raw_indices = window_plan(query, indices)
        if not reuse_vision:
            cache.clear()
        wanted = set(raw_indices)
        cache = {key: value for key, value in cache.items() if key in wanted}
        missing = [index for index in raw_indices if index not in cache]
        decode_started = time.perf_counter()
        arrays = reader.get_batch(missing).asnumpy() if missing else []
        decode_seconds = time.perf_counter() - decode_started
        adapter.torch.cuda.synchronize()
        vision_started = time.perf_counter()
        encoded = adapter.encode_images(arrays) if missing else []
        adapter.torch.cuda.synchronize()
        vision_seconds = time.perf_counter() - vision_started
        cache.update(zip(missing, encoded))
        if len(cache) != len(raw_indices) or len(cache) > 4:
            raise RuntimeError("Feature window is not exactly the selected recent frames")
        response, metadata = adapter.answer([cache[index] for index in raw_indices], query["prompt"])
        row = {key: query[key] for key in IDENTITY}
        row.update(metadata, response=response,
                   correct=official.official_correct(query["task"], response, query["target"]),
                   frames=frames, window_real_frames=len(raw_indices),
                   window_sample_ordinals=sample_ordinals, window_source_frame_indices=raw_indices,
                   window_timestamps_seconds=[index / reader.get_avg_fps() for index in raw_indices],
                   source_video_frames=len(reader), source_video_avg_fps=reader.get_avg_fps(),
                   encoded_frames=len(missing), reused_image_frames=len(raw_indices) - len(missing),
                   decode_seconds=decode_seconds, vision_seconds=vision_seconds,
                   reader_setup_seconds=reader_seconds if ordinal == 0 else 0,
                   seconds=time.perf_counter() - started + (reader_seconds if ordinal == 0 else 0),
                   execution_mode="grouped_visual_features_only" if reuse_vision else "fresh_visual_features",
                   prompt_sha256=digest(query["prompt"]))
        validate_result(row, query)
        yield row


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ids")
    parser.add_argument("--concurrent-processes", type=int, choices=(1, 2), default=2)
    parser.add_argument("--max-new", type=int, default=100000)
    parser.add_argument("--fresh-vision", dest="reuse_vision", action="store_false", default=True)
    args = parser.parse_args(argv)
    if args.max_new < 1:
        parser.error("--max-new must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    expected = official.queries()
    selected = schedule_queries(select_queries(expected, args.ids))
    directory = args.output_root.expanduser().resolve() / ARM
    directory.mkdir(parents=True, exist_ok=True)
    protocol = build_protocol(args, selected, expected)
    path = directory / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Experiment source/protocol changed; refusing to resume")
    rows = read_progress(directory / "queries.jsonl", selected)
    if rows and not path.exists():
        raise ValueError("Existing results lack a pinned protocol")
    for row, query in zip(rows, selected):
        validate_result(row, query)
    write_json(path, protocol)
    write_json(directory / "query_manifest.json", selected)
    for name in SOURCE_FILES:
        destination = directory / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    if len(rows) == len(selected):
        return
    from sparsemm_native.gpu_slots import gpu_lease
    with gpu_lease(directory, concurrent_processes=args.concurrent_processes) as admission:
        write_json(directory / "gpu_admission.json", admission)
        adapter = RecentFourRefill()
        remaining = selected[len(rows):len(rows) + args.max_new]
        groups = official.grouped_batches(remaining, len(remaining))
        with (directory / "queries.jsonl").open("a") as stream:
            for group in groups:
                for row in run_group(adapter, group, reuse_vision=args.reuse_vision):
                    validate_result(row, selected[len(rows)])
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    rows.append(row)
                    print(f"{ARM}: {len(rows)}/{len(selected)} {row['query_id']} {row['task']} "
                          f"correct={row['correct']} visual={row['visual_tokens']} "
                          f"KV/head={row['prefill_kv']['tokens_per_head']} seconds={row['seconds']:.3f}", flush=True)
        write_json(directory / "summary.json", official.summary(rows, len(selected)))


if __name__ == "__main__":
    main()
