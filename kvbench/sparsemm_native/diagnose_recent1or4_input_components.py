"""First-token EPM ablation of the text wrapper and visual position scheme.

Uses one native HF Qwen model and the same per-head one/four ragged decoder for
every arm. No historical KV is included. This leaves only the serialized
prefix/question and supplied M-RoPE coordinates as experimental variables.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import DynamicCache

from sparsemm_native.diagnose_recent1or4_epm import DEFAULT_RUN, read_cases
from sparsemm_native.diagnose_recent1or4_native_input import current_transformers_vision_encode
from sparsemm_native.simplestream_resize import resize_hwc_batch
from sparsemm_baselines.run_streamingbench import sampled_indices
from sparsemm_protokv.models.qwen2_5_vl_stream import BatchScopedVideoReader
from sparsemm_native import run_ovobench as ovo


SYSTEM = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
USER = "<|im_start|>user\n"


def frame_images(query, saved):
    reader = BatchScopedVideoReader(query["video_path"], num_threads=1)
    indices = sampled_indices(len(reader), reader.get_avg_fps(), 1.0)
    end = min(len(indices), math.ceil(query["clip_end"]))
    if end != saved["frames"]:
        raise AssertionError("Saved frame count changed")
    frames = torch.from_numpy(reader.get_batch(indices[end-4:end]).asnumpy())
    frames = resize_hwc_batch(frames, *saved["resize_hw"])
    return [Image.fromarray(frame.numpy()) for frame in frames], list(range(end-4, end))


def token_ids(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def make_sequence(qa, features, grids, prompt, ordinals, layout):
    tokenizer = qa.processor.tokenizer
    image_id = qa.image_token_id
    vision_start = qa._vision_start_id
    vision_end = qa._vision_end_id
    im_end = qa._im_end_id
    counts = (grids.prod(dim=-1) // qa.merge_size**2).tolist()
    if len(counts) != 4 or sum(counts) != features.shape[0]:
        raise AssertionError("Four per-frame image grids expected")
    system = layout in ("native_system", "native_system_manual_rope", "stream", "stream_no_system")
    prefix_text = (SYSTEM if system and layout != "stream_no_system" else "") + USER
    prefix = token_ids(tokenizer, prefix_text)
    if layout.startswith("native"):
        prefix += [vision_start]
        suffix = [vision_end] + token_ids(tokenizer, "\n" + prompt) + [im_end]
        suffix += token_ids(tokenizer, "\n<|im_start|>assistant\n")
    else:
        suffix = token_ids(tokenizer, "\n" + prompt + "<|im_end|><|im_start|>assistant\n")
    visual_start = len(prefix)
    visual_end = visual_start + sum(counts)
    ids = prefix + [image_id] * sum(counts) + suffix
    device = qa.model.device
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    embeds = qa._get_text_model().get_input_embeddings()(input_ids)
    embeds[0, visual_start:visual_end] = features.to(device=device, dtype=embeds.dtype)
    if layout.startswith("native") and layout != "native_system_manual_rope":
        positions, _ = qa._get_multimodal_model().get_rope_index(
            input_ids=input_ids, image_grid_thw=grids.to(device), video_grid_thw=None,
            attention_mask=torch.ones_like(input_ids))
    else:
        vision = qa.model.config.vision_config
        tps = float(vision.tokens_per_second)
        parts = [torch.arange(visual_start, device=device).view(1, 1, -1).expand(3, 1, -1)]
        for ordinal, grid, count in zip(ordinals, grids, counts):
            _, h, w = map(int, grid.tolist())
            h //= int(vision.spatial_merge_size)
            w //= int(vision.spatial_merge_size)
            if h * w != count:
                raise AssertionError("Feature/grid count differs")
            temporal = torch.full((count,), visual_start + math.floor(ordinal * tps), device=device)
            yy = torch.arange(h, device=device).view(h, 1).expand(h, w).flatten() + visual_start
            xx = torch.arange(w, device=device).view(1, w).expand(h, w).flatten() + visual_start
            parts.append(torch.stack((temporal, yy, xx))[:, None])
        cursor = int(torch.cat(parts, dim=2).max()) + 1
        qpos = torch.arange(cursor, cursor + len(suffix), device=device).view(1, 1, -1).expand(3, 1, -1)
        parts.append(qpos)
        positions = torch.cat(parts, dim=2)
    if positions.shape != (3, 1, len(ids)):
        raise AssertionError("Position count differs from input sequence")
    last_frame_start = visual_end - counts[-1]
    one_indices = list(range(visual_start)) + list(range(last_frame_start, visual_end))
    offsets = [visual_start]
    for count in counts:
        offsets.append(offsets[-1] + count)
    position_audit = dict(
        visual_start=visual_start,
        visual_end=visual_end,
        frame_first_positions=[positions[:, 0, offset].tolist() for offset in offsets[:-1]],
        frame_last_positions=[positions[:, 0, offset-1].tolist() for offset in offsets[1:]],
        question_first_position=positions[:, 0, visual_end].tolist(),
    )
    return embeds, positions, visual_end, one_indices, position_audit


@torch.inference_mode()
def first_token(qa, features, grids, prompt, ordinals, layout):
    from sparsemm_native.simplestream_recent_share_attention import HEADS
    from sparsemm_native.simplestream_recent_share_independent_last1 import (
        _combine_prefix_caches, _prefill_question_suffix, _ragged_helpers,
    )
    Layer, Cache, _ = _ragged_helpers()
    embeds, positions, visual_end, one_indices, position_audit = make_sequence(
        qa, features, grids, prompt, ordinals, layout)
    full_cache = DynamicCache()
    qa.model(inputs_embeds=embeds[:, :visual_end],
             attention_mask=torch.ones((1, visual_end), device=embeds.device, dtype=torch.long),
             position_ids=positions[:, :, :visual_end],
             past_key_values=full_cache, use_cache=True, return_dict=True)
    index = torch.tensor(one_indices, device=embeds.device)
    one_cache = DynamicCache()
    qa.model(inputs_embeds=embeds.index_select(1, index),
             attention_mask=torch.ones((1, len(one_indices)), device=embeds.device, dtype=torch.long),
             position_ids=positions.index_select(2, index),
             past_key_values=one_cache, use_cache=True, return_dict=True)
    ragged = Cache(_combine_prefix_caches(full_cache, one_cache, set(HEADS), Layer))
    hidden = _prefill_question_suffix(qa.model, embeds[:, visual_end:],
                                      positions[:, :, visual_end:], ragged)
    logits = qa.model.lm_head(hidden)[0, -1]
    next_id = int(logits.argmax())
    return qa.processor.tokenizer.decode([next_id], skip_special_tokens=True), next_id, position_audit


def run(output, ids, run_root, layouts=("native", "native_system", "native_system_manual_rope", "stream", "stream_no_system")):
    saved, manifests, _, _ = read_cases(run_root, ids)
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, "/home/yujun/program/SimpleStream")
    from lib.recent_window_eval_qwen25 import RecentWindowQAModel
    qa = RecentWindowQAModel(str(ovo.MODEL), device="cuda", max_new_tokens=16,
                              attn_implementation="flash_attention_2")
    qa.encode_vision = current_transformers_vision_encode.__get__(qa)
    for query_id in ids:
        images, ordinals = frame_images(manifests[query_id], saved[query_id])
        encoded = [qa.encode_vision([image]) for image in images]
        features = torch.cat([item[0] for item in encoded], dim=0)
        grids = torch.cat([item[1] for item in encoded], dim=0)
        for layout in layouts:
            response, token_id, position_audit = first_token(
                qa, features, grids, manifests[query_id]["prompt"], ordinals, layout)
            row = dict(query_id=query_id, layout=layout, response=response,
                       first_token_id=token_id, target=saved[query_id]["target"],
                       correct=ovo.official_correct("EPM", response, saved[query_id]["target"]),
                       saved_new_response=saved[query_id]["response"],
                       position_audit=position_audit)
            with (output / "answers.jsonl").open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(query_id, layout, repr(response), token_id, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--ids", nargs="+", default=("32", "192"))
    parser.add_argument("--layouts", nargs="+", choices=("native", "native_system",
                        "native_system_manual_rope", "stream", "stream_no_system"))
    args = parser.parse_args()
    run(args.output, tuple(args.ids), args.run_root, tuple(args.layouts) if args.layouts else
        ("native", "native_system", "native_system_manual_rope", "stream", "stream_no_system"))
