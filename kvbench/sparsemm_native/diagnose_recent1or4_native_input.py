"""Run the old one/four-head native Qwen input path on the new run's exact frames.

This isolates the combined input-template/M-RoPE/decoder stack while holding
source sampled frame indices and recorded per-video resize fixed. It does not
claim to isolate any single member of that stack.
"""

import argparse
import json
import math
import sys
import types
from pathlib import Path

import torch
from PIL import Image

from sparsemm_native.diagnose_recent1or4_epm import DEFAULT_RUN, IDS, OLD_RUN, old_prompt, read_cases
from sparsemm_native import run_ovobench as ovo
from sparsemm_native.simplestream_resize import resize_hwc_batch
from sparsemm_baselines.run_streamingbench import sampled_indices
from sparsemm_protokv.models.qwen2_5_vl_stream import BatchScopedVideoReader


def exact_last_four(query, saved, source_mode, old_observation):
    if source_mode == "old_clip":
        reader = BatchScopedVideoReader(old_observation["source_clip"], num_threads=1)
        indices = old_observation["selected_raw_frame_indices"]
    elif source_mode == "current_source_old_indices":
        reader = BatchScopedVideoReader(query["video_path"], num_threads=1)
        indices = old_observation["selected_raw_frame_indices"]
    else:
        reader = BatchScopedVideoReader(query["video_path"], num_threads=1)
        indices = sampled_indices(len(reader), reader.get_avg_fps(), 1.0)
        target = min(len(indices), math.ceil(query["clip_end"]))
        if target != saved["frames"]:
            raise AssertionError("Frame count differs from saved current run")
        indices = indices[max(0, target - 4):target]
    frames = torch.from_numpy(reader.get_batch(indices).asnumpy())
    height, width = saved["resize_hw"]
    frames = resize_hwc_batch(frames, height, width)
    return [Image.fromarray(frame.numpy()) for frame in frames]


@torch.inference_mode()
def current_transformers_vision_encode(self, frames):
    """Old image-input construction with the newer get_image_features return type."""
    content = [{"type": "image", "image": frame} for frame in frames]
    content.append({"type": "text", "text": "."})
    messages = [{"role": "user", "content": content}]
    inputs = self.processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        return_dict=True, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(self.model.device)
    grid = inputs["image_grid_thw"].to(self.model.device)
    result = self._get_multimodal_model().get_image_features(pixel_values, grid)
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, list):
        result = torch.cat(result, dim=0)
    return result, grid


def run(output, ids, run_root, source_mode, penalty, compat_current_env):
    saved, manifests, old, annotations = read_cases(run_root, ids)
    observations = {row["query_id"]: row for shard in ("shard0", "shard1")
                    for line in (OLD_RUN / "shards" / shard / "query_observations.jsonl").read_text().splitlines()
                    if (row := json.loads(line))["query_id"] in ids}
    output.mkdir(parents=True, exist_ok=True)
    simplestream_root = Path("/home/yujun/program/SimpleStream")
    sys.path.insert(0, str(simplestream_root))
    from lib.recent_window_eval_qwen25 import RecentWindowQAModel
    from sparsemm_native.simplestream_recent_share_independent_last1 import install
    install(output)
    qa = RecentWindowQAModel(str(ovo.MODEL), device="cuda", max_new_tokens=256,
                              attn_implementation="flash_attention_2")
    if compat_current_env:
        qa.encode_vision = types.MethodType(current_transformers_vision_encode, qa)
    if penalty is not None:
        qa.model.generation_config.repetition_penalty = penalty
    for query_id in ids:
        frames = exact_last_four(manifests[query_id], saved[query_id], source_mode, observations[query_id])
        # Released SimpleStream encodes each one-second chunk separately.
        encoded = [qa.encode_vision([frame]) for frame in frames]
        features = torch.cat([item[0] for item in encoded], dim=0)
        grid = torch.cat([item[1] for item in encoded], dim=0)
        tokens = int(features.shape[0])
        for name, prompt in (("old_prompt", old_prompt(annotations[query_id])),
                             ("official_prompt", manifests[query_id]["prompt"])):
            response = qa.generate_with_cached_vision(features, grid, prompt)
            row = dict(query_id=query_id, variant="native_one_four_" + name,
                       source_mode=source_mode,
                       repetition_penalty=qa.model.generation_config.repetition_penalty,
                       transformers_vision_compat=compat_current_env,
                       response=response, target=saved[query_id]["target"],
                       correct=ovo.official_correct("EPM", response, saved[query_id]["target"]),
                       frames=len(frames), resize_hw=saved[query_id]["resize_hw"],
                       four_frame_visual_tokens=tokens,
                       saved_new_response=saved[query_id]["response"],
                       saved_old_response=old[query_id]["response"])
            with (output / "answers.jsonl").open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{query_id} {name} {response!r} target={row['target']} correct={row['correct']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ids", nargs="+", default=IDS)
    parser.add_argument("--source-mode", choices=("current_source", "current_source_old_indices", "old_clip"), default="current_source")
    parser.add_argument("--penalty", type=float, default=None)
    parser.add_argument("--compat-current-env", action="store_true")
    args = parser.parse_args()
    run(args.output, tuple(args.ids), args.run_root, args.source_mode, args.penalty,
        args.compat_current_env)
