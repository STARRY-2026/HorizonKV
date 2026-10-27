"""Qwen3-VL SparseMM OCR-box/top-1 scoring with per-image checkpoints."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image
from transformers import DynamicCache, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from sparsemm_baselines.run_streamingbench import gpu_lease, write_json
from sparsemm_baselines.qwen3_window4 import enable_window4_vision, FULL_BLOCKS, WINDOW_PATCHES
from sparsemm_native.policy import allocate_head_budgets, load_head_scores


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def box_tokens(entry, original_hw, grid, patch_size=14, merge=2):
    """Adapt the upstream integer OCR-box mapping to Qwen3's merged image grid."""
    _, raw_h, raw_w = map(int, grid)
    height, width = raw_h // merge, raw_w // merge
    original_width, original_height = original_hw
    scaled_width, scaled_height = raw_w * patch_size, raw_h * patch_size
    cell_width = scaled_width // width
    cell_height = scaled_height // height
    (x0, y0), (x1, y1) = entry["coord"]
    x0, x1 = int((x0 * scaled_width / original_width) // cell_width), int((x1 * scaled_width / original_width) // cell_width + 1)
    y0, y1 = int((y0 * scaled_height / original_height) // cell_height), int((y1 * scaled_height / original_height) // cell_height + 1)
    mask = torch.zeros((height, width), dtype=torch.bool)
    mask[y0:y1, x0:x1] = True
    return torch.where(mask.flatten())[0]


@torch.inference_mode()
def evaluate(model, processor, record, max_pixels, max_decode):
    image_path = Path(record["image_name"])
    with Image.open(image_path) as image:
        original_size = image.size
        rgb = image.convert("RGB")
    prompt = processor.apply_chat_template(
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": [{"type": "image"},
                                      {"type": "text", "text": "Provide all the OCR results of this image."}]}],
        tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[rgb], return_tensors="pt",
                       size={"shortest_edge": processor.image_processor.size["shortest_edge"],
                             "longest_edge": max_pixels})
    grid = inputs["image_grid_thw"][0].tolist()
    first_image_token = int(torch.where(inputs["input_ids"][0] == model.config.image_token_id)[0][0])
    boxes = [(item["chunk"], box_tokens(item, original_size, grid,
                                        patch_size=model.config.vision_config.patch_size,
                                        merge=model.config.vision_config.spatial_merge_size))
             for item in record["2_coord"]]
    layers = model.config.text_config.num_hidden_layers
    heads = model.config.text_config.num_attention_heads
    scores = np.zeros((layers, heads), dtype=np.float64)
    hits = np.zeros((layers, heads), dtype=np.int64)
    top_indices = [None] * layers
    hooks = []
    for index, decoder in enumerate(model.language_model.layers):
        def capture(module, arguments, output, index=index):
            weights = output[1]
            if weights is None:
                raise RuntimeError("Qwen3 eager attention did not expose weights")
            top_indices[index] = weights[0, :, -1].topk(1, dim=-1).indices[:, 0].cpu()
        hooks.append(decoder.self_attn.register_forward_hook(capture))
    try:
        past = DynamicCache()
        inputs = inputs.to(model.device)
        outputs = model(**inputs, past_key_values=past, use_cache=True,
                        cache_position=torch.arange(inputs["input_ids"].shape[1], device=model.device),
                        logits_to_keep=1)
        generated, unique = [], 0
        for step in range(max_decode):
            token = int(outputs.logits[0, -1].argmax())
            generated.append(token)
            if token == processor.tokenizer.eos_token_id:
                break
            decoded = processor.tokenizer.decode([token])
            matched = [indices for chunk, indices in boxes if decoded in chunk]
            if len(matched) == 1 and len(matched[0]) > 0:
                unique += 1
                bbox = matched[0] + first_image_token
                for layer, maxima in enumerate(top_indices):
                    if maxima is None:
                        raise RuntimeError("Missing Qwen3 attention head")
                    found = torch.isin(maxima, bbox)
                    scores[layer] += found.numpy().astype(np.float64) / len(bbox)
                    hits[layer] += found.numpy().astype(np.int64)
            if step + 1 == max_decode:
                break
            next_pos = past.get_seq_length()
            outputs = model(input_ids=torch.tensor([[token]], device=model.device),
                            past_key_values=past, use_cache=True,
                            cache_position=torch.tensor([next_pos], device=model.device),
                            logits_to_keep=1)
        return scores, hits, {"image": str(image_path), "grid_thw": grid,
                              "visual_tokens": grid[0] * (grid[1] // 2) * (grid[2] // 2),
                              "generated_tokens": len(generated), "unique_ocr_matches": unique,
                              "decoded": processor.tokenizer.decode(generated, skip_special_tokens=True)}
    finally:
        for hook in hooks:
            hook.remove()


def main(args):
    out = args.output_dir.resolve()
    if out.exists() and not args.resume:
        raise FileExistsError(out)
    dataset = json.loads(args.annotations.read_text())
    if not 1 <= args.num_images <= len(dataset):
        raise ValueError("Image count must fit the real SynthDOG dataset")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "running", "method": "sparsemm", "model": str(args.model_path.resolve()),
                "code_sha256": digest(__file__),
                "vision_adapter_sha256": digest(Path(__file__).resolve().parents[1] / "sparsemm_baselines/qwen3_window4.py") if args.vision_topology == "window4" else None,
                "model_config_sha256": digest(args.model_path / "config.json"),
                "dataset_sha256": digest(args.annotations), "selected_ids": [x["id"] for x in dataset[:args.num_images]],
                "max_pixels": args.max_pixels, "max_decode": args.max_decode,
                "processor_use_fast": False,
                "attention_implementation": "eager",
                "vision_topology": args.vision_topology,
                "model_variant": args.vision_topology == "window4",
                "window_patches": WINDOW_PATCHES if args.vision_topology == "window4" else None,
                "full_vision_blocks": list(FULL_BLOCKS) if args.vision_topology == "window4" else None,
                "vision_patch_operator": "linear_flattened_conv3d_bf16" if args.vision_topology == "window4" else "native_conv3d",
                "pixel_protocol": "qwen2_5_official_12845056" if args.max_pixels == 12_845_056 else "qwen3_custom_cap",
                "scoring_rule": "unique decoded OCR substring; top1 attention inside merged bbox += 1/bbox_tokens",
                "calibration_status": ("experimental_window4_qwen3_model_variant_1000_images" if args.vision_topology == "window4" else "formal_1000_image_qwen3_adaptation") if args.num_images == 1000 else "bounded_pilot_not_official_1000_image_calibration",
                "full_benchmarks_paused": True, "images": []}
    if args.resume:
        saved = json.loads((out / "manifest.json").read_text())
        for key in ("method", "model", "code_sha256", "vision_adapter_sha256", "model_config_sha256", "dataset_sha256",
                    "selected_ids", "max_pixels", "max_decode", "processor_use_fast",
                    "attention_implementation", "vision_topology", "model_variant", "window_patches", "full_vision_blocks",
                    "vision_patch_operator",
                    "pixel_protocol", "scoring_rule"):
            if saved.get(key) != manifest[key]:
                raise ValueError(f"SparseMM resume protocol mismatch: {key}")
        manifest = saved
        if len(manifest["images"]) > args.num_images:
            raise ValueError("SparseMM checkpoint exceeds selection")
        for index, record in enumerate(manifest["images"]):
            if record["id"] != dataset[index]["id"]:
                raise ValueError(f"SparseMM checkpoint image order mismatch at {index}")
            source = out / f"image_{record['id']}.json"
            if not source.is_file() or digest(source) != record.get("sha256"):
                raise ValueError(f"SparseMM checkpoint hash mismatch at {index}")
            saved_image = json.loads(source.read_text())
            if np.asarray(saved_image["scores"]).shape != (36, 32):
                raise ValueError(f"SparseMM checkpoint score shape mismatch at {index}")
        if manifest["status"].startswith("complete"):
            if len(manifest["images"]) != args.num_images or not (out / "head_scores.csv").is_file():
                raise ValueError("Completed SparseMM checkpoint is missing results")
            print(json.dumps({"status": manifest["status"], "images": len(manifest["images"])}), flush=True)
            return
        manifest.update(status="running", resumed_utc=datetime.now(timezone.utc).isoformat())
        manifest.pop("error", None)
    write_json(out / "manifest.json", manifest)
    started = time.perf_counter()
    try:
        with gpu_lease(out):
            processor = Qwen3VLProcessor.from_pretrained(args.model_path, use_fast=False,
                                                         local_files_only=True)
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                args.model_path, dtype=torch.bfloat16, attn_implementation="eager",
                low_cpu_mem_usage=True, device_map="cuda", local_files_only=True).eval()
            if args.vision_topology == "window4":
                enable_window4_vision(model)
            layers, heads = model.config.text_config.num_hidden_layers, model.config.text_config.num_attention_heads
            raw = {f"{layer}-{head}": [] for layer in range(layers) for head in range(heads)}
            for record in manifest["images"]:
                saved_image = json.loads((out / f"image_{record['id']}.json").read_text())
                saved_scores = np.asarray(saved_image["scores"], dtype=np.float64)
                for layer in range(layers):
                    for head in range(heads):
                        raw[f"{layer}-{head}"].append(float(saved_scores[layer, head]))
            for item in dataset[len(manifest["images"]):args.num_images]:
                scores, hits, metadata = evaluate(model, processor, item, args.max_pixels, args.max_decode)
                image_file = out / f"image_{item['id']}.json"
                write_json(image_file, {**metadata, "scores": scores.tolist(), "hits": hits.tolist()})
                for layer in range(layers):
                    for head in range(heads):
                        raw[f"{layer}-{head}"].append(float(scores[layer, head]))
                manifest["images"].append({"id": item["id"], **metadata,
                                           "score_sum": float(scores.sum()), "hit_count": int(hits.sum()),
                                           "sha256": digest(image_file)})
                write_json(out / "manifest.json", manifest)
                print(f"SPARSEMM_QWEN3_OCR id={item['id']} hits={int(hits.sum())} tokens={metadata['visual_tokens']}", flush=True)
            write_json(out / "raw_head_scores.json", raw)
            with (out / "head_scores.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("layer", "head", "score_sum", "score_mean", "nonzero", "max"))
                for layer in range(layers):
                    for head in range(heads):
                        values = raw[f"{layer}-{head}"]
                        writer.writerow((layer, head, sum(values), float(np.mean(values)),
                                         sum(v != 0 for v in values), max(values)))
            scores = load_head_scores(out / "head_scores.csv", layers, heads)
            capacities, allocation = allocate_head_budgets(
                scores, budget=500, window_size=32,
                num_kv_heads=model.config.text_config.num_key_value_heads)
            write_json(out / "allocation_k500.json", allocation)
            completion = ("complete_variant" if args.vision_topology == "window4" else "complete_formal") if args.num_images == 1000 else "complete_pilot"
            manifest.update(status=completion, finished_utc=datetime.now(timezone.utc).isoformat(),
                            seconds=round(time.perf_counter() - started, 3),
                            decode_limit_count=sum(item["generated_tokens"] >= args.max_decode for item in manifest["images"]),
                            head_scores_sha256=digest(out / "head_scores.csv"),
                            score_total=float(scores.sum()), capacity_shape=list(capacities.shape))
            write_json(out / "manifest.json", manifest)
            print(json.dumps({"status": manifest["status"], "images": len(manifest["images"]),
                              "score_total": manifest["score_total"], "seconds": manifest["seconds"]}), flush=True)
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(out / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-images", type=int, default=3)
    parser.add_argument("--max-pixels", type=int, default=1_200_000)
    parser.add_argument("--max-decode", type=int, default=64)
    parser.add_argument("--vision-topology", choices=("native", "window4"), default="native")
    parser.add_argument("--resume", action="store_true")
    main(parser.parse_args())
