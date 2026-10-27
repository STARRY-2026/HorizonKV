"""Bounded Qwen3-VL full-context probe on KVBench MCQA annotations.

This checks model/data compatibility. It does not implement KV compression or
claim equivalence to KVBench's streaming methods.
"""

import argparse
from contextlib import nullcontext
import json
import math
import time
from pathlib import Path


def run(args):
    import torch
    from decord import VideoReader
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
    from transformers.video_utils import VideoMetadata

    from sparsemm_baselines.run_streamingbench import gpu_lease, sampled_indices

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    videos = json.loads(args.anno_path.read_text())[args.video_offset: args.video_offset + args.max_videos]
    if not videos or sum(min(len(v["conversations"]), args.max_questions) for v in videos) > 8:
        raise ValueError("Probe requires 1–8 questions and at most 3 videos")
    for video in videos:
        for question in video["conversations"][: args.max_questions]:
            if not 0 < float(question["end_time"]) <= 120:
                raise ValueError("Question endpoint must be in (0, 120] seconds")
    output.mkdir(parents=True)
    with (nullcontext() if args.processor_only else gpu_lease(output)):
        model_path = args.model_path.resolve()
        if json.loads((model_path / "config.json").read_text()).get("model_type") != "qwen3_vl":
            raise ValueError("Only dense Qwen3-VL is supported by this probe")
        processor = Qwen3VLProcessor.from_pretrained(model_path, local_files_only=True)
        model = None if args.processor_only else Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
            low_cpu_mem_usage=True, device_map="cuda",
        ).eval()
        results = []
        for video in videos:
            reader = VideoReader(video["video_path"], num_threads=1)
            indices = sampled_indices(len(reader), reader.get_avg_fps(), args.sample_fps, video.get("clip"))
            for question in video["conversations"][: args.max_questions]:
                target = min(len(indices), math.ceil(float(question["end_time"]) * args.sample_fps))
                if target < 1:
                    raise ValueError("Question has no sampled frames")
                frames = reader.get_batch(indices[:target]).asnumpy()
                # Qwen3-VL's video processor rejects fewer than two frames.
                # Treat a one-frame prefix as an image to avoid inventing a
                # second timestamp or changing the actual visual observation.
                modality = "image" if target == 1 else "video"
                options = "\n".join(f"({chr(65 + i)}) {choice}" for i, choice in enumerate(question["choices"]))
                text = (f"Question: {question['question']}\nOptions:\n{options}\n"
                        "Only give the best option.")
                messages = [{"role": "user", "content": [{"type": modality}, {"type": "text", "text": text}]}]
                prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                started = time.perf_counter()
                if modality == "image":
                    image_kwargs = {} if args.video_max_pixels is None else {"size": {
                        "shortest_edge": processor.image_processor.size["shortest_edge"],
                        "longest_edge": args.video_max_pixels,
                    }}
                    inputs = processor(text=[prompt], images=[frames[0]], return_tensors="pt",
                                       **image_kwargs)
                    grid = inputs["image_grid_thw"][0].tolist()
                else:
                    metadata = VideoMetadata(total_num_frames=target, fps=args.sample_fps,
                                             width=frames.shape[2], height=frames.shape[1],
                                             frames_indices=list(range(target)))
                    video_kwargs = {} if args.video_max_pixels is None else {"size": {
                        "shortest_edge": processor.video_processor.size["shortest_edge"],
                        "longest_edge": args.video_max_pixels,
                    }}
                    inputs = processor(text=[prompt], videos=[frames], video_metadata=[metadata],
                                       return_tensors="pt", **video_kwargs)
                    grid = inputs["video_grid_thw"][0].tolist()
                merge = int(json.loads((model_path / "config.json").read_text())["vision_config"]["spatial_merge_size"])
                visual_tokens = grid[0] * (grid[1] // merge) * (grid[2] // merge)
                answer = None
                if model is not None:
                    with torch.inference_mode():
                        answer_ids = model.generate(**inputs.to("cuda"), max_new_tokens=16, do_sample=False)
                    torch.cuda.synchronize()
                    answer = processor.batch_decode(answer_ids[:, inputs["input_ids"].shape[1]:],
                                                    skip_special_tokens=True)[0]
                row = {"video_id": video["video_id"], "question": question["question"],
                       "answer": answer, "target": question["answer"], "sampled_frames": target,
                       "input_modality": modality,
                       "grid_thw": grid, "visual_tokens": visual_tokens,
                       "input_tokens": int(inputs["input_ids"].shape[1]),
                       "seconds": round(time.perf_counter() - started, 3)}
                results.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
                with (output / "results.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        (output / "results.json").write_text(json.dumps({
            "kind": "qwen3_vl_processor_probe" if args.processor_only else "qwen3_vl_full_context_probe",
            "model_path": str(model_path),
            "annotation_path": str(args.anno_path.resolve()), "sample_fps": args.sample_fps,
            "video_max_pixels": args.video_max_pixels,
            "results": results,
        }, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--anno-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-fps", type=float, default=0.5)
    parser.add_argument("--video-max-pixels", type=int)
    parser.add_argument("--max-videos", type=int, default=2, choices=(1, 2, 3))
    parser.add_argument("--video-offset", type=int, default=0)
    parser.add_argument("--max-questions", type=int, default=1, choices=range(1, 9))
    parser.add_argument("--processor-only", action="store_true")
    run(parser.parse_args())
