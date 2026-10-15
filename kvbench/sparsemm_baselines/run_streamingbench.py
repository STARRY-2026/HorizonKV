"""Evaluate SparseMM AdaKV/PyramidKV selectors with native-resolution video.

This is a separate streaming adaptation, not an upstream image-benchmark run.
The --dry-run path uses only Python's standard library and never loads a model.
"""

import argparse
import ast
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
DEFAULT_HERMES = WORKSPACE / "hermes"
UPSTREAM_COMMIT = "33f2f0ab96d1d99516bcd2092caccab60516ff2c"
LETTERS = "ABCDEFGH"
FIELDS = ["video_id", "question", "choices", "answer", "correct_choice",
          "pred_answer", "pred_choice", "qa_acc", "task", "duration_category"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def verify_upstream():
    upstream = ROOT / "upstream"
    commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError(f"SparseMM source must be pinned to {UPSTREAM_COMMIT}; found {commit}")
    changed = subprocess.run(["git", "-C", str(upstream), "diff", "--quiet", "HEAD", "--"], check=False)
    if changed.returncode:
        raise RuntimeError("Pinned SparseMM tracked files changed; refusing to label modified source as upstream")
    return commit


def code_hashes():
    files = [ROOT / name for name in ("core.py", "cache.py", "adapter.py", "run_streamingbench.py", "requirements-streaming.txt")]
    files += sorted((ROOT / "upstream/sparsemm").glob("*.py"))
    return {str(p.relative_to(ROOT)): sha256(p) for p in files if p.exists()}


def dependency_versions():
    versions = {}
    for package in ("torch", "transformers", "decord", "numpy", "pillow", "flash-attn", "qwen-vl-utils", "pandas"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def prepare_annotations(path, max_videos=250, max_questions=None, video_root=None):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("Annotations must be a nonempty list of video records")
    if max_videos is not None and max_videos < 1 or max_questions is not None and max_questions < 1:
        raise ValueError("Video and per-video question limits must be positive")
    out, ids = [], set()
    for original in data if max_videos is None else data[:max_videos]:
        video = dict(original)
        vid = str(video["video_id"])
        if vid in ids:
            raise ValueError(f"Duplicate video ID {vid}")
        ids.add(vid)
        vp = Path(video["video_path"])
        if video_root:
            vp = Path(video_root) / vp.name
        elif not vp.is_absolute():
            vp = Path(path).resolve().parent / vp
        if not vp.is_file():
            raise FileNotFoundError(f"Video {vid}: {vp}")
        if vp.suffix.lower() == ".npy":
            raise ValueError("This runner requires encoded videos with measured FPS; .npy sampling is ambiguous")
        video["video_path"] = str(vp.resolve())
        questions = video["conversations"]
        if max_questions is not None:
            questions = questions[:max_questions]
        if not questions:
            raise ValueError(f"Video {vid} has no questions")
        last_time = 0.0
        for question in questions:
            choices = question.get("choices")
            if not question.get("question") or not isinstance(choices, list) or not 1 <= len(choices) <= len(LETTERS):
                raise ValueError(f"Video {vid}: requires an MCQA question with 1..8 choices")
            if len(set(choices)) != len(choices) or question.get("answer") not in choices:
                raise ValueError(f"Video {vid}: choices must be unique and contain the exact answer")
            end = float(question["end_time"])
            if not math.isfinite(end) or end <= 0 or end < last_time:
                raise ValueError(f"Video {vid}: end_time must be positive and nondecreasing")
            last_time = end
        video["conversations"] = questions
        out.append(video)
    return out


def extract_choice(answer):
    # Match the established StreamingBench/BaseVQA parser exactly.
    answer = answer.strip()
    if ")" in answer:
        i = answer.index(")")
        return answer[i - 1:i]
    return answer[:1]


def format_mcqa(model, question):
    options = "\n".join(f"({LETTERS[i]}) {choice}" for i, choice in enumerate(question["choices"]))
    formatted = f"Question: {question['question']}\nOptions:\n{options}\nOnly give the best option."
    return {"question": question["question"], "formatted_question": formatted,
            "prompt": model.get_prompt(formatted, mc=True)}


def validate_row(row, video, question):
    if str(row.get("video_id")) != str(video["video_id"]) or row.get("question") != question["question"]:
        raise RuntimeError("Result CSV does not match the selected annotation order")
    expected = LETTERS[question["choices"].index(question["answer"])]
    try:
        choices = ast.literal_eval(row["choices"]) if isinstance(row["choices"], str) else row["choices"]
        accuracy = float(row["qa_acc"])
    except (KeyError, ValueError, TypeError, SyntaxError) as exc:
        raise RuntimeError("Result CSV has invalid MCQA fields") from exc
    if choices != question["choices"] or row.get("answer") != question["answer"] or row.get("correct_choice") != expected:
        raise RuntimeError("Result CSV has mismatched choices or ground truth")
    if row.get("pred_answer") is None or row.get("pred_choice") is None or accuracy not in (0.0, 100.0):
        raise RuntimeError("Result CSV has missing predictions or invalid accuracy")
    if row["pred_choice"] != extract_choice(row["pred_answer"]):
        raise RuntimeError("Result CSV predicted choice disagrees with its answer text")
    if accuracy != 100.0 * (row["pred_choice"] == expected):
        raise RuntimeError("Result CSV accuracy disagrees with its predicted choice")
    expected_task = question.get("task", question.get("question_type", video.get("task", "")))
    if row.get("task", "") != (expected_task or ""):
        raise RuntimeError("Result CSV task annotation differs")


def load_resume(path, annotations, require_complete=False):
    if not Path(path).exists():
        return [], 0
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    cursor, completed = 0, 0
    for video in annotations:
        if cursor == len(rows):
            break
        for question in video["conversations"]:
            if cursor >= len(rows):
                raise RuntimeError("Resume CSV ends inside a video; recover from a complete video or use a new directory")
            validate_row(rows[cursor], video, question)
            cursor += 1
        completed += 1
    if cursor != len(rows) or require_complete and completed != len(annotations):
        raise RuntimeError("Result CSV contains extra rows or does not cover all selected videos")
    return rows, completed


def write_results(path, rows):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, FIELDS, quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def check_pause(args, out):
    marker = ROOT / "results/PAUSED"
    if marker.exists() and not (args.resume_experiments or os.environ.get("SPARSEMM_RESUME") == "1"):
        (out / "run.status").write_text("paused\n")
        raise RuntimeError("Experiments are paused. Explicitly pass --resume-experiments or SPARSEMM_RESUME=1 when resuming is authorized.")


@contextmanager
def gpu_lease(out):
    """Use the same cooperative GPU lock as other local StreamingBench runs."""
    with (out / ".run.lock").open("w") as output_lock, Path("/tmp/hermes-gpu0.lock").open("a") as gpu_lock:
        for stream in (output_lock, gpu_lock):
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Output directory or GPU is busy; no job has been queued") from exc
        gpu = os.environ.get("SPARSEMM_GPU_INDEX", "0")
        report = subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
        free_mib = int(report.strip())
        minimum = int(os.environ.get("SPARSEMM_MIN_FREE_MIB", "28000"))
        if free_mib < minimum:
            raise RuntimeError(f"GPU {gpu} has {free_mib} MiB free; requires {minimum} MiB. No job has been queued")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        yield


def sampled_indices(frame_count, avg_fps, sample_fps, clip=None):
    # Preserve the established local StreamingBench sampling convention.
    fps = round(avg_fps)
    if fps <= 0 or sample_fps > fps:
        raise ValueError("Invalid video FPS or sampling FPS exceeds rounded video FPS")
    start = max(0, int(clip[0] * fps)) if clip else 0
    end = min(frame_count, int(clip[1] * fps) + 1) if clip else frame_count
    step = int(fps / sample_fps)
    return list(range(start, end, step))


def analyze_video(model, video, sample_fps, chunk_size, torch_module, reader_factory=None):
    if reader_factory is None:
        from decord import VideoReader
        reader_factory = VideoReader
    reader = reader_factory(video["video_path"], num_threads=1)
    indices = sampled_indices(len(reader), reader.get_avg_fps(), sample_fps, video.get("clip"))
    if not indices:
        raise RuntimeError(f"No sampled frames in video {video['video_id']}")
    model._active_video_id = video["video_id"]
    model.clear_cache()
    model.encode_init_prompt()
    current, rows = 0, []
    for qi, question in enumerate(video["conversations"]):
        model._active_question_index = qi
        model._active_question = question["question"]
        model._active_question_end_time = question["end_time"]
        target = min(len(indices), math.ceil(float(question["end_time"]) * sample_fps))
        while current < target:
            end = min(current + chunk_size, target)
            frames = torch_module.from_numpy(reader.get_batch(indices[current:end]).asnumpy())
            model.encode_video_chunk(frames, is_last_chunk=end == target)
            model.predict_and_compress()
            current = end
        pred_answer = str(model.question_answering(format_mcqa(model, question), max_new_tokens=16))
        pred_choice = extract_choice(pred_answer)
        correct = LETTERS[question["choices"].index(question["answer"])]
        rows.append({"video_id": video["video_id"], "question": question["question"],
                     "choices": question["choices"], "answer": question["answer"],
                     "correct_choice": correct, "pred_answer": pred_answer, "pred_choice": pred_choice,
                     "qa_acc": 100.0 * (pred_choice == correct),
                     "task": question.get("task", question.get("question_type", video.get("task", ""))) or "",
                     "duration_category": video.get("duration_category", "") or ""})
        print(f"video={video['video_id']} question={qi + 1} pred={pred_choice!r} correct={correct}", flush=True)
    model.clear_cache()
    return rows


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("adakv", "pyramidkv"), required=True)
    parser.add_argument("--anno-path", type=Path, default=DEFAULT_HERMES / "data/streamingbench/streamingbench_realtime_half_userpaths.json")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_HERMES / "models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--video-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-videos", type=int, default=250)
    parser.add_argument("--max-questions", type=int, help="Per-video question limit for smoke tests")
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--budget-scope", choices=("prompt", "visual"), default="prompt")
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--floor-alpha", type=float, default=0.2)
    parser.add_argument("--pooling", choices=("avgpool", "maxpool"), default="avgpool")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--beta", type=int, default=20)
    parser.add_argument("--gqa-reduce", choices=("mean", "max"), default="mean")
    parser.add_argument("--score-precision", choices=("float32", "input"), default="float32")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--sample-fps", type=float, default=0.5)
    parser.add_argument("--encode-chunk-size", type=int, default=16)
    parser.add_argument("--resume-experiments", action="store_true", help="Explicitly bypass the previous experiment pause for this invocation")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_videos < 1 or args.max_questions is not None and args.max_questions < 1:
        parser.error("Video and question limits must be positive")
    if not math.isfinite(args.sample_fps) or args.sample_fps <= 0 or args.encode_chunk_size < 1:
        parser.error("FPS and chunk size must be positive")
    if not 0 <= args.floor_alpha <= 1 or args.window_size < 1 or args.budget <= args.window_size or args.beta < 1:
        parser.error("Require budget > window_size > 0, beta >= 1, floor_alpha in [0,1]")
    if args.kernel_size < 1 or args.kernel_size % 2 != 1:
        parser.error("Kernel size must be positive and odd")
    commit = verify_upstream()
    data = prepare_annotations(args.anno_path, args.max_videos, args.max_questions, args.video_root)
    selection = "smoke" if args.max_videos == 1 and args.max_questions is not None else f"first{args.max_videos}"
    out = (args.output_dir or ROOT / f"results/{args.method}_{selection}_k{args.budget}_{args.budget_scope}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    subset_hash = hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    keys = ("method", "budget", "window_size", "kernel_size", "floor_alpha", "pooling", "normalize", "beta", "gqa_reduce", "budget_scope", "score_precision")
    config = {key: getattr(args, key) for key in keys}
    protocol = dict(config, dtype=args.dtype, sample_fps=args.sample_fps, encode_chunk_size=args.encode_chunk_size,
                    subset_sha256=subset_hash, source_sha256=sha256(args.anno_path),
                    model_path=str(args.model_path.resolve()), native_resolution=True,
                    sampling="rounded_source_fps_integer_stride; endpoint=ceil(end_time*sample_fps)",
                    eviction_granularity="vision_temporal_patch", question_cache="temporary_then_restore_video_cache",
                    initial_text="inside_prompt_budget" if args.budget_scope == "prompt" else "outside_visual_budget",
                    reproduction="SparseMM-selectors-streaming-adaptation", upstream_commit=commit, version=1)
    result_path, protocol_path = out / "1_0.csv", out / "protocol.json"
    manifest_path = out / "run_manifest.json"
    if result_path.exists() and (not protocol_path.exists() or not manifest_path.exists()):
        raise RuntimeError("Resume requires protocol.json and run_manifest.json; use a new output directory")
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise RuntimeError("Output directory has another protocol; choose a new --output-dir")
    rows, completed = load_resume(result_path, data)
    audit = {"source": str(args.anno_path.resolve()), "source_sha256": sha256(args.anno_path),
             "subset_sha256": subset_hash, "videos": len(data), "questions": sum(len(v["conversations"]) for v in data),
             "video_ids": [v["video_id"] for v in data], "completed_videos": completed, "missing_videos": 0}
    evaluator = DEFAULT_HERMES / "eval/eval_multiple_choice.py"
    manifest = {"python": sys.executable, "python_version": platform.python_version(),
                "dependencies": dependency_versions(), "upstream_commit": commit, "code_sha256": code_hashes(),
                "evaluator_sha256": sha256(evaluator), "model_files_sha256": {
                    name: sha256(args.model_path / name) for name in ("config.json", "generation_config.json", "preprocessor_config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors.index.json")
                    if (args.model_path / name).exists()}}
    if manifest_path.exists() and result_path.exists():
        previous = json.loads(manifest_path.read_text())
        for key, value in manifest.items():
            if previous.get(key) != value:
                raise RuntimeError(f"Cannot resume across changed {key}; use a new output directory")
    write_json(protocol_path, protocol)
    write_json(out / "annotations.json", data)
    write_json(out / "dataset_audit.json", audit)
    print(json.dumps({"protocol": protocol, "dataset": audit, "runtime": manifest}, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        return 0
    check_pause(args, out)
    if not (args.model_path / "config.json").exists():
        raise FileNotFoundError(f"No local model config at {args.model_path}")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("DECORD_EOF_RETRY_MAX", "40960")
    for flag in ("PROTOKV_ENABLE", "PROTOKV_FULL_ENABLE", "INFINIPOT_V_PUBLIC"):
        os.environ[flag] = "0"
    try:
        with gpu_lease(out):
            sys.path.insert(0, str(WORKSPACE))
            import torch
            from sparsemm_baselines.adapter import load_model
            from sparsemm_baselines.core import BaselineConfig
            if not torch.cuda.is_available():
                raise RuntimeError("Qwen2.5-VL inference requires CUDA; --dry-run works without CUDA")
            manifest.update(started_utc=datetime.now(timezone.utc).isoformat(), argv=sys.argv,
                            gpu=torch.cuda.get_device_name())
            write_json(manifest_path, manifest)
            (out / "run.status").write_text("running\n")
            if completed < len(data):
                model = load_model(str(args.model_path.resolve()), BaselineConfig(**config), sample_fps=args.sample_fps,
                                   trace_path=out / "cache_trace.jsonl", device="cuda", dtype=args.dtype)
                with torch.inference_mode():
                    for video in data[completed:]:
                        video_rows = analyze_video(model, video, args.sample_fps, args.encode_chunk_size, torch)
                        # Commit only complete videos; interruption cannot create a partial resume.
                        rows.extend(video_rows)
                        write_results(result_path, rows)
            rows, _ = load_resume(result_path, data, require_complete=True)
            (out / "results.csv").write_bytes(result_path.read_bytes())
            with (out / "eval_results.txt").open("w") as log:
                subprocess.run([sys.executable, str(evaluator), "general", "--results_path", str(out / "results.csv")],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            summary = {"videos": len(data), "questions": len(rows), "accuracy_percent": sum(float(r["qa_acc"]) for r in rows) / len(rows),
                       "is_first250_complete": args.anno_path.resolve() == (DEFAULT_HERMES / "data/streamingbench/streamingbench_realtime_half_userpaths.json").resolve()
                           and args.max_questions is None and len(data) == 250 and len(rows) == 1255,
                       "finished_utc": datetime.now(timezone.utc).isoformat(), "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}
            write_json(out / "summary.json", summary)
            (out / "run.status").write_text("complete\n")
            print(json.dumps(summary, indent=2), flush=True)
    except BaseException as exc:
        (out / "run.status").write_text(f"failed {type(exc).__name__}: {exc}\n")
        raise
    return 0


if __name__ == "__main__":
    main()
