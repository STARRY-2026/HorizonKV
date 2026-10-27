"""Qwen3-VL LU calibration on evaluation-disjoint videos, with checkpoints."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from sparsemm_baselines.core import BaselineConfig
from sparsemm_baselines.qwen3_adapter import load_model
from sparsemm_baselines.run_streamingbench import analyze_video, gpu_lease, write_json
from sparsemm_lukv.calibration_data import DEFAULT_EXCLUDE, DEFAULT_SOURCE, select_calibration
from sparsemm_lukv.profile import allocate_profile, build_profile, validate_profile
from sparsemm_lukv.qwen3_observer import Qwen3LUObserver


class _CopiedBatch:
    def __init__(self, pixels):
        self.pixels = pixels

    def asnumpy(self):
        return self.pixels


class BatchScopedVideoReader:
    """Release Decord's background decoder before each GPU vision forward."""

    def __init__(self, path, num_threads=1):
        from decord import VideoReader
        self.path, self.num_threads = path, num_threads
        reader = VideoReader(path, num_threads=num_threads)
        try:
            self.count, self.fps = len(reader), reader.get_avg_fps()
        finally:
            reader.__del__()
            reader._handle = None

    def __len__(self):
        return self.count

    def get_avg_fps(self):
        return self.fps

    def get_batch(self, indices):
        from decord import VideoReader
        reader = VideoReader(self.path, num_threads=self.num_threads)
        try:
            pixels = reader.get_batch(indices).asnumpy()
        finally:
            reader.__del__()
            reader._handle = None
        return _CopiedBatch(pixels)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    out = args.output_dir.resolve()
    if out.exists() and not args.resume:
        raise FileExistsError(out)
    selected, selection = select_calibration(args.source, args.exclude, args.num_videos, args.max_time)
    if not selected:
        raise ValueError("No evaluation-disjoint calibration video")
    out.mkdir(parents=True, exist_ok=True)
    model_files = {name: digest(args.model_path / name) for name in
                   ("config.json", "model.safetensors.index.json", "preprocessor_config.json")
                   if (args.model_path / name).is_file()}
    manifest = {"status": "running", "method": "lukv", "model": str(args.model_path.resolve()),
                "code_sha256": {"collector": digest(__file__),
                                "observer": digest(Path(__file__).resolve().parents[1] / "sparsemm_lukv/qwen3_observer.py"),
                                "adapter": digest(Path(__file__).resolve().parents[1] / "sparsemm_baselines/qwen3_adapter.py"),
                                "hermes_ranking": digest(Path(__file__).resolve().parents[1] / "sparsemm_lukv/qwen3_hermes_ranking.py"),
                                "hermes_reference": digest(Path(__file__).resolve().parents[1] / "sparsemm_baselines/qwen3_hermes.py"),
                                "profile_solver": digest(Path(__file__).resolve().parents[1] / "sparsemm_lukv/profile.py")},
                "model_files_sha256": model_files, "started_utc": datetime.now(timezone.utc).isoformat(),
                "selection": selection, "video_max_pixels": args.video_max_pixels,
                "sample_fps": 0.5, "encode_chunk_size": 16,
                "input_protocol": "Qwen3 whole_chunk_video, singleton_as_image, no timestamp text markers",
                "calibration_status": "formal_10_video_qwen3_adaptation" if args.num_videos == 10 else "bounded_pilot_not_full_method_calibration",
                "ranking_mode": args.ranking_mode,
                "full_benchmarks_paused": True, "observations": []}
    observations = []
    completed = {}
    if args.resume:
        saved = json.loads((out / "manifest.json").read_text())
        for key in ("method", "model", "code_sha256", "model_files_sha256", "selection",
                    "video_max_pixels", "sample_fps", "encode_chunk_size", "input_protocol",
                    "ranking_mode"):
            if saved.get(key) != manifest[key]:
                raise ValueError(f"LU resume protocol mismatch: {key}")
        if json.loads((out / "annotations.json").read_text()) != selected:
            raise ValueError("LU resume annotations mismatch")
        manifest = saved
        for record in manifest["observations"]:
            key = (str(record["video_id"]), int(record["source_question_index"]))
            source = Path(record["path"])
            if key in completed or not source.is_file() or digest(source) != record["sha256"]:
                raise ValueError(f"Invalid LU checkpoint {key}")
            metadata_path = source.with_suffix(".json")
            metadata = json.loads(metadata_path.read_text())
            if str(metadata["video_id"]) != key[0] or int(metadata["source_question_index"]) != key[1]:
                raise ValueError(f"LU checkpoint metadata mismatch {key}")
            with np.load(source, allow_pickle=False) as arrays:
                ranking, utility = arrays["ranking"].copy(), arrays["utility_projected"].copy()
            if ranking.shape[0] != 36 * 8 or utility.shape[0] != 36 * 8 or not np.isfinite(utility).all():
                raise ValueError(f"Invalid LU checkpoint array {key}")
            observations.append({"video_id": record["video_id"], "ranking": ranking, "utility": utility})
            completed[key] = metadata["pred_answer"]
        if manifest["status"].startswith("complete"):
            if len(completed) != selection["selected_question_count"] or not (out / "profile.json").is_file():
                raise ValueError("Completed LU checkpoint is missing results")
            print(json.dumps({"status": manifest["status"], "observations": len(completed)}), flush=True)
            return
        manifest.update(status="running", resumed_utc=datetime.now(timezone.utc).isoformat())
        manifest.pop("error", None)
    else:
        write_json(out / "annotations.json", selected)
    write_json(out / "manifest.json", manifest)
    start = time.perf_counter()
    try:
        os.environ.setdefault("DECORD_EOF_RETRY_MAX", "40960")
        with gpu_lease(out):
            bridge = load_model(args.model_path,
                                config=BaselineConfig(method="uniform", budget=1_000_000,
                                                      budget_scope="visual"),
                                sample_fps=0.5, trace_path=out / "cache_trace.jsonl",
                                video_max_pixels=args.video_max_pixels)
            observer = Qwen3LUObserver(bridge.num_layers, bridge.num_kv_heads,
                                      bridge.num_query_heads, ranking_mode=args.ranking_mode)
            bridge.calibration_observer = observer
            ordinary_answer = bridge.question_answering
            current_video = [None]

            def observed_answer(prompt, *positional, **keyword):
                video = current_video[0]
                question = video["conversations"][bridge._active_question_index]
                observation_key = (str(video["video_id"]), int(question["source_question_index"]))
                if observation_key in completed:
                    return completed[observation_key]
                observer.begin(bridge)
                answer = ordinary_answer(prompt, *positional, **keyword)
                record = observer.end(bridge)
                key = f"v{video['video_id']}_q{question['source_question_index']}"
                arrays = {name: record[name].numpy() for name in
                          ("ranking", "score", "utility_attention", "utility_projected")}
                if any(not np.isfinite(a).all() for a in arrays.values()):
                    raise FloatingPointError("Non-finite Qwen3 LU calibration array")
                np.savez_compressed(out / f"{key}.npz", **arrays)
                metadata = {**record["metadata"], "video_id": video["video_id"],
                            "source_question_index": question["source_question_index"],
                            "pred_answer": answer}
                write_json(out / f"{key}.json", metadata)
                manifest["observations"].append({"video_id": video["video_id"],
                                                  "source_question_index": question["source_question_index"],
                                                  "path": str(out / f"{key}.npz"),
                                                  "sha256": digest(out / f"{key}.npz"),
                                                  "visual_tokens": record["metadata"]["visual_tokens"]})
                observations.append({"video_id": video["video_id"],
                                     "ranking": arrays["ranking"],
                                     "utility": arrays["utility_projected"]})
                completed[observation_key] = answer
                write_json(out / "manifest.json", manifest)
                print(f"LU_QWEN3_OBSERVATION {key} tokens={record['metadata']['visual_tokens']}", flush=True)
                return answer

            bridge.question_answering = observed_answer
            for video in selected:
                if all((str(video["video_id"]), int(q["source_question_index"])) in completed
                       for q in video["conversations"]):
                    continue
                current_video[0] = video
                analyze_video(bridge, video, 0.5, 16, torch,
                              reader_factory=BatchScopedVideoReader)
            if len(observations) != selection["selected_question_count"]:
                raise RuntimeError("Incomplete LU calibration observations")
            profile = build_profile(observations, bridge.num_layers, bridge.num_kv_heads)
            profile["calibration"] = {"model_path": str(args.model_path.resolve()),
                                      "model_files_sha256": model_files,
                                      "manifest_path": str(out / "manifest.json"),
                                      "source_sha256": selection["source_hashes"],
                                      "selector": args.ranking_mode, "utility_mode": "projected",
                                      "runtime_score_formula": ("hermes_pseudo" if args.ranking_mode == "hermes_pseudo"
                                                                else "attention"),
                                      "input_protocol": manifest["input_protocol"]}
            validate_profile(profile, bridge.num_layers, bridge.num_kv_heads, 32)
            write_json(out / "profile.json", profile)
            reference_t = int(np.median([x["visual_tokens"] for x in manifest["observations"]]))
            budgets, allocation = allocate_profile(profile, reference_t, 500, 32)
            import csv
            with (out / f"budget_k500_at_T{reference_t}.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("layer", "kv_head", "budget"))
                for layer in range(bridge.num_layers):
                    for head in range(bridge.num_kv_heads):
                        writer.writerow((layer, head, int(budgets[layer, head])))
            write_json(out / "reference_allocation.json", allocation)
            manifest.update(status="complete_formal" if args.num_videos == 10 else "complete_pilot", finished_utc=datetime.now(timezone.utc).isoformat(),
                            profile_sha256=digest(out / "profile.json"),
                            reference_T=reference_t,
                            reference_total_slots=int(budgets.sum()),
                            seconds=round(time.perf_counter() - start, 3))
            write_json(out / "manifest.json", manifest)
            print(json.dumps({"status": manifest["status"], "observations": len(observations),
                              "reference_T": reference_t, "slots": int(budgets.sum()),
                              "seconds": manifest["seconds"]}), flush=True)
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(out / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--exclude", type=Path, default=DEFAULT_EXCLUDE)
    parser.add_argument("--num-videos", type=int, default=1)
    parser.add_argument("--max-time", type=float, default=8)
    parser.add_argument("--video-max-pixels", type=int, default=1_200_000)
    parser.add_argument("--ranking-mode", choices=("pure_attention", "hermes_pseudo"),
                        default="hermes_pseudo")
    parser.add_argument("--resume", action="store_true")
    main(parser.parse_args())
