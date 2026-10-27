"""Short, isolated single-versus-two-process GPU benchmark; never starts a full run."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def worker(args):
    import torch
    from sparsemm_native.run_ovobench import (DP_BUDGET, MODEL, queries, run_query,
                                              grouped_batches, run_grouped_video)
    from sparsemm_native.streaming import load_model
    from sparsemm_protokv.config import StreamingConfig

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    selected = {q["query_id"]: q for q in queries()}
    work = [selected[qid] for qid in args.query_ids.split(",")]
    config = StreamingConfig(method="sparsemm", budget=500, budget_scope="visual",
                             window_size=32, kernel_size=5, pooling="avgpool",
                             gqa_reduce="mean", score_precision="float32",
                             sparsemm_budget_path=str(DP_BUDGET) if args.arm == "dp" else "")
    started = time.monotonic()
    bridge = load_model(MODEL, config, sample_fps=.5, device="cuda", dtype="bfloat16",
                        video_max_pixels=151200)
    torch.cuda.synchronize()
    load_seconds = time.monotonic() - started
    counts = {"compressed_layer_updates": 0, "layer_updates": 0}
    original_forward = bridge.forward_chunk

    def measured_forward(*a, **kw):
        before = len(bridge.validation_checks["selector_checks"])
        result = original_forward(*a, **kw)
        new = bridge.validation_checks["selector_checks"][before:]
        counts["compressed_layer_updates"] += sum(bool(row["compressed"]) for row in new)
        counts["layer_updates"] += len(new)
        return result

    bridge.forward_chunk = measured_forward
    # Warm up model/vision/selection kernels on an actual selected query, outside timing.
    warm_started = time.monotonic()
    run_query(bridge, work[0], .5, 16, 16)
    torch.cuda.synchronize()
    warm_seconds = time.monotonic() - warm_started
    write_json(output / "ready.json", {"pid": os.getpid(), "load_seconds": load_seconds,
                                       "warmup_seconds": warm_seconds})
    barrier = Path(args.barrier)
    deadline = time.monotonic() + 300
    while not barrier.exists():
        if time.monotonic() > deadline:
            raise TimeoutError("Benchmark barrier timed out")
        time.sleep(.05)
    counts.update(compressed_layer_updates=0, layer_updates=0)
    torch.cuda.reset_peak_memory_stats()
    benchmark_started = time.monotonic()
    rows = []
    for group in grouped_batches(work, len(work)):
        answers = run_grouped_video(bridge, group, .5, 16, 16)
        torch.cuda.synchronize()
        for query in group:
            response, frames, seconds, encoded_frames = answers[query["query_id"]]
            row = {"query_id": query["query_id"], "task": query["task"],
                   "response": response, "frames": frames, "encoded_frames": encoded_frames,
                   "seconds": seconds}
            rows.append(row)
            print(json.dumps(row), flush=True)
    benchmark_ended = time.monotonic()
    write_json(output / "result.json", {
        "arm": args.arm, "pid": os.getpid(), "load_seconds": load_seconds,
        "warmup_seconds": warm_seconds, "benchmark_started": benchmark_started,
        "benchmark_ended": benchmark_ended,
        "steady_seconds": benchmark_ended - benchmark_started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        **counts, "rows": rows})


class GPUMonitor:
    def __init__(self, output):
        self.output, self.samples, self.stop = Path(output), [], threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop.is_set():
            try:
                gpu = subprocess.check_output([
                    "nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits"], text=True).strip()
                apps = subprocess.check_output([
                    "nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                    "--format=csv,noheader,nounits"], text=True).strip()
                mem, utilization = (int(x.strip()) for x in gpu.split(","))
                processes = {str(int(parts[0])): int(parts[1])
                             for line in apps.splitlines() if line.strip()
                             for parts in [line.split(",")]}
                self.samples.append({"monotonic": time.monotonic(), "memory_mib": mem,
                                     "utilization_percent": utilization, "processes_mib": processes})
            except (subprocess.CalledProcessError, ValueError):
                pass
            self.stop.wait(.4)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(5)
        write_json(self.output, self.samples)


def wait_ready(processes, paths):
    deadline = time.monotonic() + 300
    while not all(path.exists() for path in paths):
        if any(proc.poll() is not None for proc in processes):
            raise RuntimeError("Benchmark worker exited before ready; see worker log")
        if time.monotonic() > deadline:
            raise TimeoutError("Benchmark model load/warmup timed out")
        time.sleep(.2)


def main(args):
    from sparsemm_native.supervise_ovobench_dual import worker_environment
    from sparsemm_native.run_ovobench import queries, sha256

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    qids = args.query_ids.split(",")
    query_map = {q["query_id"]: q for q in queries()}
    write_json(output / "manifest.json", {
        "model": "Qwen2.5-VL-7B-Instruct", "dtype": "bfloat16", "budget": 500,
        "fps": .5, "video_max_pixels": 151200, "chunk_size": 16,
        "max_new_tokens": 16, "prompt_scoring": "official OVO-Bench",
        "query_manifest": [query_map[qid] for qid in qids],
        "script_sha256": sha256(__file__), "warmup": "first selected query, excluded",
        "timing": "monotonic after model load and warmup; same-video queries share prefixes",
        "grouped": True,
        "scope": "short benchmark only; full experiment remains paused"})
    processes, logs = [], []

    def launch(phase, arm, barrier):
        target = output / phase / arm
        target.mkdir(parents=True)
        log = (target / "worker.log").open("w")
        logs.append(log)
        process = subprocess.Popen([
            sys.executable, "-u", "-m", "sparsemm_native.benchmark_dual_process", "--worker",
            "--output", str(target), "--arm", arm, "--barrier", str(barrier),
            "--query-ids", args.query_ids], cwd=ROOT, env=worker_environment(),
            stdout=log, stderr=subprocess.STDOUT)
        processes.append(process)
        return process, target

    try:
        for arm in ("native", "dp"):
            barrier = output / "sequential" / f"start_{arm}"
            with GPUMonitor(output / f"gpu_sequential_{arm}.json"):
                proc, target = launch("sequential", arm, barrier)
                wait_ready([proc], [target / "ready.json"])
                barrier.touch()
                if proc.wait(timeout=300):
                    raise RuntimeError(f"Sequential {arm} failed")
            print(f"Completed sequential {arm}", flush=True)
        barrier = output / "parallel" / "start"
        with GPUMonitor(output / "gpu_parallel.json"):
            launched = [launch("parallel", arm, barrier) for arm in ("native", "dp")]
            wait_ready([p for p, _ in launched], [t / "ready.json" for _, t in launched])
            barrier.touch()
            for proc, _ in launched:
                if proc.wait(timeout=300):
                    raise RuntimeError("Parallel worker failed")
        single = [json.loads((output / "sequential" / arm / "result.json").read_text())
                  for arm in ("native", "dp")]
        parallel = [json.loads((output / "parallel" / arm / "result.json").read_text())
                    for arm in ("native", "dp")]
        sequential_seconds = sum(item["steady_seconds"] for item in single)
        parallel_seconds = (max(item["benchmark_ended"] for item in parallel)
                            - min(item["benchmark_started"] for item in parallel))
        samples = json.loads((output / "gpu_parallel.json").read_text())
        measured_samples = [sample for sample in samples
                            if min(item["benchmark_started"] for item in parallel)
                            <= sample["monotonic"] <= max(item["benchmark_ended"] for item in parallel)]
        summary = {
            "sequential_steady_seconds": sequential_seconds,
            "parallel_steady_makespan_seconds": parallel_seconds,
            "speedup": sequential_seconds / parallel_seconds,
            "questions_per_phase": 2 * len(qids),
            "sequential_questions_per_minute": 120 * len(qids) / sequential_seconds,
            "parallel_questions_per_minute": 120 * len(qids) / parallel_seconds,
            "parallel_gpu_peak_mib_all_phases": max(s["memory_mib"] for s in samples),
            "parallel_gpu_peak_mib_steady": max(s["memory_mib"] for s in measured_samples),
            "responses_exact_equal": all(
                [(r["query_id"], r["response"]) for r in a["rows"]]
                == [(r["query_id"], r["response"]) for r in b["rows"]]
                for a, b in zip(single, parallel)),
            "sequential": single, "parallel": parallel,
            "oom": False,
            "limitation": "Four-query/two-video short/mid grouped benchmark, not full-dataset throughput estimate."}
        write_json(output / "summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
        for proc in processes:
            if proc.poll() is None:
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        for log in logs:
            log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--query-ids", default="9,10,104,105")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--arm", choices=("native", "dp"))
    parser.add_argument("--barrier")
    arguments = parser.parse_args()
    worker(arguments) if arguments.worker else main(arguments)
