#!/usr/bin/env python3
"""Run the matched, isolated DP/Hermes diagnostic (one local GPU, offline)."""
import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/myobservations/dp_token_frame_study_20261010"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("dp", "hermes", "both"), default="both")
    args = parser.parse_args()
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               CUDA_VISIBLE_DEVICES="0", DECORD_EOF_RETRY_MAX="40960")
    libs = glob.glob(str(Path(sys.prefix) / "lib/python*/site-packages/nvidia/cu[0-9]*/lib"))
    env["LD_LIBRARY_PATH"] = os.pathsep.join(libs + [env.get("LD_LIBRARY_PATH", "")])
    policies = ("dp", "hermes") if args.policy == "both" else (args.policy,)
    for policy in policies:
        dest = OUT / policy
        dest.mkdir(parents=True, exist_ok=True)
        common = [sys.executable, "-u", "scripts/trace_dp_token_frames.py",
                  "--model", "qwen2.5_vl_7b", "--anno_path", str(OUT / "annotation.json"),
                  "--save_dir", str(dest / "predictions"), "--sample_fps", "0.5",
                  "--kv_size", "1000", "--streaming", "true", "--compress_mode", "hermes",
                  "--encode_chunk_size", "16", "--kv_head_budget_scheme", "sparsemm",
                  "--kv_head_budget_sparsemm_ratio", "0.2",
                  "--kv_head_budget_sparsemm_window_size", "32",
                  "--kv_head_budget_union_cap_ratio", "1.0",
                  "--kv_head_ragged_prefill", "true", "--debug", "false",
                  "--ragged_readout_trace_last_n", "0",
                  "--ragged_token_trace_path", str(dest / "tokens.jsonl"),
                  "--ragged_readout_trace_path", str(dest / "readout.jsonl"),
                  "--source_manifest_path", str(dest / "source_manifest.jsonl")]
        profile = ROOT / "results/myobservations/k1000_dp_cachecomp_score_20260924"
        scores = (profile / "dp_k1000_exact_budget_scores.csv" if policy == "dp" else
                  ROOT / "results/observations/memory_core_budget_scores/uniform.csv")
        common += ["--kv_head_budget_scores", str(scores)]
        if policy == "dp":
            common += ["--kv_head_recency_scores", str(profile / "recent_ratio_profile.csv"),
                       "--kv_head_recency_metric", "recent_ratio", "--kv_head_recency_delta", "0.1",
                       "--kv_head_recency_tau", "1.0", "--kv_head_recency_alpha_min", "0.0",
                       "--kv_head_recency_alpha_max", "1.0", "--kv_head_recency_k_delta", "2.0",
                       "--kv_head_recency_k_min", "0.0", "--kv_head_recency_k_max", "20.0"]
        sources = [ROOT / "scripts/trace_dp_token_frames.py", OUT / "annotation.json", scores,
                   ROOT / "head_analysis/hermes_kv_head_ragged_prefill.py",
                   ROOT / "head_analysis/hermes_kv_head_ragged.py",
                   ROOT / "video_qa/base.py", ROOT / "video_qa/hermes_vqa.py"]
        if policy == "dp":
            sources.append(profile / "recent_ratio_profile.csv")
        (dest / "command.json").write_text(json.dumps({"argv": common,
            "sources_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}, indent=2) + "\n")
        print(f"Running {policy}; log: {dest / 'run.log'}", flush=True)
        with (dest / "run.log").open("x") as log:
            result = subprocess.run(common, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise SystemExit(f"{policy} failed: {result.returncode}; inspect {dest / 'run.log'}")
        if len(list((dest / "predictions").glob("*.csv"))) == 0:
            raise SystemExit(f"{policy}: no predictions exported")
        print(f"Completed {policy}", flush=True)


if __name__ == "__main__":
    main()
