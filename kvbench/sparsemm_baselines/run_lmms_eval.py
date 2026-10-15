"""Run the unmodified, pinned SparseMM lmms-eval image-benchmark framework.

Use a separate Python environment matching upstream Transformers 4.46.2.
This wrapper never installs packages, downloads model weights, or queues jobs.
"""

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
from datetime import datetime, timezone

if __package__:
    from .run_streamingbench import ROOT, check_pause, gpu_lease, sha256, verify_upstream, write_json
else:
    from run_streamingbench import ROOT, check_pause, gpu_lease, sha256, verify_upstream, write_json


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("adakv", "pyramidkv", "snapkv", "fullkv"), required=True)
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--model", choices=("qwen2_vl", "llava"), default="qwen2_vl")
    parser.add_argument("--model-args", required=True, help="Official lmms-eval model args, e.g. pretrained=/local/Qwen2-VL-7B-Instruct")
    parser.add_argument("--tasks", default="ocrbench")
    parser.add_argument("--python", default=sys.executable, help="Python executable in the separate native SparseMM environment")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=float, help="Official lmms-eval sample limit; use only for smoke checks")
    parser.add_argument("--resume-experiments", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_command(args, output):
    command = [args.python, "-u", "-m", "lmms_eval", "--model", args.model,
               "--model_args", args.model_args, "--tasks", args.tasks,
               "--batch_size", "1", "--log_samples", "--log_samples_suffix", f"{args.model}_{args.method}_k{args.budget}",
               "--output_path", str(output), "--gen_kwargs", "temperature=0", "--verbosity", "DEBUG"]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    environment = {"METHOD": args.method, "BUDGET": str(args.budget), "RATIO": "0.1", "MASK_RATIO": "0.1",
                   "PYTHONPATH": os.pathsep.join([str(ROOT / "upstream"), str(ROOT / "upstream/lmms-eval")]),
                   "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
    return command, environment


def source_hashes():
    roots = [ROOT / "upstream/sparsemm", ROOT / "upstream/lmms-eval/lmms_eval"]
    files = [Path(__file__), ROOT / "run_streamingbench.py", ROOT / "upstream/pyproject.toml", ROOT / "upstream/lmms-eval/pyproject.toml"]
    for folder in roots:
        files.extend(p for p in folder.rglob("*") if p.suffix in (".py", ".yaml", ".yml"))
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(files)}


def preflight(python, environment, method):
    code = '''import importlib.metadata as m, json, pathlib
versions = {name:m.version(name) for name in ('torch','transformers','flash-attn','numpy','lmms_eval')}
assert versions['transformers'] == '4.46.2', 'Native SparseMM requires transformers==4.46.2; use a separate environment'
assert versions['torch'].split('+')[0] == '2.5.1', 'Native SparseMM reproduction requires upstream torch==2.5.1'
assert versions['flash-attn'] == '2.4.1', 'Native SparseMM reproduction requires upstream flash-attn==2.4.1'
import torch, transformers, flash_attn, lmms_eval
from sparsemm.monkeypatch import replace_qwen, replace_llama, replace_mistral
assert torch.cuda.is_available(), 'CUDA unavailable'
'''
    if method == "adakv":
        code += "import tiny_api_cuda\n"
    code += "print(json.dumps(dict(versions=versions, lmms_eval_file=str(pathlib.Path(lmms_eval.__file__).resolve()), gpu=torch.cuda.get_device_name())))"
    return json.loads(subprocess.check_output([python, "-c", code], cwd=ROOT / "upstream", env=environment, text=True))


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.budget <= 32:
        parser.error("Upstream selectors use window_size=32 and require budget > 32")
    if args.limit is not None and (not math.isfinite(args.limit) or args.limit <= 0):
        parser.error("--limit must be finite and positive")
    if not args.tasks.strip() or not args.model_args.strip():
        parser.error("Model args and tasks must be nonempty")
    commit = verify_upstream()
    out = (args.output_dir or ROOT / f"results/native_{args.model}_{args.tasks.replace(',', '_')}_{args.method}_k{args.budget}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    command, overrides = build_command(args, out)
    protocol = {"reproduction": "unmodified-SparseMM-lmms-eval", "upstream_commit": commit,
                "method": args.method, "budget": args.budget, "budget_scope": "entire_prompt_including_window32",
                "model": args.model, "model_args": args.model_args, "tasks": args.tasks, "limit": args.limit,
                "batch_size": 1, "command": command, "environment": overrides, "source_sha256": source_hashes()}
    protocol_file = out / "protocol.json"
    if protocol_file.exists() and json.loads(protocol_file.read_text()) != protocol:
        raise RuntimeError("Native output directory belongs to another protocol/source version; choose a new --output-dir")
    write_json(protocol_file, protocol)
    print(json.dumps({"protocol": protocol, "shell_command": shlex.join(command)}, indent=2), flush=True)
    if args.dry_run:
        return 0
    check_pause(args, out)
    environment = dict(os.environ, **overrides)
    try:
        with gpu_lease(out):
            # The GPU index is selected by gpu_lease after environment creation.
            environment["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
            runtime = preflight(args.python, environment, args.method)
            expected_lmms = ROOT / "upstream/lmms-eval/lmms_eval/__init__.py"
            if Path(runtime["lmms_eval_file"]).resolve() != expected_lmms.resolve():
                raise RuntimeError("Selected Python did not import the bundled official lmms-eval")
            write_json(out / "run_manifest.json", dict(runtime, started_utc=datetime.now(timezone.utc).isoformat(), protocol_sha256=sha256(protocol_file)))
            (out / "run.status").write_text("running\n")
            with (out / "run.log").open("a") as log:
                subprocess.run(command, cwd=ROOT / "upstream", env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            (out / "run.status").write_text("complete\n")
    except BaseException as exc:
        (out / "run.status").write_text(f"failed {type(exc).__name__}: {exc}\n")
        raise
    return 0


if __name__ == "__main__":
    main()
