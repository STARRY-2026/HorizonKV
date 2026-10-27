"""Command line interface shared with the public RunConfig API."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from .api import RunConfig, audit, run, summarize
from .ovo import OvoRunConfig, audit_ovo, run_ovo
from .registry import METHODS, OVO_METHODS


def build_parser():
    parser = argparse.ArgumentParser(prog="kvbench", description="Unified Qwen2.5-VL/LLaVA-OneVision KV methods; only bounded validation is enabled.")
    commands = parser.add_subparsers(dest="command", required=True)
    methods = commands.add_parser("methods", help="List the method registry; no model is loaded")
    methods.add_argument("--json", action="store_true")
    execute = commands.add_parser("run", help="Run and audit methods sequentially; default: five non-ProtoKV methods")
    selection = execute.add_mutually_exclusive_group()
    selection.add_argument("--method", choices=tuple(METHODS), help="Select one method")
    selection.add_argument("--methods", nargs="+", help="Select several methods, separated by spaces or commas")
    execute.add_argument("--anno-path", type=Path, help="Native MCQA annotations (default: bundled OVO smoke subset)")
    execute.add_argument("--model-path", type=Path, help="Local Qwen2.5-VL, Qwen3-VL or LLaVA-OneVision model; backend detected from config.json")
    execute.add_argument("--output-dir", type=Path, help="New collection directory; existing paths are rejected")
    execute.add_argument("--budget", type=int, default=1000, help="Visual KV budget; full control always uses 1,000,000")
    execute.add_argument("--max-videos", type=int, default=3)
    execute.add_argument("--max-questions", type=int, default=1, help="Per-video limit; at most 8 selected questions in total")
    execute.add_argument("--sample-fps", type=float, default=0.5)
    execute.add_argument("--encode-chunk-size", type=int, default=16)
    execute.add_argument("--video-chunk-mode", choices=("auto", "temporal_patch", "hermes_chunk"),
                         default="auto", help="auto uses HERMES chunk cadence when HERMES is selected")
    execute.add_argument("--video-max-pixels", type=int, help="Qwen2.5 ProtoKV/SparseMM/DP/DP-dual or Qwen3 HERMES single-method runs; changes visual inputs")
    execute.add_argument("--fixed-frame-size", type=int,
                         help="Shared Qwen2.5-VL square canvas for every method; 896 gives 1024 visual tokens per temporal unit")
    execute.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    execute.add_argument("--lukv-profile-path", type=Path)
    execute.add_argument("--sparsemm-score-path", type=Path)
    execute.add_argument("--sparsemm-budget-path", type=Path, help="External layer,kv_head,capacity_n CSV for SparseMM")
    execute.add_argument("--dp-budget-path", type=Path, help="DP K=500 capacity_n CSV; default is bundled calibrated table")
    execute.add_argument("--dp-dual-budget-path", type=Path, help="DP K=500 recent/history two-bank CSV")
    execute.add_argument("--config-json", type=Path, help="Native StreamingConfig JSON; fixed method/budget/scope must agree")
    execute.add_argument("--small-validation", action="store_true", default=True, help="Always enabled: <=3 videos, <=8 questions, <=120 seconds")
    execute.add_argument("--dry-run", action="store_true", help="Prepare native protocols and commands without GPU inference or audits")
    execute.add_argument("--allow-calibration-transfer", action="store_true", help="Explicit LU audit opt-in for external datasets, with calibration/video hash checks")
    execute.add_argument("--min-compression-events-per-video", type=int, default=0)
    ovo = commands.add_parser("run-ovo", help="Run a bounded official OVO nine-task method; explicit IDs required")
    ovo.add_argument("--method", required=True, choices=tuple(OVO_METHODS))
    ovo.add_argument("--ids", required=True, help="Comma-separated official OVO query IDs, at most 8 queries / 3 source videos")
    ovo.add_argument("--output-dir", type=Path, required=True, help="New output directory")
    ovo.add_argument("--dry-run", action="store_true")
    ovo_audit = commands.add_parser("audit-ovo", help="Audit a completed bounded OVO run")
    ovo_audit.add_argument("--output-dir", type=Path, required=True)
    inspect = commands.add_parser("audit", help="Audit one native run or a collection without inference")
    inspect.add_argument("--output-dir", type=Path, required=True)
    inspect.add_argument("--allow-calibration-transfer", action="store_true")
    inspect.add_argument("--min-compression-events-per-video", type=int, default=0)
    summary = commands.add_parser("summarize", help="Write summary.json and report.md from existing runs")
    summary.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "methods":
            records = [{"name": method.name, "benchmark": "streamingbench_mcqa", "default": method.default,
                        "runtime_method": method.runtime_method, "description": method.description,
                        "fixed_budget": method.fixed_budget} for method in METHODS.values()]
            records += [{"name": name, "benchmark": "ovobench_nine_task", "default": False,
                         "runtime_method": name, "description": description, "fixed_budget": None}
                        for name, description in OVO_METHODS.items()]
            if args.json:
                print(json.dumps(records, indent=2, ensure_ascii=False))
            else:
                for method in records:
                    print(f"{method['name']:12} {'default' if method['default'] else 'explicit':8} {method['description']}")
            return 0
        if args.command == "run-ovo":
            result = run_ovo(OvoRunConfig(output_dir=args.output_dir,
                        ids=tuple(part.strip() for part in args.ids.split(",")),
                        method=args.method, dry_run=args.dry_run))
        elif args.command == "audit-ovo":
            result = audit_ovo(args.output_dir)
        elif args.command == "run":
            options = vars(args).copy()
            options.pop("command")
            options.pop("small_validation")
            if options["methods"] is not None:
                options["methods"] = tuple(part.strip() for item in options["methods"] for part in item.split(","))
            for key in ("anno_path", "model_path", "lukv_profile_path", "sparsemm_score_path", "sparsemm_budget_path", "dp_budget_path", "dp_dual_budget_path"):
                if options[key] is None:
                    options.pop(key)
            result = run(RunConfig(**options))
        elif args.command == "audit":
            result = audit(args.output_dir, allow_calibration_transfer=args.allow_calibration_transfer,
                           min_compression_events_per_video=args.min_compression_events_per_video)
        else:
            result = summarize(args.output_dir)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (ValueError, TypeError, FileExistsError, FileNotFoundError, KeyError) as error:
        print(f"kvbench: {error}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as error:
        print(f"kvbench: native subprocess failed with exit code {error.returncode}; inspect the method's kvbench_controller.log or kvbench_audit.log", file=sys.stderr)
        return 1
    except (RuntimeError, OSError) as error:
        print(f"kvbench: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
