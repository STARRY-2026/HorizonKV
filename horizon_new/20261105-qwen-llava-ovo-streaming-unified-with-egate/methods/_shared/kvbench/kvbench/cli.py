"""Command line interface shared with the public RunConfig API."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from .api import RunConfig, audit, run, summarize
from .registry import METHODS


def build_parser():
    parser = argparse.ArgumentParser(prog="kvbench", description="Unified Qwen2.5-VL KV methods; only bounded validation is enabled.")
    commands = parser.add_subparsers(dest="command", required=True)
    methods = commands.add_parser("methods", help="List the method registry; no model is loaded")
    methods.add_argument("--json", action="store_true")
    execute = commands.add_parser("run", help="Run and audit methods sequentially; default: five non-ProtoKV methods")
    selection = execute.add_mutually_exclusive_group()
    selection.add_argument("--method", choices=tuple(METHODS), help="Select one method")
    selection.add_argument("--methods", nargs="+", help="Select several methods, separated by spaces or commas")
    execute.add_argument("--anno-path", type=Path, help="Native MCQA annotations (default: bundled OVO smoke subset)")
    execute.add_argument("--model-path", type=Path)
    execute.add_argument("--output-dir", type=Path, help="New collection directory; existing paths are rejected")
    execute.add_argument("--budget", type=int, default=1000, help="Visual KV budget; full control always uses 1,000,000")
    execute.add_argument("--max-videos", type=int, default=3)
    execute.add_argument("--max-questions", type=int, default=1, help="Per-video limit; at most 8 selected questions in total")
    execute.add_argument("--sample-fps", type=float, default=0.5)
    execute.add_argument("--encode-chunk-size", type=int, default=16)
    execute.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    execute.add_argument("--lukv-profile-path", type=Path)
    execute.add_argument("--sparsemm-score-path", type=Path)
    execute.add_argument("--config-json", type=Path, help="Native StreamingConfig JSON; fixed method/budget/scope must agree")
    execute.add_argument("--small-validation", action="store_true", default=True, help="Always enabled: <=3 videos, <=8 questions, <=120 seconds")
    execute.add_argument("--dry-run", action="store_true", help="Prepare native protocols and commands without GPU inference or audits")
    execute.add_argument("--allow-calibration-transfer", action="store_true", help="Explicit LU audit opt-in for external datasets, with calibration/video hash checks")
    execute.add_argument("--min-compression-events-per-video", type=int, default=0)
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
            records = [{"name": method.name, "default": method.default,
                        "runtime_method": method.runtime_method, "description": method.description,
                        "fixed_budget": method.fixed_budget} for method in METHODS.values()]
            if args.json:
                print(json.dumps(records, indent=2, ensure_ascii=False))
            else:
                for method in records:
                    print(f"{method['name']:12} {'default' if method['default'] else 'explicit':8} {method['description']}")
            return 0
        if args.command == "run":
            options = vars(args).copy()
            options.pop("command")
            options.pop("small_validation")
            if options["methods"] is not None:
                options["methods"] = tuple(part.strip() for item in options["methods"] for part in item.split(","))
            for key in ("anno_path", "model_path", "lukv_profile_path", "sparsemm_score_path"):
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
