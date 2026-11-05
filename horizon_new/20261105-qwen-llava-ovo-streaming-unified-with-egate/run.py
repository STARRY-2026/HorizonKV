#!/usr/bin/env python3
"""Unified 1 fps Qwen / LLaVA baseline launcher with validated dispatch."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from core.dispatch import check_saved_plan, make_plan, save_completed_plan
from core.registry import BENCHMARKS, METHODS, MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', required=True, choices=BENCHMARKS)
    parser.add_argument('--model', required=True, choices=MODELS)
    parser.add_argument('--method', required=True, choices=METHODS)
    parser.add_argument('--gpus', default='0')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--annotation', type=Path)
    parser.add_argument('--head-map', type=Path, help='Frozen per-backbone H2SVC head classification')
    parser.add_argument('--max-videos', type=int)
    parser.add_argument('--cohort', choices=['heldout', 'full'], default='heldout')
    parser.add_argument('--allow-calibration-overlap', action='store_true')
    parser.add_argument('--accept-equivalent-choices', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Run the backend data/configuration checks without inference')
    parser.add_argument('--plan-only', action='store_true', help='Print normalized configuration and command without reading datasets or starting workers')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    try:
        plan = make_plan(args, root)
        check_saved_plan(plan)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.plan_only:
        return 0
    command = plan['command'] + (['--dry-run'] if args.dry_run else [])
    code = subprocess.call(command)
    if code == 0 and not args.dry_run:
        save_completed_plan(plan, root)
    return code


if __name__ == '__main__':
    sys.exit(main())
