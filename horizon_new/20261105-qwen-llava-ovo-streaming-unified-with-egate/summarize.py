#!/usr/bin/env python3
"""Dispatch summaries using recorded provenance, not inferred question counts."""
import argparse
import subprocess
import sys
from pathlib import Path
from core.dispatch import summary_backend
from core.registry import BACKENDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--backend', choices=BACKENDS, help='Explicit backend for legacy runs without launcher_plan.json')
    args = parser.parse_args()
    try:
        backend = summary_backend(args.output_dir, args.backend)
    except ValueError as error:
        parser.error(str(error))
    return subprocess.call([sys.executable, str(Path(__file__).resolve().parent/backend/'summarize.py'), str(args.output_dir)])


if __name__ == '__main__':
    sys.exit(main())
