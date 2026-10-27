"""Run the pinned SimpleStream one/four path with 500 visual KV slots/head.

Uses the released pre-cut OVO clips and the existing official worker/scorer.
This arm is the recent-only budget baseline; historical KV is never admitted.
"""

import argparse
import sys
from pathlib import Path

from sparsemm_native import supervise_simplestream_upstream as base
from sparsemm_native.simplestream_recent_share_budget500 import install


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clips", type=Path, default=base.DATA / "chunked_videos")
    args = parser.parse_args()
    sys.path.insert(0, str(base.UPSTREAM))
    install(args.output)
    worker_args = argparse.Namespace(output=str(args.output), annotation=str(args.annotation),
                                     clips=str(args.clips), check_pins=False)
    base.worker(worker_args)


if __name__ == "__main__":
    main()
