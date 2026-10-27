"""Run fps=1 SimpleStream-frame OVO with independent, mild-budget history KV."""

import argparse
import random
import sys
from pathlib import Path

from sparsemm_native import supervise_simplestream_upstream as base
from sparsemm_native.simplestream_recent_share_pseudo_history_mild500 import install


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--clips', type=Path, default=base.DATA / 'chunked_videos')
    parser.add_argument('--recent-only-control', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(base.UPSTREAM))
    install(args.output, recent_only_control=args.recent_only_control)
    original_shuffle = random.shuffle
    def group_same_video(items):
        items.sort(key=lambda item: (item['video'], item['id']))
    random.shuffle = group_same_video
    try:
        base.worker(argparse.Namespace(output=str(args.output), annotation=str(args.annotation),
                                       clips=str(args.clips), check_pins=False))
    finally:
        random.shuffle = original_shuffle


if __name__ == '__main__':
    main()
