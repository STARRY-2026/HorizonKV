"""Run the one-frame 250 recent + 250 history OVO pilot at mean K=500."""

import argparse
import sys
from pathlib import Path

from sparsemm_native import supervise_simplestream_upstream as base
from sparsemm_native.simplestream_recent_share_pseudo_history250 import install


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--clips', type=Path, default=base.DATA / 'chunked_videos')
    args = parser.parse_args()
    sys.path.insert(0, str(base.UPSTREAM))
    install(args.output)
    base.worker(argparse.Namespace(output=str(args.output), annotation=str(args.annotation),
                                   clips=str(args.clips), check_pins=False))


if __name__ == '__main__':
    main()
