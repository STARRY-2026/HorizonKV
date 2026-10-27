"""Launch the unchanged pinned OVO recent-1/4 implementation with bundled quotas."""

from pathlib import Path


def main(argv=None):
    from sparsemm_native import capped_history

    capped_history.SOURCE = (Path(__file__).resolve().parents[1] /
                             "assets/recent1or4/per_head_allocations.csv").resolve()
    from sparsemm_native.run_ovobench_recent1or4_cap4000 import main as native_main

    return native_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
