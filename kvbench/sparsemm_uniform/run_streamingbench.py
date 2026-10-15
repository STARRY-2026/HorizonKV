"""Bounded native Uniform (fixed-budget SnapKV) validation."""
import sys
from sparsemm_protokv.run_streamingbench import main

if __name__ == "__main__":
    raise SystemExit(main(["--method", "uniform", *sys.argv[1:]]))
