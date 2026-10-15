"""Bounded native SparseMM validation; full benchmark execution stays paused."""
import sys
from sparsemm_protokv.run_streamingbench import main

if __name__ == "__main__":
    raise SystemExit(main(["--method", "sparsemm", *sys.argv[1:]]))
