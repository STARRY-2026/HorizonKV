"""LU-specific convenience entry for the shared bounded native CLI controller."""

import sys

from sparsemm_protokv.run_streamingbench import main as shared_main


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if any(arg == "--method" or arg.startswith("--method=") for arg in args):
        raise SystemExit("This entry selects method=lukv; use the shared controller to select another method")
    return shared_main(["--method", "lukv", *args])


if __name__ == "__main__":
    raise SystemExit(main())
