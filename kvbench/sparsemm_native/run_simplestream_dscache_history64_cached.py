"""Compatibility entrypoint for fixed-896 DSCache history with disk embeddings.

All allocation, audit, baseline, and upstream checks live in the main runner.
This entrypoint requires --vision-cache-dir and forwards the remaining options.
"""

import sys

from sparsemm_native.run_simplestream_dscache_history64 import main as run_main


def main():
    if not any(arg in ('-h', '--help') for arg in sys.argv[1:]) and not any(
            arg == '--vision-cache-dir' or arg.startswith('--vision-cache-dir=')
               for arg in sys.argv[1:]):
        raise SystemExit('--vision-cache-dir is required for the cached entrypoint')
    return run_main()


if __name__ == '__main__':
    main()
