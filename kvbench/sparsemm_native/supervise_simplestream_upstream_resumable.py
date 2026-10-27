"""Run the pinned SimpleStream supervisor with the resumable download launcher.

The original supervisor, experiment workers, and their protocol pins are not
modified. Only its fallback downloader command is replaced at process start.
"""

import subprocess

from sparsemm_native import supervise_simplestream_upstream as original


class SubprocessProxy:
    def __getattr__(self, name):
        return getattr(subprocess, name)

    @staticmethod
    def Popen(command, *args, **kwargs):
        if (isinstance(command, (list, tuple)) and len(command) >= 4
                and command[-2:] == ['-m', 'sparsemm_native.download_ovobench_chunks']):
            command = [*command[:-1], 'sparsemm_native.download_ovobench_chunks_resumable']
        return subprocess.Popen(command, *args, **kwargs)


if __name__ == '__main__':
    original.subprocess = SubprocessProxy()
    original.main()
