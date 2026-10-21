"""Nonblocking GPU leases compatible with legacy exclusive StreamingBench jobs.

Two-process mode holds the existing global lock in shared mode and one of two
exclusive slot locks. Legacy jobs keep using the same global lock exclusively.
All locks are released on admission failure, context exit, or process death.
"""

from contextlib import ExitStack, contextmanager
import fcntl
import os
from pathlib import Path
import subprocess


GPU_LOCK = Path("/tmp/hermes-gpu0.lock")
SLOT_LOCKS = (Path("/tmp/hermes-gpu0.slot0.lock"),
              Path("/tmp/hermes-gpu0.slot1.lock"))


def _lock(stream, mode, message):
    try:
        fcntl.flock(stream.fileno(), mode | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(message + "; no job has been queued") from exc


@contextmanager
def gpu_lease(output, concurrent_processes=1):
    """Admit one exclusive job or up to two shared-GPU jobs without queuing.

    GPU selection and the minimum free-memory override retain the legacy
    SPARSEMM_GPU_INDEX/SPARSEMM_MIN_FREE_MIB environment variable names.
    Locks prevent cooperating jobs from exceeding the selected concurrency;
    free memory is a pre-load check rather than an external memory reservation.
    """
    if (not isinstance(concurrent_processes, int)
            or isinstance(concurrent_processes, bool)
            or concurrent_processes not in (1, 2)):
        raise ValueError("concurrent_processes must be 1 or 2")
    gpu = os.environ.get("SPARSEMM_GPU_INDEX", "0")
    default_minimum = 28000 if concurrent_processes == 1 else 18000
    minimum = int(os.environ.get("SPARSEMM_MIN_FREE_MIB", str(default_minimum)))
    if minimum < 0:
        raise ValueError("SPARSEMM_MIN_FREE_MIB must be nonnegative")
    output = Path(output)
    with ExitStack() as stack:
        # Acquire GPU capacity first: a rejected third worker never claims an
        # output directory. flock itself atomically chooses each slot.
        global_lock = stack.enter_context(GPU_LOCK.open("a"))
        _lock(global_lock, fcntl.LOCK_EX if concurrent_processes == 1 else fcntl.LOCK_SH,
              "GPU is busy with an incompatible lease")
        slot = None
        if concurrent_processes == 2:
            for index, path in enumerate(SLOT_LOCKS):
                candidate = path.open("a")
                try:
                    fcntl.flock(candidate.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    candidate.close()
                    continue
                except BaseException:
                    candidate.close()
                    raise
                stack.callback(candidate.close)
                slot = index
                break
            if slot is None:
                raise RuntimeError("Both GPU process slots are busy; no job has been queued")
        output.mkdir(parents=True, exist_ok=True)
        output_lock = stack.enter_context((output / ".run.lock").open("a"))
        _lock(output_lock, fcntl.LOCK_EX, "Output directory is busy")
        report = subprocess.check_output([
            "nvidia-smi", "-i", gpu, "--query-gpu=memory.free",
            "--format=csv,noheader,nounits"], text=True)
        free_mib = int(report.strip())
        if free_mib < minimum:
            raise RuntimeError(f"GPU {gpu} has {free_mib} MiB free; requires {minimum} MiB. "
                               "No job has been queued")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        yield {"gpu_index": gpu, "slot": slot,
               "concurrent_processes": concurrent_processes,
               "free_mib_at_admission": free_mib, "minimum_free_mib": minimum}
