"""CPU-only, real-process flock admission and cleanup checks."""

import fcntl
import multiprocessing
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from sparsemm_native import gpu_slots


def configure_paths(root):
    root = Path(root)
    gpu_slots.GPU_LOCK = root / "global.lock"
    gpu_slots.SLOT_LOCKS = tuple(root / f"slot{i}.lock" for i in range(2))


def hold_lease(root, output, count, queue, release):
    configure_paths(root)
    with patch.object(gpu_slots.subprocess, "check_output", return_value="40960\n"):
        try:
            with gpu_slots.gpu_lease(output, count) as lease:
                queue.put(("ready", lease))
                release.wait(20)
        except Exception as error:
            queue.put(("error", str(error)))


def hold_legacy(root, queue, release):
    configure_paths(root)
    with gpu_slots.GPU_LOCK.open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        queue.put(("ready", None))
        release.wait(20)


class GPUSlotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.addCleanup(patch.stopall)
        patch.object(gpu_slots, "GPU_LOCK", self.root / "global.lock").start()
        patch.object(gpu_slots, "SLOT_LOCKS", tuple(self.root / f"slot{i}.lock" for i in range(2))).start()
        patch.dict(os.environ).start()
        os.environ.pop("SPARSEMM_MIN_FREE_MIB", None)
        os.environ["SPARSEMM_GPU_INDEX"] = "0"
        self.report = patch.object(gpu_slots.subprocess, "check_output", return_value="40960\n").start()
        self.context = multiprocessing.get_context("spawn")

    def start_holder(self, name, count=2, legacy=False):
        queue, release = self.context.Queue(), self.context.Event()
        target = hold_legacy if legacy else hold_lease
        arguments = ((str(self.root), queue, release) if legacy else
                     (str(self.root), str(self.root / name), count, queue, release))
        process = self.context.Process(target=target, args=arguments)
        process.start()

        def cleanup():
            release.set()
            process.join(3)
            if process.is_alive():
                process.kill()
                process.join(3)
            queue.close()

        self.addCleanup(cleanup)
        kind, detail = queue.get(timeout=10)
        self.assertEqual(kind, "ready", detail)
        return process, release, detail

    def assert_locks_free(self, *outputs):
        for path in (gpu_slots.GPU_LOCK, *gpu_slots.SLOT_LOCKS,
                     *(Path(output) / ".run.lock" for output in outputs)):
            with path.open("a") as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_two_real_processes_coexist_third_rejected_without_claiming_output(self):
        first, release_first, first_lease = self.start_holder("first")
        second, release_second, second_lease = self.start_holder("second")
        self.assertNotEqual(first.pid, second.pid)
        self.assertEqual({first_lease["slot"], second_lease["slot"]}, {0, 1})
        rejected = self.root / "third"
        with self.assertRaisesRegex(RuntimeError, "Both GPU process slots"):
            with gpu_slots.gpu_lease(rejected, 2):
                self.fail("A third process was admitted")
        self.assertFalse(rejected.exists())
        release_first.set()
        first.join(3)
        self.assertFalse(first.is_alive())
        with gpu_slots.gpu_lease(rejected, 2) as lease:
            self.assertEqual(lease["slot"], first_lease["slot"])
        release_second.set()
        second.join(3)
        self.assert_locks_free(self.root / "first", self.root / "second", rejected)

    def test_parallel_holder_blocks_legacy_exclusive_lock_and_new_single_mode(self):
        process, release, _ = self.start_holder("shared")
        with gpu_slots.GPU_LOCK.open("a") as stream:
            with self.assertRaises(BlockingIOError):
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with self.assertRaisesRegex(RuntimeError, "incompatible"):
            with gpu_slots.gpu_lease(self.root / "single", 1):
                self.fail("Exclusive job entered alongside shared job")
        release.set()
        process.join(3)
        with gpu_slots.gpu_lease(self.root / "single", 1) as lease:
            self.assertIsNone(lease["slot"])
            self.assertEqual(lease["minimum_free_mib"], 28000)

    def test_legacy_exclusive_holder_blocks_both_new_modes(self):
        self.start_holder("legacy", legacy=True)
        for count in (1, 2):
            with self.subTest(count=count), self.assertRaisesRegex(RuntimeError, "incompatible"):
                with gpu_slots.gpu_lease(self.root / f"mode{count}", count):
                    self.fail("Legacy exclusive lock was bypassed")
            self.assertFalse((self.root / f"mode{count}").exists())

    def test_duplicate_output_does_not_truncate_lock_and_releases_claimed_slot(self):
        self.start_holder("same")
        lock = self.root / "same/.run.lock"
        lock.write_text("existing holder metadata\n")
        with self.assertRaisesRegex(RuntimeError, "Output directory"):
            with gpu_slots.gpu_lease(self.root / "same", 2):
                self.fail("Duplicate output was admitted")
        self.assertEqual(lock.read_text(), "existing holder metadata\n")
        with gpu_slots.gpu_lease(self.root / "other", 2) as lease:
            self.assertEqual(lease["slot"], 1)

    def test_memory_query_and_body_failures_release_every_lock(self):
        output = self.root / "failure"
        cases = [("17999\n", None, "requires 18000"),
                 (None, OSError("query failed"), "query failed")]
        for report, error, message in cases:
            with self.subTest(message=message), patch.object(
                    gpu_slots.subprocess, "check_output", return_value=report, side_effect=error):
                with self.assertRaisesRegex((RuntimeError, OSError), message):
                    with gpu_slots.gpu_lease(output, 2):
                        self.fail("Invalid free-memory check admitted work")
            self.assert_locks_free(output)
        with self.assertRaisesRegex(ValueError, "body failure"):
            with gpu_slots.gpu_lease(output, 2):
                raise ValueError("body failure")
        self.assert_locks_free(output)

    def test_defaults_override_gpu_selection_and_validation(self):
        self.report.return_value = "18000\n"
        with gpu_slots.gpu_lease(self.root / "two", 2) as lease:
            self.assertEqual(lease["minimum_free_mib"], 18000)
        with self.assertRaisesRegex(RuntimeError, "requires 28000"):
            with gpu_slots.gpu_lease(self.root / "one"):
                self.fail("Single mode used parallel memory threshold")
        os.environ.update(SPARSEMM_GPU_INDEX="3", SPARSEMM_MIN_FREE_MIB="17000")
        with gpu_slots.gpu_lease(self.root / "custom", 2) as lease:
            self.assertEqual(lease["gpu_index"], "3")
            self.assertEqual(lease["minimum_free_mib"], 17000)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "3")
        self.assertEqual(self.report.call_args.args[0][1:3], ["-i", "3"])
        for invalid in (0, 3, True, "2"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                with gpu_slots.gpu_lease(self.root / "invalid", invalid):
                    self.fail("Invalid concurrency accepted")


if __name__ == "__main__":
    unittest.main()
