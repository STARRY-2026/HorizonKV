"""CPU checks for transferring a DP head-capacity table to native SparseMM."""

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from sparsemm_native.policy import load_fixed_head_budgets
from sparsemm_protokv.audit import AuditError, external_sparsemm_table
from kvbench import RunConfig, build_command


class FixedDPBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "dp.csv"
        self.rows = [
            {"layer": layer, "kv_head": head, "capacity_n": 20,
             "recent_r": 0 if (layer, head) == (0, 0) else 4,
             "history_q": 20 if (layer, head) == (0, 0) else 16}
            for layer in range(2) for head in range(2)
        ]

    def write(self):
        with self.path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def test_exact_table_and_bank_split_is_explicitly_unused(self):
        self.write()
        runtime, metadata = load_fixed_head_budgets(self.path, 2, 2, 20, 4)
        independently_audited, audit_metadata = external_sparsemm_table(self.path, 2, 2, 20, 4)
        self.assertEqual(runtime.tolist(), independently_audited)
        self.assertEqual(metadata, audit_metadata)
        self.assertEqual(metadata["recent_below_sparsemm_window_heads"], 1)
        self.assertFalse(metadata["dp_bank_split_used_online"])

    def test_wrong_global_total_is_rejected_by_runtime_and_audit(self):
        self.rows[0]["capacity_n"] = 21
        self.rows[0]["history_q"] = 21
        self.write()
        with self.assertRaisesRegex(ValueError, "total"):
            load_fixed_head_budgets(self.path, 2, 2, 20, 4)
        with self.assertRaises(AuditError):
            external_sparsemm_table(self.path, 2, 2, 20, 4)

    def test_inconsistent_dp_split_is_rejected(self):
        self.rows[0]["history_q"] = 19
        self.write()
        with self.assertRaisesRegex(ValueError, "split"):
            load_fixed_head_budgets(self.path, 2, 2, 20, 4)
        with self.assertRaises(AuditError):
            external_sparsemm_table(self.path, 2, 2, 20, 4)

    def test_unified_command_passes_external_budget_path(self):
        config = RunConfig(method="sparsemm", budget=250, sparsemm_budget_path=self.path)
        command = build_command(config, "sparsemm", Path(self.temp.name) / "output")
        index = command.index("--sparsemm-budget-path")
        self.assertEqual(command[index + 1], str(self.path.resolve()))


if __name__ == "__main__":
    unittest.main()
