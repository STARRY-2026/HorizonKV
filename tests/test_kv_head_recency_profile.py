import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from head_analysis.hermes_kv_head_ragged_prefill import (
    _apply_kv_head_recency_alpha,
    _apply_kv_head_recency_k_decay,
    install_kv_head_recency_profile,
)


class KVHeadRecencyProfileTests(unittest.TestCase):
    def _model(self, num_layers=2, num_kv_heads=4):
        config = SimpleNamespace(
            num_attention_heads=28,
            num_key_value_heads=num_kv_heads,
        )
        return SimpleNamespace(
            num_layers=num_layers,
            language_model=SimpleNamespace(config=config),
            kv_cache=SimpleNamespace(layers=[object()] * num_layers),
        )

    def _write_profile(self, rows):
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", newline="", delete=False
        )
        with tmp:
            writer = csv.DictWriter(
                tmp,
                fieldnames=["layer", "kv_head", "b_log_per_token_ratio"],
            )
            writer.writeheader()
            writer.writerows(rows)
        self.addCleanup(Path(tmp.name).unlink, missing_ok=True)
        return tmp.name

    def test_layer_centered_score_adjusts_alpha_in_expected_direction(self):
        rows = []
        for layer_idx in range(2):
            for kv_head, score in enumerate((-2.0, -1.0, 1.0, 2.0)):
                rows.append({
                    "layer": layer_idx,
                    "kv_head": kv_head,
                    "b_log_per_token_ratio": score,
                })
        model = self._model()
        install_kv_head_recency_profile(model, self._write_profile(rows))

        alphas = [
            _apply_kv_head_recency_alpha(model, 0, kv_head, 0.5)
            for kv_head in range(4)
        ]
        self.assertLess(alphas[0], alphas[1])
        self.assertLess(alphas[1], 0.5)
        self.assertGreater(alphas[2], 0.5)
        self.assertLess(alphas[2], alphas[3])
        self.assertAlmostEqual(alphas[0] + alphas[3], 1.0, places=7)

    def test_alpha_bounds_clip_layer_extremes(self):
        rows = [
            {
                "layer": 0,
                "kv_head": kv_head,
                "b_log_per_token_ratio": score,
            }
            for kv_head, score in enumerate((-10.0, -1.0, 1.0, 10.0))
        ]
        model = self._model(num_layers=1)
        install_kv_head_recency_profile(
            model,
            self._write_profile(rows),
            delta=0.4,
            alpha_min=0.05,
            alpha_max=0.95,
        )

        self.assertEqual(_apply_kv_head_recency_alpha(model, 0, 0, 0.0), 0.05)
        self.assertEqual(_apply_kv_head_recency_alpha(model, 0, 3, 1.0), 0.95)

    def test_k_decay_uses_same_temporal_direction_as_alpha(self):
        rows = [
            {
                "layer": 0,
                "kv_head": kv_head,
                "b_log_per_token_ratio": score,
            }
            for kv_head, score in enumerate((-2.0, -1.0, 1.0, 2.0))
        ]
        model = self._model(num_layers=1)
        install_kv_head_recency_profile(
            model,
            self._write_profile(rows),
            k_delta=8.0,
        )

        self.assertLess(_apply_kv_head_recency_k_decay(model, 0, 0, 10.0), 10.0)
        self.assertGreater(_apply_kv_head_recency_k_decay(model, 0, 3, 10.0), 10.0)
        self.assertEqual(_apply_kv_head_recency_k_decay(model, 0, 0, 0.0), 0.0)
        self.assertGreater(_apply_kv_head_recency_k_decay(model, 0, 3, 0.0), 0.0)

    def test_zero_k_delta_preserves_layer_decay(self):
        rows = [
            {
                "layer": 0,
                "kv_head": kv_head,
                "b_log_per_token_ratio": float(kv_head),
            }
            for kv_head in range(4)
        ]
        model = self._model(num_layers=1)
        install_kv_head_recency_profile(model, self._write_profile(rows))
        for base_k in (0.0, 8.0, 20.0):
            for kv_head in range(4):
                self.assertEqual(
                    _apply_kv_head_recency_k_decay(
                        model, 0, kv_head, base_k
                    ),
                    base_k,
                )

    def test_neutral_profile_preserves_hermes_layer_alpha(self):
        rows = [
            {
                "layer": 0,
                "kv_head": kv_head,
                "b_log_per_token_ratio": 1.0,
            }
            for kv_head in range(4)
        ]
        model = self._model(num_layers=1)
        install_kv_head_recency_profile(model, self._write_profile(rows))

        for base_alpha in (0.0, 0.35, 1.0):
            for kv_head in range(4):
                self.assertEqual(
                    _apply_kv_head_recency_alpha(
                        model, 0, kv_head, base_alpha
                    ),
                    base_alpha,
                )

    def test_install_before_kv_cache_initialization(self):
        rows = [
            {
                "layer": 0,
                "kv_head": kv_head,
                "b_log_per_token_ratio": float(kv_head),
            }
            for kv_head in range(4)
        ]
        model = self._model(num_layers=1)
        model.kv_cache = None
        install_kv_head_recency_profile(model, self._write_profile(rows))
        self.assertEqual(
            model._kv_head_recency_profile["centered_scores"].shape,
            (1, 4),
        )

    def test_profile_requires_every_kv_head_exactly_once(self):
        incomplete = [
            {
                "layer": 0,
                "kv_head": kv_head,
                "b_log_per_token_ratio": float(kv_head),
            }
            for kv_head in range(3)
        ]
        with self.assertRaisesRegex(ValueError, "covers 3/4"):
            install_kv_head_recency_profile(
                self._model(num_layers=1), self._write_profile(incomplete)
            )

        duplicate = incomplete + [dict(incomplete[0])]
        with self.assertRaisesRegex(ValueError, "duplicate head"):
            install_kv_head_recency_profile(
                self._model(num_layers=1), self._write_profile(duplicate)
            )

    def test_install_does_not_modify_budget_scores(self):
        rows = [
            {
                "layer": 0,
                "kv_head": kv_head,
                "b_log_per_token_ratio": float(kv_head),
            }
            for kv_head in range(4)
        ]
        model = self._model(num_layers=1)
        model._kv_head_budget_scores = np.arange(28, dtype=np.float64).reshape(1, 28)
        before = model._kv_head_budget_scores.copy()
        install_kv_head_recency_profile(model, self._write_profile(rows))
        np.testing.assert_array_equal(model._kv_head_budget_scores, before)


if __name__ == "__main__":
    unittest.main()
