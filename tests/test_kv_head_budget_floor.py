import numpy as np

from head_analysis.hermes_kv_head_budget import _apply_budget_floor


def test_budget_floor_preserves_total_and_order():
    original = np.asarray([[466, 800, 1000, 1507]], dtype=int)
    adjusted = _apply_budget_floor(original, num_keep=1000, floor_ratio=0.75)

    assert adjusted.shape == original.shape
    assert int(adjusted.sum()) == int(original.sum())
    assert int(adjusted.min()) >= 750
    assert list(adjusted.reshape(-1)) == sorted(adjusted.reshape(-1))


def test_zero_budget_floor_is_identity():
    original = np.asarray([[466, 800, 1000, 1507]], dtype=int)
    adjusted = _apply_budget_floor(original, num_keep=1000, floor_ratio=0.0)

    assert np.array_equal(adjusted, original)
