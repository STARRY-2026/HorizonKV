"""Offline LU history-retention profiles and exact streaming capacity lookup.

The protected recent window is outside the profiled history domain. Offline
token utilities may come from a chosen calibration objective; online lookup
uses only cumulative uncompressed visual length and a fixed global budget.
PAVA optimizes the concave-envelope relaxation, not the original nonconcave
integer utility problem. Capacity targets do not reconstruct evicted tokens.
"""
from __future__ import annotations

from collections import defaultdict
import heapq
import math
from numbers import Integral

import numpy as np
import torch


PROFILE_SCHEMA = "sparsemm_lukv.history_profile"
PROFILE_VERSION = 1
DEFAULT_RETENTION_GRID = (
    0., .001, .002, .005, .01, .02, .04, .06, .08, .10,
    .15, .20, .30, .40, .50, .60, .70, .80, .90, 1.,
)


def _integer(value, name, minimum=0):
    if not isinstance(value, Integral) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _array(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _grid(value):
    grid = np.asarray(value, dtype=np.float64)
    if (grid.ndim != 1 or grid.size < 2 or not np.isfinite(grid).all()
            or grid[0] != 0. or grid[-1] != 1. or np.any(np.diff(grid) <= 0.)):
        raise ValueError("retention_grid must strictly increase from 0 to 1")
    return grid


def _concave_blocks(gains):
    """PAVA blocks of non-increasing marginal utility, in rank order."""
    blocks = []
    for value in gains:
        blocks.append([float(value), 1])
        while len(blocks) >= 2:
            left, right = blocks[-2:]
            if left[0] / left[1] > right[0] / right[1]:
                break
            blocks[-2:] = [[left[0] + right[0], left[1] + right[1]]]
    return blocks


def _solve_history(ordered_utility, grid):
    """Incrementally allocate envelope segments at every increasing grid point."""
    heads, history_tokens = ordered_utility.shape
    curves = np.concatenate((np.zeros((heads, 1)), np.cumsum(ordered_utility, axis=1)), axis=1)
    if not np.isfinite(curves).all():
        raise ValueError("Offline utility sums exceed the supported finite numeric range")
    blocks = [_concave_blocks(row) for row in ordered_utility]
    heap = []
    for head, sequence in enumerate(blocks):
        gain, count = sequence[0]
        heapq.heappush(heap, (-gain / count, head, 0, count))
    capacities = np.zeros(heads, dtype=np.int64)
    assigned, envelope_utility = 0, 0.
    allocations, gaps = [], []
    for retention in grid:
        total = round(float(retention) * heads * history_tokens)
        while assigned < total:
            neg_slope, head, block, available = heapq.heappop(heap)
            take = min(total - assigned, available)
            capacities[head] += take
            assigned += take
            envelope_utility -= neg_slope * take
            if take < available:
                heapq.heappush(heap, (neg_slope, head, block, available - take))
            elif block + 1 < len(blocks[head]):
                gain, count = blocks[head][block + 1]
                heapq.heappush(heap, (-gain / count, head, block + 1, count))
        actual = float(curves[np.arange(heads), capacities].sum())
        gap = envelope_utility - actual
        if gap < -1e-7 * max(1., abs(actual)):
            raise AssertionError("Original-curve utility exceeds its concave envelope")
        allocations.append(capacities.copy())
        gaps.append(max(0., gap))
    return np.stack(allocations), np.asarray(gaps)


def build_profile(observations, num_layers, num_kv_heads, window_size=32,
                  retention_grid=None):
    """Build a JSON-safe profile from attention rankings and offline utilities.

    Each observation is ``{video_id, ranking[H,N-W], utility[H,N]}``, with
    ``H=L*Hkv`` and more than W visual tokens. Ranking indexes the unprotected
    history and must be a complete permutation of ``0..N-W-1`` for each head.
    The retained recent W tokens contribute a constant utility, so allocation
    optimizes only history, at round(rho*H*(N-W)) slots for each grid point.

    Per-observation capacities are divided by that observation's history
    length, then averaged within videos and equally across videos. No blend,
    fixed head clipping, or evaluation-question information is added here.
    """
    num_layers = _integer(num_layers, "num_layers", 1)
    num_kv_heads = _integer(num_kv_heads, "num_kv_heads", 1)
    window_size = _integer(window_size, "window_size", 1)
    grid = _grid(DEFAULT_RETENTION_GRID if retention_grid is None else retention_grid)
    heads = num_layers * num_kv_heads
    by_video = defaultdict(list)
    gaps, history_lengths = [], []
    for observation in observations:
        if not isinstance(observation, dict) or any(name not in observation for name in ("video_id", "ranking", "utility")):
            raise ValueError("Each observation must contain video_id, ranking and utility")
        video_id = observation["video_id"]
        if not isinstance(video_id, (str, Integral)) or isinstance(video_id, bool) or not str(video_id):
            raise ValueError("video_id must be a nonempty string or integer")
        ranking = _array(observation["ranking"])
        utility = _array(observation["utility"])
        if (utility.ndim != 2 or utility.shape[0] != heads or utility.shape[1] <= window_size
                or not np.issubdtype(utility.dtype, np.number) or np.iscomplexobj(utility)):
            raise ValueError("utility must be a numeric [L*Hkv,N] matrix with N > W")
        utility = utility.astype(np.float64, copy=False)
        if not np.isfinite(utility).all() or np.any(utility < 0):
            raise ValueError("Offline token utilities must be finite and nonnegative")
        history_tokens = utility.shape[1] - window_size
        if (ranking.shape != (heads, history_tokens)
                or not np.issubdtype(ranking.dtype, np.integer)):
            raise ValueError("ranking must be an integer [L*Hkv,N-W] matrix")
        expected = np.arange(history_tokens)
        for row in ranking:
            if not np.array_equal(np.sort(row), expected):
                raise ValueError("Every history ranking must be a complete permutation")
        ordered = np.take_along_axis(utility[:, :history_tokens], ranking, axis=1)
        allocation, observation_gaps = _solve_history(ordered, grid)
        by_video[str(video_id)].append(allocation.astype(np.float64) / history_tokens)
        gaps.append(observation_gaps)
        history_lengths.append(history_tokens)
    if not by_video:
        raise ValueError("At least one calibration observation is required")
    video_means = [np.mean(values, axis=0) for _, values in sorted(by_video.items())]
    ratios = np.mean(video_means, axis=0).reshape(grid.size, num_layers, num_kv_heads)
    return {
        "schema": PROFILE_SCHEMA, "version": PROFILE_VERSION,
        "selector": "sparsemm_attention", "budget_scope": "visual",
        "profile_domain": "unprotected_history", "window_size": window_size,
        "num_layers": num_layers, "num_kv_heads": num_kv_heads,
        "retention_grid": grid.tolist(), "history_retention": ratios.tolist(),
        "allocation_solver": "pava_concave_envelope_marginal_greedy",
        "aggregation": "mean_ratios_within_video_then_equal_video_mean",
        "online_utility": False,
        "calibration_summary": {
            "num_videos": len(by_video), "num_observations": len(history_lengths),
            "observations_per_video": {video: len(values) for video, values in sorted(by_video.items())},
            "minimum_history_tokens": min(history_lengths), "maximum_history_tokens": max(history_lengths),
            "maximum_relaxation_gap": float(np.max(gaps)),
            "mean_relaxation_gap": float(np.mean(gaps)),
            "original_nonconcave_integer_optimality_claimed": False,
        },
    }


def _validate_profile(profile, window_size):
    if (not isinstance(profile, dict) or profile.get("schema") != PROFILE_SCHEMA
            or isinstance(profile.get("version"), bool) or profile.get("version") != PROFILE_VERSION):
        raise ValueError("Unsupported LU history profile schema/version")
    if profile.get("selector") != "sparsemm_attention" or profile.get("profile_domain") != "unprotected_history":
        raise ValueError("LU profile must use direct attention in the history domain")
    if _integer(profile.get("window_size"), "profile window_size", 1) != window_size:
        raise ValueError("Profile protection window differs from the runtime window")
    layers = _integer(profile.get("num_layers"), "profile num_layers", 1)
    heads = _integer(profile.get("num_kv_heads"), "profile num_kv_heads", 1)
    grid = _grid(profile.get("retention_grid"))
    ratios = np.asarray(profile.get("history_retention"), dtype=np.float64)
    if (ratios.shape != (grid.size, layers, heads) or not np.isfinite(ratios).all()
            or np.any(ratios < 0.) or np.any(ratios > 1.)):
        raise ValueError("history_retention must be finite [grid,L,Hkv] ratios in [0,1]")
    if not np.all(ratios[0] == 0.) or not np.all(ratios[-1] == 1.):
        raise ValueError("Profile endpoints must retain none/all of history")
    if np.any(np.diff(ratios, axis=0) < -1e-12):
        raise ValueError("Each head's history retention must be nondecreasing")
    return grid, ratios, layers, heads


def validate_profile(profile, num_layers=None, num_kv_heads=None, window_size=None):
    """Validate a loaded profile and optional model shape, preserving metadata.

    Additional fields such as a ``calibration`` provenance manifest are allowed
    and returned unchanged. No runtime model, utility, or GPU work is needed.
    """
    if not isinstance(profile, dict):
        raise ValueError("LU history profile must be a dictionary")
    window = _integer(profile.get("window_size") if window_size is None else window_size,
                      "window_size", 1)
    _, _, layers, heads = _validate_profile(profile, window)
    if num_layers is not None and _integer(num_layers, "num_layers", 1) != layers:
        raise ValueError("Profile layer count differs from the model")
    if num_kv_heads is not None and _integer(num_kv_heads, "num_kv_heads", 1) != heads:
        raise ValueError("Profile KV-head count differs from the model")
    return profile


def _integer_projection(raw, total, lower, upper):
    """Euclidean box/sum projection, then exact largest-remainder rounding."""
    raw = np.asarray(raw, dtype=np.float64).reshape(-1)
    left, right = lower - float(raw.max()), upper - float(raw.min())
    for _ in range(80):
        shift = (left + right) / 2.
        if math.fsum(float(value) for value in np.clip(raw + shift, lower, upper)) < total:
            left = shift
        else:
            right = shift
    projected = np.clip(raw + (left + right) / 2., lower, upper)
    capacities = np.floor(projected).astype(np.int64)
    missing = total - int(capacities.sum())
    eligible = np.flatnonzero(capacities < upper)
    order = eligible[np.argsort(-(projected[eligible] - capacities[eligible]), kind="stable")]
    if not 0 <= missing <= order.size:
        raise ArithmeticError("Exact LU integer projection failed due to numeric range")
    capacities[order[:missing]] += 1
    if int(capacities.sum()) != total or np.any(capacities < lower) or np.any(capacities > upper):
        raise AssertionError("LU projection violated its exact total or capacity bounds")
    return capacities


def allocate_profile(profile, seen_tokens, budget, window_size=32):
    """Look up capacities using uncompressed visual T, with no online utility.

    For T>K, rho=(K-W)/(T-W) is the fraction of unprotected history affordable
    under the average per-head visual budget K. Interpolated head ratios become
    W+ratio*(T-W), then are projected/rounded to exactly L*Hkv*K slots, bounded
    by W and T. For T<=K, every visual token is retained. Prefix text is extra.

    These are target capacities. Dynamic growth of an individual target cannot
    recover tokens previously evicted from a streaming cache, so actual cache
    occupancy may be smaller; runtime tracing must report that independently.
    """
    seen_tokens = _integer(seen_tokens, "seen_tokens")
    budget = _integer(budget, "budget", 1)
    window_size = _integer(window_size, "window_size", 1)
    if budget < window_size:
        raise ValueError("The visual budget must cover recent-window protection")
    grid, ratios, layers, heads = _validate_profile(profile, window_size)
    count = layers * heads
    target = count * min(budget, seen_tokens)
    rho = min(1., (budget - window_size) / (seen_tokens - window_size)) if seen_tokens > window_size else None
    if seen_tokens <= budget:
        capacities = np.full((layers, heads), seen_tokens, dtype=np.int64)
        endpoints, weight = [1., 1.] if rho is not None else None, 0.
    else:
        upper_index = int(np.searchsorted(grid, rho, side="left"))
        if grid[upper_index] == rho:
            lower_index = upper_index
            weight = 0.
        else:
            lower_index = upper_index - 1
            weight = float((rho - grid[lower_index]) / (grid[upper_index] - grid[lower_index]))
        retained = ratios[lower_index] * (1. - weight) + ratios[upper_index] * weight
        raw = window_size + retained.reshape(-1) * (seen_tokens - window_size)
        capacities = _integer_projection(raw, target, window_size, seen_tokens).reshape(layers, heads)
        endpoints = [float(grid[lower_index]), float(grid[upper_index])]
    metadata = {
        "source_T": seen_tokens, "rho": rho, "grid_endpoints": endpoints,
        "interpolation_weight": weight, "target_slots": target,
        "allocated_slots": int(capacities.sum()), "budget": budget,
        "window_size": window_size, "protected_tokens_per_head": min(window_size, seen_tokens),
        "profile_domain": "unprotected_history", "budget_scope": "visual",
        "profile_bypassed": seen_tokens <= budget, "no_online_utility": True,
        "actual_occupancy_may_be_lower": True,
        "projection": "bounded_euclidean_shift_largest_remainder",
        "head_capacities": capacities.tolist(),
    }
    return torch.from_numpy(capacities.copy()), metadata
