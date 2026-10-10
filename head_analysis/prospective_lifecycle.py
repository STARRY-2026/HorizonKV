"""Utilities for prospective, frame-level KV lifecycle traces.

The trace follows a source frame from admission through later observations.
This is deliberately different from an age-bin attention snapshot: lifecycle
scores are fit on training videos and are evaluated against future reads of the
same source frame on held-out videos.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 1
REQUIRED_FRAME_FIELDS = {
    "source_frame_id",
    "age_sec",
    "token_count",
    "arrival_salience_top_p",
    "value_l1_mean",
    "attention_mass",
    "effective_readout_mass",
}
DEFAULT_AGE_EDGES = np.asarray(
    [0.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, np.inf],
    dtype=np.float64,
)


def validate_trace_record(record):
    """Validate the on-disk contract and reject impossible/future provenance."""
    if int(record.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema_version: {record.get('schema_version')}")
    required = {
        "video_id", "query_source", "observation_frame_id", "sample_fps",
        "full_cache", "heads",
    }
    missing = required.difference(record)
    if missing:
        raise ValueError(f"Trace record is missing fields: {sorted(missing)}")
    fps = float(record["sample_fps"])
    if fps <= 0:
        raise ValueError("sample_fps must be positive")
    observation = int(record["observation_frame_id"])
    for head in record["heads"]:
        if not {"layer", "kv_head", "frames"}.issubset(head):
            raise ValueError("Each head requires layer, kv_head, and frames")
        for frame in head["frames"]:
            missing_frame = REQUIRED_FRAME_FIELDS.difference(frame)
            if missing_frame:
                raise ValueError(f"Frame entry is missing fields: {sorted(missing_frame)}")
            source = int(frame["source_frame_id"])
            if source > observation:
                raise ValueError("source_frame_id cannot be later than observation_frame_id")
            expected_age = (observation - source) / fps
            if not math.isclose(float(frame["age_sec"]), expected_age, abs_tol=1e-6):
                raise ValueError("age_sec is inconsistent with frame provenance")
    return record


def load_trace_jsonl(paths, require_full_cache=True, query_sources=None):
    """Load and flatten one or more prospective JSONL traces."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    allowed = None if query_sources is None else set(query_sources)
    rows = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = validate_trace_record(json.loads(line))
                if require_full_cache and not bool(record["full_cache"]):
                    raise ValueError(
                        f"{path}:{line_number} was collected after compression; "
                        "prospective fitting requires a full-cache replay"
                    )
                if allowed is not None and record["query_source"] not in allowed:
                    continue
                for head in record["heads"]:
                    for frame in head["frames"]:
                        base = {
                            "video_id": str(record["video_id"]),
                            "question_index": record.get("question_index"),
                            "query_source": str(record["query_source"]),
                            "observation_frame_id": int(record["observation_frame_id"]),
                            "layer": int(head["layer"]),
                            "kv_head": int(head["kv_head"]),
                            **frame,
                        }
                        content_bins = base.pop("content_bins", None)
                        if content_bins:
                            for content_row in content_bins:
                                rows.append({
                                    **base,
                                    **content_row,
                                    "content_bin": int(content_row["content_bin"]),
                                })
                        else:
                            rows.append(base)
    if not rows:
        raise ValueError("No eligible prospective lifecycle observations")
    return rows


def add_remaining_future_utility(rows):
    """Attach strictly-future cumulative effective readout to every row."""
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row["video_id"], int(row["layer"]), int(row["kv_head"]),
            int(row["source_frame_id"]), str(row["query_source"]),
            int(row.get("content_bin", -1)),
        )
        grouped[key].append(row)
    output = []
    for trajectory in grouped.values():
        trajectory.sort(key=lambda x: int(x["observation_frame_id"]), reverse=True)
        future = 0.0
        for row in trajectory:
            enriched = dict(row)
            enriched["future_effective_readout"] = future
            output.append(enriched)
            future += max(0.0, float(row["effective_readout_mass"]))
    return output


def align_predictor_to_future_targets(
    rows, predictor_source="hermes", target_source="true_question",
    include_same_observation=True,
):
    """Label decision-time predictor rows with later task-facing readout.

    By default a true question at the same observation frame is included
    because HERMES compression runs after encoding and before answering that
    question. Set ``include_same_observation=False`` for a strict-future
    diagnostic that excludes the current question.
    """
    target_groups = defaultdict(list)
    for row in rows:
        if row["query_source"] != target_source:
            continue
        key = (
            row["video_id"], int(row["layer"]), int(row["kv_head"]),
            int(row["source_frame_id"]), int(row.get("content_bin", -1)),
        )
        target_groups[key].append((
            int(row["observation_frame_id"]),
            max(0.0, float(row["effective_readout_mass"])),
        ))

    output = []
    for row in rows:
        if row["query_source"] != predictor_source:
            continue
        key = (
            row["video_id"], int(row["layer"]), int(row["kv_head"]),
            int(row["source_frame_id"]), int(row.get("content_bin", -1)),
        )
        targets = target_groups.get(key)
        if not targets:
            continue
        observation = int(row["observation_frame_id"])
        item = dict(row)
        item["future_effective_readout"] = sum(
            value for target_observation, value in targets
            if (
                target_observation >= observation
                if include_same_observation
                else target_observation > observation
            )
        )
        item["score_hermes"] = math.log(
            max(float(row.get("attention_density", 0.0)), 0.0) + 1e-12
        )
        output.append(item)
    if not output:
        raise ValueError(
            f"No aligned {predictor_source!r} predictors and {target_source!r} targets"
        )
    return output


def age_bin_indices(ages, edges=DEFAULT_AGE_EDGES):
    ages = np.asarray(ages, dtype=np.float64)
    return np.clip(np.searchsorted(edges, ages, side="right") - 1, 0, len(edges) - 2)


def fit_remaining_lifecycle(rows, train_video_ids, edges=DEFAULT_AGE_EDGES):
    """Fit per-head and global remaining-read curves using training videos only."""
    train_ids = {str(value) for value in train_video_ids}
    num_bins = len(edges) - 1
    head_values = defaultdict(lambda: [[] for _ in range(num_bins)])
    global_values = [[] for _ in range(num_bins)]
    conditional_head_values = defaultdict(lambda: [[] for _ in range(num_bins)])
    conditional_global_values = defaultdict(lambda: [[] for _ in range(num_bins)])
    content_values = defaultdict(list)
    for row in rows:
        if str(row["video_id"]) not in train_ids:
            continue
        idx = int(age_bin_indices([row["age_sec"]], edges)[0])
        value = max(0.0, float(row["future_effective_readout"]))
        value /= max(1, int(row.get("token_count", 1)))
        head_values[(int(row["layer"]), int(row["kv_head"]))][idx].append(value)
        global_values[idx].append(value)
        content_bin = int(row.get("content_bin", -1))
        if content_bin >= 0:
            key = (int(row["layer"]), int(row["kv_head"]), content_bin)
            conditional_head_values[key][idx].append(value)
            conditional_global_values[content_bin][idx].append(value)
            content_values[content_bin].append(value)

    def robust_curve(buckets):
        observed = np.asarray([
            np.median(bucket) if bucket else np.nan for bucket in buckets
        ], dtype=np.float64)
        valid = np.flatnonzero(np.isfinite(observed))
        if valid.size == 0:
            return np.ones(num_bins, dtype=np.float64)
        curve = np.interp(np.arange(num_bins), valid, observed[valid])
        # Remaining utility cannot increase as a token gets older.
        curve = np.maximum.accumulate(curve[::-1])[::-1]
        return (curve + 1e-12) / (curve[0] + 1e-12)

    global_curve = robust_curve(global_values)
    head_curves = {key: robust_curve(value) for key, value in head_values.items()}
    conditional_global = {
        key: robust_curve(value) for key, value in conditional_global_values.items()
    }
    conditional_heads = {}
    for key, buckets in conditional_head_values.items():
        curve = robust_curve(buckets)
        global_fallback = conditional_global.get(key[2], global_curve)
        counts = np.asarray([len(bucket) for bucket in buckets], dtype=np.float64)
        shrink = counts / (counts + 20.0)
        conditional_heads[key] = shrink * curve + (1.0 - shrink) * global_fallback
    content = {
        key: float(np.median(value)) if value else 0.0
        for key, value in content_values.items()
    }
    if content:
        scale = max(max(content.values()), 1e-12)
        content = {key: value / scale for key, value in content.items()}
    return {
        "edges": np.asarray(edges),
        "global": global_curve,
        "heads": head_curves,
        "conditional_global": conditional_global,
        "conditional_heads": conditional_heads,
        "content": content,
    }


def assign_scores(rows, profiles, lifecycle_weight=1.0, shuffle_seed=20260907):
    """Score content/global/head lifecycle plus shuffle and reversed-time controls."""
    edges = profiles["edges"]
    global_curve = profiles["global"]
    heads = profiles["heads"]
    conditional_global = profiles.get("conditional_global", {})
    conditional_heads = profiles.get("conditional_heads", {})
    learned_content = profiles.get("content", {})
    shuffled = {}
    rng = np.random.default_rng(shuffle_seed)
    by_layer = defaultdict(list)
    for key in heads:
        by_layer[key[0]].append(key)
    for layer, keys in by_layer.items():
        keys = sorted(keys)
        permutation = rng.permutation(len(keys))
        for destination, source_idx in zip(keys, permutation):
            shuffled[destination] = heads[keys[int(source_idx)]]
    shuffled_conditional = {}
    head_keys = sorted({(key[0], key[1]) for key in conditional_heads})
    by_layer = defaultdict(list)
    for key in head_keys:
        by_layer[key[0]].append(key)
    rng = np.random.default_rng(shuffle_seed)
    for layer, keys in by_layer.items():
        keys = sorted(keys)
        permutation = rng.permutation(len(keys))
        for destination, source_idx in zip(keys, permutation):
            source = keys[int(source_idx)]
            for content_bin in learned_content:
                source_key = (source[0], source[1], content_bin)
                if source_key in conditional_heads:
                    shuffled_conditional[(destination[0], destination[1], content_bin)] = (
                        conditional_heads[source_key]
                    )

    scored = []
    for row in rows:
        item = dict(row)
        idx = int(age_bin_indices([row["age_sec"]], edges)[0])
        key = (int(row["layer"]), int(row["kv_head"]))
        head_curve = heads.get(key, global_curve)
        shuffled_curve = shuffled.get(key, global_curve)
        content = math.log(max(float(row["arrival_salience_top_p"]), 0.0) + 1e-12)
        item["score_content"] = content
        item["score_global"] = content + lifecycle_weight * math.log(global_curve[idx] + 1e-12)
        item["score_head"] = content + lifecycle_weight * math.log(head_curve[idx] + 1e-12)
        item["score_shuffled"] = content + lifecycle_weight * math.log(shuffled_curve[idx] + 1e-12)
        item["score_reversed"] = content + lifecycle_weight * math.log(head_curve[-1 - idx] + 1e-12)
        content_bin = int(row.get("content_bin", -1))
        if content_bin >= 0 and conditional_global:
            global_cond = conditional_global.get(content_bin, global_curve)
            head_cond = conditional_heads.get(
                (key[0], key[1], content_bin), global_cond
            )
            shuffled_cond = shuffled_conditional.get(
                (key[0], key[1], content_bin), global_cond
            )
            content_prior = learned_content.get(content_bin, 1.0)
            item["score_content"] = math.log(content_prior + 1e-12)
            item["score_global"] = item["score_content"] + lifecycle_weight * math.log(
                global_cond[idx] + 1e-12
            )
            item["score_head"] = item["score_content"] + lifecycle_weight * math.log(
                head_cond[idx] + 1e-12
            )
            item["score_shuffled"] = item["score_content"] + lifecycle_weight * math.log(
                shuffled_cond[idx] + 1e-12
            )
            item["score_reversed"] = item["score_content"] + lifecycle_weight * math.log(
                head_cond[-1 - idx] + 1e-12
            )
        scored.append(item)
    return scored


def _dcg(relevance):
    relevance = np.asarray(relevance, dtype=np.float64)
    if relevance.size == 0:
        return 0.0
    return float(np.sum(relevance / np.log2(np.arange(2, relevance.size + 2))))


def _binary_auc(scores, labels):
    positive = np.asarray(scores)[np.asarray(labels, dtype=bool)]
    negative = np.asarray(scores)[~np.asarray(labels, dtype=bool)]
    if positive.size == 0 or negative.size == 0:
        return None
    comparisons = positive[:, None] - negative[None, :]
    wins = (comparisons > 0).astype(np.float64)
    ties = (comparisons == 0).astype(np.float64)
    return float(np.mean(wins + 0.5 * ties))


def _tie_averaged_dcg(scores, relevance):
    """Expected DCG under a uniform random ordering inside score ties."""
    scores = np.asarray(scores, dtype=np.float64)
    relevance = np.asarray(relevance, dtype=np.float64)
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_relevance = relevance[order]
    discounts = 1.0 / np.log2(np.arange(2, len(order) + 2))
    total = 0.0
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        total += float(sorted_relevance[start:end].mean() * discounts[start:end].sum())
        start = end
    return total


def _tie_averaged_topk_recall(scores, labels, k):
    """Expected top-k recall under random ordering inside score ties."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    expected_hits = 0.0
    consumed = 0
    start = 0
    while start < len(order) and consumed < int(k):
        end = start + 1
        while end < len(order) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        group_size = end - start
        selected = min(group_size, int(k) - consumed)
        expected_hits += (
            selected * float(sorted_labels[start:end].sum()) / group_size
        )
        consumed += selected
        start = end
    return expected_hits / max(int(k), 1)


def ranking_metrics(rows, score_key, top_fraction=0.1):
    """Macro-average AUC, NDCG, and top-k recall per head observation."""
    groups = defaultdict(list)
    for row in rows:
        key = (
            row["video_id"], row["query_source"], int(row["observation_frame_id"]),
            int(row["layer"]), int(row["kv_head"]),
        )
        groups[key].append(row)
    aucs, ndcgs, recalls = [], [], []
    for group in groups.values():
        if len(group) < 2:
            continue
        truth = np.asarray([
            max(0.0, float(x["future_effective_readout"]))
            / max(1, int(x.get("token_count", 1)))
            for x in group
        ])
        if float(truth.max()) <= 0:
            continue
        score = np.asarray([float(x[score_key]) for x in group])
        ideal = np.argsort(-truth, kind="stable")
        ideal_dcg = _dcg(truth[ideal])
        ndcgs.append(_tie_averaged_dcg(score, truth) / max(ideal_dcg, 1e-12))
        k = max(1, int(math.ceil(len(group) * float(top_fraction))))
        labels = np.zeros(len(group), dtype=bool)
        labels[ideal[:k]] = True
        auc = _binary_auc(score, labels)
        if auc is not None:
            aucs.append(auc)
        recalls.append(_tie_averaged_topk_recall(score, labels, k))
    if not ndcgs:
        raise ValueError("No evaluable groups with positive future readout")
    return {
        "groups": len(ndcgs),
        "frame_auc": float(np.mean(aucs)),
        "ndcg": float(np.mean(ndcgs)),
        "future_topk_recall": float(np.mean(recalls)),
    }
