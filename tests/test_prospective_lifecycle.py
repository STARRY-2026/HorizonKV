import json

import pytest
import torch

from head_analysis.hermes_kv_head_ragged_prefill import (
    _write_prospective_lifecycle_trace,
)
from head_analysis.prospective_lifecycle import (
    add_remaining_future_utility,
    align_predictor_to_future_targets,
    assign_scores,
    fit_remaining_lifecycle,
    load_trace_jsonl,
    ranking_metrics,
    validate_trace_record,
    _tie_averaged_dcg,
    _tie_averaged_topk_recall,
)


def frame(source, age, salience, readout):
    return {
        "source_frame_id": source,
        "age_sec": age,
        "token_count": 2,
        "arrival_salience_top_p": salience,
        "value_l1_mean": 1.0,
        "attention_mass": readout,
        "effective_readout_mass": readout,
    }


def record(observation, frames, full_cache=True):
    return {
        "schema_version": 1,
        "video_id": 7,
        "question_index": 0,
        "query_source": "pseudo_mixed",
        "observation_frame_id": observation,
        "sample_fps": 1.0,
        "full_cache": full_cache,
        "heads": [{"layer": 0, "kv_head": 0, "frames": frames}],
    }


def test_schema_rejects_future_frame_and_wrong_age():
    with pytest.raises(ValueError, match="later than"):
        validate_trace_record(record(2, [frame(3, -1.0, 0.5, 0.1)]))
    with pytest.raises(ValueError, match="age_sec"):
        validate_trace_record(record(2, [frame(1, 2.0, 0.5, 0.1)]))


def test_loader_rejects_survivor_biased_compressed_trace(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text(json.dumps(record(2, [frame(1, 1.0, 0.5, 0.1)], False)) + "\n")
    with pytest.raises(ValueError, match="full-cache replay"):
        load_trace_jsonl(path)


def test_loader_expands_within_frame_content_bins(tmp_path):
    item = record(2, [frame(1, 1.0, 0.5, 0.1)])
    item["heads"][0]["frames"][0]["content_bins"] = [
        {
            "content_bin": 0, "token_count": 1,
            "arrival_salience_mean": 0.1, "value_l1_mean": 1.0,
            "attention_mass": 0.01, "effective_readout_mass": 0.01,
        },
        {
            "content_bin": 1, "token_count": 1,
            "arrival_salience_mean": 0.9, "value_l1_mean": 1.0,
            "attention_mass": 0.09, "effective_readout_mass": 0.09,
        },
    ]
    path = tmp_path / "trace.jsonl"
    path.write_text(json.dumps(item) + "\n")
    rows = load_trace_jsonl(path)
    assert [row["content_bin"] for row in rows] == [0, 1]
    assert [row["token_count"] for row in rows] == [1, 1]


def test_runtime_writer_preserves_source_frame_provenance(tmp_path):
    class Layer:
        def get_horizon_aux_segment(self, kv_head):
            return torch.tensor([-1, 3, 3, 5]), torch.tensor([0.0, 0.2, 0.8, 0.5])

        def get_segment(self, kv_head):
            values = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
            return torch.zeros_like(values), values

    class Cache:
        layers = [Layer()]

    class Model:
        kv_cache = Cache()
        sample_fps = 0.5
        _active_video_id = 9
        _active_question_index = 1
        _prospective_lifecycle_trace_config = {
            "trace_path": str(tmp_path / "prospective.jsonl"),
            "score_last_n": 1,
            "top_fraction": 0.5,
            "current_frame_id": 5,
            "cache_compressed": False,
        }

    scores = [[torch.tensor([0.0, 0.1, 0.2, 0.7])]]
    _write_prospective_lifecycle_trace(Model(), scores, "pseudo_mixed")
    traced = json.loads((tmp_path / "prospective.jsonl").read_text())
    validate_trace_record(traced)
    frames = traced["heads"][0]["frames"]
    assert [item["source_frame_id"] for item in frames] == [3, 5]
    assert [item["age_sec"] for item in frames] == [4.0, 0.0]
    assert frames[0]["arrival_salience_top_p"] == pytest.approx(0.8)
    assert frames[0]["effective_readout_mass"] == pytest.approx(0.8)


def test_remaining_utility_uses_only_strictly_future_reads():
    rows = [
        {"video_id": "v", "layer": 0, "kv_head": 0, "source_frame_id": 0,
         "query_source": "pseudo_mixed", "observation_frame_id": 0,
         "effective_readout_mass": 1.0},
        {"video_id": "v", "layer": 0, "kv_head": 0, "source_frame_id": 0,
         "query_source": "pseudo_mixed", "observation_frame_id": 1,
         "effective_readout_mass": 2.0},
        {"video_id": "v", "layer": 0, "kv_head": 0, "source_frame_id": 0,
         "query_source": "pseudo_mixed", "observation_frame_id": 2,
         "effective_readout_mass": 4.0},
    ]
    enriched = sorted(add_remaining_future_utility(rows), key=lambda x: x["observation_frame_id"])
    assert [row["future_effective_readout"] for row in enriched] == [6.0, 4.0, 0.0]


def test_hermes_alignment_includes_question_at_same_observation():
    common = {
        "video_id": "v", "layer": 0, "kv_head": 0,
        "source_frame_id": 1, "content_bin": 2, "token_count": 1,
    }
    rows = [
        {**common, "query_source": "hermes", "observation_frame_id": 4,
         "attention_density": 0.25},
        {**common, "query_source": "true_question", "observation_frame_id": 4,
         "effective_readout_mass": 2.0},
        {**common, "query_source": "true_question", "observation_frame_id": 8,
         "effective_readout_mass": 3.0},
    ]
    aligned = align_predictor_to_future_targets(rows)
    assert aligned[0]["future_effective_readout"] == 5.0
    assert aligned[0]["score_hermes"] == pytest.approx(-1.38629436112)


def test_hermes_alignment_can_use_strictly_future_questions():
    common = {
        "video_id": "v", "layer": 0, "kv_head": 0,
        "source_frame_id": 1, "content_bin": 2, "token_count": 1,
    }
    rows = [
        {**common, "query_source": "hermes", "observation_frame_id": 4,
         "attention_density": 0.25},
        {**common, "query_source": "true_question", "observation_frame_id": 4,
         "effective_readout_mass": 2.0},
        {**common, "query_source": "true_question", "observation_frame_id": 8,
         "effective_readout_mass": 3.0},
    ]
    aligned = align_predictor_to_future_targets(
        rows, include_same_observation=False
    )
    assert aligned[0]["future_effective_readout"] == 3.0


def test_ranking_metrics_average_ties_instead_of_using_source_order():
    assert _tie_averaged_dcg([1.0, 1.0], [2.0, 0.0]) == pytest.approx(
        _tie_averaged_dcg([1.0, 1.0], [0.0, 2.0])
    )
    assert _tie_averaged_topk_recall([1.0, 1.0], [True, False], 1) == 0.5


def test_head_lifecycle_can_beat_global_on_held_out_rankings():
    rows = []
    for video in ("train", "test"):
        for head, late in ((0, False), (1, True)):
            for source, age, salience in ((0, 64.0, 1.0), (1, 4.0, 0.5)):
                if source == 1:
                    future = 1.0
                else:
                    future = 0.8 if late else 0.1
                rows.append({
                    "video_id": video, "query_source": "pseudo_mixed",
                    "observation_frame_id": 100, "layer": 0, "kv_head": head,
                    "source_frame_id": source, "age_sec": age,
                    "arrival_salience_top_p": salience,
                    "future_effective_readout": future,
                })
    profiles = fit_remaining_lifecycle(rows, {"train"})
    scored = assign_scores([row for row in rows if row["video_id"] == "test"], profiles)
    head = ranking_metrics(scored, "score_head")
    content = ranking_metrics(scored, "score_content")
    assert head["ndcg"] > content["ndcg"]
