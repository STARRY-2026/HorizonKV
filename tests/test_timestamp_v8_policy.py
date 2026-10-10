import numpy as np
import torch
import json

from head_analysis.timestamp_v8_policy import (
    CURRENT_ROLE,
    MEMORY_ROLE,
    build_source_frame_ids,
    load_v8_head_profiles,
    select_annealed_recency_tokens,
    select_recent_frames,
    select_temporal_utility_tokens,
    select_prospective_lifecycle_tokens,
    select_temporal_survival_tokens,
    select_temporal_residual_tokens,
    select_temporal_zone_tokens,
    select_timestamp_v8_keep_indices,
    write_ragged_frame_trace,
    _percentile_rank_tensor,
)


def test_prospective_lifecycle_uses_content_and_age_bins():
    visual = torch.arange(3)
    frame_ids = torch.tensor([10, 0, 10])
    admission = torch.tensor([0.1, 0.1, 0.9])
    lifecycle = torch.ones((4, 8))
    lifecycle[0, 0] = 0.1
    lifecycle[0, 2] = 0.9
    lifecycle[3, 0] = 0.8

    selected, counts = select_prospective_lifecycle_tokens(
        visual_indices=visual,
        frame_ids=frame_ids,
        admission_scores=admission,
        current_frame_id=10,
        sample_fps=1.0,
        budget=2,
        content_priors=[1.0, 1.0, 1.0, 1.0],
        lifecycle_values=lifecycle,
        lifecycle_weight=1.0,
    )

    assert selected.tolist() == [1, 2]
    assert counts == (2, 0, 0)


def test_annealed_recency_yields_to_attention_and_honors_protection():
    visual = torch.arange(4)
    frame_ids = torch.tensor([100, 98, 90, 80], dtype=torch.int32)
    attention = torch.tensor([0.0, 0.1, 0.8, 0.9])
    recency = torch.tensor([1.0, 0.8, 0.2, 0.0])
    chosen, counts = select_annealed_recency_tokens(
        visual,
        frame_ids,
        attention,
        recency,
        current_frame_id=100,
        sample_fps=1.0,
        budget=2,
        alpha_initial=0.8,
        alpha_floor=0.0,
        alpha_tau_sec=2.0,
        protected=torch.tensor([0]),
    )
    assert chosen.tolist() == [0, 3]
    assert counts == (1, 1, 0)


def test_constant_annealed_alpha_matches_fixed_combination():
    visual = torch.arange(4)
    frame_ids = torch.tensor([100, 98, 90, 80], dtype=torch.int32)
    attention = torch.tensor([0.0, 0.1, 0.8, 0.9])
    recency = torch.tensor([1.0, 0.8, 0.2, 0.0])
    chosen, _ = select_annealed_recency_tokens(
        visual,
        frame_ids,
        attention,
        recency,
        current_frame_id=100,
        sample_fps=1.0,
        budget=2,
        alpha_initial=0.5,
        alpha_floor=0.5,
        alpha_tau_sec=8.0,
        alpha_grace_sec=2.0,
        protected=torch.empty(0, dtype=torch.long),
    )
    expected = torch.topk(0.5 * attention + 0.5 * recency, 2).indices.sort().values
    assert chosen.tolist() == expected.tolist()


def test_temporal_utility_uses_one_shared_budget():
    visual = torch.arange(6)
    frame_ids = torch.tensor([100, 98, 96, 80, 60, 20], dtype=torch.int32)
    utility = torch.tensor([0.1, 0.2, 0.3, 0.8, 0.7, 0.6])
    selected, counts = select_temporal_utility_tokens(
        visual,
        frame_ids,
        utility,
        current_frame_id=100,
        sample_fps=1.0,
        budget=3,
        temporal_values=np.ones(7),
        current_affinity=0.0,
        temporal_weight=0.2,
        novelty_weight=0.0,
        novelty_tau_sec=4.0,
    )
    assert selected.tolist() == [3, 4, 5]
    assert counts == (0, 2, 1)


def test_temporal_utility_novelty_can_retain_new_token():
    visual = torch.arange(3)
    frame_ids = torch.tensor([100, 90, 80], dtype=torch.int32)
    utility = torch.tensor([0.1, 0.9, 0.8])
    selected, _ = select_temporal_utility_tokens(
        visual,
        frame_ids,
        utility,
        current_frame_id=100,
        sample_fps=1.0,
        budget=1,
        temporal_values=np.zeros(7),
        current_affinity=1.0,
        temporal_weight=0.0,
        novelty_weight=2.0,
        novelty_tau_sec=4.0,
    )
    assert selected.tolist() == [0]


def test_temporal_survival_combines_content_and_remaining_lifetime():
    visual = torch.arange(4)
    frame_ids = torch.tensor([100, 96, 60, 20], dtype=torch.int32)
    admission = torch.tensor([0.7, 0.6, 0.9, 0.8])
    probe = admission.clone()
    selected, counts, updated = select_temporal_survival_tokens(
        visual, frame_ids, admission, probe,
        current_frame_id=100,
        sample_fps=1.0,
        budget=2,
        survival_values=np.asarray([1.0, 0.9, 0.8, 0.7, 0.5, 0.2, 0.1]),
        temporal_weight=1.0,
        ema_decay=1.0,
    )
    assert selected.tolist() == [0, 1]
    assert counts == (2, 0, 0)
    torch.testing.assert_close(updated, admission)


def test_relative_temporal_uses_head_deviation_from_global_curve():
    visual = torch.arange(3)
    frame_ids = torch.tensor([100, 60, 20], dtype=torch.int32)
    content = torch.tensor([0.5, 0.5, 0.5])
    selected, _, _ = select_temporal_survival_tokens(
        visual, frame_ids, content, content,
        current_frame_id=100,
        sample_fps=1.0,
        budget=1,
        survival_values=np.asarray([1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8]),
        global_survival_values=np.asarray([1.0, 1.0, 1.0, 1.0, 0.5, 0.3, 0.2]),
        temporal_weight=0.2,
        relative_temporal=True,
        temporal_clip=0.5,
        ema_decay=1.0,
    )
    assert selected.tolist() == [2]


def test_percentile_rank_tensor_averages_ties():
    ranks = _percentile_rank_tensor(torch.tensor([0.0, 0.0, 1.0, 2.0]))
    torch.testing.assert_close(
        ranks, torch.tensor([1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0, 1.0])
    )


def test_zero_temporal_residual_preserves_hermes_topk():
    visual = torch.arange(6)
    frame_ids = torch.tensor([100, 98, 96, 80, 60, 20], dtype=torch.int32)
    utility = torch.tensor([0.1, 0.9, 0.4, 0.8, 0.2, 0.7])
    selected, _ = select_temporal_residual_tokens(
        visual,
        frame_ids,
        utility,
        current_frame_id=100,
        sample_fps=1.0,
        budget=3,
        temporal_values=np.asarray([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]),
        global_temporal_values=np.asarray([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
        residual_weight=0.0,
    )
    assert selected.tolist() == [1, 3, 5]


def test_original_v8_roles_and_ratios():
    profile = load_v8_head_profiles(
        "results/myobservations/relocatev3/v8_budget.csv"
    )
    assert profile["role_counts"] == {
        MEMORY_ROLE: 25,
        "mixed": 62,
        CURRENT_ROLE: 25,
    }
    ratios = profile["attention_ratios"]
    assert ratios.shape == (28, 4)
    assert np.all((0.0 <= ratios) & (ratios <= 1.0))
    assert np.all(ratios[profile["roles"] == MEMORY_ROLE] == 1.0)
    assert np.all(ratios[profile["roles"] == CURRENT_ROLE] == 0.0)


def test_temporal_zone_profile_schema(tmp_path):
    path = tmp_path / "zones.csv"
    rows = ["layer,kv_head,recent_ratio,middle_ratio,far_ratio"]
    for layer in range(2):
        for head in range(2):
            rows.append(f"{layer},{head},2,1,1")
    path.write_text("\n".join(rows) + "\n")
    profile = load_v8_head_profiles(path, num_layers=2, num_kv_heads=2)
    assert profile["policy_mode"] == "temporal_zones"
    np.testing.assert_allclose(
        profile["temporal_ratios"],
        np.tile([0.5, 0.25, 0.25], (2, 2, 1)),
    )


def test_temporal_zone_selective_protection_schema(tmp_path):
    path = tmp_path / "selective.csv"
    rows = [
        "layer,kv_head,recent_ratio,middle_ratio,far_ratio,protect_latest,current_affinity"
    ]
    for layer in range(2):
        for head in range(2):
            rows.append(f"{layer},{head},2,1,1,{int(layer == head)},0.2")
    path.write_text("\n".join(rows) + "\n")
    profile = load_v8_head_profiles(path, num_layers=2, num_kv_heads=2)
    assert profile["protect_latest"].tolist() == [[True, False], [False, True]]


def test_temporal_zones_assign_by_seconds_and_spill_budget():
    visual = torch.arange(8)
    frame_ids = torch.tensor([100, 98, 94, 93, 80, 70, 60, 40], dtype=torch.int32)
    utility = torch.arange(8, dtype=torch.float32)
    selected, counts = select_temporal_zone_tokens(
        visual, frame_ids, utility,
        current_frame_id=100,
        sample_fps=0.5,
        budget=6,
        ratios=np.asarray([0.5, 0.25, 0.25]),
        protected=torch.tensor([0]),
    )
    assert selected.numel() == 6
    assert counts == (3, 2, 1)


def test_temporal_zone_protection_is_capped_by_recent_budget():
    visual = torch.arange(12)
    frame_ids = torch.tensor(
        [100, 100, 100, 100, 95, 95, 90, 90, 90, 60, 60, 60],
        dtype=torch.int32,
    )
    selected, _ = select_temporal_zone_tokens(
        visual, frame_ids, torch.arange(12, dtype=torch.float32),
        current_frame_id=100,
        sample_fps=0.5,
        budget=6,
        ratios=np.asarray([0.2, 0.4, 0.4]),
        protected=torch.arange(4),
    )
    assert selected.numel() == 6
    # Recent receives one of six slots, so only its highest-utility protected
    # candidate is guaranteed; Middle/Far quotas are never borrowed.
    assert 3 in selected.tolist()
    assert len(set(range(4)) & set(selected.tolist())) == 1


def test_frame_ids_map_temporal_patches_to_latest_raw_frame():
    ids = build_source_frame_ids(
        q_len=12,
        temporal_grid=3,
        raw_frame_count=6,
        first_frame_id=10,
    )
    assert ids.tolist() == [11] * 4 + [13] * 4 + [15] * 4


def test_recent_selection_keeps_whole_frames_then_attention_boundary():
    history = torch.arange(8)
    frame_ids = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2], dtype=torch.int32)
    utility = torch.tensor([0.0, 0.0, 0.0, 0.1, 0.9, 0.2, 0.8, 0.3])
    selected = select_recent_frames(history, frame_ids, utility, target_k=4)
    assert selected.tolist() == [4, 5, 6, 7]


def test_selector_protects_current_and_respects_per_head_budget():
    class Layer:
        def __init__(self):
            self.head_lens = torch.tensor([8, 8], dtype=torch.int32)
            self.ids = torch.tensor(
                [-1, -1, 0, 0, 1, 1, 2, 2], dtype=torch.int32
            )

        def get_horizon_aux_segment(self, kv_head):
            return self.ids, torch.zeros_like(self.ids, dtype=torch.float32)

    class Cache:
        layers = [Layer()]

    class Model:
        device = torch.device("cpu")
        kv_cache = Cache()
        short_term_threshold = 8
        long_term_threshold = 19
        _timestamp_v8_config = {
            "current_frame_id": 2,
            "grace_frames": 1,
            "attention_ratios": np.asarray([[1.0, 0.0]]),
        }

    # Head 0 Attention picks frame-0 tokens; Head 1 Recency keeps frame 1.
    local_scores = [[
        torch.tensor([0.0, 0.0, 0.9, 0.8, 0.2, 0.1, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.9, 0.8, 0.2, 0.1, 0.0, 0.0]),
    ]]
    keep, _ = select_timestamp_v8_keep_indices(
        Model(), local_scores, local_scores, local_scores,
        budget_table=np.asarray([[4, 4]]),
    )
    assert keep[0][0].tolist() == [0, 1, 2, 3, 6, 7]
    assert keep[0][1].tolist() == [0, 1, 4, 5, 6, 7]


def test_frame_trace_records_before_and_after_histograms(tmp_path):
    class Layer:
        head_lens = torch.tensor([6], dtype=torch.int32)
        ids = torch.tensor([-1, 0, 0, 1, 1, 2], dtype=torch.int32)

        def get_horizon_aux_segment(self, kv_head):
            return self.ids, torch.zeros_like(self.ids, dtype=torch.float32)

    class Cache:
        layers = [Layer()]

    trace_path = tmp_path / "trace.jsonl"

    class Model:
        kv_cache = Cache()
        sample_fps = 0.5
        _ragged_frame_trace_is_last_chunk = True
        _active_video_id = 237
        _active_question_index = 4
        _active_question = "How many clips?"
        _active_question_end_time = 333.0
        _ragged_frame_trace_config = {
            "trace_path": str(trace_path),
            "current_frame_id": 2,
            "next_frame_id": 3,
        }

    write_ragged_frame_trace(
        Model(), [[torch.tensor([0, 2, 4, 5])]],
        np.asarray([[3]]), "default-ragged",
    )
    record = json.loads(trace_path.read_text())
    assert record["video_id"] == 237
    assert record["heads"][0]["before_frame_tokens"] == {"0": 2, "1": 2, "2": 1}
    assert record["heads"][0]["after_frame_tokens"] == {"0": 1, "1": 1, "2": 1}
