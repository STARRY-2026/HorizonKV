"""CPU checks for native HERMES continuous real-history evaluation and resume."""

import copy
import re
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from kvbench.hermes_backend.abstract_hermes import Abstract_Hermes
from kvbench.fixed_inputs import fixed_input_metadata
from sparsemm_native import run_ovobench_hermes_fixed896 as fixed


REAL_QUESTION_METADATA = fixed.question_metadata


@pytest.fixture(autouse=True)
def synthetic_question_metadata():
    with patch.object(fixed, "question_metadata", side_effect=lambda q: (q["prompt"], ["one", "two"])):
        yield


def query(qid, frames):
    return dict(query_id=str(qid), source_id=qid, probe_id=None, task="EPM", group="backward",
                timestamp=frames * 2., clip_end=frames * 2., target="A", video_path="source.mp4",
                prompt=f"Question {qid}?\nOptions: A. one; B. two;\nRespond only with a letter.")


class Reader:
    def __init__(self, path, num_threads):
        pass

    def __len__(self):
        return 160

    def get_avg_fps(self):
        return 1.

    def get_batch(self, indices):
        return SimpleNamespace(asnumpy=lambda: np.asarray(indices))


class Model:
    num_layers = 28
    get_prompt = Abstract_Hermes.get_prompt

    def __init__(self, *, corrupt=None):
        self.conv_history, self.fixed_input_checks, self.answers, self.encoded, self.compressions = [], [], [], [], []
        self.inputs = []
        self.corrupt = corrupt

    def clear_cache(self):
        self.kv_cache, self.conv_history = None, []

    def encode_init_prompt(self):
        self.visual_start_idx = 2
        self.kv_cache = [(torch.zeros(1, 4, 2, 1), torch.zeros(1, 4, 2, 1)) for _ in range(28)]
        self._position_ids_cache = [torch.arange(2.).repeat(3, 1) for _ in range(28)]
        self._layer_position_ids = {}
        self.total_processed_frames = self.last_encoded_frames = 0

    def encode_video_chunk(self, frames, is_last_chunk):
        self.encoded.append(frames.tolist())
        self.last_encoded_frames = len(frames)
        self.total_processed_frames += len(frames)
        next_value = frames.float().reshape(1, 1, -1, 1).repeat(1, 4, 1, 1)
        self.kv_cache = [(torch.cat((key, next_value), dim=2), torch.cat((value, next_value + 1), dim=2))
                         for key, value in self.kv_cache]
        self._position_ids_cache = [torch.cat((pos, frames.float().repeat(3, 1)), dim=1)
                                    for pos in self._position_ids_cache]
        self.fixed_input_checks.append(dict(raw_frames=len(frames), visual_tokens=((len(frames) + 1) // 2) * 1024,
                                           grid_thw=[(len(frames) + 1) // 2, 64, 64]))

    def predict_and_compress(self):
        self.compressions.append((self.total_processed_frames, copy.deepcopy(self.conv_history)))
        # Stateful, chunk-sensitive compression makes wrong reuse boundaries
        # and any leaked prior answer visible in the generated response.
        factor = .75 + .01 * self.last_encoded_frames + len(self.conv_history)
        self.kv_cache = [(torch.cat((key[:, :, :2], key[:, :, -5:] * factor), dim=2),
                          torch.cat((value[:, :, :2], value[:, :, -5:] * factor), dim=2))
                         for key, value in self.kv_cache]
        self._position_ids_cache = [torch.cat((pos[:, :2], pos[:, -5:]), dim=1)
                                    for pos in self._position_ids_cache]

    def question_answering(self, input_text, **kwargs):
        assert kwargs.get("max_new_tokens") == 16
        assert kwargs.get("temperature", 0) == 0
        assert kwargs.get("repetition_penalty", 1.1) == 1.1
        assert kwargs.get("pseudo_forward", False) is False
        assert input_text["prompt"].endswith("<|im_start|>assistant\n")
        assert "Best option" not in input_text["prompt"]
        response = f"A {self.kv_cache[0][0].sum().item():.7f} {self.total_processed_frames}"
        self.inputs.append(input_text)
        self.answers.append((input_text["question"], response))
        matches = re.findall(r'\([A-Z]\)\s*(.+?)(?=\n\([A-Z]\)|\nThe best answer|\n*$)',
                             input_text.get("formatted_question", ""), re.DOTALL)
        self.conv_history.append((input_text["question"], response, [x.strip() for x in matches] if matches else None))
        if self.corrupt == "cache":
            self.kv_cache[0][0].add_(1)
        elif self.corrupt == "positions":
            self._position_ids_cache[0].add_(1)
        elif self.corrupt == "frames":
            self.total_processed_frames += 1
        return response


@pytest.fixture
def cpu_runner():
    with patch("sparsemm_native.video_reader.BatchScopedVideoReader", Reader), \
         patch.object(torch.cuda, "synchronize"), patch.object(torch.cuda, "reset_peak_memory_stats"), \
         patch.object(torch.cuda, "max_memory_allocated", return_value=1234):
        yield


def test_continuous_cursor_and_real_history_match_original_analyze_video(cpu_runner):
    from sparsemm_baselines import run_streamingbench as original
    group = [query(i, frames) for i, frames in enumerate((1, 16, 17, 31, 32, 32, 33, 34))]
    shared = Model()
    commits = []

    def commit(row):
        assert shared.total_processed_frames == row["frames"]
        assert len(shared.conv_history) == row["question_history_length"]
        assert shared.conv_history[-1][1] == row["response"]
        commits.append(row["query_id"])

    rows = fixed.run_group(shared, group, on_result=commit)
    native = Model()
    video = dict(video_id="source", video_path="source.mp4", conversations=[
        dict(question=q["prompt"], end_time=q["clip_end"], choices=["one", "two"], answer="one")
        for q in group])
    with patch.object(original, "format_mcqa", side_effect=lambda model, q: dict(
            prompt=model.get_prompt(q["question"], mc=False), question=q["question"],
            formatted_question=fixed.formatted_options(q["choices"]))):
        original.analyze_video(native, video, .5, 16, torch, reader_factory=Reader)
    assert [r["response"] for r in rows] == [response for question, response in native.answers]
    assert shared.encoded == native.encoded
    assert shared.compressions == native.compressions
    for i, (result, q) in enumerate(zip(rows, group)):
        assert result["frames"] == q["clip_end"] // 2
        assert result["question_history_length"] == i + 1
        assert result["validation"]["history_length_before"] == i
        assert result["validation"]["history_length_after"] == i + 1
        assert result["validation"]["history_mode"] == "native_previous_real_qa"
        assert result["validation"]["real_qa_history_preserved"] is True
    assert commits == [str(i) for i in range(8)]
    assert shared.compressions[0][1] == []
    assert len(shared.compressions[1][1]) == 1
    assert len(shared.compressions[-1][1]) == 7
    # Source frames are sampled every 2 raw frames at fps=.5.
    assert shared.encoded == [[0], list(range(2, 32, 2)), [32], list(range(34, 62, 2)), [62], [64], [66]]
    assert sum(row["encoded_frames"] for row in rows) == 34
    assert sum(row["encoded_temporal_units"] for row in rows) == 20
    assert rows[-1]["input_audit"]["video_temporal_units"] == 20
    assert rows[-1]["input_audit"]["temporal_padding_units_from_question_boundaries"] == 3
    assert shared.kv_cache is None and shared.fixed_input_checks == []


def test_resume_replays_completed_boundaries_with_saved_answers_without_regenerating(cpu_runner):
    group = [query(i, frames) for i, frames in enumerate((1, 16, 17, 31, 32, 32, 33, 34))]
    completed = fixed.run_group(Model(), group)
    resumed = Model()
    delivered = []
    rows = fixed.run_group(resumed, group, replay_rows=completed[:4], on_result=delivered.append)
    assert len(resumed.answers) == len(rows) == len(delivered) == 4
    for actual, expected in zip(rows, completed[4:]):
        assert actual["response"] == expected["response"]
        assert actual["input_audit"] == expected["input_audit"]
        assert actual["budget_audit"] == expected["budget_audit"]
        assert actual["validation"] == expected["validation"]
    corrupt = copy.deepcopy(completed[:4])
    corrupt[-1]["input_audit"]["prefix_encoded_visual_tokens"] += 1024
    with pytest.raises(ValueError, match="Replayed video state"):
        fixed.run_group(Model(), group, replay_rows=corrupt)


def test_large_question_delta_uses_native_sixteen_frame_chunks_without_tail_replay(cpu_runner):
    model = Model()
    rows = fixed.run_group(model, [query(i, n) for i, n in enumerate((17, 50, 66))])
    assert [len(x) for x in model.encoded] == [16, 1, 16, 16, 1, 16]
    assert [row["encoded_frames"] for row in rows] == [17, 33, 16]
    assert [row["encoded_temporal_units"] for row in rows] == [9, 17, 8]
    assert [len(history) for frames, history in model.compressions] == [0, 0, 1, 1, 1, 2]


@pytest.mark.parametrize("corrupt", ["cache", "positions", "frames"])
def test_native_qa_state_corruption_fails_and_restores_everything(corrupt):
    model = Model(corrupt=corrupt)
    model.encode_init_prompt()
    model.fixed_input_checks = []
    before = fixed.snapshot_state(model, clone=True)
    with pytest.raises(AssertionError, match="failed to restore"):
        fixed.answer_native(model, query(0, 1))
    assert fixed._cache_equal(model.kv_cache, before["kv_cache"])
    assert fixed._nested_equal(model._position_ids_cache, before["_position_ids_cache"])
    assert model.total_processed_frames == 0
    assert model.conv_history == []


def test_native_qa_keeps_real_question_and_answer_for_next_compression():
    model = Model()
    model.encode_init_prompt()
    model.fixed_input_checks = []
    first, flags = fixed.answer_native(model, query(0, 1))
    second, flags2 = fixed.answer_native(model, query(0, 1))
    assert first == second
    assert flags["history_length_after"] == 1
    assert flags2["history_length_before"] == 1
    assert flags2["history_length_after"] == 2
    assert model.conv_history == [(query(0, 1)["prompt"], first, ["one", "two"])] * 2


def test_raw_question_and_options_metadata_do_not_modify_official_generation_prompt():
    model = Model()
    model.encode_init_prompt()
    model.fixed_input_checks = []
    q = query(0, 1)
    with patch.object(fixed, "question_metadata", return_value=("RAW_QUESTION_METADATA", ["red", "blue"])):
        response, flags = fixed.answer_native(model, q)
    assert model.inputs[-1]["prompt"] == model.get_prompt(q["prompt"], mc=False)
    assert "RAW_QUESTION_METADATA" not in model.inputs[-1]["prompt"]
    assert model.conv_history == [("RAW_QUESTION_METADATA", response, ["red", "blue"])]
    assert flags["raw_question_and_options_saved"] is True


def test_question_metadata_reads_official_source_id_and_native_option_stripping(tmp_path):
    import json
    path = tmp_path / "official.json"
    path.write_text(json.dumps([dict(id=17, task="EPM", question="Original question?", options=[" red ", "blue"])]))
    fixed.source_question_metadata.cache_clear()
    try:
        with patch.object(fixed.official, "ANNOTATIONS", path):
            assert REAL_QUESTION_METADATA(dict(source_id=17)) == ("Original question?", ["red", "blue"])
    finally:
        fixed.source_question_metadata.cache_clear()


def test_physical_budget_counts_all_shared_heads_and_deep_summary_slot():
    prefix = 14
    # Deep layer's 499 ordinary positions plus its summary are all physical
    # visual slots. A short-prefix layer may legitimately use fewer than 500.
    sizes = [500] * 27 + [493]
    model = SimpleNamespace(visual_start_idx=prefix,
        kv_cache=[(torch.zeros(1, 4, n + prefix, 2, dtype=torch.float16),
                   torch.zeros(1, 4, n + prefix, 2, dtype=torch.float16)) for n in sizes])
    audit = fixed.cache_budget_audit(model)
    assert audit["actual_tokens_per_head"] == sum(sizes) / 28
    assert audit["actual_visual_slots"] == sum(sizes) * 4
    assert audit["per_head_visual_slots"][-1] == [493] * 4
    assert audit["physical_kv_bytes"] == sum((n + prefix) * 4 * 2 * 2 * 2 for n in sizes)
    assert audit["actual_includes_question_tokens"] is False
    assert audit["native_summary_tokens_included"] is True


def test_fixed_wrapper_keeps_one_native_forward_and_audits_each_temporal_unit():
    from kvbench.fixed_inputs import frame_rgb_sha256, prepare_fixed_frames

    class PixelModel:
        def __init__(self):
            self.config = SimpleNamespace(vision_config=SimpleNamespace(
                patch_size=14, spatial_merge_size=2, temporal_patch_size=2))
            self.processor = SimpleNamespace(image_processor=SimpleNamespace(do_resize=True),
                                             video_processor=SimpleNamespace(do_resize=True))
            self.total_processed_frames = 16
            self._position_ids_cache = [torch.empty(3, 0)] * 28
            self.seen = []

        def get_video_features(self, pixels, grid):
            self.seen.append((pixels.clone(), grid.clone()))
            return (torch.zeros(int(grid[0, 0]) * 1024, 1),)

        def encode_video_chunk(self, frames, is_last_chunk=False):
            assert tuple(frames.shape) == (3, 896, 896, 3)
            grid = torch.tensor([[2, 64, 64]])
            self.get_video_features(frames, grid)
            self._position_ids_cache = [torch.arange(2048.).repeat(3, 1) for _ in range(28)]
            self.total_processed_frames += len(frames)

    model = fixed.install_fixed_input(PixelModel())
    raw = torch.arange(3 * 2 * 4 * 3).to(torch.uint8).reshape(3, 2, 4, 3)
    prepared = prepare_fixed_frames(raw, canvas_size=896, pad_value=127)
    model.encode_video_chunk(raw, is_last_chunk=True)
    assert len(model.seen) == 1
    assert torch.equal(model.seen[0][0], prepared)
    audit = model.fixed_input_checks[0]
    assert audit["grid_thw"] == [2, 64, 64]
    assert audit["visual_tokens"] == 2048
    assert audit["raw_frames"] == 3
    assert audit["sampled_ordinal_start"] == 16
    assert audit["original_source_hw"] == [2, 4]
    assert audit["prepared_rgb_sha256"] == frame_rgb_sha256(prepared)
    assert audit["temporal_unit_rgb_sha256"] == [frame_rgb_sha256(prepared[:2]), frame_rgb_sha256(prepared[2:])]
    assert len(audit["position_sha256_by_layer"]) == 28
    assert model.processor.video_processor.do_resize is False


def test_manifest_rejects_wrong_official_identity_and_decreasing_endpoints(tmp_path):
    # Use a small stand-in official set without weakening the actual 1468/512
    # source guard: repeat distinct videos to satisfy the source dimensions.
    expected = []
    for i in range(1468):
        q = query(i, i % 3 + 1)
        q["video_path"] = f"v{i % 512}.mp4"
        expected.append(q)
    path = tmp_path / "manifest.json"
    first = dict(expected[0])
    second = dict(expected[512])
    first["clip_end"] = expected[0]["clip_end"]
    import json
    with patch.object(fixed.official, "queries", return_value=expected):
        path.write_text(json.dumps([second, first]))
        with pytest.raises(ValueError, match="prefix length"):
            fixed.load_manifest(path)
        first["target"] = "C"
        path.write_text(json.dumps([first]))
        with pytest.raises(ValueError, match="differs"):
            fixed.load_manifest(path)
