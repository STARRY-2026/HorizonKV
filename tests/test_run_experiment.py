"""Release protocol integrity, real-prefix selection, and CPU-only preflight."""

import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_experiment
from run_experiment import budget_preflight, compression_evidence, model_preflight, read_heads, select_smoke
from export_frozen_dp_profiles import derive


REGISTRY = json.loads((ROOT / "configs/experiments/protocols.json").read_text())["methods"]


@pytest.mark.parametrize("budget", [100, 250, 500, 800, 1000])
def test_dp_runtime_matches_frozen_allocation_and_banks(budget):
    _, report, _ = budget_preflight(REGISTRY["dp"], budget, strict=True)
    _, _, _, profiles = derive(budget)
    assert report["per_head_visual_slots"] == [row["capacity_n"] for row in profiles]
    assert report["runtime_total_visual_slots"] == 112 * budget
    assert all(row["recent_r"] + row["history_q"] == row["capacity_n"] for row in profiles)


def test_dense_and_uniform_paths_remain_distinct():
    dense, dense_report, _ = budget_preflight(REGISTRY["hermes"], 1000)
    uniform, uniform_report, _ = budget_preflight(REGISTRY["uniform"], 1000)
    assert dense == {"kv_head_ragged_prefill": False}
    assert uniform["kv_head_ragged_prefill"] is True
    assert "kv_head_budget_scores" in uniform
    assert dense_report["runtime_total_visual_slots"] == uniform_report["runtime_total_visual_slots"] == 112000
    assert dense_report["rounding_delta"] == 0


def test_horizon_uses_real_fixed_region_policy_and_rejects_rescaling():
    options, report, _ = budget_preflight(REGISTRY["horizonkv"], 6000, strict=True)
    assert "horizonkv_policy" in options and "timestamp_v8_policy" not in options
    assert report["runtime_total_visual_slots"] == 672000
    with pytest.raises(ValueError, match="supports only"):
        budget_preflight(REGISTRY["horizonkv"], 1000)


def test_historical_v8_rounding_is_reported_and_can_be_rejected():
    _, report, _ = budget_preflight(REGISTRY["v8"], 1000)
    assert report["rounding_delta"] == 4
    assert report["exact_conservation"] is False
    with pytest.raises(ValueError, match="Budget mismatch"):
        budget_preflight(REGISTRY["v8"], 1000, strict=True)
    with pytest.raises(ValueError, match="supports only"):
        budget_preflight(REGISTRY["v8"], 500)


def test_memheadkv_native_allocation_is_distinct_from_timestamp_policy():
    native, _, _ = budget_preflight(REGISTRY["v8"], 1000)
    temporal, _, _ = budget_preflight(REGISTRY["timestamp-v8"], 1000)
    assert "timestamp_v8_policy" not in native and "fixed_frame_canvas_size" not in native
    assert "timestamp_v8_policy" in temporal and temporal["fixed_frame_canvas_size"] == 448
    assert native["kv_head_budget_scores"] == temporal["kv_head_budget_scores"]


def test_smoke_preserves_real_questions_and_complete_prefixes():
    data = json.loads((ROOT / "data/streamingbench/streamingbench_realtime_half.json").read_text())
    original = copy.deepcopy(data)
    selected, indices = select_smoke(data)
    assert data == original
    assert len(selected) == 1 and len(selected[0]["conversations"]) == 1
    index = indices[0]
    question = selected[0]["conversations"][0]
    assert question == data[index["source_video_index"]]["conversations"][index["source_question_indices"][0]]
    assert 64 <= question["end_time"] <= 120
    assert selected[0]["clip"] == [0, question["end_time"]]
    assert index["unused_suffix_trimmed"] is True


def test_smoke_does_not_rewrite_existing_clip():
    data = [{"video_id": 1, "clip": [10, 200], "conversations": [{"end_time": 70}]}]
    selected, indices = select_smoke(data)
    assert selected[0]["clip"] == [10, 200]
    assert indices[0]["unused_suffix_trimmed"] is False


def test_short_annotation_is_rejected_instead_of_fabricating_long_timestamp():
    with pytest.raises(ValueError, match="real questions"):
        select_smoke([{"conversations": [{"end_time": 1}]}])


@pytest.mark.parametrize("method,marker", [
    ("hermes", "Strict-shrunk KV cache. New length:"),
    ("dp", "[kv_head_ragged_prefill] prune stats:"),
    ("v8", "[kv_head_ragged_prefill] prune stats:"),
    ("timestamp-v8", "[TimestampV8] prune stats:"),
    ("horizonkv", "[HorizonKV] prune stats:"),
])
def test_smoke_requires_actual_compression_instead_of_policy_installation(method, marker):
    installation = "[HorizonKV] Installed policy\n[TimestampV8] Installed policy\n[KV] no compression needed\n"
    assert compression_evidence(method, installation)["events"] == 0
    assert compression_evidence(method, installation + marker + "\n" + marker)["events"] == 2


def test_profile_requires_all_physical_heads_and_unique_keys(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("layer,kv_head,score\n0,0,1\n0,0,2\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_heads(path, ["score"])
    path.write_text("layer,kv_head,score\n0,0,1\n")
    with pytest.raises(ValueError, match="112"):
        read_heads(path, ["score"])


def test_dry_run_works_without_torch_or_gpu_and_preserves_source(tmp_path):
    # The standard-library interpreter may have no ML dependencies. No model
    # weights or video are opened during a dry run; all assets still must exist.
    model = tmp_path / "model"
    model.mkdir()
    config = {"model_type": "qwen2_5_vl", "num_hidden_layers": 28,
              "num_attention_heads": 28, "num_key_value_heads": 4}
    for name, value in (("config.json", config), ("tokenizer_config.json", {}), ("preprocessor_config.json", {})):
        (model / name).write_text(json.dumps(value))
    (model / "model.safetensors").touch()
    video_root = tmp_path / "videos"
    video_root.mkdir()
    (video_root / "real.mp4").touch()
    source = tmp_path / "source.json"
    data = [{"video_id": 42, "video_path": "/old/location/real.mp4", "conversations": [
        {"question": "Real?", "answer": "Yes", "choices": ["Yes", "No"], "end_time": 70}]}]
    source.write_text(json.dumps(data))
    before = source.read_bytes()
    output = tmp_path / "output"
    command = [sys.executable, "-S", str(ROOT / "tools/run_experiment.py"), "--method", "dp", "--budget", "800",
               "--model-path", str(model), "--annotation", str(source), "--video-root", str(video_root),
               "--output-dir", str(output), "--dry-run"]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    manifest = json.loads((output / "experiment_manifest.json").read_text())
    assert manifest["status"] == "preflight_passed"
    assert manifest["annotation"]["questions"] == 1
    assert manifest["budget_check"]["runtime_total_visual_slots"] == 89600
    assert manifest["runtime_dtype"] == "float16"
    assert manifest["environment"]["PROTOKV_ENABLE"] == "0"
    assert manifest["environment"]["INFINIPOT_V_PUBLIC"] == "0"
    assert manifest["environment"]["PYTHONPATH"] == str(ROOT)
    assert source.read_bytes() == before
    assert not (output / "inference.log").exists()
    assert not list(output.glob("*.csv"))


def test_missing_weight_shard_is_rejected(tmp_path):
    config = {"model_type": "qwen2_5_vl", "num_hidden_layers": 28,
              "num_attention_heads": 28, "num_key_value_heads": 4}
    for name, value in (("config.json", config), ("tokenizer_config.json", {}), ("preprocessor_config.json", {}),
                        ("model.safetensors.index.json", {"weight_map": {"weight": "missing.safetensors"}})):
        (tmp_path / name).write_text(json.dumps(value))
    with pytest.raises(ValueError, match="weight shards missing"):
        model_preflight(tmp_path)


@pytest.mark.parametrize("filename", ["annotation.json", "annotation.source.json"])
def test_output_directory_cannot_overwrite_source_annotation(tmp_path, filename):
    model = tmp_path / "model"
    model.mkdir()
    config = {"model_type": "qwen2_5_vl", "num_hidden_layers": 28,
              "num_attention_heads": 28, "num_key_value_heads": 4}
    for name, value in (("config.json", config), ("tokenizer_config.json", {}), ("preprocessor_config.json", {})):
        (model / name).write_text(json.dumps(value))
    (model / "model.safetensors").touch()
    output = tmp_path / "output"
    output.mkdir()
    source = output / filename
    source.write_text(json.dumps([{"video_path": "/unused.mp4", "conversations": [
        {"question": "Real?", "answer": "Yes", "choices": ["Yes", "No"], "end_time": 70}]}]))
    original = source.read_bytes()
    completed = subprocess.run([sys.executable, "-S", str(ROOT / "tools/run_experiment.py"),
                                "--method", "hermes", "--model-path", str(model), "--annotation", str(source),
                                "--output-dir", str(output), "--dry-run"], capture_output=True, text=True)
    assert completed.returncode != 0
    assert "cannot contain the source" in completed.stderr
    assert source.read_bytes() == original
    assert list(output.iterdir()) == [source]


@pytest.mark.parametrize("compressed", [False, True])
def test_successful_child_and_complete_predictions_do_not_bypass_smoke_compression(tmp_path, monkeypatch, compressed):
    video = tmp_path / "real.mp4"
    video.touch()
    source = tmp_path / "input.json"
    source.write_text(json.dumps([{"video_path": str(video), "conversations": [
        {"question": "Real?", "answer": "Yes", "choices": ["Yes", "No"], "end_time": 70}]}]))
    output = tmp_path / "output"
    monkeypatch.setattr(run_experiment, "model_preflight", lambda value: (tmp_path / "model", {}))

    def successful_child(command, *, cwd, env, stdout, stderr):
        (output / "results.csv").write_text("qa_pred,qa_acc\nA,1\n")
        stdout.write("[kv_head_ragged_prefill] Installed physical per-KV-head ragged prefill/chunk/decode path.\n")
        if compressed:
            stdout.write("[kv_head_ragged_prefill] prune stats: kv_budget=[750, 1500]\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_experiment.subprocess, "run", successful_child)
    result = run_experiment.main(["--method", "dp", "--model-path", "unused", "--annotation", str(source),
                                  "--output-dir", str(output), "--smoke"])
    manifest = json.loads((output / "experiment_manifest.json").read_text())
    assert manifest["result_rows"] == 1
    assert manifest["compression_evidence"]["events"] == int(compressed)
    assert result == (0 if compressed else 1)
    assert manifest["status"] == ("completed" if compressed else "failed")
