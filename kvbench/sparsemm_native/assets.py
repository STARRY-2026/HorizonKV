"""Preflight the completed local Qwen2.5 SynthDOG calibration assets.

Only annotation/score/provenance files and the model configuration are read.
The previous complete audit checked all 1000 JPEGs/checkpoints; inference
preflight binds that evidence without decoding images or loading model weights.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import re

from sparsemm_native.policy import DEFAULT_RATIO, DEFAULT_SCORE_PATH, DEFAULT_SCORE_SHA256


_LAYERS, _QUERY_HEADS, _KV_HEADS, _CALIBRATION_COUNT = 28, 28, 4, 1000
_UPSTREAM_COMMIT = "33f2f0ab96d1d99516bcd2092caccab60516ff2c"
_OFFICIAL_SOURCE_SHA256 = "b05399110865e54434ef5c6122a86a1a467843593f7a610db71390a32bbad6ef"
_DEFAULT_MANIFEST_SHA = "254af2300f1c6ea742a4d05f24d003ee6d9599849454f9a7581b7307d653d52d"
_DEFAULT_RAW_SHA = "1e02713bc3e582e4a8adde7eaa27188c657361cbc6603463cd67eee224a5cd5f"
_DEFAULT_CANONICAL_PROTOCOL_SHA = "9ba783db0468b2bd7696b5279ca620b3101ffecacd1d2bb80b5785fb8802d1a4"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, f"Duplicate JSON key in {path}: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def _canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _bound_file(name, expected, label):
    path = Path(name).expanduser()
    _require(path.is_absolute() and path.is_file(), f"Missing absolute {label} path")
    _require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected)
             and _sha256(path) == expected, f"{label} SHA-256 mismatch")
    return path.resolve()


def _csv_scores(path, raw):
    expected = {(layer, head) for layer in range(_LAYERS) for head in range(_QUERY_HEADS)}
    _require(isinstance(raw, dict) and set(raw) == {f"{l}-{h}" for l, h in expected},
             "Raw visual-score shape must cover every Qwen2.5 query head")
    means, positive_samples = {}, [False] * _CALIBRATION_COUNT
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = ["layer", "head", "score_sum", "score_mean", "nonzero", "max"]
        _require(reader.fieldnames is not None and len(reader.fieldnames) == len(fields)
                 and set(reader.fieldnames) == set(fields), "Fresh calibration CSV schema mismatch")
        for row in reader:
            _require(None not in row and all(value is not None for value in row.values()), "Malformed calibration CSV row")
            _require(re.fullmatch(r"\d+", row["layer"]) and re.fullmatch(r"\d+", row["head"]),
                     "Calibration CSV indices must be nonnegative integers")
            key = int(row["layer"]), int(row["head"])
            _require(key in expected and key not in means, "Calibration CSV has duplicate or out-of-range query heads")
            samples = raw[f"{key[0]}-{key[1]}"]
            _require(isinstance(samples, list) and len(samples) == _CALIBRATION_COUNT,
                     "Every query head requires the complete calibration image count")
            _require(all(isinstance(value, (int, float)) and not isinstance(value, bool)
                         and math.isfinite(value) and value >= 0 for value in samples),
                     "Raw visual scores must be finite and nonnegative")
            total = math.fsum(samples)
            mean = float(row["score_mean"])
            _require(math.isfinite(mean) and math.isclose(mean, total / _CALIBRATION_COUNT, rel_tol=1e-12, abs_tol=1e-15),
                     "Calibration CSV score_mean differs from mean(raw)")
            _require(math.isclose(float(row["score_sum"]), total, rel_tol=1e-12, abs_tol=1e-15)
                     and int(row["nonzero"]) == sum(value != 0 for value in samples)
                     and math.isclose(float(row["max"]), max(samples), rel_tol=1e-12, abs_tol=1e-15),
                     "Calibration CSV statistics differ from raw observations")
            means[key] = mean
            for index, value in enumerate(samples):
                positive_samples[index] |= value > 0
    _require(set(means) == expected and math.fsum(means.values()) > 0,
             "Calibration CSV is incomplete or has no positive visual scores")
    return means, sum(positive_samples)


def asset_metadata(path=None):
    """Validate full calibration proof and return JSON-safe inference metadata.

    Explicit paths must also have a complete sibling manifest/protocol/raw
    export. Tiny CPU policy tests may load their synthetic scores separately;
    this function never labels such tables as an official completed calibration.
    """
    score_path = Path(path or DEFAULT_SCORE_PATH).expanduser().resolve()
    _require(score_path.is_file() and score_path.suffix.lower() == ".csv", "Fresh Qwen2.5 calibration score CSV is missing")
    manifest_path, raw_path = score_path.with_name("manifest.json"), score_path.with_name("raw_head_scores.json")
    manifest = _read_json(manifest_path)
    _require(isinstance(manifest, dict) and manifest.get("status") == "completed"
             and manifest.get("count") == _CALIBRATION_COUNT
             and manifest.get("requested_count") == _CALIBRATION_COUNT
             and manifest.get("official_full_calibration") is True
             and manifest.get("last_completed_image_index") == _CALIBRATION_COUNT - 1,
             "SparseMM requires a completed full official-flow calibration")
    score_hash, manifest_hash, raw_hash = _sha256(score_path), _sha256(manifest_path), _sha256(raw_path)
    _require(score_hash == manifest.get("head_scores_sha256") and raw_hash == manifest.get("raw_head_scores_sha256"),
             "Calibration CSV/raw SHA-256 differs from completed manifest")
    protocol_path = Path(manifest.get("protocol_path", ""))
    _require(protocol_path.is_absolute() and protocol_path.is_file(), "Calibration protocol path is missing")
    protocol = _read_json(protocol_path)
    canonical_hash = _canonical_sha256(protocol)
    _require(canonical_hash == manifest.get("protocol_sha256"), "Canonical calibration protocol SHA-256 mismatch")
    if score_path == DEFAULT_SCORE_PATH.resolve():
        _require(score_hash == DEFAULT_SCORE_SHA256 and manifest_hash == _DEFAULT_MANIFEST_SHA
                 and raw_hash == _DEFAULT_RAW_SHA and canonical_hash == _DEFAULT_CANONICAL_PROTOCOL_SHA,
                 "Pinned default Qwen2.5 calibration asset fingerprint changed")
    _require(protocol.get("official_commit") == _UPSTREAM_COMMIT
             and protocol.get("official_dataset_verified") is True
             and protocol.get("official_full_calibration_requested") is True
             and manifest.get("official_source_sha256") == _OFFICIAL_SOURCE_SHA256,
             "Calibration is not bound to the pinned official SparseMM workflow")
    _require(protocol.get("processor_use_fast") is False
             and protocol.get("attention_implementation") == "eager"
             and manifest.get("actual_attention_implementation") == "eager"
             and manifest.get("runtime_dtype") == "torch.bfloat16",
             "Calibration processor/attention/precision protocol differs from the Qwen2.5 asset")
    model_path = Path(protocol.get("model_path", "")).expanduser().resolve()
    _require(model_path == Path(manifest.get("model_path", "")).expanduser().resolve(), "Calibration model paths disagree")
    model_files = protocol.get("model_files", {})
    _require(isinstance(model_files, dict) and isinstance(model_files.get("config.json"), dict),
             "Calibration HF model configuration fingerprint is missing")
    config_hash = model_files["config.json"].get("sha256")
    config_path = _bound_file(str(model_path / "config.json"), config_hash, "HF model configuration")
    for key in ("model_config_sha256", "config_sha256"):
        _require(key not in manifest or manifest[key] == config_hash, "Calibration manifest HF config fingerprint mismatch")
    declared_model_files = manifest.get("model_files_sha256", {})
    _require("config.json" not in declared_model_files or declared_model_files["config.json"] == config_hash,
             "Calibration manifest HF config fingerprint mismatch")
    config = _read_json(config_path)
    text = config.get("text_config", config)
    _require(config.get("model_type") == "qwen2_5_vl"
             and (text.get("num_hidden_layers"), text.get("num_attention_heads"), text.get("num_key_value_heads"))
             == (_LAYERS, _QUERY_HEADS, _KV_HEADS), "Calibration model is not the required Qwen2.5-VL architecture")
    dataset = protocol.get("dataset", {})
    _require(dataset.get("requested_count") == _CALIBRATION_COUNT
             and dataset.get("available_records", 0) >= _CALIBRATION_COUNT,
             "Calibration dataset does not cover the full requested image count")
    sources = {str(score_path): score_hash, str(raw_path): raw_hash, str(config_path): config_hash}
    dataset_path = _bound_file(dataset.get("path", ""), dataset.get("sha256"), "SynthDOG annotation")
    dataset_records = _read_json(dataset_path)
    _require(isinstance(dataset_records, list) and len(dataset_records) >= _CALIBRATION_COUNT,
             "SynthDOG annotation contains fewer records than the declared calibration")
    _require(manifest.get("dataset_sha256") == dataset["sha256"], "Calibration dataset fingerprints disagree")
    sources[str(dataset_path)] = dataset["sha256"]
    for name, manifest_key in (("official_source", "official_source_sha256"), ("implementation", "implementation_sha256")):
        entry = protocol.get(name, {})
        source_path = _bound_file(entry.get("path", ""), entry.get("sha256"), name)
        _require(manifest.get(manifest_key) == entry["sha256"], f"Calibration {name} fingerprints disagree")
        sources[str(source_path)] = entry["sha256"]
    provenance = protocol.get("data_provenance", {})
    provenance_path = _bound_file(provenance.get("path", ""), provenance.get("sha256"), "SynthDOG data manifest")
    data = _read_json(provenance_path)
    _require(provenance.get("dataset") == data.get("dataset") == "nnethercott/synthdog-en-detection"
             and data.get("official_commit") == _UPSTREAM_COMMIT
             and provenance.get("dataset_revision") == data.get("dataset_revision")
             and provenance.get("validated_rows") == _CALIBRATION_COUNT
             and provenance.get("ordered_image_hashes_verified") is True
             and provenance.get("annotation_sha256") == data.get("annotation_sha256") == dataset["sha256"],
             "Official SynthDOG data provenance is incomplete")
    images, rows = protocol.get("images", []), data.get("rows", [])
    _require(len(images) == _CALIBRATION_COUNT and len(rows) >= _CALIBRATION_COUNT,
             "Calibration image fingerprint list is incomplete")
    for index, image in enumerate(images):
        _require(image.get("index") == image.get("id") == rows[index].get("id")
                 == rows[index].get("original_train_row_index") == index
                 and image.get("sha256") == rows[index].get("jpeg_sha256")
                 and image.get("record_sha256") == _canonical_sha256(dataset_records[index]),
                 "Calibration ordered image fingerprints disagree with source data")
    sources[str(provenance_path)] = provenance["sha256"]
    means, nonzero_images = _csv_scores(score_path, _read_json(raw_path))
    protocol_hash = _sha256(protocol_path)
    return {
        "path": str(score_path), "sha256": score_hash, "asset_type": "visual_head_scores",
        "score_column": "score_mean", "shape": [_LAYERS, _KV_HEADS],
        "query_head_shape": [_LAYERS, _QUERY_HEADS], "num_images": _CALIBRATION_COUNT,
        "nonzero_score_images": nonzero_images, "nonzero_query_heads": sum(value > 0 for value in means.values()),
        "calibration_manifest_path": str(manifest_path), "calibration_manifest_sha256": manifest_hash,
        "protocol_path": str(protocol_path), "protocol_file_sha256": protocol_hash,
        "protocol_canonical_sha256": canonical_hash, "raw_path": str(raw_path), "raw_sha256": raw_hash,
        "model_path": str(model_path), "model_config_sha256": config_hash,
        "calibration_manifests": {str(manifest_path): manifest_hash, str(protocol_path): protocol_hash},
        "calibration_sources": sources, "official_commit": _UPSTREAM_COMMIT,
        "dataset_sha256": dataset["sha256"], "official_source_sha256": manifest["official_source_sha256"],
        "implementation_sha256": manifest["implementation_sha256"],
        "processor_use_fast": False, "online_processor_use_fast": True,
        "online_processor_adaptation": "Official image calibration uses the slow processor; native streaming inference uses the shared fast video processor.",
        "ratio": float(DEFAULT_RATIO), "static_gqa_reduction": "sum", "online_gqa_reduction": "mean",
        "allocation_rounding": "upstream_round_to_even", "exact_total_correction": False,
        "weights_loaded": False, "weight_shards_rehashed": False, "image_pixels_rehashed": False,
    }
