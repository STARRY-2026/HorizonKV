"""Create clearly synthetic LLaVA head scores for SparseMM interface smoke tests.

These scores do not measure OCR attention and must never be used as calibrated
SparseMM results. The runtime and auditor bind them to one local model config.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    model = args.model_path.expanduser().resolve()
    config_path = model / "config.json"
    config = json.loads(config_path.read_text())
    if config.get("model_type") != "llava_onevision":
        raise ValueError("Smoke-score generator requires LLaVA-OneVision")
    text = config["text_config"]
    layers, query_heads = int(text["num_hidden_layers"]), int(text["num_attention_heads"])
    kv_heads = int(text["num_key_value_heads"])
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    scores = out / "head_scores.csv"
    with scores.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("layer", "head", "score_mean"))
        for layer in range(layers):
            for head in range(query_heads):
                writer.writerow((layer, head, "1.0"))
    manifest = {
        "schema": "kvbench.sparsemm_synthetic_smoke.v1",
        "official_calibration": False, "smoke_only": True,
        "score_rule": "constant_one_per_query_head; no image or OCR observation",
        "model_path": str(model), "model_config_sha256": sha256(config_path),
        "score_sha256": sha256(scores),
        "query_head_shape": [layers, query_heads], "shape": [layers, kv_heads],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"scores": str(scores), "manifest": str(out / "manifest.json"),
                      "smoke_only": True}))


if __name__ == "__main__":
    main()
