"""Local model selection shared by all KVBench streaming methods."""

import json
from pathlib import Path

SUPPORTED = {"qwen2_5_vl", "llava_onevision"}


def model_type(model_path):
    path = Path(model_path).expanduser().resolve() / "config.json"
    kind = json.loads(path.read_text(encoding="utf-8"))["model_type"]
    if kind not in SUPPORTED:
        raise ValueError(f"Unsupported KVBench model_type={kind!r}; expected {sorted(SUPPORTED)}")
    return kind


def load_pretrained(model_path, *, device="cuda", dtype="bfloat16"):
    import torch
    from transformers import (AutoProcessor, LlavaOnevisionForConditionalGeneration,
                              Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor)
    if dtype not in ("bfloat16", "float16", "float32"):
        raise ValueError("Unsupported model dtype")
    kind = model_type(model_path)
    processor_cls, model_cls = (
        (AutoProcessor, LlavaOnevisionForConditionalGeneration) if kind == "llava_onevision"
        else (Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration)
    )
    processor = processor_cls.from_pretrained(model_path, local_files_only=True)
    model = model_cls.from_pretrained(
        model_path, dtype=getattr(torch, dtype), attn_implementation="sdpa",
        local_files_only=True,
    ).to(device).eval()
    return model, processor
