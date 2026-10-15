"""Tiny CUDA check of the actual streaming ragged GQA attention implementation.

Does not load model weights, videos, or run an experiment. Each of six cases
uses four KV heads, seven query groups per head, head dimension 64, and ragged
historical lengths [0,2,9,21]. FlashAttention is compared to the same adapter's
CUDA fallback and to an independent CPU float64 causal reference.

Run from /home/yujun/program:
  /home/yujun/.conda/envs/hermes-qwen/bin/python -m sparsemm_baselines.validation_attention
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from unittest.mock import patch

import torch

from sparsemm_baselines import adapter
from sparsemm_baselines.cache import RaggedLayer


def cpu_reference(queries, keys, values, layer, scale):
    """Independent float64 oracle, with each query's allowed keys explicit."""
    q = queries.cpu().double()[0]
    k = keys.cpu().double()[0]
    v = values.cpu().double()[0]
    q_len, dim = q.shape[1:]
    groups = q.shape[0] // k.shape[0]
    out = torch.empty_like(q)
    for head in range(k.shape[0]):
        old_k, old_v = layer.get_segment(head)
        old_len = old_k.shape[0]
        head_k = torch.cat((old_k.cpu().double(), k[head]))
        head_v = torch.cat((old_v.cpu().double(), v[head]))
        for group in range(groups):
            query_head = head * groups + group
            for position in range(q_len):
                causal_keys = old_len + position + 1
                scores = head_k[:causal_keys] @ q[query_head, position] * scale
                probs = torch.softmax(scores, dim=0)
                out[query_head, position] = probs @ head_v[:causal_keys]
    return out.transpose(0, 1).reshape(1, q_len, -1)


def error(actual, expected):
    delta = actual.detach().cpu().double() - expected.detach().cpu().double()
    return {"max_abs": float(delta.abs().max()), "mean_abs": float(delta.abs().mean()),
            "rms": float(delta.square().mean().sqrt())}


@torch.inference_mode()
def validate(output):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this isolated kernel check")
    if adapter.flash_attn_varlen_func is None:
        raise RuntimeError("FlashAttention is required to validate its real dispatch")
    torch.manual_seed(20261014)
    torch.cuda.reset_peak_memory_stats()
    heads, groups, dim = 4, 7, 64
    old_lengths = [0, 2, 9, 21]
    records = []
    for dtype in (torch.float16, torch.bfloat16):
        for q_len in (1, 3, 17):
            # Host generation makes the input data reproducible independently
            # of GPU RNG implementation; no model or dataset is accessed.
            queries = torch.randn(1, heads * groups, q_len, dim)
            keys = torch.randn(1, heads, q_len, dim)
            values = torch.randn_like(keys)
            for head in range(heads):
                queries[:, head * groups:(head + 1) * groups] += head * .08
                values[:, head] += head * .3
            queries = queries.to(device="cuda", dtype=dtype)
            keys = keys.to(device="cuda", dtype=dtype)
            values = values.to(device="cuda", dtype=dtype)
            layer = RaggedLayer(heads, dim, "cuda", dtype)
            layer.flat_k = torch.cat([torch.randn(n, dim) + h * .1 for h, n in enumerate(old_lengths)]).to(device="cuda", dtype=dtype)
            layer.flat_v = torch.cat([torch.randn(n, dim) + h * .3 for h, n in enumerate(old_lengths)]).to(device="cuda", dtype=dtype)
            layer.flat_pos = torch.zeros(3, sum(old_lengths), device="cuda")
            layer.head_lens = torch.tensor(old_lengths, device="cuda", dtype=torch.int32)
            layer._refresh()
            scale = dim ** -.5
            flash = adapter.ragged_attention(queries, keys, values, layer, scale)
            torch.cuda.synchronize()
            with patch.object(adapter, "flash_attn_varlen_func", None):
                fallback = adapter.ragged_attention(queries, keys, values, layer, scale)
            torch.cuda.synchronize()
            reference = cpu_reference(queries, keys, values, layer, scale)
            atol = rtol = .002 if dtype == torch.float16 else .02
            torch.testing.assert_close(flash, fallback, atol=atol, rtol=rtol)
            torch.testing.assert_close(flash.cpu().double(), reference, atol=atol, rtol=rtol)
            torch.testing.assert_close(fallback.cpu().double(), reference, atol=atol, rtol=rtol)
            assert flash.shape == (1, q_len, heads * groups * dim)
            assert flash.dtype == dtype
            assert torch.isfinite(flash).all() and torch.isfinite(fallback).all()
            assert layer.head_lens.tolist() == old_lengths, "Attention mutated the history cache"
            records.append({
                "dtype": str(dtype).removeprefix("torch."), "query_length": q_len,
                "kv_heads": heads, "query_groups": groups, "head_dim": dim,
                "history_lengths": old_lengths, "output_shape": list(flash.shape),
                "flash_vs_cuda_fallback": error(flash, fallback),
                "flash_vs_cpu_float64": error(flash, reference),
                "cuda_fallback_vs_cpu_float64": error(fallback, reference),
                "atol": atol, "rtol": rtol, "passed": True,
            })
            del queries, keys, values, layer, flash, fallback, reference
    peak = torch.cuda.max_memory_allocated()
    assert peak < 100 * 1024 ** 2, f"Unexpected tensor memory footprint {peak}"
    import flash_attn
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "isolated unit kernel validation; no model weights or dataset loaded",
        "implementation": "sparsemm_baselines.adapter.ragged_attention",
        "torch_version": torch.__version__, "flash_attn_version": flash_attn.__version__,
        "cuda_version": torch.version.cuda, "device": torch.cuda.get_device_name(),
        "seed": 20261014, "peak_tensor_memory_bytes": peak,
        "passed_cases": len(records), "total_cases": len(records), "cases": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed_cases": len(records), "peak_tensor_memory_bytes": peak,
                      "result": str(output)}, ensure_ascii=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results/validation/flash_attention_check.json")
    validate(parser.parse_args().output)
