"""Qwen3-VL adaptation of HERMES predictive, layer-shared visual eviction.

This uses the existing Qwen3 streaming bridge for Q/K norm, M-RoPE and
DeepStack. It preserves HERMES's local/global/mixed pseudo-query scores,
depth-dependent recency and cross-layer refinement. Retained keys keep their
original M-RoPE positions instead of Qwen2.5 HERMES's position compaction.
"""

import re

import torch
import torch.nn.functional as F
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

from sparsemm_baselines.core import BaselineConfig
from sparsemm_baselines.qwen3_adapter import SparseMMQwen3


SHALLOW_RATIO = 0.10
DEEP_RATIO = 0.30


def layer_groups(num_layers):
    if num_layers < 3:
        raise ValueError("HERMES requires at least three decoder layers")
    shallow_end = int(num_layers * SHALLOW_RATIO)
    deep_start = int(num_layers * (1 - DEEP_RATIO))
    if not 0 < shallow_end < deep_start < num_layers:
        raise ValueError("Empty HERMES depth group")
    return shallow_end, deep_start


def predicted_queries(history):
    if history:
        last_q, last_a, last_options = history[-1]
        option = re.match(r"^\s*\(?([A-Z])\)?\.?\s*$", last_a)
        if option and last_options is not None:
            index = ord(option.group(1)) - ord("A")
            if 0 <= index < len(last_options):
                last_a = last_options[index]
        last_a = re.sub(r"[A-Z]\) ", "", last_a)
        context = f"Question: {last_q} Answer: {last_a}"
        global_query = (f"Context summary: {context}. Summarize the video narrative, "
                        "identifying main characters, key events, timeline changes, and the overall theme.")
        local_query = (f"Find recent details related to: {context}. Describe the current scene in detail, "
                       "focusing on specific objects, fine-grained actions, and spatial relationships.")
    else:
        global_query = ("Summarize the video narrative, identifying main characters, key events, "
                        "timeline changes, and the overall theme.")
        local_query = ("Describe the current scene in detail, focusing on specific objects, "
                       "fine-grained actions, and spatial relationships.")
    return local_query, global_query


def hermes_layer_scores(local, global_, mixed, visual_lengths, shallow_end, deep_start):
    """Reproduce HERMES's min-max/recency mix and backward refinement."""
    if not (len(local) == len(global_) == len(mixed) == len(visual_lengths)):
        raise ValueError("HERMES pseudo-query scores are missing a layer")
    raw, types = [], []
    for index, (short, long, combined, length) in enumerate(zip(local, global_, mixed, visual_lengths)):
        if index < shallow_end:
            values, alpha, decay, kind = short, 1.0, 20.0, "short-term"
        elif index >= deep_start:
            values, alpha, decay, kind = long, 0.0, 0.0, "long-term"
        else:
            progress = (index - shallow_end) / (deep_start - shallow_end)
            values, alpha, decay, kind = combined, 0.75 - 0.6 * progress, 20.0 - 12.0 * progress, "mid-term"
        if values.ndim != 1 or len(values) != length or length < 1 or not torch.isfinite(values).all():
            raise ValueError("HERMES visual attention vector is invalid")
        positions = torch.arange(length, device=values.device, dtype=torch.float32)
        recency = torch.exp(-decay * (length - 1 - positions) / max(length - 1, 1))
        attn = (values - values.min()) / (values.max() - values.min() + 1e-6)
        recent = (recency - recency.min()) / (recency.max() - recency.min() + 1e-6)
        raw.append(attn * (1 - alpha) + recent * alpha)
        types.append(kind)
    refined = [item.clone() for item in raw]
    for index in range(len(refined) - 2, -1, -1):
        gamma = {"short-term": 0.1, "mid-term": 0.3, "long-term": 0.4}[types[index]]
        following = refined[index + 1]
        if len(following) != len(refined[index]):
            following = F.interpolate(following.reshape(1, 1, -1), size=len(refined[index]),
                                      mode="linear", align_corners=False).flatten()
        refined[index] = (1 - gamma) * refined[index] + gamma * following
    return refined, types


class Qwen3Hermes(SparseMMQwen3):
    def __init__(self, model, processor, kv_size=500, sample_fps=0.5, trace_path=None,
                 video_max_pixels=None):
        if not isinstance(kv_size, int) or kv_size < 1:
            raise ValueError("kv_size must be positive")
        config = BaselineConfig(method="uniform", budget=max(kv_size, 33), budget_scope="visual")
        super().__init__(model, processor, config=config, sample_fps=sample_fps,
                         trace_path=trace_path, video_max_pixels=video_max_pixels)
        self.kv_size = kv_size
        self.short_term_threshold, self.long_term_threshold = layer_groups(self.num_layers)

    def forward_chunk(self, inputs_embeds, positions, compress=True, deepstack=None):
        # HERMES scores the accumulated cache after the complete video chunk.
        return super().forward_chunk(inputs_embeds, positions, compress=False, deepstack=deepstack)

    def _get_cache_seq_len_per_layer(self):
        lengths = []
        for layer in self.kv_cache.layers:
            row = layer.head_lens.tolist()
            if len(set(row)) != 1:
                raise RuntimeError("HERMES requires layer-shared KV positions")
            lengths.append(int(row[0]))
        return lengths

    @torch.inference_mode()
    def _pseudo_scores(self, query):
        ids = self.processor.tokenizer(query, return_tensors="pt").input_ids.to(self.device)
        q_len = ids.shape[1]
        position = torch.arange(self._next_position, self._next_position + q_len,
                                device=self.device).view(1, 1, -1).expand(3, 1, -1)
        # Original HERMES's SDPA path computes QK scores directly from the
        # pseudo-query embeddings at every decoder layer; it does not advance
        # the decoder hidden state or mutate the persistent cache.
        embeddings = self.model.get_input_embeddings()(ids)
        cos, sin = self.language_model.rotary_emb(embeddings, position)
        groups = self.num_query_heads // self.num_kv_heads
        scores = []
        for index, decoder in enumerate(self.language_model.layers):
            norm = decoder.input_layernorm(embeddings)
            attn = decoder.self_attn
            queries = attn.q_norm(attn.q_proj(norm).view(
                1, q_len, self.num_query_heads, self.head_dim)).transpose(1, 2)
            current_keys = attn.k_norm(attn.k_proj(norm).view(
                1, q_len, self.num_kv_heads, self.head_dim)).transpose(1, 2)
            queries, current_keys = apply_rotary_pos_emb(queries, current_keys, cos, sin)
            pieces = []
            for head in range(self.num_kv_heads):
                old_keys, _ = self.kv_cache.layers[index].get_segment(head)
                if len(old_keys) <= self.visual_start_idx:
                    raise RuntimeError("HERMES pseudo-query requires visual cache")
                keys = torch.cat((old_keys, current_keys[0, head]), dim=0)
                q = queries[0, head * groups:(head + 1) * groups]
                logits = q.float() @ keys.float().T * attn.scaling
                visual = logits.softmax(dim=-1)[:, :, self.visual_start_idx:len(old_keys)]
                pieces.append(visual.mean(dim=(0, 1)))
            if any(len(value) != len(pieces[0]) for value in pieces):
                raise RuntimeError("HERMES requires equal visual lengths across KV heads")
            scores.append(torch.stack(pieces).mean(dim=0))
        return scores

    @torch.inference_mode()
    def predict_and_compress(self):
        lengths_before = self._get_cache_seq_len_per_layer()
        visual_lengths = [length - self.visual_start_idx for length in lengths_before]
        if min(visual_lengths) <= 0:
            return
        local, global_ = predicted_queries(self.conv_history)
        local_scores = self._pseudo_scores(local)
        global_scores = self._pseudo_scores(global_)
        mixed_scores = self._pseudo_scores(local + "; " + global_)
        refined, types = hermes_layer_scores(
            local_scores, global_scores, mixed_scores, visual_lengths,
            self.short_term_threshold, self.long_term_threshold)
        metadata = []
        for index, (layer, score, kind) in enumerate(zip(self.kv_cache.layers, refined, types)):
            capacity = max(0, self.kv_size - (kind == "long-term"))
            keep_visual = min(capacity, len(score))
            selected_visual = torch.topk(score, keep_visual, sorted=False).indices + self.visual_start_idx
            keep = torch.cat((torch.arange(self.visual_start_idx, device=self.device),
                              selected_visual.sort().values)).long()
            before = int(layer.head_lens[0])
            layer.prune([keep for _ in range(self.num_kv_heads)])
            metadata.append({"layer": index, "layer_type": kind, "before": before,
                             "after": int(layer.head_lens[0]), "visual_capacity": capacity,
                             "compressed": int(layer.head_lens[0]) < before})
        self.last_metadata = metadata
        self._prefills += 1
        self._compressions += int(any(item["compressed"] for item in metadata))
        self._trace({"event": "hermes_predictive_compression", "local_query": local,
                     "global_query": global_, "head_lengths": self.kv_cache.head_lengths(),
                     "layers": metadata})


def load_qwen3_hermes(model_path, kv_size=500, sample_fps=0.5, trace_path=None,
                     video_max_pixels=None, dtype="bfloat16", vision_topology="window4"):
    if dtype not in ("bfloat16", "float16"):
        raise ValueError("Qwen3 HERMES dtype must be bfloat16 or float16")
    if vision_topology not in ("native", "window4"):
        raise ValueError("Qwen3 HERMES vision topology must be native or window4")
    processor = Qwen3VLProcessor.from_pretrained(model_path, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, dtype=getattr(torch, dtype), attn_implementation="sdpa",
        low_cpu_mem_usage=True, device_map="cuda", local_files_only=True).eval()
    if vision_topology == "window4":
        from sparsemm_baselines.qwen3_window4 import enable_window4_vision
        enable_window4_vision(model)
    return Qwen3Hermes(model, processor, kv_size=kv_size, sample_fps=sample_fps,
                       trace_path=trace_path, video_max_pixels=video_max_pixels)
