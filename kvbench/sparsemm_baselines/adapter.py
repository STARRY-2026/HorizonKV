"""Qwen2.5-VL port of SparseMM's GQA, M-RoPE and flattened-cache path.

SparseMM upstream patches Qwen2-VL on transformers 4.46.2. This bridge uses
the same decoder computation and physical per-KV-head segments on the local
4.57.1 Qwen2.5-VL model. Eviction is delegated to the SparseMM baseline core.
No HERMES attention implementation or eviction policy is imported.
"""

import json
from dataclasses import asdict
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb

from sparsemm_baselines.cache import RaggedCache
from sparsemm_baselines.core import BaselineConfig, select_layer

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None


def ragged_attention(queries, keys, values, layer_cache, scaling):
    """Attend to each head's history + current causal chunk using GQA."""
    _, query_heads, q_len, dim = queries.shape
    kv_heads = keys.shape[1]
    groups = query_heads // kv_heads
    old_lengths = layer_cache.head_lens.tolist()
    all_k, all_v = [], []
    for head in range(kv_heads):
        hk, hv = layer_cache.get_segment(head)
        all_k.append(torch.cat((hk, keys[0, head])))
        all_v.append(torch.cat((hv, values[0, head])))
    if (flash_attn_varlen_func is not None and queries.is_cuda
            and queries.dtype in (torch.float16, torch.bfloat16)):
        q = queries[0].reshape(kv_heads, groups, q_len, dim).permute(0, 2, 1, 3).reshape(-1, groups, dim).contiguous()
        k = torch.cat(all_k)[:, None].contiguous()
        v = torch.cat(all_v)[:, None].contiguous()
        lens = torch.tensor([n + q_len for n in old_lengths], dtype=torch.int32, device=queries.device)
        cu_k = torch.cat((lens.new_zeros(1), lens.cumsum(0, dtype=torch.int32)))
        cu_q = torch.arange(kv_heads + 1, dtype=torch.int32, device=queries.device) * q_len
        output = flash_attn_varlen_func(q, k, v, cu_q, cu_k, q_len, int(lens.max()),
                                       softmax_scale=scaling, causal=True)
        return output.reshape(kv_heads, q_len, groups, dim).permute(0, 2, 1, 3).reshape(query_heads, q_len, dim).transpose(0, 1).reshape(1, q_len, -1)
    outputs = []
    for head in range(kv_heads):
        q = queries[0, head * groups:(head + 1) * groups]
        k, v = all_k[head], all_v[head]
        logits = q.float() @ k.float().T * scaling
        key_indices = torch.arange(k.shape[0], device=k.device)
        query_indices = old_lengths[head] + torch.arange(q_len, device=k.device)
        logits = logits.masked_fill(key_indices[None] > query_indices[:, None], -torch.inf)
        probs = F.softmax(logits, dim=-1).to(q.dtype)
        outputs.append(probs @ v)
    return torch.cat(outputs).transpose(0, 1).reshape(1, q_len, -1)


class SparseMMQwen:
    def __init__(self, model, processor, config=None, sample_fps=0.5, trace_path=None):
        self.model = model.eval()
        self.processor = processor
        self.config = config or BaselineConfig()
        self.model_config = model.config
        self.language_model = model.language_model
        if any(getattr(layer.self_attn, "sliding_window", None) is not None
               for layer in self.language_model.layers):
            raise ValueError("The SparseMM port currently requires full-attention Qwen2.5-VL layers")
        text_config = self.language_model.config
        self.num_layers = int(text_config.num_hidden_layers)
        self.num_kv_heads = int(text_config.num_key_value_heads)
        self.num_query_heads = int(text_config.num_attention_heads)
        self.head_dim = int(self.language_model.layers[0].self_attn.head_dim)
        self._mrope_section = text_config.rope_scaling["mrope_section"]
        self.sample_fps = sample_fps
        self.kv_size = self.config.budget
        self.trace_path = None if trace_path is None else Path(trace_path)
        if self.trace_path:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_prompt_ids = None
        self.clear_cache()

    @property
    def device(self):
        return self.model.device

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    def _trace(self, entry):
        if self.trace_path is None:
            return
        with self.trace_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(entry, video_id=getattr(self, "_active_video_id", None)), ensure_ascii=False) + "\n")

    def clear_cache(self):
        self.kv_cache = RaggedCache(self.num_layers, self.num_kv_heads, self.head_dim, self.device, self.dtype)
        self.visual_start_idx = 0
        self._next_position = 0
        self._seen_visual_tokens = 0
        self.total_processed_frames = 0
        self.last_encoded_frames = 0
        self._prefills = 0
        self._compressions = 0
        self.conv_history = []
        self.last_metadata = []

    @torch.inference_mode()
    def encode_init_prompt(self):
        # A fresh video always starts with a new model cache and budget state.
        self.clear_cache()
        prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
        ids = self.processor.tokenizer(prompt, return_tensors="pt").input_ids
        self.init_prompt_ids = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        n = self.init_prompt_ids.shape[-1]
        positions = torch.arange(n, device=self.device).view(1, 1, -1).expand(3, 1, -1)
        self.forward_chunk(self.model.get_input_embeddings()(self.init_prompt_ids), positions, compress=False)
        self.visual_start_idx = n
        self._next_position = n
        self._trace({"event": "init", "text_tokens": n, "config": asdict(self.config),
                     "num_layers": self.num_layers, "num_kv_heads": self.num_kv_heads})

    @torch.inference_mode()
    def forward_chunk(self, inputs_embeds, positions, compress=True):
        if inputs_embeds.shape[0] != 1 or positions.shape != (3, 1, inputs_embeds.shape[1]):
            raise ValueError("SparseMM requires batch 1 and explicit [3,1,Q] positions")
        if inputs_embeds.shape[1] == 0:
            raise ValueError("Cannot forward an empty chunk")
        hidden = inputs_embeds
        all_metadata = []
        for layer_index, decoder in enumerate(self.language_model.layers):
            residual = hidden
            norm = decoder.input_layernorm(hidden)
            attn = decoder.self_attn
            q_len = norm.shape[1]
            q = attn.q_proj(norm).view(1, q_len, self.num_query_heads, self.head_dim).transpose(1, 2)
            k = attn.k_proj(norm).view(1, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = attn.v_proj(norm).view(1, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            cos, sin = self.language_model.rotary_emb(norm, positions)
            q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, self._mrope_section)
            layer = self.kv_cache.layers[layer_index]
            # The current chunk sees the full pre-eviction cache, as upstream.
            attended = ragged_attention(q, k, v, layer, getattr(attn, "scaling", self.head_dim ** -0.5))
            layer.append_chunk(k, v, positions[:, 0])
            if compress:
                prefix = self.visual_start_idx if self.config.budget_scope == "visual" else 0
                seen = self._seen_visual_tokens + q_len + self.visual_start_idx
                selected, metadata = select_layer(
                    [layer.get_segment(h)[0] for h in range(self.num_kv_heads)], q[0], self.config,
                    layer_idx=layer_index, num_layers=self.num_layers,
                    prefix_tokens=prefix, seen_tokens=seen,
                )
                layer.prune(selected)
                all_metadata.append(metadata)
            hidden = residual + attn.o_proj(attended.contiguous())
            hidden = hidden + decoder.mlp(decoder.post_attention_layernorm(hidden))
        if compress:
            self.last_metadata = all_metadata
            self._prefills += 1
            self._compressions += int(any(m.get("compressed", False) for m in all_metadata))
            self._trace({"event": "video_prefill", "prefill": self._prefills, "input_tokens": inputs_embeds.shape[1],
                         "head_lengths": self.kv_cache.head_lengths(), "kv_bytes": self.kv_cache.nbytes(),
                         "layers": all_metadata})
        return self.language_model.norm(hidden)

    def _video_positions(self, grid, q_len):
        t, h, w = [int(v) for v in grid]
        vision = self.model_config.vision_config
        h //= int(vision.spatial_merge_size)
        w //= int(vision.spatial_merge_size)
        if t * h * w != q_len:
            raise RuntimeError("Video feature count does not match the merged Qwen grid")
        # Qwen's full-video M-RoPE uses one spatial origin for all frames.
        # Advance time by the raw sampled frame index, never by retained-cache
        # length or spatial grid size. The question cursor is separate.
        offset = self.visual_start_idx
        frame_indices = self.total_processed_frames + torch.arange(t, device=self.device, dtype=torch.float32) * int(vision.temporal_patch_size)
        temporal = (frame_indices / self.sample_fps * float(vision.tokens_per_second)).floor()
        temporal = temporal[:, None].expand(t, h * w).flatten()
        height = torch.arange(h, device=self.device, dtype=torch.float32)[None, :, None].expand(t, h, w).flatten()
        width = torch.arange(w, device=self.device, dtype=torch.float32)[None, None, :].expand(t, h, w).flatten()
        return torch.stack((temporal, height, width))[:, None] + offset

    @torch.inference_mode()
    def encode_video_chunk(self, video_chunk, is_last_chunk=False):
        if video_chunk is None or video_chunk.shape[0] == 0:
            return
        stride = int(self.model_config.vision_config.temporal_patch_size)
        for start in range(0, video_chunk.shape[0], stride):
            frames = video_chunk[start:start + stride]
            if frames.shape[-1] == 3:
                frames = frames.permute(0, 3, 1, 2)
            inputs = self.processor(text=[""], videos=frames, return_tensors="pt").to(self.device, self.dtype)
            grid = inputs["video_grid_thw"]
            features = self.model.get_video_features(inputs["pixel_values_videos"], grid)[0].unsqueeze(0)
            positions = self._video_positions(grid[0].tolist(), features.shape[1])
            self.forward_chunk(features, positions, compress=True)
            self._seen_visual_tokens += features.shape[1]
            self._next_position = max(self._next_position, int(positions.max()) + 1)
            self.total_processed_frames += frames.shape[0]
        self.last_encoded_frames = video_chunk.shape[0]

    def predict_and_compress(self):
        return None

    def get_prompt(self, text, mc=False):
        prompt = f"\n{text}<|im_end|><|im_start|>assistant\n"
        return prompt + "Best option: (" if mc else prompt

    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=128, temperature=0, repetition_penalty=1.1):
        if temperature != 0:
            raise ValueError("Baseline comparison uses greedy decoding (temperature=0)")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        saved = self.kv_cache.snapshot()
        cursor = self._next_position
        saved_seen = self._seen_visual_tokens
        tokens = []
        start = time.perf_counter()
        try:
            ids = self.processor.tokenizer(input_text["prompt"]).input_ids
            ids = torch.as_tensor([ids], device=self.device, dtype=torch.long)
            q_len = ids.shape[-1]
            positions = torch.arange(cursor, cursor + q_len, device=self.device).view(1, 1, -1).expand(3, 1, -1)
            hidden = self.forward_chunk(self.model.get_input_embeddings()(ids), positions, compress=False)
            logits = self.model.lm_head(hidden[:, -1])
            for step in range(max_new_tokens):
                scores = logits[0].clone()
                for token in set(tokens):
                    scores[token] = scores[token] * repetition_penalty if scores[token] < 0 else scores[token] / repetition_penalty
                token = int(scores.argmax())
                tokens.append(token)
                if step == 0:
                    print(f"TTFT: {time.perf_counter() - start:.6f} seconds", flush=True)
                if token == self.processor.tokenizer.eos_token_id or step + 1 == max_new_tokens:
                    break
                pos = torch.full((3, 1, 1), cursor + q_len + step, device=self.device, dtype=torch.long)
                new_id = torch.tensor([[token]], device=self.device, dtype=torch.long)
                hidden = self.forward_chunk(self.model.get_input_embeddings()(new_id), pos, compress=False)
                logits = self.model.lm_head(hidden[:, -1])
            answer = self.processor.tokenizer.decode(tokens, skip_special_tokens=True,
                spaces_between_special_tokens=False, clean_up_tokenization_spaces=True)
            self.conv_history.append((input_text.get("question", ""), answer, None))
            return answer
        finally:
            self.kv_cache.restore(saved)
            self._next_position = cursor
            self._seen_visual_tokens = saved_seen
            self._trace({"event": "answer_restored", "next_position": cursor,
                         "head_lengths": self.kv_cache.head_lengths(), "kv_bytes": self.kv_cache.nbytes()})


def load_model(model_path, config=None, sample_fps=0.5, trace_path=None, device="cuda", dtype="bfloat16"):
    if dtype not in ("bfloat16", "float16", "float32"):
        raise ValueError("Unsupported model dtype")
    processor = Qwen2_5_VLProcessor.from_pretrained(model_path, local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=getattr(torch, dtype), attn_implementation="sdpa", local_files_only=True,
    ).to(device).eval()
    return SparseMMQwen(model, processor, config, sample_fps=sample_fps, trace_path=trace_path)
