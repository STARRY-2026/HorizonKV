"""Bounded Qwen3-VL bridge for the model-independent SparseMM selectors.

This keeps Qwen3's Q/K normalization, interleaved M-RoPE, and DeepStack
features. It deliberately does not load Qwen2.5-specific calibration assets.
"""

import ctypes
import gc
import os
from pathlib import Path

import torch
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb
from transformers.video_utils import VideoMetadata

from sparsemm_baselines.adapter import SparseMMQwen, ragged_attention
from sparsemm_baselines.core import BaselineConfig, select_layer


class SparseMMQwen3(SparseMMQwen):
    def __init__(self, model, processor, config=None, sample_fps=0.5, trace_path=None,
                 video_max_pixels=None, policy_method=None, head_budgets=None,
                 recent_budgets=None, calibration_observer=None, lukv_profile=None):
        self.video_max_pixels = video_max_pixels
        self.policy_method = policy_method or (config.method if config is not None else "adakv")
        self.calibration_observer = calibration_observer
        self.lukv_profile = lukv_profile
        super().__init__(model, processor, config, sample_fps, trace_path)
        if self.policy_method not in ("adakv", "pyramidkv", "uniform", "full",
                                      "lukv", "sparsemm", "dp", "dp_dual"):
            raise ValueError(f"Unsupported Qwen3 selector {self.policy_method}")
        self.head_budgets = None if head_budgets is None else torch.as_tensor(head_budgets, dtype=torch.long)
        self.recent_budgets = None if recent_budgets is None else torch.as_tensor(recent_budgets, dtype=torch.long)
        if self.lukv_profile is not None:
            if self.policy_method != "lukv":
                raise ValueError("A LU profile requires the lukv policy")
            from sparsemm_lukv.profile import validate_profile
            validate_profile(self.lukv_profile, self.num_layers, self.num_kv_heads,
                             self.config.window_size)
            calibration = self.lukv_profile.get("calibration", {})
            mode = calibration.get("runtime_score_formula", "attention")
            if mode not in ("attention", "hermes_pseudo") or (
                    mode == "hermes_pseudo" and calibration.get("selector") != "hermes_pseudo"):
                raise ValueError("LU profile ranking and runtime scorer disagree")
        if self.policy_method in ("lukv", "sparsemm", "dp", "dp_dual"):
            shape = (self.num_layers, self.num_kv_heads)
            if (self.head_budgets is None and self.lukv_profile is None or
                    self.head_budgets is not None and tuple(self.head_budgets.shape) != shape):
                raise ValueError(f"{self.policy_method} needs a model-bound {shape} capacity table")
            if self.policy_method == "dp_dual" and (self.recent_budgets is None
                    or tuple(self.recent_budgets.shape) != shape):
                raise ValueError(f"dp_dual needs a model-bound {shape} recent table")

    def _debug_memory(self, stage):
        if os.environ.get("QWEN3_DEBUG_MEMORY") == "1":
            current = int(Path("/sys/fs/cgroup/memory.current").read_text())
            print(f"QWEN3_MEMORY {stage} cgroup_gib={current / 2**30:.2f} "
                  f"gpu_allocated_gib={torch.cuda.memory_allocated() / 2**30:.2f} "
                  f"gpu_reserved_gib={torch.cuda.memory_reserved() / 2**30:.2f}", flush=True)

    @torch.inference_mode()
    def forward_chunk(self, inputs_embeds, positions, compress=True, deepstack=None):
        if inputs_embeds.shape[0] != 1 or positions.shape != (3, 1, inputs_embeds.shape[1]):
            raise ValueError("Qwen3 streaming requires batch 1 and [3,1,Q] positions")
        if (compress and self.policy_method == "lukv" and self.lukv_profile is not None
                and self.lukv_profile.get("calibration", {}).get("runtime_score_formula") == "hermes_pseudo"):
            # HERMES probes the accumulated video cache, so prune after the
            # ordinary visual forward rather than during each decoder layer.
            hidden = self.forward_chunk(inputs_embeds, positions, compress=False, deepstack=deepstack)
            from sparsemm_lukv.profile import allocate_profile
            from sparsemm_lukv.qwen3_hermes_ranking import collect_hermes_head_scores
            budgets, allocation = allocate_profile(
                self.lukv_profile, self._seen_visual_tokens + inputs_embeds.shape[1],
                self.config.budget, self.config.window_size)
            scores, probe = collect_hermes_head_scores(
                self, start_position=max(self._next_position, int(positions.max()) + 1))
            prefix = self.visual_start_idx
            metadata = []
            for layer_idx, layer in enumerate(self.kv_cache.layers):
                before = layer.head_lens.tolist()
                chosen = []
                for head in range(self.num_kv_heads):
                    visual = before[head] - prefix
                    recent = min(self.config.window_size, visual)
                    capacity = int(budgets[layer_idx, head])
                    score = scores[layer_idx][head]
                    if score.numel() != visual:
                        raise RuntimeError("HERMES pseudo score length differs from LU cache")
                    history = visual - recent
                    count = min(max(capacity - recent, 0), history)
                    ranked = torch.argsort(score[:history], descending=True, stable=True)[:count]
                    keep = torch.cat((torch.arange(prefix, device=self.device),
                                      ranked + prefix,
                                      torch.arange(before[head] - recent, before[head], device=self.device)))
                    chosen.append(keep.sort().values.long())
                layer.prune(chosen)
                after = layer.head_lens.tolist()
                metadata.append({"method": "lukv", "selector": "hermes_pseudo_per_head",
                                 "layer_idx": layer_idx, "before_head_lengths": before,
                                 "head_lengths": after, "head_budgets": budgets[layer_idx].tolist(),
                                 "prefix_tokens": prefix, "window_size": self.config.window_size,
                                 "compressed": any(a < b for a, b in zip(after, before)),
                                 "allocation_target_slots": allocation["target_slots"]})
            self.last_metadata = metadata
            self._prefills += 1
            self._compressions += int(any(item["compressed"] for item in metadata))
            self._trace({"event": "video_prefill", "prefill": self._prefills,
                         "input_tokens": inputs_embeds.shape[1],
                         "grid_thw": getattr(self, "_pending_visual_grid", None),
                         "head_lengths": self.kv_cache.head_lengths(),
                         "kv_bytes": self.kv_cache.nbytes(), "layers": metadata,
                         "lukv_probe": probe})
            return hidden
        hidden = inputs_embeds
        all_metadata = []
        active_budgets = self.head_budgets
        if compress and self.lukv_profile is not None:
            from sparsemm_lukv.profile import allocate_profile
            active_budgets, _ = allocate_profile(
                self.lukv_profile, self._seen_visual_tokens + inputs_embeds.shape[1],
                self.config.budget, self.config.window_size)
        for layer_index, decoder in enumerate(self.language_model.layers):
            residual = hidden
            norm = decoder.input_layernorm(hidden)
            attn = decoder.self_attn
            q_len = norm.shape[1]
            q = attn.q_norm(attn.q_proj(norm).view(1, q_len, self.num_query_heads, self.head_dim)).transpose(1, 2)
            k = attn.k_norm(attn.k_proj(norm).view(1, q_len, self.num_kv_heads, self.head_dim)).transpose(1, 2)
            v = attn.v_proj(norm).view(1, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            cos, sin = self.language_model.rotary_emb(norm, positions)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            layer = self.kv_cache.layers[layer_index]
            attended = ragged_attention(q, k, v, layer, attn.scaling)
            if self.calibration_observer is not None and self.calibration_observer.active and not compress:
                self.calibration_observer.observe_answer_layer(layer_index, layer, q, k,
                                                               attn.o_proj.weight, attn.scaling)
            layer.append_chunk(k, v, positions[:, 0])
            if self.calibration_observer is not None and compress:
                self.calibration_observer.observe_visual_layer(layer_index, layer, q)
            if compress:
                prefix = self.visual_start_idx if self.config.budget_scope == "visual" else 0
                seen = self._seen_visual_tokens + q_len + self.visual_start_idx
                keys = [layer.get_segment(h)[0] for h in range(self.num_kv_heads)]
                if self.policy_method == "lukv":
                    from sparsemm_lukv.policy import attention_select_layer
                    selected, metadata = attention_select_layer(
                        keys, q[0], active_budgets[layer_index], prefix,
                        window_size=self.config.window_size)
                elif self.policy_method in ("sparsemm", "dp"):
                    from sparsemm_native.policy import select_layer as native_select
                    selected, metadata = native_select(
                        keys, q[0], self.head_budgets[layer_index], prefix,
                        budget=self.config.budget, window_size=self.config.window_size,
                        layer_idx=layer_index, external_budget=self.policy_method == "dp")
                elif self.policy_method == "dp_dual":
                    from sparsemm_native.two_bank import select_two_bank_layer
                    selected, metadata = select_two_bank_layer(
                        keys, q[0], self.head_budgets[layer_index],
                        self.recent_budgets[layer_index], prefix,
                        budget=self.config.budget, window_size=self.config.window_size,
                        layer_idx=layer_index, short_cache_gate="legacy_global")
                else:
                    selected, metadata = select_layer(
                        keys, q[0], self.config,
                        layer_idx=layer_index, num_layers=self.num_layers,
                        prefix_tokens=prefix, seen_tokens=seen,
                    )
                layer.prune(selected)
                all_metadata.append(metadata)
            hidden = residual + attn.o_proj(attended.contiguous())
            hidden = hidden + decoder.mlp(decoder.post_attention_layernorm(hidden))
            if deepstack is not None and layer_index < len(deepstack):
                hidden = hidden + deepstack[layer_index].unsqueeze(0).to(hidden.dtype)
        if compress:
            self.last_metadata = all_metadata
            self._prefills += 1
            self._compressions += int(any(m.get("compressed", False) for m in all_metadata))
            self._trace({"event": "video_prefill", "prefill": self._prefills,
                         "input_tokens": inputs_embeds.shape[1],
                         "grid_thw": getattr(self, "_pending_visual_grid", None),
                         "head_lengths": self.kv_cache.head_lengths(),
                         "kv_bytes": self.kv_cache.nbytes(), "layers": all_metadata})
        return self.language_model.norm(hidden)

    def _video_positions(self, grid, q_len):
        t, h, w = (int(v) for v in grid)
        merge = int(self.model_config.vision_config.spatial_merge_size)
        h, w = h // merge, w // merge
        if t * h * w != q_len:
            raise RuntimeError("Qwen3 visual feature count differs from video grid")
        # Qwen3 represents timestamps as text markers. This bounded streaming
        # bridge instead advances the temporal coordinate between patches;
        # all compared selectors receive identical positions.
        offset = self._next_position
        time = torch.arange(t, device=self.device)[:, None].expand(t, h * w).flatten()
        height = torch.arange(h, device=self.device)[None, :, None].expand(t, h, w).flatten()
        width = torch.arange(w, device=self.device)[None, None, :].expand(t, h, w).flatten()
        return torch.stack((time, height, width))[:, None] + offset

    @torch.inference_mode()
    def encode_video_chunk(self, video_chunk, is_last_chunk=False):
        if video_chunk is None or video_chunk.shape[0] == 0:
            return
        # Use the same complete frame block as the native Qwen3 processor.
        # Its pixel limit is shared across the whole block, so splitting a
        # block into pairs would silently change every frame's resolution.
        self._debug_memory("chunk_before")
        frames = video_chunk.cpu().numpy()
        if len(frames) == 1:
            image_kwargs = {} if self.video_max_pixels is None else {"size": {
                "shortest_edge": self.processor.image_processor.size["shortest_edge"],
                "longest_edge": self.video_max_pixels,
            }}
            inputs = self.processor(text=[""], images=[frames[0]], return_tensors="pt",
                                    **image_kwargs)
            grid = inputs["image_grid_thw"]
            self._debug_memory("chunk_processed")
            features, deepstack = self.model.get_image_features(
                inputs["pixel_values"].to(self.device), grid.to(self.device))
        else:
            metadata = VideoMetadata(total_num_frames=len(frames), fps=self.sample_fps,
                                     width=frames.shape[2], height=frames.shape[1],
                                     frames_indices=list(range(len(frames))))
            video_kwargs = {} if self.video_max_pixels is None else {"size": {
                "shortest_edge": self.processor.video_processor.size["shortest_edge"],
                "longest_edge": self.video_max_pixels,
            }}
            inputs = self.processor(text=[""], videos=[frames], video_metadata=[metadata],
                                    return_tensors="pt", **video_kwargs)
            grid = inputs["video_grid_thw"]
            self._debug_memory("chunk_processed")
            features, deepstack = self.model.get_video_features(
                inputs["pixel_values_videos"].to(self.device), grid.to(self.device))
        visual = features[0].unsqueeze(0)
        self._debug_memory("chunk_vision")
        self._pending_visual_grid = grid[0].tolist()
        positions = self._video_positions(grid[0].tolist(), visual.shape[1])
        self.forward_chunk(visual, positions, compress=True, deepstack=deepstack)
        self._pending_visual_grid = None
        self._debug_memory("chunk_forward")
        self._seen_visual_tokens += visual.shape[1]
        self._next_position = max(self._next_position, int(positions.max()) + 1)
        self.total_processed_frames += len(frames)
        del inputs, features, deepstack, visual, positions
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        torch.cuda.empty_cache()
        self._debug_memory("chunk_released")
        self.last_encoded_frames = video_chunk.shape[0]


def load_model(model_path, *, config=None, sample_fps=0.5, trace_path=None,
               video_max_pixels=None):
    if config is None:
        config = BaselineConfig()
    processor = Qwen3VLProcessor.from_pretrained(model_path, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="sdpa",
        low_cpu_mem_usage=True, device_map="cuda", local_files_only=True,
    ).eval()
    return SparseMMQwen3(model, processor, config, sample_fps=sample_fps,
                         trace_path=trace_path, video_max_pixels=video_max_pixels)
