"""Streaming orchestration around real HF decoder calls and SparseMM patches."""

from dataclasses import asdict
from contextlib import contextmanager, nullcontext
import time

import torch
from transformers import (AutoProcessor, LlavaOnevisionForConditionalGeneration,
                          Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor)
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb

from sparsemm_baselines.adapter import SparseMMQwen
from sparsemm_protokv.cache import StreamingCache
from sparsemm_protokv.config import StreamingConfig
from sparsemm_protokv.monkeypatch import replace_qwen2_5


class StreamingQwen(SparseMMQwen):
    """Shared streaming bridge for Qwen2.5-VL and LLaVA-OneVision."""
    def __init__(self, model, processor, config=None, sample_fps=0.5, trace_path=None,
                 validate_state=True, video_max_pixels=None):
        config = config or StreamingConfig()
        self.is_llava = model.config.model_type == "llava_onevision"
        if self.is_llava and config.video_chunk_mode != "temporal_patch":
            raise ValueError("HERMES chunk mode is Qwen2.5-VL only")
        if video_max_pixels is not None and (self.is_llava or config.method not in ("protokv", "sparsemm")):
            raise ValueError("video_max_pixels is supported only for Qwen2.5-VL ProtoKV or SparseMM")
        self.video_max_pixels = video_max_pixels
        if self.is_llava:
            from sparsemm_protokv.qwen2 import install
            install()
        else:
            replace_qwen2_5(config.method)
        self.validate_state = validate_state
        self.timing_enabled = False
        # Cumulative across videos; clear_cache must not reset these measurements.
        self.timings = {"vision_seconds": 0.0, "language_video_seconds": 0.0,
                        "bank_update_seconds": 0.0, "qa_seconds": 0.0,
                        "video_prefills": 0, "raw_frames": 0, "visual_tokens": 0,
                        "questions": 0, "ttft_seconds": []}
        self.validation_checks = {"finite_hidden_checks": 0, "finite_logits_checks": 0,
                                  "qa_restore_checks": 0, "qa_checks": [], "failures": []}
        super().__init__(model, processor, config, sample_fps, trace_path)

    @contextmanager
    def _measure(self, field):
        if not self.timing_enabled:
            yield
            return
        if torch.device(self.device).type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        try:
            yield
        finally:
            if torch.device(self.device).type == "cuda":
                torch.cuda.synchronize(self.device)
            self.timings[field] += time.perf_counter() - started

    def clear_cache(self):
        self.kv_cache = StreamingCache(self.config, self.num_layers, self.num_kv_heads,
                                       self.head_dim, self.device, self.dtype)
        self.visual_start_idx = 0
        self._next_position = self._seen_visual_tokens = self.total_processed_frames = 0
        self.last_encoded_frames = self._prefills = self._compressions = 0
        self._pending_visual_input = None
        self.conv_history, self.last_metadata = [], []

    @torch.inference_mode()
    def encode_init_prompt(self):
        self.clear_cache()
        prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
        ids = self.processor.tokenizer(prompt, return_tensors="pt").input_ids
        self.init_prompt_ids = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        count = self.init_prompt_ids.shape[-1]
        positions = torch.arange(count, device=self.device).view(1, 1, -1).expand(3, 1, -1)
        self.forward_chunk(self.model.get_input_embeddings()(self.init_prompt_ids), positions, compress=False)
        self.visual_start_idx = self._next_position = count
        self.kv_cache.finish_init(count)
        self._trace({"event": "init", "text_tokens": count, "config": asdict(self.config),
                     "num_layers": self.num_layers, "num_kv_heads": self.num_kv_heads,
                     "num_query_heads": self.num_query_heads, "head_dim": self.head_dim,
                     "dtype": str(self.dtype),
                     "memory": self.kv_cache.memory_stats()})

    @torch.inference_mode()
    def forward_chunk(self, inputs_embeds, positions, compress=True, xy=None, time=None):
        count = inputs_embeds.shape[1]
        if inputs_embeds.shape[0] != 1 or positions.shape != (3, 1, count) or count < 1:
            raise ValueError("Native streaming forward requires batch 1, nonempty tokens and [3,1,Q] positions")
        cache = self.kv_cache
        phase = "video" if compress else cache.phase
        if phase == "video" and xy is None:
            y, x = positions[1, 0], positions[2, 0]
            x, y = x - x.min(), y - y.min()
            xy = torch.stack((x / x.max().clamp_min(1), y / y.max().clamp_min(1)), dim=-1)
        cache.set_context(phase, positions[:, 0], xy=xy,
                          time=self.total_processed_frames if time is None else time)
        logical = cache.get_seq_length()
        # Query forwards belong to the enclosing QA timer.
        model_positions = positions[0] if self.is_llava else positions
        with self._measure("language_video_seconds") if compress else nullcontext():
            result = self.language_model(
                inputs_embeds=inputs_embeds, position_ids=model_positions, past_key_values=cache,
                cache_position=torch.arange(logical, logical + count, device=self.device),
                attention_mask={"full_attention": None}, use_cache=True, return_dict=True,
            )
        hidden = result.last_hidden_state
        self.validation_checks["finite_hidden_checks"] += 1
        if not torch.isfinite(hidden).all().item():
            raise FloatingPointError("Non-finite native decoder final hidden states")
        if compress:
            with self._measure("bank_update_seconds"):
                cache.finish_video()
            self._prefills += 1
            metadata = getattr(cache, "last_metadata", {})
            self.last_metadata = metadata.get("layers", [])
            absorbed = 0 if cache.bank is None else cache.bank.total_absorbed
            self._compressions += int(metadata.get("compressed", False))
            self._trace({"event": "video_prefill", "prefill": self._prefills, "input_tokens": count,
                         "visual_input": self._pending_visual_input,
                         "head_lengths": cache.head_lengths(), "kv_bytes": cache.nbytes(),
                         "memory": cache.memory_stats(), "layers": self.last_metadata,
                         "method_metadata": metadata,
                         "bank_total_absorbed": absorbed,
                         "active_prototypes": 0 if cache.bank is None else int(cache.bank.active.sum()),
                         "pq_ready_groups": 0 if cache.bank is None else int(cache.bank.code_ready.sum()),
                         "native_attention_calls": getattr(cache, "native_attention_calls", 0)})
        elif phase == "query":
            cache.advance_query(count)
        return hidden

    @torch.inference_mode()
    def encode_video_chunk(self, video_chunk, is_last_chunk=False):
        if video_chunk is None or video_chunk.shape[0] == 0:
            return
        if self.is_llava:
            for frame in video_chunk:
                if frame.shape[-1] == 3:
                    frame = frame.permute(2, 0, 1)
                with self._measure("vision_seconds"):
                    pixels = self.processor.video_processor(
                        frame.unsqueeze(0), return_tensors="pt"
                    ).pixel_values_videos.to(self.device, self.dtype)
                    features = self.model.get_video_features(
                        pixels, vision_feature_layer=self.model.config.vision_feature_layer,
                        vision_feature_select_strategy=self.model.config.vision_feature_select_strategy,
                    )
                count = features.shape[1]
                positions = torch.arange(self._next_position, self._next_position + count,
                                         device=self.device).view(1, 1, -1).expand(3, 1, -1)
                side = int(count ** 0.5)
                if side * side != count:
                    raise ValueError(f"LLaVA pooled frame requires a square grid, got {count} tokens")
                grid = torch.stack(torch.meshgrid(
                    torch.arange(side, device=self.device), torch.arange(side, device=self.device),
                    indexing="ij"), dim=-1).reshape(-1, 2)
                xy = grid[:, [1, 0]].float() / max(side - 1, 1)
                self.forward_chunk(features, positions, compress=True, xy=xy,
                                   time=self.total_processed_frames + 1)
                self._seen_visual_tokens += count
                self._next_position += count
                self.total_processed_frames += 1
                self.timings["video_prefills"] += 1
                self.timings["raw_frames"] += 1
                self.timings["visual_tokens"] += count
            self.last_encoded_frames = video_chunk.shape[0]
            return
        # The legacy path commits one temporal patch at a time.  The explicit
        # HERMES comparison protocol commits the complete runner chunk once,
        # matching HERMES's visual-forward and eviction cadence.
        stride = (video_chunk.shape[0] if self.config.video_chunk_mode == "hermes_chunk"
                  else int(self.model_config.vision_config.temporal_patch_size))
        for start in range(0, video_chunk.shape[0], stride):
            frames = video_chunk[start:start + stride]
            if frames.shape[-1] == 3:
                frames = frames.permute(0, 3, 1, 2)
            with self._measure("vision_seconds"):
                video_kwargs = {} if self.video_max_pixels is None else {"size": {
                    "shortest_edge": self.processor.video_processor.size["shortest_edge"],
                    "longest_edge": self.video_max_pixels,
                }}
                inputs = self.processor(text=[""], videos=frames, return_tensors="pt",
                                        **video_kwargs).to(self.device, self.dtype)
                grid = inputs["video_grid_thw"]
                features = self.model.get_video_features(inputs["pixel_values_videos"], grid)[0].unsqueeze(0)
            positions = self._video_positions(grid[0].tolist(), features.shape[1])
            self._pending_visual_input = {
                "raw_frames": int(frames.shape[0]),
                "source_hw": [int(frames.shape[-2]), int(frames.shape[-1])],
                "grid_thw": [int(value) for value in grid[0].tolist()],
                "resized_hw": [int(grid[0, 1]) * int(self.model_config.vision_config.patch_size),
                               int(grid[0, 2]) * int(self.model_config.vision_config.patch_size)],
                "visual_tokens": int(features.shape[1]),
            }
            self.forward_chunk(features, positions, compress=True,
                               time=self.total_processed_frames + frames.shape[0])
            self._seen_visual_tokens += features.shape[1]
            self._next_position = max(self._next_position, int(positions.max()) + 1)
            self.total_processed_frames += frames.shape[0]
            self.timings["video_prefills"] += 1
            self.timings["raw_frames"] += frames.shape[0]
            self.timings["visual_tokens"] += features.shape[1]
        self.last_encoded_frames = video_chunk.shape[0]

    def _rotate_pseudo(self, keys, positions):
        keys = keys.to(self.dtype)
        dummy = keys.new_empty(keys.shape[0], keys.shape[1], 1)
        if self.is_llava:
            cos, sin = self.language_model.rotary_emb(dummy, positions[0])
            return apply_rotary_pos_emb(keys[:, None], keys[:, None], cos, sin)[1][:, 0]
        cos, sin = self.language_model.rotary_emb(dummy, positions)
        return apply_multimodal_rotary_pos_emb(keys[:, None], keys[:, None], cos, sin,
                                               self._mrope_section)[1][:, 0]

    def _logits(self, hidden):
        logits = self.model.lm_head(hidden[:, -1])
        self.validation_checks["finite_logits_checks"] += 1
        if not torch.isfinite(logits).all().item():
            raise FloatingPointError("Non-finite native Qwen answer logits")
        return logits

    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=128, temperature=0,
                          repetition_penalty=1.1):
        with self._measure("qa_seconds"):
            self._question_timing_started = time.perf_counter()
            answer = self._question_answering(input_text, max_new_tokens, temperature, repetition_penalty)
            self.timings["questions"] += 1
            return answer

    def _question_answering(self, input_text, max_new_tokens=128, temperature=0,
                           repetition_penalty=1.1):
        if temperature != 0 or max_new_tokens < 1:
            raise ValueError("Streaming comparisons require greedy decoding and max_new_tokens >= 1")
        cache = self.kv_cache
        saved_cursors = (self._next_position, self._seen_visual_tokens, self.total_processed_frames)
        state = cache.state_tensors()
        saved = {name: (tensor, tensor.clone()) for name, tensor in state.items()} if self.validate_state else {}
        tokens, check = [], {"video_id": getattr(self, "_active_video_id", None),
                            "question_index": getattr(self, "_active_question_index", None),
                            "restored": False, "returned": False}
        started = time.perf_counter()
        try:
            with cache.query(self._rotate_pseudo):
                check["query_memory"] = cache.memory_stats()
                before_native = getattr(cache, "query_attention_calls", 0)
                before_pseudo = getattr(cache, "query_pseudo_attention_calls", 0)
                ids = self.processor.tokenizer(input_text["prompt"]).input_ids
                ids = torch.as_tensor([ids], device=self.device, dtype=torch.long)
                count, cursor = ids.shape[-1], self._next_position
                positions = torch.arange(cursor, cursor + count, device=self.device).view(1, 1, -1).expand(3, 1, -1)
                hidden = self.forward_chunk(self.model.get_input_embeddings()(ids), positions, compress=False)
                check["first_answer_native_layers"] = getattr(cache, "query_attention_calls", 0) - before_native
                check["first_answer_pseudo_layers"] = getattr(cache, "query_pseudo_attention_calls", 0) - before_pseudo
                if check["first_answer_native_layers"] != self.num_layers:
                    raise AssertionError("Question prefill did not execute every patched native decoder layer")
                if self.config.method == "protokv" and bool(cache.bank.active.any()):
                    if check["first_answer_pseudo_layers"] != self.num_layers:
                        raise AssertionError("The first answer token did not read ProtoKV memory")
                logits = self._logits(hidden)
                for step in range(max_new_tokens):
                    scores = logits[0].clone()
                    for token in set(tokens):
                        scores[token] = scores[token] * repetition_penalty if scores[token] < 0 else scores[token] / repetition_penalty
                    token = int(scores.argmax())
                    tokens.append(token)
                    if step == 0:
                        if self.timing_enabled:
                            self.timings["ttft_seconds"].append(time.perf_counter() - self._question_timing_started)
                        print(f"TTFT: {time.perf_counter() - started:.6f} seconds", flush=True)
                    if token == self.processor.tokenizer.eos_token_id or step + 1 == max_new_tokens:
                        break
                    pos = torch.full((3, 1, 1), cursor + count + step, device=self.device, dtype=torch.long)
                    new_id = torch.tensor([[token]], device=self.device, dtype=torch.long)
                    hidden = self.forward_chunk(self.model.get_input_embeddings()(new_id), pos, compress=False)
                    logits = self._logits(hidden)
                answer = self.processor.tokenizer.decode(tokens, skip_special_tokens=True,
                    spaces_between_special_tokens=False, clean_up_tokenization_spaces=True)
                self.conv_history.append((input_text.get("question", ""), answer, None))
                check["returned"] = True
                return answer
        except BaseException as error:
            self.validation_checks["failures"].append(f"{type(error).__name__}: {error}")
            raise
        finally:
            self._next_position, self._seen_visual_tokens, self.total_processed_frames = saved_cursors
            current = cache.state_tensors()
            if self.validate_state:
                if current.keys() != saved.keys() or any(current[name] is not ref or not torch.equal(current[name], value)
                                                        for name, (ref, value) in saved.items()):
                    self.validation_checks["failures"].append("Persistent KV/bank state changed during query")
                    raise AssertionError("Persistent KV/bank state changed during query")
                self.validation_checks["qa_restore_checks"] += 1
                check["restored"] = True
            self.validation_checks["qa_checks"].append(check)
            self._trace({"event": "answer_restored", "next_position": self._next_position,
                         "head_lengths": cache.head_lengths(), "kv_bytes": cache.nbytes(),
                         "memory": cache.memory_stats(), "query_check": check})


def load_model(model_path, config=None, sample_fps=0.5, trace_path=None,
               device="cuda", dtype="bfloat16", validate_state=True, video_max_pixels=None):
    if dtype not in ("bfloat16", "float16", "float32"):
        raise ValueError("Unsupported model dtype")
    from sparsemm_protokv.model_loader import load_pretrained
    model, processor = load_pretrained(model_path, device=device, dtype=dtype)
    return StreamingQwen(model, processor, config, sample_fps, trace_path, validate_state,
                         video_max_pixels=video_max_pixels)
