"""Eager answer-time attention observer for LLaVA-OneVision HERMES."""

import torch

from inference.abstract_hermes import Abstract_Hermes
from inference.llavaov_hermes import LlavaOneVision_Hermes


class EagerObservedLlavaOV(LlavaOneVision_Hermes):
    def get_video_features(self, pixel_values_videos):
        return self.model.get_video_features(
            pixel_values=pixel_values_videos,
            vision_feature_layer=self.config.vision_feature_layer,
            vision_feature_select_strategy=self.config.vision_feature_select_strategy,
        )

    def _language_layers(self):
        if hasattr(self.language_model, "layers"):
            return self.language_model.layers
        return self.language_model.model.layers

    def _register_forward_hooks(self):
        # Full-prefix profiling keeps every layer at the same logical position,
        # so hooks only need to support the legacy nested language-model layout.
        if hasattr(self.language_model, "layers"):
            return
        super()._register_forward_hooks()

    def _patch_rotary_embeddings(self):
        if hasattr(self.language_model, "layers"):
            return
        super()._patch_rotary_embeddings()

    def _set_rotary_required_len(self, required_len):
        if hasattr(self.language_model, "layers"):
            return
        super()._set_rotary_required_len(required_len)

    @torch.inference_mode()
    def encode_video_chunk(self, video_chunk, **kwargs):
        if video_chunk is None or int(video_chunk.shape[0]) == 0:
            return
        if len(video_chunk.shape) == 4 and video_chunk.shape[-1] == 3:
            video_chunk = video_chunk.permute(0, 3, 1, 2)

        pixel_values = self.processor.video_processor(
            video_chunk, return_tensors="pt"
        ).pixel_values_videos.to(self.device, self.dtype)
        video_features = self.get_video_features(pixel_values)
        start_pos_per_layer = self._get_next_start_pos_per_layer()
        q_len = int(video_features.shape[1])
        batch = int(video_features.shape[0])
        position_ids = self._build_position_ids(start_pos_per_layer[0], q_len, batch)
        # Video ingestion only needs the updated KV cache. SDPA avoids
        # materializing the quadratic attention matrix for long prefixes;
        # answer-time profiling below still explicitly uses eager attention.
        text_config = self.language_model.config
        previous_attn_implementation = text_config._attn_implementation
        text_config._attn_implementation = "sdpa"
        try:
            output = self.language_model(
                inputs_embeds=video_features,
                past_key_values=self.kv_cache,
                use_cache=True,
                return_dict=True,
                position_ids=position_ids,
                cache_position=None,
                attention_mask=None,
            )
        finally:
            text_config._attn_implementation = previous_attn_implementation
        self.kv_cache = output.past_key_values
        self._append_position_ids(start_pos_per_layer, q_len)
        self.last_encoded_frames = int(video_chunk.shape[0])
        self.total_processed_frames += self.last_encoded_frames

    @torch.inference_mode()
    def compute_eager_attentions(self, input_ids):
        start_pos_per_layer = self._get_next_start_pos_per_layer()
        original_cache_len = int(self.kv_cache.get_seq_length())
        q_len = int(input_ids.shape[1])
        batch = int(input_ids.shape[0])

        self._layer_position_ids.clear()
        for layer_idx in range(self.num_layers):
            self._layer_position_ids[layer_idx] = self._build_position_ids(
                start_pos_per_layer[layer_idx], q_len, batch
            )
        position_ids = self._build_position_ids(start_pos_per_layer[0], q_len, batch)
        self._set_rotary_required_len(max(start_pos_per_layer, default=0) + q_len)

        try:
            output = self.language_model(
                input_ids=input_ids,
                past_key_values=self.kv_cache,
                use_cache=False,
                output_attentions=True,
                return_dict=True,
                position_ids=position_ids,
                cache_position=None,
                attention_mask=None,
            )
        finally:
            self.kv_cache.crop(original_cache_len)
            self._layer_position_ids.clear()
        if output.attentions is None or any(attn is None for attn in output.attentions):
            raise RuntimeError("LLaVA eager output_attentions returned None.")
        return output.attentions


def build_observed_llava_model(args, model_path):
    from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

    device = f"cuda:{args.device}" if args.device >= 0 else "cuda"
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    processor.tokenizer.padding_side = "left"
    system_prompt = (
        "<|im_start|>system \nYou are a helpful assistant.<|im_end|>"
        "<|im_start|>user "
    )
    init_prompt_ids = processor.tokenizer(
        system_prompt, return_tensors="pt"
    ).input_ids.to(device)

    raw_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager",
        local_files_only=True,
    )
    model = EagerObservedLlavaOV.__new__(EagerObservedLlavaOV)
    model.__dict__ = raw_model.__dict__.copy()
    Abstract_Hermes.__init__(
        model, processor, init_prompt_ids.tolist(), args.kv_size
    )
    model.streaming = True
    model.sample_fps = args.sample_fps
    model.compress_mode = args.compress_mode
    model.num_layers = int(raw_model.language_model.config.num_hidden_layers)
    model._position_ids_cache = [None for _ in range(model.num_layers)]
    model.token_activity_cache = [None for _ in range(model.num_layers)]
    model.short_term_ratio = 0.1
    model.long_term_ratio = 0.3
    model.short_term_threshold = int(model.num_layers * model.short_term_ratio)
    model.long_term_threshold = int(model.num_layers * (1 - model.long_term_ratio))
    model.total_processed_frames = 0
    model._layer_position_ids = {}
    model._hook_handles = []
    model._register_forward_hooks()
    model._patch_rotary_embeddings()
    model._temporal_profile_family = "llava_ov"
    model.eval()
    return model, processor
