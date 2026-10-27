"""One/four independently recomputed recent image units plus capped history.

Each sampled source frame uses Qwen's independent image-feature path, as in
the completed SimpleStream one/four-frame experiment.  The streaming branch
selects historical KV with the existing SparseMM online attention scorer.
At a question, independent last-one and last-four branches replace the
streaming branch's recent KV before the question is prefetched once.
"""

from dataclasses import dataclass
import hashlib
import time

import torch
from transformers.cache_utils import DynamicCache

from sparsemm_native import capped_history
from sparsemm_native.assets import asset_metadata
from sparsemm_native.cache import SparseMMCache
from sparsemm_native.decoupled import dense_prefix
from sparsemm_native.policy import DEFAULT_SCORE_PATH
from sparsemm_protokv.cache import StreamingCache
from sparsemm_protokv.streaming import StreamingQwen


@dataclass
class ImageUnit:
    features: torch.Tensor
    positions: torch.Tensor
    frame_index: int

    @property
    def tokens(self):
        return int(self.features.shape[1])


def quota_matrices(rows, units):
    """Exact per-head split after a newly sampled image enters the stream."""
    if not 1 <= len(units) <= 4:
        raise ValueError("One to four recent images are required")
    history = torch.zeros((28, 4), dtype=torch.long)
    recent = torch.zeros_like(history)
    for row in rows:
        layer, head = row["layer"], row["kv_head"]
        recent[layer, head] = (units[-1].tokens if row["recent_group"] == "one_frame"
                               else sum(unit.tokens for unit in units))
        history[layer, head] = row["history_quota"]
    if int(history.sum()) != capped_history.TOTAL_HISTORY or int(history.max()) > capped_history.CAP:
        raise AssertionError("Capped historical quotas changed")
    if bool((recent < 32).any()):
        raise ValueError("A visual unit did not contain the protected 32 tokens")
    return recent + history, recent


@torch.inference_mode()
def prefill_window(language_model, prefix, units):
    embeddings = torch.cat([unit.features for unit in units], dim=1)
    positions = torch.cat([unit.positions for unit in units], dim=2)
    p = int(prefix[0][0].shape[2])
    cache = DynamicCache.from_legacy_cache(prefix)
    output = language_model(
        inputs_embeds=embeddings, position_ids=positions, past_key_values=cache,
        cache_position=torch.arange(p, p + embeddings.shape[1], device=embeddings.device),
        attention_mask=torch.ones((1, p + embeddings.shape[1]), device=embeddings.device, dtype=torch.long),
        use_cache=True, return_dict=True)
    if any(key.shape[2] != p + embeddings.shape[1] for key, _ in output.past_key_values):
        raise AssertionError("Independent recent prefill contains the wrong number of tokens")
    return output.past_key_values


class CappedHistoryCache(SparseMMCache):
    def __init__(self, *args, rows, **kwargs):
        self.rows = rows
        self.units = []
        self.prefix_dense = None
        super().__init__(*args, **kwargs)

    def enter_image(self, unit):
        if self.units and unit.frame_index != self.units[-1].frame_index + 1:
            raise ValueError("Sampled image ordinals are not consecutive")
        self.units = (self.units + [unit])[-4:]
        self.budgets, self.recent_budgets = quota_matrices(self.rows, self.units)
        self.allocation = dict(method="recent1or4_capped_history",
                               history_total=capped_history.TOTAL_HISTORY,
                               history_head_cap=capped_history.CAP,
                               recent_budgets=self.recent_budgets.tolist(),
                               history_budgets=(self.budgets - self.recent_budgets).tolist())


def assemble(history, one, four, logical_tokens):
    """One prefix, attention-selected older KV, and independent recent KV."""
    fused = StreamingCache(history.config, history.num_layers, history.num_kv_heads,
                           history.head_dim, history.device, history.dtype)
    p = history.prefix_tokens
    actual_history = actual_recent = 0
    for layer_idx, layer in enumerate(history.layers):
        keys, values, positions, lengths = [], [], [], []
        for head in range(history.num_kv_heads):
            r = int(history.recent_budgets[layer_idx, head])
            hk, hv = layer.get_segment(head)
            hp = layer.get_position_segment(head)
            hist_count = hk.shape[0] - p - r
            if hist_count < 0 or hist_count > int(history.budgets[layer_idx, head]) - r:
                raise AssertionError("Historical bank occupancy exceeds the capped quota")
            branch = one if history.rows[layer_idx * 4 + head]["recent_group"] == "one_frame" else four
            rk, rv = branch[layer_idx]
            if rk.shape[2] != p + r:
                raise AssertionError("Independent recent branch has a different visual length")
            cutoff = p + hist_count
            recent_positions = torch.cat([unit.positions[:, 0] for unit in (
                history.units[-1:] if branch is one else history.units)], dim=1)
            if hist_count and bool((hp[0, p:cutoff] >= recent_positions[0].min()).any()):
                raise AssertionError("Historical and recent token positions overlap")
            keys.append(torch.cat((hk[:cutoff], rk[0, head, p:])))
            values.append(torch.cat((hv[:cutoff], rv[0, head, p:])))
            positions.append(torch.cat((hp[:, :cutoff], recent_positions), dim=1))
            lengths.append(cutoff + r)
            actual_history += hist_count
            actual_recent += r
        target = fused.layers[layer_idx]
        target.flat_k, target.flat_v = torch.cat(keys), torch.cat(values)
        target.flat_pos = torch.cat(positions, dim=1)
        target.head_lens = target.head_lens.new_tensor(lengths)
        target._refresh()
    fused.prefix_tokens, fused.logical_tokens = p, int(logical_tokens)
    fused._initialized, fused._context_done = True, True
    fused.phase = "video"
    return fused, dict(actual_history_visual_slots=actual_history,
                       actual_recent_visual_slots=actual_recent,
                       actual_visual_slots=actual_history + actual_recent,
                       configured_history_visual_slots=capped_history.TOTAL_HISTORY,
                       history_head_cap=capped_history.CAP)


class RecentOneFourHistoryQwen(StreamingQwen):
    def __init__(self, model, processor, config, *args, **kwargs):
        self.rows = capped_history.allocation()
        self.score_asset = asset_metadata(config.sparsemm_score_path or None)
        self.last_question_metrics = {}
        super().__init__(model, processor, config, *args, **kwargs)
        self.validation_checks["selector_checks"] = []

    @property
    def budgets(self):
        return self.kv_cache.budgets

    def clear_cache(self):
        initial = torch.full((28, 4), self.config.window_size, dtype=torch.long)
        self.kv_cache = CappedHistoryCache(
            self.config, self.num_layers, self.num_kv_heads, self.head_dim, self.device, self.dtype,
            rows=self.rows, budgets=initial, recent_budgets=initial,
            allocation=dict(method="recent1or4_uninitialized"), asset_metadata=self.score_asset)
        self.visual_start_idx = 0
        self._next_position = self._seen_visual_tokens = self.total_processed_frames = 0
        self.last_encoded_frames = self._prefills = self._compressions = 0
        self.conv_history, self.last_metadata = [], []

    @torch.inference_mode()
    def encode_init_prompt(self):
        super().encode_init_prompt()
        self.kv_cache.prefix_dense = dense_prefix(self.kv_cache)

    @torch.inference_mode()
    def forward_chunk(self, *args, **kwargs):
        cache = self.kv_cache
        if isinstance(cache, CappedHistoryCache):
            cache.timing_enabled = self.timing_enabled
            before = cache.selector_seconds
            hidden = StreamingQwen.forward_chunk(self, *args, **kwargs)
            self.timings.setdefault("selector_seconds", 0.0)
            self.timings["selector_seconds"] += cache.selector_seconds - before
            if kwargs.get("compress", True) and cache.phase == "video":
                for layer in cache.last_metadata["layers"]:
                    self.validation_checks["selector_checks"].append(dict(
                        layer_idx=layer["layer_idx"],
                        head_budgets=layer["head_budgets"],
                        validation_passed=all(layer["position_checks"].values())))
            return hidden
        return StreamingQwen.forward_chunk(self, *args, **kwargs)

    def predict_and_compress(self):
        return None

    @torch.inference_mode()
    def encode_video_chunk(self, video_chunk, is_last_chunk=False):
        if video_chunk is None or video_chunk.shape[0] == 0:
            return
        from PIL import Image
        for frame in video_chunk:
            if frame.ndim != 3 or frame.shape[-1] != 3:
                raise ValueError("Expected a decoded HWC RGB source frame")
            image = Image.fromarray(frame.cpu().numpy())
            with self._measure("vision_seconds"):
                image_kwargs = {} if self.video_max_pixels is None else {"size": {
                    "shortest_edge": self.processor.image_processor.size["shortest_edge"],
                    "longest_edge": self.video_max_pixels}}
                inputs = self.processor(text=[""], images=[image], return_tensors="pt",
                                        **image_kwargs).to(self.device, self.dtype)
                grid = inputs["image_grid_thw"]
                encoded = self.model.get_image_features(inputs["pixel_values"], grid)
                if isinstance(encoded, tuple):
                    encoded = encoded[0]
                features = (torch.cat(encoded, dim=0) if isinstance(encoded, list)
                            else encoded).unsqueeze(0)
            positions = self._video_positions(grid[0].tolist(), features.shape[1])
            unit = ImageUnit(features.detach(), positions.detach(), self.total_processed_frames)
            self.kv_cache.enter_image(unit)
            self.forward_chunk(features, positions, compress=True,
                               time=self.total_processed_frames + 1)
            self._seen_visual_tokens += unit.tokens
            self._next_position = max(self._next_position, int(positions.max()) + 1)
            self.total_processed_frames += 1
            self.timings["video_prefills"] += 1
            self.timings["raw_frames"] += 1
            self.timings["visual_tokens"] += unit.tokens
        self.last_encoded_frames = video_chunk.shape[0]

    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=16, temperature=0, repetition_penalty=1.1):
        history = self.kv_cache
        if not history.units or history.prefix_dense is None:
            raise RuntimeError("Missing recent units or common prefix")
        started = time.perf_counter()
        one = prefill_window(self.language_model, history.prefix_dense, history.units[-1:])
        four = prefill_window(self.language_model, history.prefix_dense, history.units)
        fused, audit = assemble(history, one, four,
                                self.visual_start_idx + self._seen_visual_tokens)
        del one, four
        self.kv_cache = fused
        try:
            answer = StreamingQwen.question_answering(
                self, input_text, max_new_tokens=max_new_tokens,
                temperature=temperature, repetition_penalty=repetition_penalty)
        finally:
            self.kv_cache = history
        audit["recent_rebuild_and_assembly_seconds"] = time.perf_counter() - started
        audit["history_recent_independent_prefill"] = True
        audit["recent_one_frame_heads"] = 56
        audit["recent_four_frame_heads"] = 56
        self.last_question_metrics = audit
        return answer


def load_model(model_path, config, **kwargs):
    from sparsemm_protokv.model_loader import load_pretrained
    dtype = kwargs.pop("dtype", "bfloat16")
    device = kwargs.pop("device", "cuda")
    model, processor = load_pretrained(model_path, device=device, dtype=dtype)
    asset = asset_metadata(config.sparsemm_score_path or None)
    requested = hashlib.sha256((model_path / "config.json").read_bytes()).hexdigest()
    if requested != asset["model_config_sha256"]:
        raise ValueError("SparseMM score table was calibrated for a different Qwen model")
    return RecentOneFourHistoryQwen(model, processor, config, **kwargs)
