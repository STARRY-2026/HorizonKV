"""Full-prefix calibration through the native model-specific attention patch.

Video KV is never pruned. Selector scores are recomputed only at observation
boundaries from the last visual patch's actual post-M-RoPE queries. Oracle
scores observe the last query row of each ordinary answer-prediction forward;
the collector neither replaces the decoder nor adds a scoring forward.
"""

from __future__ import annotations

import math

import torch

from sparsemm_baselines.core import _head_scores
from sparsemm_protokv.cache import StreamingCache
from sparsemm_protokv.config import StreamingConfig
from sparsemm_protokv.streaming import StreamingQwen


@torch.inference_mode()
def projected_value_norms(values, weight, num_query_heads, chunk_size=4096):
    """Return ||V_h W^O_q||_2 [Hq,N], without creating [N,hidden]."""
    if values.ndim != 3 or weight.ndim != 2:
        raise ValueError("values and o_proj weight must be [Hkv,N,D] and [out,Hq*D]")
    heads, count, dim = values.shape
    if (not isinstance(num_query_heads, int) or num_query_heads < heads
            or num_query_heads % heads or weight.shape[1] != num_query_heads * dim
            or not isinstance(chunk_size, int) or chunk_size < 1):
        raise ValueError("Invalid projection dimensions or chunk size")
    groups = num_query_heads // heads
    result = torch.empty((num_query_heads, count), device=values.device, dtype=torch.float32)
    with torch.autocast(device_type=values.device.type, enabled=False):
        for query_head in range(num_query_heads):
            block = weight[:, query_head * dim:(query_head + 1) * dim].float()
            gram = block.T @ block
            for start in range(0, count, chunk_size):
                value = values[query_head // groups, start:start + chunk_size].float()
                squared = ((value @ gram) * value).sum(-1).clamp_min(0)
                result[query_head, start:start + chunk_size] = squared.sqrt()
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("Non-finite projected value norm")
    return result


class UncompressedCalibrationCache(StreamingCache):
    """A native cache with full visual history and transactional QA observers."""

    def __init__(self, config, num_layers, num_kv_heads, head_dim, device, dtype,
                 *, o_proj_weights=None, utility_mode="projected"):
        if (config.method != "adakv" or config.budget_scope != "visual"
                or config.score_precision != "float32" or config.gqa_reduce != "mean"
                or config.pooling != "avgpool"):
            raise ValueError("Calibration requires adakv routing, visual scope, FP32/mean/avgpool")
        if utility_mode not in ("attention", "projected", "both"):
            raise ValueError("utility_mode must be attention, projected or both")
        if utility_mode != "attention" and (o_proj_weights is None or len(o_proj_weights) != num_layers):
            raise ValueError("Projected utility requires one native o_proj weight per layer")
        super().__init__(config, num_layers, num_kv_heads, head_dim, device, dtype)
        self.o_proj_weights = o_proj_weights
        self.utility_mode = utility_mode
        self.last_visual_queries = [None] * num_layers
        self._observation = None

    @torch.inference_mode()
    def append_layer(self, layer_idx, k, v, raw_k=None, q=None):
        if self.phase != "video":
            if self.phase == "query" and self._observation is not None:
                self._observe_query(layer_idx, k, q)
            return super().append_layer(layer_idx, k, v, raw_k=raw_k, q=q)
        if self._observation is not None:
            raise RuntimeError("End or abort the observation before adding video tokens")
        if self._positions is None or self._context_done:
            raise RuntimeError("set_context must precede a forward")
        if not 0 <= layer_idx < self.num_layers or layer_idx in self._appended:
            raise ValueError("Each layer must append exactly once per forward")
        expected = (1, self.num_kv_heads, self._positions.shape[1], self.head_dim)
        if k.shape != expected or v.shape != expected:
            raise ValueError(f"KV shape must be {expected}")
        if (q is None or q.ndim != 4 or q.shape[0] != 1 or q.shape[2:] != k.shape[2:]
                or q.shape[1] % self.num_kv_heads):
            raise ValueError("Calibration requires actual native [1,Hq,Q,D] queries")
        layer = self.layers[layer_idx]
        layer.append_chunk(k, v, self._positions)
        observed = min(self.config.window_size, q.shape[2])
        self.last_visual_queries[layer_idx] = q[0, :, -observed:].detach().clone()
        self._pending_metadata[layer_idx] = {
            "method": "calibration_uncompressed", "compressed": False,
            "head_lengths": layer.head_lens.tolist(), "prefix_tokens": self.prefix_tokens,
            "visual_tokens": self.seen_visual_tokens + k.shape[2],
        }
        self._appended.add(layer_idx)
        # finish_video in StreamingCache performs all ordinary cursor updates.

    @torch.inference_mode()
    def collect_ranking(self):
        if not self._initialized or not self._context_done or self._query_saved is not None:
            raise RuntimeError("Ranking requires completed full-prefix video state")
        if self.seen_visual_tokens < 1 or any(q is None for q in self.last_visual_queries):
            raise RuntimeError("A visual patch must precede an observation")
        recent = min(self.config.window_size, self.seen_visual_tokens)
        rows, ranking_rows = [], []
        for layer_idx, layer in enumerate(self.layers):
            queries = self.last_visual_queries[layer_idx]
            groups = queries.shape[0] // self.num_kv_heads
            for head in range(self.num_kv_heads):
                key, _ = layer.get_segment(head)
                if key.shape[0] != self.prefix_tokens + self.seen_visual_tokens:
                    raise AssertionError("Calibration visual history was compressed or contaminated")
                score = _head_scores(
                    key, queries[head * groups:(head + 1) * groups], self.prefix_tokens,
                    recent, self.config, layer_idx, head,
                )
                ranking_rows.append(torch.sort(score, descending=True).indices.detach().cpu())
                rows.append(score.detach().cpu())
        scores = torch.stack(rows)
        ranking = torch.stack(ranking_rows)
        return {"ranking": ranking, "score": scores, "recent": recent}

    @torch.inference_mode()
    def begin_observation(self):
        if self._observation is not None:
            raise RuntimeError("An observation is already active")
        ranked = self.collect_ranking()
        count = self.seen_visual_tokens
        self._observation = {
            **ranked, "visual_tokens": count, "prefix_tokens": self.prefix_tokens,
            "prefill": self.prefills, "state": self.state_tensors(),
            "attention": [torch.zeros((self.num_kv_heads, count), device=self.device,
                                      dtype=torch.float32) for _ in range(self.num_layers)],
            "projected": (None if self.utility_mode == "attention" else
                          [torch.zeros((self.num_kv_heads, count), device=self.device,
                                       dtype=torch.float32) for _ in range(self.num_layers)]),
            "norms": [None] * self.num_layers, "norm_denominators": [None] * self.num_layers,
            "query_forwards": [0] * self.num_layers,
        }

    @torch.inference_mode()
    def _observe_query(self, layer_idx, new_keys, queries):
        observation = self._observation
        if (queries is None or queries.ndim != 4 or queries.shape[0] != 1
                or queries.shape[1] % self.num_kv_heads
                or queries.shape[-1] != self.head_dim):
            raise ValueError("Oracle collection requires actual native projected queries")
        layer = self.layers[layer_idx]
        count, prefix = observation["visual_tokens"], observation["prefix_tokens"]
        heads = queries.shape[1]
        groups = heads // self.num_kv_heads
        if observation["projected"] is not None and observation["norms"][layer_idx] is None:
            context_values = torch.stack([layer.get_segment(h)[1][:prefix + count]
                                          for h in range(self.num_kv_heads)])
            raw_norms = projected_value_norms(context_values, self.o_proj_weights[layer_idx], heads)
            # LU-KV's per-layer normalization includes every query head and all
            # persistent prefix/video tokens, before selecting visual utility.
            denominator = raw_norms.sum() + 1e-9
            observation["norms"][layer_idx] = (raw_norms / denominator)[:, prefix:prefix + count]
            observation["norm_denominators"][layer_idx] = float(denominator)
        with torch.autocast(device_type=self.device.type, enabled=False):
            for head in range(self.num_kv_heads):
                old_keys, _ = layer.get_segment(head)
                all_keys = torch.cat((old_keys, new_keys[0, head]))
                query = queries[0, head * groups:(head + 1) * groups, -1].float()
                logits = query @ all_keys.float().T / math.sqrt(self.head_dim)
                if not bool(torch.isfinite(logits).all()):
                    raise FloatingPointError("Non-finite full-prefix oracle QK logits")
                # The final row causally sees every key, including question and
                # generated text. Normalize over those keys before slicing video.
                attention = torch.softmax(logits, dim=-1, dtype=torch.float32)[:, prefix:prefix + count]
                utility = attention.mean(0)
                buffer = observation["attention"][layer_idx][head]
                torch.maximum(buffer, utility, out=buffer)
                if observation["projected"] is not None:
                    norms = observation["norms"][layer_idx][head * groups:(head + 1) * groups]
                    projected = (attention * norms).mean(0)
                    buffer = observation["projected"][layer_idx][head]
                    torch.maximum(buffer, projected, out=buffer)
        observation["query_forwards"][layer_idx] += 1

    def abort_observation(self):
        self._observation = None

    @torch.inference_mode()
    def end_observation(self):
        observation = self._observation
        if observation is None:
            raise RuntimeError("No observation is active")
        if self._query_saved is not None or not self._context_done:
            raise RuntimeError("QA must restore the video state before ending an observation")
        counts = observation["query_forwards"]
        if min(counts) < 1 or len(set(counts)) != 1:
            raise RuntimeError("Every native layer must observe the same nonzero answer forwards")
        state = self.state_tensors()
        if (state.keys() != observation["state"].keys()
                or any(state[name] is not value for name, value in observation["state"].items())
                or self.seen_visual_tokens != observation["visual_tokens"]
                or self.prefills != observation["prefill"]):
            raise AssertionError("The observation changed persistent full-prefix video state")
        attention = torch.cat(observation["attention"]).cpu()
        projected = (None if observation["projected"] is None else
                     torch.cat(observation["projected"]).cpu())
        result = {
            "ranking": observation["ranking"], "score": observation["score"],
            "utility_attention": attention, "utility_projected": projected,
            "utility": projected if self.utility_mode == "projected" else attention,
            "metadata": {
                "collector": "native_full_prefix", "selector": "sparsemm_visual_attention",
                "visual_tokens": observation["visual_tokens"], "prefix_tokens": self.prefix_tokens,
                "num_layers": self.num_layers, "num_kv_heads": self.num_kv_heads,
                "num_query_heads": self.last_visual_queries[0].shape[0], "prefill": self.prefills,
                "recent": observation["recent"], "ranking_indices": "visual_history_local",
                "score_precision": "float32", "gqa_reduce": "mean", "pooling": "avgpool",
                "kernel_size": self.config.kernel_size, "window_size": self.config.window_size,
                "utility_mode": self.utility_mode, "answer_prediction_forwards": counts[0],
                "utility_attention_definition": "max_answer_position mean_shared_Q attention",
                "utility_projected_definition": "max_answer_position mean_shared_Q attention * layer_normalized_L2(V @ Wo_Q)",
                "projected_norm_normalization": "layer_sum",
                "projected_norm_context": "persistent text prefix plus visual tokens, before question",
                "projected_norm_denominators": observation["norm_denominators"],
                "oracle_normalization": "all prefix, visual, question and generated keys",
                "additional_scoring_forwards": 0, "compressed": False, "query_restored": True,
                **self.memory_stats(),
            },
        }
        self._observation = None
        return result


class CalibrationQwen(StreamingQwen):
    """Native full-prefix streaming, independent of LU's deployed head budgets."""

    def __init__(self, model, processor, config=None, sample_fps=0.5, trace_path=None,
                 validate_state=False, utility_mode="projected"):
        self.utility_mode = utility_mode
        config = config or StreamingConfig(method="adakv", budget=1000, lukv_history="none")
        super().__init__(model, processor, config, sample_fps, trace_path, validate_state)

    def clear_cache(self):
        self.kv_cache = UncompressedCalibrationCache(
            self.config, self.num_layers, self.num_kv_heads, self.head_dim, self.device, self.dtype,
            o_proj_weights=[layer.self_attn.o_proj.weight for layer in self.language_model.layers],
            utility_mode=self.utility_mode,
        )
        self.visual_start_idx = 0
        self._next_position = self._seen_visual_tokens = self.total_processed_frames = 0
        self.last_encoded_frames = self._prefills = self._compressions = 0
        self.conv_history, self.last_metadata = [], []

    def predict_and_compress(self):
        """Runner chunk boundaries never trigger a calibration compression."""

    def begin_observation(self):
        return self.kv_cache.begin_observation()

    def end_observation(self):
        return self.kv_cache.end_observation()

    def abort_observation(self):
        return self.kv_cache.abort_observation()

    def question_answering(self, *args, **kwargs):
        # Native cache.query restores references even on an exception. Avoid the
        # base bridge's optional full-KV clone by default during long calibration.
        state = self.kv_cache.state_tensors()
        history = list(self.conv_history)
        cursors = (self._next_position, self._seen_visual_tokens, self.total_processed_frames)
        checks_before = len(self.validation_checks["qa_checks"])
        try:
            return super().question_answering(*args, **kwargs)
        finally:
            self.conv_history[:] = history
            current = self.kv_cache.state_tensors()
            restored = (current.keys() == state.keys()
                        and all(current[name] is value for name, value in state.items())
                        and cursors == (self._next_position, self._seen_visual_tokens,
                                        self.total_processed_frames))
            if not restored:
                self.validation_checks["failures"].append("Calibration QA changed persistent video state")
                raise AssertionError("Calibration QA changed persistent video state")
            if not self.validate_state and len(self.validation_checks["qa_checks"]) > checks_before:
                self.validation_checks["qa_restore_checks"] += 1
                self.validation_checks["qa_checks"][-1]["restored"] = True
                self.validation_checks["qa_checks"][-1]["restore_validation"] = "tensor_identity_and_cursors"


def load_calibration_model(model_path, config=None, sample_fps=0.5, trace_path=None,
                           device="cuda", dtype="bfloat16", validate_state=False,
                           utility_mode="projected"):
    if dtype not in ("bfloat16", "float16", "float32"):
        raise ValueError("Unsupported model dtype")
    from sparsemm_protokv.model_loader import load_pretrained
    model, processor = load_pretrained(model_path, device=device, dtype=dtype)
    return CalibrationQwen(model, processor, config, sample_fps, trace_path,
                           validate_state, utility_mode)
