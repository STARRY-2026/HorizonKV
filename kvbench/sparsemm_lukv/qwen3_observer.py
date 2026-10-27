"""Full-cache Qwen3-VL LU observations for bounded, disjoint calibration.

Ranking uses only visual-query attention. The oracle measures attention times
the layer-normalized projected Value norm during ordinary answer forwards.
"""

import math

import torch

from sparsemm_baselines.core import BaselineConfig, _head_scores
from sparsemm_lukv.calibration import projected_value_norms


class Qwen3LUObserver:
    def __init__(self, layers, kv_heads, query_heads, window=32,
                 ranking_mode="pure_attention"):
        if ranking_mode not in ("pure_attention", "hermes_pseudo"):
            raise ValueError("Unknown LU ranking mode")
        self.ranking_mode = ranking_mode
        self.layers = layers
        self.kv_heads = kv_heads
        self.query_heads = query_heads
        self.window = window
        self.config = BaselineConfig(method="uniform", budget=500, budget_scope="visual",
                                     window_size=window, gqa_reduce="mean", pooling="avgpool",
                                     score_precision="float32")
        self.last_visual_queries = [None] * layers
        self.active = False
        self.projected = None
        self.attention = None
        self.norms = [None] * layers
        self.query_forwards = [0] * layers

    def observe_visual_layer(self, index, layer, queries):
        if self.active:
            raise RuntimeError("Cannot add visual tokens during a LU observation")
        self.last_visual_queries[index] = queries[0, :, -self.window:].detach().clone()

    @torch.inference_mode()
    def begin(self, bridge):
        if self.active or bridge._seen_visual_tokens <= self.window:
            raise ValueError("LU observation requires an idle full visual prefix longer than W")
        self.prefix = bridge.visual_start_idx
        self.count = bridge._seen_visual_tokens
        self.state = bridge.kv_cache.snapshot()
        groups = self.query_heads // self.kv_heads
        ranking_rows = []
        score_rows = []
        if self.ranking_mode == "hermes_pseudo":
            from sparsemm_lukv.qwen3_hermes_ranking import collect_hermes_head_scores
            hermes_scores, self.hermes_metadata = collect_hermes_head_scores(bridge)
        else:
            hermes_scores, self.hermes_metadata = None, None
        for index, layer in enumerate(bridge.kv_cache.layers):
            queries = self.last_visual_queries[index]
            if queries is None:
                raise RuntimeError("Missing native Qwen3 visual queries")
            for head in range(self.kv_heads):
                key = layer.get_segment(head)[0]
                if key.shape[0] != self.prefix + self.count:
                    raise RuntimeError("LU calibration requires an unpruned video cache")
                if hermes_scores is None:
                    score = _head_scores(key, queries[head * groups:(head + 1) * groups],
                                         self.prefix, self.window, self.config, index, head)
                else:
                    if hermes_scores[index][head].numel() != self.count:
                        raise RuntimeError("HERMES score length differs from uncompressed video")
                    score = hermes_scores[index][head][:-self.window]
                ranking_rows.append(torch.argsort(score, descending=True, stable=True).cpu())
                score_rows.append(score.cpu())
        self.ranking = torch.stack(ranking_rows)
        self.score = torch.stack(score_rows)
        self.attention = torch.zeros((self.layers, self.kv_heads, self.count), dtype=torch.float32)
        self.projected = torch.zeros_like(self.attention)
        self.norms = [None] * self.layers
        self.query_forwards = [0] * self.layers
        self.active = True

    @torch.inference_mode()
    def observe_answer_layer(self, index, layer, queries, new_keys, output_weight, scaling):
        if not self.active:
            return
        groups = self.query_heads // self.kv_heads
        if self.norms[index] is None:
            values = torch.stack([layer.get_segment(h)[1][:self.prefix + self.count]
                                  for h in range(self.kv_heads)])
            raw = projected_value_norms(values, output_weight, self.query_heads)
            self.norms[index] = (raw / (raw.sum() + 1e-9))[:, self.prefix:self.prefix + self.count]
        with torch.autocast(device_type=queries.device.type, enabled=False):
            for head in range(self.kv_heads):
                old_key = layer.get_segment(head)[0]
                all_key = torch.cat((old_key, new_keys[0, head]))
                query = queries[0, head * groups:(head + 1) * groups, -1].float()
                logits = query @ all_key.float().T * scaling
                attention = torch.softmax(logits, dim=-1, dtype=torch.float32)[:,
                                          self.prefix:self.prefix + self.count]
                plain = attention.mean(0).cpu()
                projected = (attention * self.norms[index][head * groups:(head + 1) * groups]).mean(0).cpu()
                self.attention[index, head] = torch.maximum(self.attention[index, head], plain)
                self.projected[index, head] = torch.maximum(self.projected[index, head], projected)
        self.query_forwards[index] += 1

    def end(self, bridge):
        if not self.active or min(self.query_forwards) < 1 or len(set(self.query_forwards)) != 1:
            raise RuntimeError("Incomplete Qwen3 LU answer observation")
        after = bridge.kv_cache.snapshot()
        if any(any(x is not y for x, y in zip(before, current))
               for before, current in zip(self.state, after)):
            raise RuntimeError("LU observation changed the persistent video cache")
        self.active = False
        return {"ranking": self.ranking.reshape(-1, self.count - self.window),
                "score": self.score.reshape(-1, self.count - self.window),
                "utility_attention": self.attention.reshape(-1, self.count),
                "utility_projected": self.projected.reshape(-1, self.count),
                "metadata": {"visual_tokens": self.count, "prefix_tokens": self.prefix,
                             "num_layers": self.layers, "num_kv_heads": self.kv_heads,
                             "num_query_heads": self.query_heads, "window_size": self.window,
                             "answer_prediction_forwards": self.query_forwards[0],
                             "ranking": ("pure_visual_attention_avgpool5" if self.ranking_mode == "pure_attention"
                                         else "hermes_local_global_mixed_pseudo_per_kv_head_refined"),
                             "hermes_pseudo": self.hermes_metadata,
                             "utility": "max_answer_attention_times_layer_normalized_projected_Value",
                             "additional_scoring_forwards": (3 if self.ranking_mode == "hermes_pseudo" else 0),
                             "query_restored": True}}
