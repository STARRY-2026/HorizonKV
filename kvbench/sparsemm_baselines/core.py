"""SparseMM AdaKV/PyramidKV/Uniform selectors with a streaming extension.

The reference classes are loaded from the untouched pinned SparseMM checkout,
not from the older local AdaKV experiment.  Dense prefill uses the reference
formulas; streaming adds ragged head histories, short query chunks, optional
prefix protection, and FP32 scoring.  No exact-budget redistribution is added:
AdaKV's integer floor/round and PyramidKV's integer layer step are preserved.
Uniform uses the original SnapKV selector with its pyramid schedule disabled:
every layer and KV head gets the same capacity, with attention-ranked tokens.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib.util
import math
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
import torch.nn.functional as F


UPSTREAM_COMMIT = "33f2f0ab96d1d99516bcd2092caccab60516ff2c"
UPSTREAM_SOURCE = Path(__file__).resolve().parent / "upstream/sparsemm/sparsemm_utils.py"


@dataclass(frozen=True)
class BaselineConfig:
    method: str = "adakv"
    budget: int = 1000
    window_size: int = 32
    kernel_size: int = 5
    pooling: str = "avgpool"
    floor_alpha: float = 0.2
    normalize: bool = True
    beta: int = 20
    gqa_reduce: str = "mean"
    budget_scope: str = "prompt"
    score_precision: str = "float32"

    def __post_init__(self) -> None:
        if self.method not in ("adakv", "pyramidkv", "uniform"):
            raise ValueError("method must be 'adakv', 'pyramidkv', or 'uniform'")
        for name in ("budget", "window_size", "kernel_size", "beta"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.budget <= self.window_size:
            raise ValueError("budget must be greater than window_size, as in SparseMM")
        if self.kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd to preserve the score length")
        if self.pooling not in ("avgpool", "maxpool"):
            raise ValueError("pooling must be 'avgpool' or 'maxpool'")
        if not math.isfinite(self.floor_alpha) or not 0 <= self.floor_alpha <= 1:
            raise ValueError("floor_alpha must be finite and between 0 and 1")
        if not isinstance(self.normalize, bool):
            raise ValueError("normalize must be boolean")
        if self.gqa_reduce not in ("mean", "max"):
            raise ValueError("gqa_reduce must be 'mean' or 'max'")
        if self.budget_scope not in ("prompt", "visual"):
            raise ValueError("budget_scope must be 'prompt' or 'visual'")
        if self.score_precision not in ("float32", "input"):
            raise ValueError("score_precision must be 'float32' or 'input'")


@lru_cache(maxsize=1)
def load_upstream() -> ModuleType:
    """Load the original selectors without importing its legacy model patches."""
    if not UPSTREAM_SOURCE.is_file():
        raise FileNotFoundError(f"Pinned SparseMM source is missing: {UPSTREAM_SOURCE}")
    spec = importlib.util.spec_from_file_location(
        "_sparsemm_baselines_original_selectors", UPSTREAM_SOURCE
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load SparseMM source: {UPSTREAM_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def upstream_source_sha256() -> str:
    return hashlib.sha256(UPSTREAM_SOURCE.read_bytes()).hexdigest()


def make_dense_cluster(
    config: BaselineConfig,
    layer_idx: int,
    num_layers: int,
    num_key_value_groups: int = 1,
) -> Any:
    """Construct the actual unmodified SparseMM selector for dense validation.

    Its update_kv method expects [batch=1, heads, tokens, dim] tensors and a
    complete prompt as queries. AdaKV returns flattened, head-concatenated KV;
    PyramidKV/Uniform return a dense [1, KV heads, retained tokens, dim] pair.
    score_precision is not applied here: this is intentionally the original.
    """
    _validate_layer(layer_idx, num_layers, config)
    if config.budget_scope != "prompt":
        raise ValueError("Original SparseMM dense selectors support prompt scope only")
    if num_key_value_groups < 1:
        raise ValueError("num_key_value_groups must be positive")
    original = load_upstream()
    common = dict(
        window_size=config.window_size,
        kernel_size=config.kernel_size,
        pooling=config.pooling,
        layer_idx=layer_idx,
        num_hidden_layers=num_layers,
        num_key_value_groups=num_key_value_groups,
        gqa_func=config.gqa_reduce,
    )
    if config.method == "adakv":
        return original.AdaKVCluster(
            **common,
            base_capacity=config.budget,
            floor_alpha=config.floor_alpha,
            skip=0,
            normalize=config.normalize,
        )
    return original.SnapKVCluster(
        **common,
        max_capacity_prompt=config.budget,
        pyram_mode=config.method == "pyramidkv",
        pyram_beta=config.beta,
    )


def dense_reference(
    keys: torch.Tensor,
    queries: torch.Tensor,
    values: torch.Tensor,
    config: BaselineConfig,
    layer_idx: int,
    num_layers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the actual upstream dense implementation; no ported code is used."""
    if keys.ndim != 4 or queries.ndim != 4 or values.shape != keys.shape:
        raise ValueError("Dense reference expects K/V [1,Hkv,N,D], Q [1,Hq,N,D]")
    if keys.shape[0] != 1 or queries.shape[0] != 1:
        raise ValueError("Only batch size 1 is supported")
    if queries.shape[1] % keys.shape[1] or queries.shape[2:] != keys.shape[2:]:
        raise ValueError("Dense reference needs complete-prompt queries and valid GQA")
    cluster = make_dense_cluster(
        config, layer_idx, num_layers, queries.shape[1] // keys.shape[1]
    )
    return cluster.update_kv(keys, queries, values)


def _validate_layer(layer_idx: int, num_layers: int, config: BaselineConfig) -> None:
    if not isinstance(num_layers, int) or num_layers < 1:
        raise ValueError("num_layers must be a positive integer")
    if not isinstance(layer_idx, int) or not 0 <= layer_idx < num_layers:
        raise ValueError("layer_idx must index num_layers")
    if config.method == "pyramidkv" and num_layers < 2:
        raise ValueError("PyramidKV's upstream schedule needs at least 2 layers")


def pyramid_layer_capacity(
    config: BaselineConfig,
    layer_idx: int,
    num_layers: int,
    seen_tokens: int,
    prefix_tokens: int = 0,
) -> int:
    """Original SnapKVCluster(pyram_mode=True) integer capacity schedule.

    seen_tokens is the cumulative uncompressed prompt length. For visual scope,
    protected prefix tokens are outside this schedule. Very short streaming
    prefixes clamp historical availability to zero instead of a negative value.
    The upstream integer step is deliberately not corrected to an exact mean.
    """
    _validate_layer(layer_idx, num_layers, config)
    if config.method != "pyramidkv":
        return config.budget
    eligible_seen = seen_tokens - (prefix_tokens if config.budget_scope == "visual" else 0)
    base = config.budget - config.window_size
    min_num = base // config.beta
    max_num = 2 * base - min_num
    available = max(0, eligible_seen - config.window_size)
    if max_num >= available:
        max_num = available
        min_num = 2 * base - max_num
    steps = (max_num - min_num) // (num_layers - 1)
    return max_num - layer_idx * steps + config.window_size


def _validate_inputs(
    keys: list[torch.Tensor],
    queries: torch.Tensor,
    config: BaselineConfig,
    layer_idx: int,
    num_layers: int,
    prefix_tokens: int,
    seen_tokens: int | None,
) -> tuple[int, int, int]:
    _validate_layer(layer_idx, num_layers, config)
    if not keys or queries.ndim != 3:
        raise ValueError("Expected nonempty per-head K [N,D] and Q [Hq,Q,D]")
    if not isinstance(prefix_tokens, int) or prefix_tokens < 0:
        raise ValueError("prefix_tokens must be a nonnegative integer")
    query_heads, query_len, dim = queries.shape
    if query_len < 1 or dim < 1 or query_heads < 1 or query_heads % len(keys):
        raise ValueError("Queries must be nonempty and query heads divisible by KV heads")
    if not queries.is_floating_point():
        raise ValueError("Queries must have floating dtype")
    if not torch.isfinite(queries).all().item():
        raise FloatingPointError(f"Non-finite queries at layer {layer_idx}")
    lengths = []
    for head, key in enumerate(keys):
        if key.ndim != 2 or key.shape[1] != dim:
            raise ValueError(f"Invalid key dimensions in head {head}")
        if key.shape[0] < max(prefix_tokens, query_len):
            raise ValueError("Each head must contain its protected prefix and current query chunk")
        if key.device != queries.device or key.dtype != queries.dtype:
            raise ValueError("All key/query tensors must share device and dtype")
        if not key.is_floating_point():
            raise ValueError("Keys must have floating dtype")
        if not torch.isfinite(key).all().item():
            raise FloatingPointError(f"Non-finite keys at layer {layer_idx}, head {head}")
        lengths.append(key.shape[0])
    if seen_tokens is None:
        seen_tokens = max(lengths)
    if not isinstance(seen_tokens, int) or seen_tokens < max(lengths):
        raise ValueError("seen_tokens must be an integer >= every current head length")
    return query_heads // len(keys), query_len, seen_tokens


def _head_scores(
    key: torch.Tensor,
    queries: torch.Tensor,
    history_start: int,
    recent: int,
    config: BaselineConfig,
    layer_idx: int,
    head: int,
) -> torch.Tensor:
    """Source-equivalent causal window scoring, for one KV head's Q group."""
    n = key.shape[0]
    history_end = n - recent
    if history_end <= history_start:
        return key.new_empty((0,), dtype=torch.float32)
    observed = min(config.window_size, queries.shape[1])
    query = queries[:, -observed:, :]
    work_key = key
    if config.score_precision == "float32":
        query = query.float()
        work_key = key.float()
    # Outer model autocast must not cast our explicit FP32 scores back down.
    with torch.autocast(device_type=key.device.type, enabled=False):
        logits = torch.matmul(query, work_key.transpose(0, 1)) / math.sqrt(key.shape[1])
    if not torch.isfinite(logits).all().item():
        raise FloatingPointError(
            f"Non-finite QK scores at layer {layer_idx}, head {head}; "
            "use score_precision='float32' for low-precision inputs"
        )
    # Queries are right-aligned to the current head history, even after older
    # positions have been evicted. Their absolute RoPE positions stay in K/Q.
    query_positions = torch.arange(n - observed, n, device=key.device)
    key_positions = torch.arange(n, device=key.device)
    future = key_positions[None, :] > query_positions[:, None]
    logits = logits.masked_fill(future[None, :, :], torch.finfo(logits.dtype).min)
    weights = torch.softmax(logits, dim=-1, dtype=torch.float32)
    if config.score_precision == "input":
        weights = weights.to(queries.dtype)
    scores = weights[:, :, history_start:history_end].mean(dim=-2)
    if config.gqa_reduce == "mean":
        scores = scores.mean(dim=0)
    else:
        scores = scores.max(dim=0).values
    scores = scores[None, None, :]
    pool = F.avg_pool1d if config.pooling == "avgpool" else F.max_pool1d
    scores = pool(
        scores, kernel_size=config.kernel_size, padding=config.kernel_size // 2, stride=1
    )[0, 0]
    if not torch.isfinite(scores).all().item():
        raise FloatingPointError(f"Non-finite attention scores at layer {layer_idx}, head {head}")
    return scores


@torch.no_grad()
def select_layer(
    keys: list[torch.Tensor],
    queries: torch.Tensor,
    config: BaselineConfig,
    layer_idx: int,
    num_layers: int,
    prefix_tokens: int = 0,
    seen_tokens: int | None = None,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    """Return retained indices into every current KV head and JSON-safe metadata.

    Q is the newly prefetched chunk, [query_heads, chunk_tokens, head_dim]. K is
    a list of [retained_tokens + chunk_tokens, head_dim] tensors. Q heads use
    contiguous GQA groups, as Qwen does. Indices retain upstream score order,
    followed by the causal recent window. Visual scope prepends the protected
    prefix; prompt scope treats prefix tokens as normal budgeted candidates.

    Dense prefill parity is guaranteed for non-tied finite scores under the
    same precision. Streaming changes are explicit: ragged histories, fewer
    than W new queries, cumulative Pyramid lengths, upstream Torch tie ordering,
    and zero historical mass yielding a normalization ratio of zero. No NaN
    or infinity is repaired; malformed tensors and non-finite QK fail loudly.
    """
    groups, query_len, seen_tokens = _validate_inputs(
        keys, queries, config, layer_idx, num_layers, prefix_tokens, seen_tokens
    )
    protected = prefix_tokens if config.budget_scope == "visual" else 0
    lengths = [key.shape[0] for key in keys]
    recents = [min(config.window_size, n - protected) for n in lengths]
    histories = [n - protected - recent for n, recent in zip(lengths, recents)]
    capacity = pyramid_layer_capacity(
        config, layer_idx, num_layers, seen_tokens, prefix_tokens
    )
    history_capacity = max(0, capacity - config.window_size)
    device = queries.device
    zero_mass_heads: list[int] = []
    proposed_counts: list[int] = []
    scores: list[torch.Tensor] = []
    sorted_indices: list[torch.Tensor] = []
    selected: list[torch.Tensor] = []

    # Match upstream's strict short-cache tests, including the equality case:
    # AdaKV adapts when historical length == base_capacity; PyramidKV sorts
    # when sequence length == scheduled layer capacity. Uniform uses the same
    # strict SnapKV test with its constant configured capacity.
    if config.method == "adakv":
        bypass = all(length < history_capacity for length in histories)
    else:
        bypass = all(n - protected < capacity for n in lengths)

    if bypass:
        selected = [torch.arange(n, device=device, dtype=torch.long) for n in lengths]
        proposed_counts = histories.copy()
    else:
        for head, (key, recent) in enumerate(zip(keys, recents)):
            score = _head_scores(
                key, queries[head * groups:(head + 1) * groups],
                protected, recent, config, layer_idx, head,
            )
            if config.method in ("pyramidkv", "uniform"):
                # SnapKVCluster uses topk directly, including Torch tie order.
                indices = torch.topk(score, k=min(history_capacity, score.numel())).indices
                scores.append(score)
                sorted_indices.append(indices)
            else:
                values, indices = torch.sort(score, descending=True)
                scores.append(values)
                sorted_indices.append(indices)
        if config.method in ("pyramidkv", "uniform"):
            proposed_counts = [history_capacity] * len(keys)
        else:
            adaptive = []
            for head, score in enumerate(scores):
                if config.normalize:
                    total = score.sum()
                    if total.item() == 0:
                        # Original 0/0 is undefined. Streaming may see no
                        # representable mass on history at all; rank it at 0.
                        zero_mass_heads.append(head)
                        ratio = total.new_zeros(())
                    else:
                        ratio = score[:history_capacity].sum() / total
                    score = score * ratio
                adaptive.append(score)
            flat = torch.cat(adaptive)
            take = min(len(keys) * history_capacity, flat.numel())
            winners = torch.topk(flat, k=take).indices
            owners = torch.repeat_interleave(
                torch.arange(len(keys), device=device),
                torch.tensor(histories, device=device),
            )
            allocations = torch.bincount(owners[winners], minlength=len(keys))
            floor_capacity = int(history_capacity * config.floor_alpha)
            # Preserve SparseMM's round-to-even and integer floor. Correcting
            # these to force an exact global budget would change the baseline.
            proposed_counts = torch.round(
                allocations * (1 - config.floor_alpha) + floor_capacity
            ).int().tolist()
        for head, key in enumerate(keys):
            past = sorted_indices[head][:proposed_counts[head]] + protected
            prefix = torch.arange(protected, device=device, dtype=torch.long)
            recent = torch.arange(
                lengths[head] - recents[head], lengths[head], device=device, dtype=torch.long
            )
            selected.append(torch.cat((prefix, past, recent)))

    kept = [index.numel() for index in selected]
    kept_history = [n - protected - recent for n, recent in zip(kept, recents)]
    expected = len(keys) * (capacity + protected)
    metadata: dict[str, Any] = {
        "method": config.method,
        "budget_scope": config.budget_scope,
        "configured_budget": config.budget,
        "layer_capacity": capacity,
        "history_capacity_per_head": history_capacity,
        "prefix_tokens_kept": protected,
        "seen_tokens": seen_tokens,
        "observed_queries": min(query_len, config.window_size),
        "gqa_groups": groups,
        "gqa_reduce": config.gqa_reduce,
        "score_precision": config.score_precision,
        "before_head_lengths": lengths,
        "head_lengths": kept,
        "recent_counts": recents,
        "proposed_history_counts": proposed_counts,
        "retained_history_counts": kept_history,
        "target_total_tokens": expected,
        "actual_total_tokens": sum(kept),
        "actual_mean_tokens": sum(kept) / len(keys),
        "allocation_drift_tokens": sum(kept) - expected,
        "bypassed_short_cache": bypass,
        "compressed": any(after < before for after, before in zip(kept, lengths)),
        "rounding": "upstream",
        "tie_policy": "upstream_torch",
        "zero_mass_normalization": 0,
        "zero_mass_heads": zero_mass_heads,
        "upstream_commit": UPSTREAM_COMMIT,
    }
    return selected, metadata
