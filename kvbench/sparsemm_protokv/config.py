"""Explicit method and budget settings for the native streaming plugins."""

from dataclasses import asdict as dataclass_asdict, dataclass
import math

from protokv.full_state import ProtoConfig
from sparsemm_baselines.core import BaselineConfig


@dataclass(frozen=True)
class StreamingConfig:
    method: str = "protokv"
    budget: int = 1000
    budget_scope: str = "visual"
    window_size: int = 32
    kernel_size: int = 5
    pooling: str = "avgpool"
    floor_alpha: float = 0.2
    normalize: bool = True
    beta: int = 20
    gqa_reduce: str = "mean"
    score_precision: str = "float32"
    sparsemm_score_path: str = ""
    sparsemm_budget_path: str = ""
    sparsemm_ratio: float = 0.1
    lukv_budget_path: str = ""
    lukv_profile_path: str = ""
    lukv_selector: str = "visual_attention"
    lukv_history: str | None = None
    modes: int = 8
    subquantizers: int = 8
    codewords: int = 16
    beam: int = 32
    lambda_sp: float = 0.1
    lambda_idle: float = 0.01
    idle_threshold: int = 120
    alpha: float = 0.05
    proto_beta: float = 0.05
    eta: float = 0.05
    gamma: float = 0.05
    merge_key: float = 0.20
    merge_value: float = 0.25
    warmup_residuals: int = 64
    pq_smoothing: float = 1.0
    covariance_ridge: float = 1e-4
    kmeans_iterations: int = 10
    kmeans_batch_size: int = 32
    sparsemm_cache_policy: str = "capacity"
    sparsemm_two_bank_gate: str = "per_head_capacity"
    video_chunk_mode: str = "temporal_patch"

    def __post_init__(self):
        if self.lukv_history is None:
            # Keep historical defaults for the other three methods' manifests.
            object.__setattr__(self, "lukv_history", "none" if self.method == "lukv" else "previous_qa")
        if self.method not in ("protokv", "adakv", "pyramidkv", "lukv", "sparsemm", "uniform"):
            raise ValueError("method must be protokv, adakv, pyramidkv, lukv, sparsemm or uniform")
        if self.video_chunk_mode not in ("temporal_patch", "hermes_chunk"):
            raise ValueError("video_chunk_mode must be temporal_patch or hermes_chunk")
        if self.sparsemm_cache_policy not in ("capacity", "two_bank"):
            raise ValueError("sparsemm_cache_policy must be capacity or two_bank")
        if self.sparsemm_two_bank_gate not in ("per_head_capacity", "legacy_global"):
            raise ValueError("Unknown SparseMM two-bank short-cache gate")
        if self.sparsemm_cache_policy == "two_bank" and (
                self.method != "sparsemm" or not self.sparsemm_budget_path):
            raise ValueError("Two-bank policy requires SparseMM and an explicit split budget CSV")
        if self.budget_scope not in ("visual", "prompt"):
            raise ValueError("budget_scope must be visual or prompt")
        for name in ("budget", "modes", "subquantizers", "codewords", "beam",
                     "warmup_residuals", "kmeans_iterations", "kmeans_batch_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (not isinstance(self.idle_threshold, int) or isinstance(self.idle_threshold, bool)
                or self.idle_threshold < 0):
            raise ValueError("idle_threshold must be a nonnegative integer")
        for name in ("alpha", "proto_beta", "eta", "gamma"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0,1]")
        for name in ("lambda_sp", "lambda_idle", "merge_key", "merge_value"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("pq_smoothing", "covariance_ridge"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.warmup_residuals < self.codewords:
            raise ValueError("warmup_residuals must be at least codewords")
        if self.beam < self.modes or self.codewords ** self.subquantizers < self.modes:
            raise ValueError("beam and number of PQ tuples must support modes")
        if self.method == "protokv":
            self.proto_config(0)
        elif self.method == "uniform":
            self.baseline_config()
            if self.score_precision != "float32" or self.pooling != "avgpool" or self.gqa_reduce != "mean":
                raise ValueError("Uniform attention uses FP32 scores, avgpool and GQA mean")
        elif self.method == "sparsemm":
            if self.budget_scope != "visual":
                raise ValueError("SparseMM requires visual budgets with the initial text prefix preserved")
            if not isinstance(self.sparsemm_score_path, str):
                raise ValueError("SparseMM score path must be a string")
            if not isinstance(self.sparsemm_budget_path, str):
                raise ValueError("SparseMM external budget path must be a string")
            if not math.isfinite(self.sparsemm_ratio) or not 0 <= self.sparsemm_ratio <= 1:
                raise ValueError("SparseMM ratio must be finite and in [0,1]")
            if self.score_precision != "float32" or self.pooling != "avgpool" or self.gqa_reduce != "mean":
                raise ValueError("SparseMM streaming scoring uses FP32, avgpool and GQA mean")
            if (not isinstance(self.kernel_size, int) or isinstance(self.kernel_size, bool)
                    or self.kernel_size < 1 or self.kernel_size % 2 != 1):
                raise ValueError("SparseMM pooling kernel must be a positive odd integer")
            if (not isinstance(self.window_size, int) or isinstance(self.window_size, bool)
                    or self.window_size < 1 or self.budget <= self.window_size):
                raise ValueError("SparseMM requires budget > window_size > 0")
        elif self.method == "lukv":
            if self.budget_scope != "visual":
                raise ValueError("LU-KV requires visual budgets with the initial text prefix preserved")
            if not isinstance(self.lukv_budget_path, str) or not isinstance(self.lukv_profile_path, str):
                raise ValueError("LU asset paths must be strings")
            if self.lukv_budget_path and self.lukv_profile_path:
                raise ValueError("Choose a LU budget profile or a legacy fixed CSV, not both")
            if self.lukv_selector != "visual_attention":
                raise ValueError("The online LU-KV selector must be visual_attention")
            if self.lukv_history != "none":
                raise ValueError("Visual-attention LU-KV does not use previous QA history")
            if self.score_precision != "float32" or self.pooling != "avgpool" or self.gqa_reduce != "mean":
                raise ValueError("Visual-attention LU-KV uses FP32 scoring, avgpool and GQA mean")
            if (not isinstance(self.kernel_size, int) or isinstance(self.kernel_size, bool)
                    or self.kernel_size < 1 or self.kernel_size % 2 != 1):
                raise ValueError("LU-KV attention pooling kernel must be a positive odd integer")
            if (not isinstance(self.window_size, int) or isinstance(self.window_size, bool)
                    or self.window_size < 1 or self.budget <= self.window_size):
                raise ValueError("LU-KV requires budget > window_size > 0")
            if not self.lukv_budget_path and not self.lukv_profile_path and self.window_size != 32:
                raise ValueError("The default LU-KV attention profile uses window_size=32")
            if not self.lukv_budget_path and self.kernel_size != 5:
                raise ValueError("LU attention profiles are calibrated with kernel_size=5")
        else:
            self.baseline_config()

    def asdict(self):
        return dataclass_asdict(self)

    def baseline_config(self):
        if self.method in ("protokv", "lukv", "sparsemm"):
            raise ValueError("ProtoKV/LU-KV/SparseMM use their own policies, not a baseline selector")
        names = ("method", "budget", "window_size", "kernel_size", "pooling",
                 "floor_alpha", "normalize", "beta", "gqa_reduce",
                 "budget_scope", "score_precision")
        return BaselineConfig(**{name: getattr(self, name) for name in names})

    def proto_config(self, prefix_tokens=0):
        if self.method != "protokv":
            raise ValueError("Only ProtoKV has a prototype configuration")
        if (not isinstance(prefix_tokens, int) or isinstance(prefix_tokens, bool)
                or prefix_tokens < 0):
            raise ValueError("prefix_tokens must be a nonnegative integer")
        visual_budget = self.budget - (prefix_tokens if self.budget_scope == "prompt" else 0)
        if visual_budget <= self.modes:
            raise ValueError("Budget after charging the text prefix cannot fit near and far memory")
        names = ("modes", "subquantizers", "codewords", "beam", "lambda_sp",
                 "lambda_idle", "idle_threshold", "alpha", "eta", "gamma",
                 "merge_key", "merge_value", "warmup_residuals", "pq_smoothing",
                 "covariance_ridge", "kmeans_iterations", "kmeans_batch_size")
        return ProtoConfig(budget=visual_budget, beta=self.proto_beta,
                           **{name: getattr(self, name) for name in names})
