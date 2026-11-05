"""One registry for the CLI and Python API; importing it never loads a model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Method:
    name: str
    description: str
    runtime_method: str
    audit_module: str = "sparsemm_protokv.audit"
    default: bool = True
    fixed_budget: int | None = None


METHODS = {
    "adakv": Method("adakv", "Attention-adaptive KV head budgets", "adakv"),
    "pyramidkv": Method("pyramidkv", "Layer-dependent pyramid KV budgets", "pyramidkv"),
    "lukv": Method("lukv", "Offline long-term utility budgets; online attention eviction", "lukv", "sparsemm_lukv.audit"),
    "sparsemm": Method("sparsemm", "Calibrated visual-head budgets; attention eviction", "sparsemm"),
    "uniform": Method("uniform", "Equal per-head budgets; attention eviction", "uniform"),
    "protokv": Method("protokv", "Prototype memory adaptation (explicit selection only)", "protokv", default=False),
    "full": Method("full", "Large-budget AdaKV control; audit must show zero evictions", "adakv", default=False, fixed_budget=1_000_000),
}
DEFAULT_METHODS = tuple(name for name, method in METHODS.items() if method.default)


def resolve_methods(method=None, methods=None):
    """Return validated, ordered method names, with no accidental ProtoKV run."""
    if method is not None and methods is not None:
        raise ValueError("Choose method or methods, not both")
    names = (method,) if method is not None else (DEFAULT_METHODS if methods is None else methods)
    if isinstance(names, str):
        names = tuple(part.strip() for part in names.split(","))
    names = tuple(names)
    if not names or any(not isinstance(name, str) or name not in METHODS for name in names):
        raise ValueError("Methods must be selected from: " + ", ".join(METHODS))
    if len(set(names)) != len(names):
        raise ValueError("Duplicate methods are not allowed")
    return names
