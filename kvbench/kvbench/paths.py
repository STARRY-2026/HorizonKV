"""Central resource paths; runtime sources stay inside this project."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resource_root():
    return Path(os.environ.get("KV_BENCH_RESOURCE_ROOT", str(PROJECT_ROOT.parent))).expanduser().resolve()


def hermes_source_dir():
    """Find Hermes beside a standalone checkout or at the enclosing repo root."""
    root = resource_root()
    sibling = root / "hermes"
    if sibling.is_dir():
        return sibling
    if (root / "eval/eval_multiple_choice.py").is_file():
        return root
    return sibling


def default_model_path():
    return Path(os.environ.get("KV_BENCH_MODEL_PATH", str(hermes_source_dir() / "models/Qwen2.5-VL-7B-Instruct"))).expanduser().resolve()


def default_annotation_path():
    return Path(os.environ.get("KV_BENCH_ANNO_PATH", str(PROJECT_ROOT / "configs/ovobench_smoke.json"))).expanduser().resolve()


def default_lukv_profile_path():
    return Path(os.environ.get("KV_BENCH_LUKV_PROFILE_PATH", str(PROJECT_ROOT / "sparsemm_lukv/assets/attention_native10v_20261014/profile.json"))).expanduser().resolve()


def default_sparsemm_score_path():
    return Path(os.environ.get("KV_BENCH_SPARSEMM_SCORE_PATH", str(PROJECT_ROOT / "assets/sparsemm/head_scores.csv"))).expanduser().resolve()


def ensure_resources():
    """Check the compatibility resource link; never silently redirect it.

    Legacy controllers resolve their evaluator and shared calibration evidence
    under PROJECT_ROOT/hermes. Only this resource link may be created here.
    """
    link = PROJECT_ROOT / "hermes"
    target = hermes_source_dir()
    if not target.is_dir():
        raise FileNotFoundError(f"Shared Hermes resource directory is missing: {target}")
    if os.path.lexists(link):
        if not link.is_symlink():
            raise ValueError(f"Expected a resource symlink at {link}; refusing to replace it")
        if not link.exists():
            raise ValueError(f"Broken Hermes resource link: {link}")
        if link.resolve() != target.resolve():
            raise ValueError(f"Hermes resource link {link} points to {link.resolve()}, but KV_BENCH_RESOURCE_ROOT requires {target.resolve()}; explicitly update the link before running")
    else:
        link.symlink_to(target, target_is_directory=True)
    return target.resolve()
