"""SparseMM-style method dispatch for the installed Qwen2.5-VL API."""

METHODS = ("protokv", "adakv", "pyramidkv", "lukv", "sparsemm", "uniform")


def replace_qwen2_5(method):
    if method not in METHODS:
        raise ValueError(f"Unknown SparseMM Qwen2.5-VL method: {method!r}")
    from sparsemm_protokv.qwen2_5 import install
    install()
    return method


replace_qwen = replace_qwen2_5
