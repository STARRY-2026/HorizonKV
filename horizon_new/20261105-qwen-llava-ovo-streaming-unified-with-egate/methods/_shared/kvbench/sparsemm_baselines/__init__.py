"""SparseMM-backed AdaKV and PyramidKV streaming reproduction."""
__all__ = ["BaselineConfig", "dense_reference", "load_upstream", "select_layer"]


def __getattr__(name):
    # CLI dry-runs must not require torch merely to import the package.
    if name in __all__:
        from . import core
        return getattr(core, name)
    raise AttributeError(name)
