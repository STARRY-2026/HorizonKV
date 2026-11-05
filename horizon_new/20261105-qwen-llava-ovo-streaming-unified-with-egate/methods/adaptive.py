"""Method dispatch for AdaKV/PyramidKV, preserving the original shared formulas."""
def select_layer(keys, queries, config, **kwargs):
    if config.method == 'adakv':
        from methods.adakv.policy import select_layer as select
    elif config.method == 'pyramidkv':
        from methods.pyramidkv.policy import select_layer as select
    else:
        # Keep the frozen shared implementation's optional Uniform path available.
        from sparsemm_baselines.core import select_layer as select
    return select(keys, queries, config, **kwargs)
