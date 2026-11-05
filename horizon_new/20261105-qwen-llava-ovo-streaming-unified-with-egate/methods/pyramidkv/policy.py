"""PyramidKV entry into the pinned shared selector, with the method checked."""
def select_layer(keys, queries, config, **kwargs):
    if config.method != 'pyramidkv':
        raise ValueError('PyramidKV policy requires method=pyramidkv')
    from sparsemm_baselines.core import select_layer as pinned_select
    return pinned_select(keys, queries, config, **kwargs)
