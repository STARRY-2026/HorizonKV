"""AdaKV entry into the pinned shared selector, with the method checked."""
def select_layer(keys, queries, config, **kwargs):
    if config.method != 'adakv':
        raise ValueError('AdaKV policy requires method=adakv')
    from sparsemm_baselines.core import select_layer as pinned_select
    return pinned_select(keys, queries, config, **kwargs)
