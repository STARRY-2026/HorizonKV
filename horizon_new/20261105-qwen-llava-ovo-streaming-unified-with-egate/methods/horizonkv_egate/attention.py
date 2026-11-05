"""GQA ragged attention shared computation, copied from the bundled baseline adapter."""
import torch
import torch.nn.functional as F
try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None

def ragged_attention(queries, keys, values, layer_cache, scaling):
    """Attend to each head's history + current causal chunk using GQA."""
    _, query_heads, q_len, dim = queries.shape
    kv_heads = keys.shape[1]
    groups = query_heads // kv_heads
    old_lengths = layer_cache.head_lens.tolist()
    all_k, all_v = [], []
    for head in range(kv_heads):
        hk, hv = layer_cache.get_segment(head)
        all_k.append(torch.cat((hk, keys[0, head])))
        all_v.append(torch.cat((hv, values[0, head])))
    if (flash_attn_varlen_func is not None and queries.is_cuda
            and queries.dtype in (torch.float16, torch.bfloat16)):
        q = queries[0].reshape(kv_heads, groups, q_len, dim).permute(0, 2, 1, 3).reshape(-1, groups, dim).contiguous()
        k = torch.cat(all_k)[:, None].contiguous()
        v = torch.cat(all_v)[:, None].contiguous()
        lens = torch.tensor([n + q_len for n in old_lengths], dtype=torch.int32, device=queries.device)
        cu_k = torch.cat((lens.new_zeros(1), lens.cumsum(0, dtype=torch.int32)))
        cu_q = torch.arange(kv_heads + 1, dtype=torch.int32, device=queries.device) * q_len
        output = flash_attn_varlen_func(q, k, v, cu_q, cu_k, q_len, int(lens.max()),
                                       softmax_scale=scaling, causal=True)
        return output.reshape(kv_heads, q_len, groups, dim).permute(0, 2, 1, 3).reshape(query_heads, q_len, dim).transpose(0, 1).reshape(1, q_len, -1)
    outputs = []
    for head in range(kv_heads):
        q = queries[0, head * groups:(head + 1) * groups]
        k, v = all_k[head], all_v[head]
        logits = q.float() @ k.float().T * scaling
        key_indices = torch.arange(k.shape[0], device=k.device)
        query_indices = old_lengths[head] + torch.arange(q_len, device=k.device)
        logits = logits.masked_fill(key_indices[None] > query_indices[:, None], -torch.inf)
        probs = F.softmax(logits, dim=-1).to(q.dtype)
        outputs.append(probs @ v)
    return torch.cat(outputs).transpose(0, 1).reshape(1, q_len, -1)
