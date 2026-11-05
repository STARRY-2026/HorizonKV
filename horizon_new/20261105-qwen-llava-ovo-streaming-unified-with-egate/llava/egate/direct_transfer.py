"""Move only selected recent KV from a four-frame batch into per-head history."""
import torch
from pae_variable_recent import compact_positions, TOKENS_PER_FRAME
from sparsemm_native.dscache_position_agnostic import remove_mrope_from_key, apply_mrope_to_key

def transfer_batch(model, recent_cache, recent_positions, one_cache, one_positions, prefix_length, allocation, probe_fn, selector_fn, tokenizer, previous):
    start = previous['seen_frames'] if previous else 0
    prior = previous['heads'] if previous else {}
    prefix_uids = list(range(-prefix_length, 0))
    full_uids = list(range(start * TOKENS_PER_FRAME, (start + 4) * TOKENS_PER_FRAME))
    frames = sorted({uid // TOKENS_PER_FRAME for entry in prior.values() for uid in entry['uids']} | set(range(start, start+4)))
    recent_probes = probe_fn(model, tokenizer, int(recent_positions.max().item())+1)
    history_probes = probe_fn(model, tokenizer, prefix_length + 32 * (len(frames)-1) + 32)
    heads = {}
    for layer_idx, layer in enumerate(model.language_model.layers):
        full_rot, full_val = recent_cache[layer_idx]
        one_rot, one_val = one_cache[layer_idx]
        sections = layer.self_attn.rope_scaling['mrope_section']
        full_raw = remove_mrope_from_key(model,full_rot,recent_positions,sections).to(full_rot.dtype)
        one_raw = remove_mrope_from_key(model,one_rot,one_positions,sections).to(one_rot.dtype)
        qs, pseudo = recent_probes[layer_idx]
        hqs, hpseudo = history_probes[layer_idx]
        group = qs.shape[0] // 4
        for h in range(4):
            config=allocation[layer_idx,h]
            old=prior.get((layer_idx,h))
            old_count=len(old['uids']) if old else 0
            history_quota=int(config['history_quota'])
            recent_budget=int(config['recent_quota']) + history_quota - old_count
            single=config['recent_group']=='one_frame'
            rot,raw,val,uids=(one_rot,one_raw,one_val,full_uids[-TOKENS_PER_FRAME:]) if single else (full_rot,full_raw,full_val,full_uids)
            idx=selector_fn(rot[0,h],qs[h*group:(h+1)*group],pseudo[h],prefix_length,recent_budget)
            selected_uids=[uids[i-prefix_length] for i in idx.tolist()]
            recent_keys=raw[0,h].index_select(0,idx)
            recent_values=val[0,h].index_select(0,idx)
            keys=torch.cat(([old['keys']] if old else [])+[recent_keys],dim=0)
            vals=torch.cat(([old['values']] if old else [])+[recent_values],dim=0)
            ids=(old['uids'] if old else [])+selected_uids
            candidate=torch.cat((raw[0,h,:prefix_length],keys),dim=0)
            positions=compact_positions(prefix_uids+ids,prefix_length,device=model.device,frame_order=frames)
            rotated=apply_mrope_to_key(model,candidate[None,None],positions,sections).to(candidate.dtype)[0,0]
            keep=selector_fn(rotated,hqs[h*group:(h+1)*group],hpseudo[h],prefix_length,history_quota,protect=0)-prefix_length
            newids=[ids[i] for i in keep.tolist()]
            if len(newids)>history_quota or len(set(newids))!=len(newids):raise RuntimeError('Invalid historical quota or duplicated positions')
            heads[layer_idx,h]={'keys':keys.index_select(0,keep).contiguous(),'values':vals.index_select(0,keep).contiguous(),'uids':newids}
    return {'heads':heads,'seen_frames':start+4}
