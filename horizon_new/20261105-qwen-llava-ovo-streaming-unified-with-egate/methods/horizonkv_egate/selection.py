"""Original EGate recent32 selector; scoring and causal mask retained unchanged."""
import math
import torch
import torch.nn.functional as F

def snap_select(key,queries,_unused,visual_start,budget,protect=32):
    n=key.shape[0]; visual=n-visual_start;budget=min(int(budget),visual)
    if visual<=budget:return torch.arange(visual_start,n,device=key.device)
    if budget<=0:return torch.empty(0,dtype=torch.long,device=key.device)
    recent=min(protect,visual,budget); end=n-recent; count=budget-recent
    scores=queries.float()@key.float().T/math.sqrt(key.shape[-1]);qlen=queries.shape[1]
    # The observation queries are the newest query tokens. Mask future keys.
    mask=torch.arange(n,device=key.device)[None,:] > (n-qlen+torch.arange(qlen,device=key.device))[:,None]
    scores.masked_fill_(mask.unsqueeze(0),torch.finfo(scores.dtype).min)
    weights=torch.softmax(scores,dim=-1).mean(dim=(0,1))[visual_start:end]
    if weights.numel():weights=F.avg_pool1d(weights[None,None],kernel_size=5,padding=2,stride=1)[0,0]
    chosen=torch.topk(weights,count).indices.sort().values+visual_start if count else torch.empty(0,dtype=torch.long,device=key.device)
    return torch.cat((chosen,torch.arange(end,n,device=key.device)))
