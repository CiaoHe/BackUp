import torch
from torch import nn, einsum
from einops import rearrange, repeat

class SinusoidalEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** torch.arange(0,dim,2).float() / dim)
        # inv_feq = 1 / (10000 ^ (i/2d))
        # shape: [d//2, ]
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, x):
        n = x.shape[-2]
        t = torch.arange(n, device=x.device).type_as(self.inv_freq)
        # shape: [n,]
        freqs = torch.einsum('i,j -> i j', t, self.inv_freq)
        # shape: [n, d//2]
        emb = torch.cat([freqs, freqs], dim = -1)
        # shape: [n, d]
        return emb[None, :, :] #shape [1, n, d]

def rotate_half(x):
    x = rearrange(x, '... (d j) -> ... d j', j = 2)
    x1,x2 = x.chunk(2, dim = -1)
    x = torch.cat([-x2, x1], dim = -1)
    return rearrange(x, '... d j -> ... (d j)')

def apply_rotary_emb_pos(q, k, freqs):
    '''
    Args:
        q: [b, n, d]
        k: [b, n, d]
        freqs: [1, n, d]
    '''
    # print(rotate_half(q).shape)
    q,k = map(lambda t: (t * freqs.cos()) + (rotate_half(t) * freqs.sin()), (q,k))
    return q,k