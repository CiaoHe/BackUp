import torch
from torch import nn, einsum
import torch.nn.functional as F 

from einops import rearrange, repeat 
from einops.layers.torch import Rearrange
from torch.nn.modules.linear import Linear 

#helpers 

def pair(t):
    if isinstance(t, tuple):
        return t 
    return (t,t)

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

#pre-layernorm

class PreNorm(nn.Module):
    def __init__(self, dim:int,fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self,x,**kwargs):
        return self.fn(self.norm(x), **kwargs)

# FFN

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

# Attention

class Attention(nn.Module):
    """docstring for  Attention"""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)

        # e.g. in Larger branch, Q comes from L's cls-token
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        # K,V comes from cat(L's cls-token, S's patch-tokens)
        self.to_kv = nn.Linear(dim, inner_dim*2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, context=None, kv_include_self = False):
        b,b,_,h = *x.shape, self.heads
        q = self.to_q(x)

        context = default(context, x) #if has no context, simple Attention
        if kv_include_self:
            # if in the cross-attention mode, 
            # context = cat(L's cls-token, S's patch-tokens)
            context = torch.cat((x, context), dim = 1)
        k,v = self.to_kv(context).chunk(2, dim=-1)

        #split to multi-heads
        q,k,v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q,k,v))

        dots = einsum('b h i d, b h j d -> b h i j', q,k)*self.scale
        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d',attn,v)
        out = rearrange(out,'b h n d -> b n (h d)')
        return self.to_out(out)

# transformer encoder, for small and large patches

class Transformer(nn.Module):
    """docstring for Transformer"""
    def __init__(
            self,
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout = 0.
        ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head,
                        dropout=dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                ])
            )
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x 
            x = ff(x) + x
        return self.norm(x)

# project CLS tokens, act as $f'()$ in the original paper
# write as Wrapper
class ProjectInOut(nn.Module):
    def __init__(self, dim_in, dim_out, fn):
        super().__init__()
        self.fn = fn 

        need_projection = dim_in != dim_out
        self.project_in = nn.Linear(dim_in, dim_out) if need_projection else nn.Identity()
        self.project_out = nn.Linear(dim_out, dim_in) if need_projection else nn.Identity()

    def forward(self, x, *args, **kwargs):
        x = self.project_in(x)
        x = self.fn(x, *args, **kwargs)
        x = self.project_out(x)
        return x

# Cross-attention Transformer

class CrossTransformer(nn.Module):
    """docstring for CrossTransformer"""
    def __init__(self, 
            sm_dim,
            lg_dim,
            depth,
            heads,
            dim_head,
            dropout
        ):
        super(CrossTransformer, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ProjectInOut(sm_dim, lg_dim, PreNorm(lg_dim, Attention(lg_dim, heads, dim_head, dropout))),
                ProjectInOut(lg_dim, sm_dim, PreNorm(sm_dim, Attention(sm_dim, heads, dim_head, dropout))),
            ]))
    
    def forward(self, sm_tokens, lg_tokens):
        (sm_cls, sm_patch_tokens), (lg_cls, lg_patch_tokens) = map(
            lambda t: (t[:,:1], t[:,1:]),
            (sm_tokens, lg_tokens)
        )
        for sm_attend_lg, lg_attend_sm in self.layers:
            sm_cls = sm_attend_lg(sm_cls, context = lg_patch_tokens, kv_include_self = True)
            lg_cls = lg_attend_sm(lg_cls, context = sm_patch_tokens, kv_include_self = True)
        
        sm_tokens = torch.cat([sm_cls, sm_patch_tokens], dim = 1)
        lg_tokens = torch.cat([lg_cls, lg_patch_tokens], dim = 1)
        return sm_tokens, lg_tokens

# multi-scale Encoder

class MultiScaleEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth, 
        sm_dim,
        lg_dim,
        sm_enc_params, #dict type
        lg_enc_params,
        cross_attn_heads,
        cross_attn_depth,
        cross_attn_dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Transformer(dim = sm_dim, dropout=dropout, **sm_enc_params),
                Transformer(dim = lg_dim, dropout=dropout, **lg_enc_params),
                CrossTransformer(sm_dim, lg_dim, depth=cross_attn_depth,
                    heads = cross_attn_heads, dim_head=cross_attn_dim_head, dropout= dropout)
            ]))
    
    def forward(self, sm_tokens, lg_tokens):
        for sm_enc, lg_enc, cross_attend in self.layers:
            sm_tokens = sm_enc(sm_tokens)
            lg_tokens = lg_enc(lg_tokens)
            sm_tokens, lg_tokens = cross_attend(sm_tokens, lg_tokens)
        
        return sm_tokens, lg_tokens

# patch-based image to tokens Embedder

class ImageEmbedder(nn.Module):
    def __init__(
        self,
        *,
        dim,
        image_size,
        patch_size,
        channel = 3,
        dropout = 0.
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image size must be divisible by patch_size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channel * patch_size**2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p) (w p) -> b (h w) (p p c)', p = patch_size),
            nn.Linear(patch_dim, dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, img):
        x = self.to_patch_embedding(img) # b n d
        b,n,d= x.shape
        cls_tokens = rearrange(self.cls_token, '{} n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:,:(n+1)]

        return self.dropout(x)
    
# cross ViT class

class CrossViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        num_classes,
        sm_dim, 
        lg_dim, 
        sm_patch_size = 12,
        sm_enc_depth = 1, # N in Figure2
        sm_enc_heads = 8,
        sm_enc_mlp_dim = 2048,
        sm_enc_dim_head = 64,
        lg_patch_size = 16,
        lg_enc_depth = 4, # M in Figure2
        lg_enc_heads = 8,
        lg_enc_mlp_dim = 2048,
        lg_enc_dim_head = 64,
        cross_attn_depth = 2,
        cross_attn_heads = 8,
        cross_attn_dim_head = 64,
        depth = 3, # K in Figure2
        dropout = 0.1,
        emb_dropout = 0.1,
    ):
        super().__init__()
        self.sm_image_embedder = ImageEmbedder(sm_dim, image_size, patch_size=sm_patch_size,
            dropout=emb_dropout)
        self.lg_image_embedder = ImageEmbedder(lg_dim, image_size, patch_size=lg_patch_size,
            dropout=emb_dropout)

        self.multi_scale_encoder = MultiScaleEncoder(
            depth=depth,
            sm_dim=sm_dim,
            lg_dim=lg_dim,
            sm_enc_params=dict(
                depth = sm_enc_depth,
                heads = sm_enc_heads,
                dim_head = sm_enc_dim_head,
                mlp_dim = sm_enc_mlp_dim,
            ),
            lg_enc_params=dict(
                depth = lg_enc_depth,
                heads = lg_enc_heads,
                dim_head = lg_enc_dim_head,
                mlp_dim = lg_enc_mlp_dim,
            ),
            cross_attn_heads=cross_attn_heads,
            cross_attn_depth=cross_attn_depth,
            cross_attn_dim_head=cross_attn_dim_head,
            dropout=dropout,
        )

        self.sm_mlp_head = nn.Sequential(nn.LayerNorm(sm_dim), nn.Linear(sm_dim, num_classes))
        self.lg_mlp_head = nn.Sequential(nn.LayerNorm(lg_dim), nn.Linear(lg_dim, num_classes))
    
    def forward(self, img):
        sm_tokens = self.sm_image_embedder(img)
        lg_tokens = self.lg_image_embedder(img)
        sm_tokens, lg_tokens = self.multi_scale_encoder(sm_tokens, lg_tokens)

        sm_cls, lg_cls = map(lambda t: t[:,0], (sm_tokens,lg_tokens))
        sm_logits = self.sm_mlp_head(sm_cls)
        lg_logits = self.lg_mlp_head(lg_cls)
        return sm_logits + lg_logits