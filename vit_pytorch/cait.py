from random import randrange
import torch
from torch import nn, einsum
from torch import Tensor
from torch import tensor
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def dropout_layers(layers, dropout):
    if dropout == 0:
        return layers
    
    num_layers = len(layers)
    to_drop = torch.zeros(num_layers).uniform_(0.,1.) < dropout

    # make sure at least one layer exists
    if all(to_drop):
        rand_index = randrange(num_layers)
        to_drop[rand_index] = False
    
    layers = [layer for (layer, drop) in zip(layers, to_drop) if not drop]
    return layers

# classes

class LayerScale(nn.Module):
    def __init__(self, dim, fn, depth):
        super().__init__()
        if depth <= 18:
            init_eps = 0.1
        elif 18<depth<=24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6
        
        scale = torch.zeros(1,1,dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale

class PreNorm(nn.Module):
    def __init__(self, dim:int, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn 
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

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

    def forward(self,x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim, bias=False)

        self.attend = nn.Softmax(dim=-1)

        # Talking heads technique
        self.mix_heads_pre_attn = nn.Parameter(torch.randn(heads, heads))
        self.mix_heads_post_attn = nn.Parameter(torch.randn(heads, heads))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x:Tensor, context:Tensor = None):
        b, n, _, h = *x.shape, self.heads

        context = x if not exists(context) else torch.cat([x, context], dim=1)

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q,k,v = map(lambda t:rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        dots = einsum('b h i j, h g -> b g i j', dots, self.mix_heads_pre_attn)
        attn = self.attend(dots)
        dots = einsum('b h i j, h g -> b g i j', dots, self.mix_heads_post_attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout = 0.,
        layer_dropout = 0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.layer_dropout = layer_dropout

        for ind in range(depth):
            self.layers.append(nn.ModuleList([
                LayerScale(dim, PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)), depth=ind+1),
                LayerScale(dim, PreNorm(dim, FeedForward(dim, hidden_dim=mlp_dim, dropout=dropout)), depth=ind+1)
            ]))
    
    def forward(self, x, context=None):
        layers = dropout_layers(self.layers, dropout=self.layer_dropout)

        for attn, ff in layers:
            x = attn(x, context) + x
            x = ff(x) + x
        
        return x

class CaiT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth, # SA depth
        cls_depth, # CA depth
        heads,
        mlp_dim,
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.,
        layer_dropout = 0.
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size**2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.patch_transformer = Transformer(dim=dim, depth=depth, heads=heads, dim_head=dim_head,  mlp_dim=mlp_dim, dropout=dropout, layer_dropout=layer_dropout)
        self.cls_transformer = Transformer(dim=dim, depth=cls_depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout, layer_dropout=layer_dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    
    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.patch_transformer(x)

        cls_tokens = repeat(self.cls_token, '() () d -> b () d', b = b)
        x = self.cls_transformer(cls_tokens, context = x)

        return self.mlp_head(x[:,0])