import torch
from torch import nn, einsum 
import torch.nn.functional as F 

from einops import rearrange, repeat 
from einops.layers.torch import Rearrange 

#helpers

def pair(t):
    if isinstance(t, tuple):
        return t 
    else:
        return (t,t) 

# classes

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

class Attetion(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = heads * dim_head
        project_out = not (heads == 1 and dim_head == dim) #make sure whether identity project

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim*3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q,k,v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super.__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attetion(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, hidden_dim=mlp_dim, dropout=dropout))
            ]))

    def forward(self,x):
        for attn, ff in self.layers:
            x = attn(x) + x 
            x = ff(x) + x 
        return x 

class ViT(nn.Module):
    def __init__(self, 
            *,
            image_size,
            patch_size,
            num_classes,
            dim,
            depth,
            heads,
            mlp_dim,
            pool = 'cls',
            channels = 3,
            dim_head = 64,
            dropout = 0.,
            embed_dropout = 0.
        ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        
        assert image_height % patch_height ==0 and image_width % patch_width == 0, 'Image dimensions must be \
            divisible by patch size'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width 
        assert pool in {'cls','mean'}, 'pool type must be either (cls token) or mean (mean pooling)'

        self.to_patch_emnbedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, 
                p2= patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, dim))
        self.cls_token = nn.Parameter(torch.randn(1,1,dim))
        self.dropout = nn.Dropout(embed_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head. mlp_dim, dropout=dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = PreNorm(dim, nn.Linear(dim, num_classes))

    def forward(self,img):
        x = self.to_patch_emnbedding(img) # b n d, b=h*w, d=c*p_h*p_w
        b,n,_ = x.shape

        cls_tokens = repeat(self.cls_token, '{} n d -> b n d', b = b)
        x = torch.cat([cls_tokens, x],dim=1)
        x += self.pos_embedding[:,:(n+1)]
        x = self.dropout(x)

        x = self.transformer(x) #b n d
        x = x.mean(dim=1) if self.pool == 'mean' else x[:,0] # b d

        x = self.to_latent(x)
        return self.mlp_head(x) 