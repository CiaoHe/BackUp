import torch
from torch import nn
from operator import mul
from functools import reduce

from einops import rearrange, repeat
from torch.nn.modules.container import ParameterList


class AxialPositionalEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        # axial shape will multiply up to the maximum sequence length allowed , Ex. (64, 64).
        axial_shape,
        # if not specified, dimensions will default to 'dim' for all axials and summed at the end. if specified, each axial will have the specified dimension and be concatted together. the concatted dimensions needs to sum up to the `dim` (256 + 256 = 512) if axial_dims = (256,256)
        axial_dims = None
    ):
        super().__init__()
        self.dim = dim
        self.shape = axial_shape
        self.max_seqlen = reduce(mul, axial_shape, 1)

        self.summed = axial_dims is None 
        axial_dims = ((dim,)*len(axial_shape) if self.summed else axial_dims)

        assert len(self.shape) == len(axial_dims), 'number of axial dimensions must equal the number of dimensions in the shape'
        assert self.summed or (not self.summed and sum(axial_dims) == dim), f'axial dimensions must sum up to the target dimension {dim}'

        self.weights = ParameterList([])

        for ind, (shape, axial_dim) in enumerate(zip(self.shape, axial_dims)):
            ax_shape = [1] * len(self.shape)
            ax_shape[ind] = shape
            ax_shape = (1, *ax_shape, axial_dim) # [1, ..., ax_shape,..., ax_dim]
            ax_emb = nn.Parameter(torch.zeros(ax_shape).normal_(0.,1.))
            self.weights.append(ax_emb)
        
    def forward(self, x:torch.Tensor):
        b, t, e = x.shape
        assert (t<=self.max_seqlen), f'Sequence length ({t}) must be less than the maximum sequence length allowed ({self.max_seq_len})'
        embs = []

        for ax_emb in self.weights:
            axial_dim = ax_emb.shape[-1]
            expand_shape = (b, *self.shape, axial_dim)
            emb = ax_emb.expand(expand_shape).reshape(b, self.max_seqlen, axial_dim)
            embs.append(emb)
        
        pos_emb = sum(embs) if self.summed else torch.cat(embs, dim=-1)
        return pos_emb[:, :t].to(x)
    
    # Axial Positional Embedding for Images

class AxialPositionalEmbeddingImage(nn.Module):
    def __init__(
        self,
        dim,
        axial_shape,
        axial_dims = None
    ):
        super().__init__()
        assert len(axial_shape) == 2, 'Axial shape for image must have 2 dimensions'
        self.pos_emb = AxialPositionalEmbedding(dim, axial_shape=axial_shape, axial_dims=axial_dims)
        # output must be (b, )
    
    def forward(self, img):
        b, c, h, w = img.shape
        img = rearrange(img, 'b c h w -> b (h w) c')
        pos_emb = self.pos_emb(img)
        return rearrange(pos_emb, 'b (h w) d -> b d h w', h = h)