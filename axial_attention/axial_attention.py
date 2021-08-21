import torch 
from torch import nn
from operator import itemgetter

from torch.nn import Parameter

from reversible import ReversibleSequence

# helper functions

def exists(val):
    return val is not None

def map_el_ind(arr, ind):
    return list(map(itemgetter(ind), arr))

def sort_and_return_indices(arr):
    indices = list(range(len(arr)))
    arr = zip(arr, indices)
    arr = sorted(arr)
    return map_el_ind(arr, 0), map_el_ind(arr, 1)

# calculates the permutation to bring the input tensor to something attend-able
# also calculates the inverse permutation to bring the tensor back to its original shape

def calculate_permutations(num_dimensions, emb_dim):
    '''
    Args:
        num_dimensions: int, number of axial dimensions (images is 2, videos is 3,.)
        emb_dim: int, embedding-dim's index
    '''
    total_dimesions = num_dimensions + 2 # add batch-dim & embed-dim
    emb_dim = emb_dim if emb_dim > 0 else (emb_dim + total_dimesions) # emb_dim's index might be negative(last x)
    axial_dims = [ind for ind in range(1, total_dimesions) if ind!=emb_dim]

    permutations = []
    for axial_dim in axial_dims:
        last_two_dims = [axial_dim, emb_dim]
        dims_rest = set(range(0,total_dimesions)) - set(last_two_dims)
        permutation = [*dims_rest, *last_two_dims]
        permutations.append(permutation)
    return permutations

# helper classes

class Rezero(nn.Module):
    def __init__(self,fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.tensor(0.))
    
    def forward(self,x):
        return self.fn(x) * self.g

class Sequential(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks
    
    def forward(self,x):
        for f,g in self.blocks:
            x = f(x) + g(x)
        return x

class PermuteToFrom(nn.Module):
    def __init__(self, permuation, fn):
        super().__init__()
        self.fn = fn
        # a very tricky trick to acquire inverse_permute
        _, inv_permuation = sort_and_return_indices(permuation)
        self.permutation = permuation
        self.inv_permuation = inv_permuation
    
    def forward(self,x,**kwargs):
        axial = x.permute(*self.permutation).contiguous()

        shape = axial.shape
        *_, t, d = shape #t: axial-dimension

        #merge all but axial dimension
        axial = axial.reshape(-1, t, d)

        #attention
        axial = self.fn(axial, **kwargs)

        #restore to original shape and permutation
        axial = axial.reshape(shape)
        axial = axial.permute(*self.inv_permuation).contiguous()
        return axial

# axial pos emb

class AxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, shape, emb_dim_index = 1):
        '''
        Args:
            dim: int, embed-dimension number \n
            shape: tuple(int), aixal-dims' shape (e.g., img: (256,256)) \n
            emb_dim_index: int, embed-dimension index
        '''
        super().__init__()
        parameters = []
        total_dimensions = len(shape) + 2
        ax_dim_indexes = [i for i in range(1,total_dimensions) if i!=emb_dim_index]
        
        # according to paper, add axial-embedding for each axial-dimension
        # 只需在每一维度的axial-dimesion's 上叠加一层embedding
        '''
        For parameter efficiency we use “additively factorized” position embeddings, meaning that we parameterize them as a broadcasted sum of H × 1 × D embeddings for rows and 1 × W × D embeddings for columns.
        '''
        for axial_dim, axial_dim_index in zip(shape, ax_dim_indexes):
            tmp_shape = [1] * total_dimensions
            tmp_shape[emb_dim_index] = dim
            tmp_shape[axial_dim_index] =  axial_dim
            parameter = Parameter(torch.randn(*tmp_shape))
            parameters.append(parameter)
        
        self.params = nn.ParameterList(parameters)
    
    def forward(self, x):
        for param in self.params:
            x = x + param
        return x

# attention

class SelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dim_heads = None
    ):
        '''
        Args:
            dim: int, embed_dimension's value
            heads: int
        '''
        super().__init__()
        self.dim_heads = (dim // heads) if dim_heads is None else dim_heads
        dim_hidden = self.dim_heads * heads

        self.heads = heads
        self.to_q = nn.Linear(dim, dim_hidden, bias=False)
        self.to_kv = nn.Linear(dim, 2*dim_hidden, bias=False)
        self.to_out = nn.Linear(dim_hidden, dim, bias=False)
    
    def forward(self,x,kv=None):
        '''
        Args:
            x:  Tensor, [b,t,d]
            kv: key&value might from other source(not x) -> form another-type attention
        '''
        kv = x if kv is None else kv
        q,k,v = self.to_q(x), *self.to_kv(kv).chunk(2,dim=-1)

        b,t,d,h,e = *q.shape, self.heads, self.dim_heads 
        #b: batch-size (融合了非axial-dim的其他axial-dim)
        #t: targeted axial-dim
        #d: dim

        #h: head
        #e: head_dim

        merge_heads = lambda t: t.reshape(b,-1,h,e).transpose(1,2).reshape(b*h,-1,e) #[bh,t,e]
        # OR lambda ts: rearrange(ts, 'b t (h e) -> (b h) t e', h=self.heads)
        q,k,v = map(merge_heads, (q,k,v))

        dots = torch.einsum('bie,bje->bij',q,k) * (e**-0.5)
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bij,bje->bie',attn,v)
        
        out = out.reshape(b,h,-1,e).transpose(1,2).reshape(b,-1,d) #放心，reshape之后绝对是contiguous的
        out = self.to_out(out)
        out = self.to_out(out)
        return out

# axial attention class 

class AxialAttention(nn.Module):
    def __init__(
        self,
        dim, #embed_dim
        num_dimensions = 2, #num of axial_dimensions
        heads = 8,
        dim_heads = None,
        dim_index = -1, #embed-dim's index
        sum_axial_out = True
    ):
        '''
        Args:
            dim, int, embed_dim's value
            num_dimensions = 2, int, num of axial_dimensions
            heads = 8,
            dim_heads = None,
            dim_index = -1, int, embed-dim's index
            sum_axial_out = True
        '''
        super().__init__()
        self.dim = dim
        self.total_dimensions = num_dimensions + 2
        self.dim_index = self.dim_index if dim_index>0 else (self.total_dimensions + dim_index)

        attentions = []
        for permutation in calculate_permutations(num_dimensions, dim_index):
            attentions.append(PermuteToFrom(permutation, SelfAttention(dim, heads, dim_heads)))
        
        self.axial_attentions = nn.ModuleList(attentions)
        self.sum_axial_out = sum_axial_out
    
    def forward(self,x):
        assert len(x.shape) == self.total_dimensions, 'input tensor does not have the correct number of dimensions'
        assert x.shape[self.dim_index] == self.dim,' input tensor does not have the correct embed dim/ OR wrongly place embed-dim'

        if self.sum_axial_out:
            return sum(map(lambda axial_attn: axial_attn(x), self.axial_attentions))
        
        for axial_attn in self.axial_attentions:
            x = axial_attn(x)
        return x

# axial image Transformer

class AxialImageTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads = 8,
        dim_heads = None,
        dim_index = 1,
        reversible = True,
        axial_pos_embed_shape = None
    ):
        '''
        Args:
            dim: embed-dim's value \n
            dim_index: embed-dim's index \n
            reversible: \n
            axial_pos_embed_shape: tuple(int), axial-dimensions's shape
        '''
        super().__init__()
        permutations = calculate_permutations(2, dim_index)

        get_ff = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1), # standard 3x3 keep-resolution conv
            nn.LeakyReLU(inplace = True),
            nn.Conv2d(dim, dim, 3, padding=1),
        )

        self.pos_emb = AxialPositionalEmbedding(dim, axial_pos_embed_shape, emb_dim_index=dim_index) if exists(axial_pos_embed_shape) else nn.Identity()

        layers = nn.ModuleList([])
        for _ in range(depth):
            attn_functions = nn.ModuleList([PermuteToFrom(permutation, Rezero(SelfAttention(dim,heads,dim_heads))) for permutation in permutations])
            conv_functions = nn.ModuleList([Rezero(get_ff()), Rezero(get_ff())])
            layers.append(attn_functions)
            layers.append(conv_functions)
        
        execute_type = ReversibleSequence if reversible else Sequential
        self.layers = layers
    
    def forward(self,x):
        x = self.pos_emb(x)
        x = torch.cat((x,x),dim=-1)
        x = self.layers(x)
        return torch.stack(x.chunk(2,dim=-1)).mean(dim=0)