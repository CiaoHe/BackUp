'''
An implementation of local windowed attention, which sets an incredibly strong baseline for language modeling. It is becoming apparent that a transformer needs local attention in the bottom layers, with the top layers reserved for global attention to integrate the findings of previous layers. This repository makes it easy to immediately employ local window attention.

This code has been battletested in multiple repositories already, alongside different implementations of sparse long-range attention.
'''

import torch
import math
from torch import nn
from torch import Tensor as T
from torch import tensor
from torch.autograd import backward
from torch.functional import einsum
import torch.nn.functional as F
from operator import mul
from functools import reduce
from einops import rearrange, repeat

from .rotary import SinusoidalEmbeddings, apply_rotary_emb_pos

# constant

TOKEN_SELF_ATTN_VALUE = -5e4  # carefully set for half precision to work

# helper functions


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def to(t: torch.Tensor):
    return {'device': t.device, 'dtype': t.dtype}


def max_neg_value(t: torch.Tensor):
    return -torch.finfo(t.dtype).max


def merge_dims(ind_from: int, ind_to: int, tensor: torch.Tensor):
    shape = list(tensor.shape)
    arr_slice = slice(ind_from, ind_to+1)
    shape[arr_slice] = [reduce(mul, shape[arr_slice])]
    return tensor.reshape(*shape)


def expand_dim(t: torch.Tensor, dim, k, unsqueeze=True):
    if unsqueeze:
        t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def pad_to_multiple(tensor: torch.Tensor, multiple, dim=-1, value=0):
    '''
    对于dim维，如果能被multiple整除，则直接返回。 否则，进行补全
    '''
    seqlen = tensor.shape[dim]
    m = seqlen / multiple
    if m.is_integer():
        return tensor
    remainder = math.ceil(m) * multiple - seqlen
    pad_offset = (0, 0) * (-1-dim)
    return F.pad(tensor, (*pad_offset, 0, remainder), value)


def look_around(x: T, backward=1, forward=0, pad_value=-1, dim=2):
    '''
    Function:
        滑动窗口，将一定范围的窗口结果统计合并
    Returm:
        返回在dim维上滑动组成的tensor族. 
        shape: [b, win, win_sz*(back+for+1), d]
    Args:
        x: Tensor. [b, win, win_sz, d]
        dim: 决定累加在哪个维度上,实际上是win_size所在的维度
        backward: int. 在前方补多少个单位
        forward: int. 在后方补多少个单位; 非常巧妙，相当于可以看到前面的tokens
    '''
    t = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value=pad_value)
    tensors = [padded_x[:, ind:(ind+t), ...]
               for ind in range(forward + backward + 1)]
    return torch.cat(tensors, dim=dim)

# main class


class LocalAttention(nn.Module):
    def __init__(
        self,
        window_size,  # window_size
        causal=False,  # auto-regressive or not
        look_backward=1,  # each window looks at the window before
        look_forward=None,  # for non-auto-regressive case, will default to 1, so each window looks at the window before and after it
        dropout=0.,  # post-attention dropout
        shared_qk=False,
        rel_pos_emb_config=None,
        # dimension of each head (need to pass this in for relative postional encoding)
        dim_head=None,
        autopad=None,  # auto pads both inputs and mask, then truncates output appropriately
        # if set True, in the causal setting, each query will see at maximum the number of keys equal to the window size
        exact_windowsize=False
    ):
        super().__init__()
        look_forward = default(look_forward, 0 if causal else 1)
        assert not (causal and look_forward>1), 'you cannot look forward when causal'

        self.window_size = window_size
        self.causal = causal
        self.look_backward = look_backward
        self.look_forward = look_forward
        self.exact_windowsize = exact_windowsize
        self.autopad = autopad

        self.dropout = nn.Dropout(dropout)

        self.shared_qk = shared_qk

        self.rel_pos = None
        if exists(rel_pos_emb_config) or exists(dim_head):
            if exists(rel_pos_emb_config):
                dim_head = rel_pos_emb_config[0]
            self.rel_pos = SinusoidalEmbeddings(dim_head)
    
    def forward(self, q:T, k:T, v:T, input_mask = None):
        shape = q.shape

        q, k, v = map(lambda t: rearrange(t, '... t d -> (...) t d'), (q,k,v))

        if exists(self.rel_pos):
            pos_emb = self.rel_pos(q)
            q,k = apply_rotary_emb_pos(q, k, pos_emb)
        
        # 当 t(orig_t) // window_size != 0: 强制启动auto_pad
        # if shape[-1] // self.window_size != 0:
        #     self.autopad = True
        
        if self.autopad:
            orig_t = shape[1]
            q,k,v = map(lambda t: pad_to_multiple(t, self.window_size, dim=-2), (q,k,v))

        window_size, causal, look_backward, look_forward, shared_qk = self.window_size, \
            self.causal, self.look_backward, self.look_forward, self.shared_qk
        b, t, d, device, dtype = *q.shape, q.device, q.dtype
        assert (t % window_size) == 0, f'sequence length {t} must be divisible by window size {window_size} for local attention'

        windows = t // window_size # window number

        if shared_qk:
            k = F.normalize(q, p=2, dim=-1).type_as(q)
        
        ticker = torch.arange(t, device = device, dtype = dtype)[None, :] #[1, t]
        b_t = rearrange(ticker, 'b (win win_sz) -> b win win_sz', win = windows)
        # b_t: [b win win_sz], b_t起到的作用也等于是mask
        bq, bk, bv = map(lambda t: rearrange(t, 'b (win win_sz) d -> b win win_sz d', win = windows,\
            win_sz = window_size), (q,k,v))
        # bq/k/v: [b win win_sz d]
        
        look_around_kwargs = {'backward':look_backward, 'forward':look_forward}
        
        bk, bv = map(lambda t: look_around(t, **look_around_kwargs), (bk, bv))
        # bk/bv : [b, win, win_sz*(b+f+1), d]

        bq_t = b_t
        bq_k = look_around(b_t, **look_around_kwargs)
        # bq_k: [b, win, win_sz*(b+f+1)]

        dots = torch.einsum('bhid,bhjd->bhij', bq, bk) * (d ** -0.5)
        # dots: [b, win, win_sz, win_sz*(b+f+1)]

        mask_value = max_neg_value(dots)

        if shared_qk:
            mask = bq_t[:,:,:,None] == bq_k[:,:,None,:]
            dots.masked_fill_(mask, TOKEN_SELF_ATTN_VALUE)
            del mask
        
        if causal:
            mask = bq_t[:,:,:,None] < bq_k[:,:,None,:]

            if self.exact_windowsize:
                max_causal_windowsize = (self.window_size * self.look_backward)
                mask = mask | (bq_t[:,:,:,None] > (bq_k[:,:,None,:] + max_causal_windowsize)) 
            
            dots.masked_fill_(mask, mask_value)
            del mask
        
        mask = bq_k[:, :, None, :] == -1
        dots.masked_fill_(mask, mask_value)
        del mask

        if input_mask is not None:
            # input_mask : [b, t]
            h = b // input_mask.shape[0] # 还原heads
            if self.autopad:
                input_mask = pad_to_multiple(input_mask, multiple=window_size, dim=-1, value=False)
            input_mask = rearrange(input_mask, 'b (w w_sz) -> b w w_sz', w=windows, w_sz = window_size)
            mq = mk = input_mask # [b, w, w_sz]
            mk = look_around(mk, pad_value=False, **look_around_kwargs) # [b, w, w_sz*(b+f+1)]
            mask = (mq[:, :, :, None] * mk[:,:,None,:]) # [b, w, w_sz, w_sz*(b+f+1)]
            mask = expand_dim(mask, 1, h) #[b, h, w_sz, w_sz*(b+f+1)]
            mask = merge_dims(0, 1, mask) #[b*h, w_sz, w_sz*(b+f+1)]
            dots.masked_fill_(~mask, mask_value)
            del mask
        
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum('bhij,bhjd->bhid', attn, bv)
        out = out.reshape(-1, t, d)

        if self.autopad:
            out = out[:, :orig_t, :]
        
        return out.reshape(*shape)