import torch
from local_attention.local_attention import LocalAttention

# test basic LocalAttention

q =  torch.randn(8, 2048, 64)
k =  torch.randn(8, 2048, 64)
v =  torch.randn(8, 2048, 64)

attn = LocalAttention(
    window_size = 512,       # window size. 512 is optimal, but 256 or 128 yields good enough results
    causal = True,           # auto-regressive or not
    look_backward = 1,       # each window looks at the window before
    look_forward = 0,        # for non-auto-regressive case, will default to 1, so each window looks at the window before and after it
    dropout = 0.1,           # post-attention dropout
    dim_head = 64,           # dimension of each head (you need to pass this in for relative positional encoding)
    exact_windowsize = False # if this is set to true, in the causal setting, each query will see at maximum the number of keys equal to the window size
)

mask = torch.ones(1,2048).bool()
out = attn(q, k, v, input_mask = mask)
print(out.shape)

############################################################

'''
This library also allows for local attention in the setting of shared query/key space. The normalization of the keys, as well as the masking of tokens to itself, will be taken care of.
'''
# test share qk mode

qk = torch.randn(8, 2048, 64)
v = torch.randn(8, 2048, 64)

attn = LocalAttention(
    dim_head=64,
    window_size=512,
    shared_qk=True,
    causal=True
)

mask = torch.ones(1,2048).bool()
out = attn(qk, qk, v, input_mask = mask)
print(out.shape)

############################################################

'''
If you wish for the module to automagically pad your query / key / values as well as the mask, simply set the autopad keyword to True
'''

import torch
from local_attention import LocalAttention

q = torch.randn(8, 2057, 64)
k = torch.randn(8, 2057, 64)
v = torch.randn(8, 2057, 64)

attn = LocalAttention(
    window_size = 512,
    causal = True,
    autopad = True      # auto pads both inputs and mask, then truncates output appropriately
)

mask = torch.ones(1, 2057).bool()
out = attn(q, k, v, input_mask = mask) # (1, 8, 2057, 64)
print(out.shape)

''''
后记：
    1. local_attention里用window的方式代替传统的multi-heads，
        -> 最重要的是采用look-around技术，将window附近的信息也利用起来
            -> bq: (b, win, win_sz, d)
            -> bk & bv: (b, win, win_sz*(back+for+1), d)
            -> (bq)^T \dot bk: (b, win, win_sz, win_sz*(back+for+1))

            ! 但是，因为利用context window的信息会存在距离问题，这里就要用到mask(bq_k)来遮挡前方padding的信息
'''