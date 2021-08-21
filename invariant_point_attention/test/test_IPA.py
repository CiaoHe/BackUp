import torch
from einops import repeat
from invariant_point_attention import InvariantPointAttention

def test_with_pairwise_repr():
    attn = InvariantPointAttention(
    dim = 64,              # single (and pairwisw) represeantion dimension
    heads = 8,             # number of heads
    scalar_key_dim=16,     # scalar query-key dimension (normal attention)
    scalar_value_dim=16,   # scalar value dimension
    point_key_dim=4,       # point query-key dimension (point attention)
    point_value_dim=4      # point value dimension
)   

    single_repr = torch.randn(1, 256, 64)           # (batch, seq, dim)
    pairwise_repr = torch.randn(1, 256, 256, 64)    # (batch, seq, seq, dim)
    mask = torch.ones(1,256).bool()                 # (batch, seq)

    rotations = torch.eye(3)
    rotations = repeat(rotations, '... -> b n ...', b=1, n=256) # (batch, seq, 3, 3)
    translations = torch.zeros(1, 256, 3)                       # (batch, seq, 3)

    attn_out = attn(
        single_repr,
        pairwise_repr = pairwise_repr,
        rotations = rotations,
        translations = translations,
        mask = mask
    )

    print(attn_out.shape)

def test_wo_pairwise_repr():
    attn = InvariantPointAttention(
        dim = 64,
        heads = 8,
        require_pairwise_repr = False   # set this to False to use the module without pairwise representations
    )

    single_repr = torch.randn(1, 256, 64)           # (batch, seq, dim)
    mask = torch.ones(1,256).bool()                 # (batch, seq)

    rotations = torch.eye(3)
    rotations = repeat(rotations, '... -> b n ...', b=1, n=256) # (batch, seq, 3, 3)
    translations = torch.zeros(1, 256, 3)                       # (batch, seq, 3)

    attn_out = attn(
        single_repr,
        rotations = rotations,
        translations = translations,
        mask = mask
    )

    print(attn_out.shape)

if __name__ == '__main__':
    test_wo_pairwise_repr()
