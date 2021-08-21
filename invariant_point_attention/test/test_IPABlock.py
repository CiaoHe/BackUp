import torch
from torch import nn
from einops import repeat
from invariant_point_attention import IPABlock

block = IPABlock(
    dim = 64,
    heads = 8,
    scalar_key_dim = 16,
    scalar_value_dim = 16,
    point_key_dim = 4,
    point_value_dim = 4
)

seq           = torch.randn(1, 256, 64)
pairwise_repr = torch.randn(1, 256, 256, 64)
mask          = torch.ones(1, 256).bool()

rotations     = repeat(torch.eye(3), 'r1 r2 -> b n r1 r2', b = 1, n = 256)
translations  = torch.randn(1, 256, 3)

block_out = block(
    seq,
    pairwise_repr = pairwise_repr,
    rotations = rotations,
    translations = translations,
    mask = mask
)

print(block_out.shape)
updates = nn.Linear(64, 6)(block_out)
quaternion_update, translation_update = updates.chunk(2, dim=-1)
print('quaternion shape: ', quaternion_update.shape)
print('translation shape: ', translation_update.shape)