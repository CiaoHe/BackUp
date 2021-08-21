from triangle_multiplicative_module import TriangleMultiolicativeModule
import torch

model = TriangleMultiolicativeModule(
    dim = 64,
    hidden_dim = 128,
    mix = 'outgoing'   # either 'ingoing' or 'outgoing'
)

fmap = torch.randn(1, 256, 256, 64)
mask = torch.randn(1, 256, 256).bool()

print(model(fmap, mask).shape)