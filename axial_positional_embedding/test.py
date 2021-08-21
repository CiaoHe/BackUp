import torch
from axial_positional_embedding import AxialPositionalEmbedding, AxialPositionalEmbeddingImage

# test AxialPositionalEmbedding

pos_emb = AxialPositionalEmbedding(
    dim = 512,
    axial_shape = (64,64),
    axial_dims = (256,256)
)
tokens = torch.randn(1, 1024, 512)
tokens = pos_emb(tokens) + tokens

print(tokens.shape)


# test AxialPositionalEmbeddingaImage 

img = torch.rand(1, 3, 28, 28)

img_emb = AxialPositionalEmbeddingImage(
    dim = 512,
    axial_shape = (28,28),
    # axial_dims=(256,256)
)
img_after = img_emb(img)
print(img_after.shape)