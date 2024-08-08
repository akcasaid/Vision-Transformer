import torch
from vit_pytorch.pit import PiT

v = PiT(
    image_size = 224,
    patch_size = 14,
    dim = 256,
    num_classes = 1000,
    depth = (3, 3, 3),     # list of depths, indicating the number of rounds of each stage before a downsample
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

# forward pass now returns predictions and the attention maps

img = torch.randn(1, 3, 224, 224)

preds = v(img) # (1, 1000)
