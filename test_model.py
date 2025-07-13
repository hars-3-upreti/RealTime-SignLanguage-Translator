# test_model.py
from models.x3d_model import build_x3d_xs
import torch

model = build_x3d_xs(num_classes=228, pretrained=True)
x = torch.randn(1, 3, 16, 224, 224)
y = model(x)
print("Output shape:", y.shape)  # Should be [1, 228]