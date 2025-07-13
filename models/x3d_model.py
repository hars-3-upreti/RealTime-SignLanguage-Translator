import torch
from pytorchvideo.models.hub import x3d_xs
import torch.nn as nn

def build_x3d_xs(num_classes=228, pretrained=True):
    model = x3d_xs(pretrained=pretrained)

    # Extract the original classifier
    original_proj = model.blocks[-1].proj
    in_features = original_proj.in_features

    # Replace with Dropout + Linear
    model.blocks[-1].proj = nn.Sequential(
        nn.Dropout(p=0.3),               
        nn.Linear(in_features, num_classes)
    )

    return model
