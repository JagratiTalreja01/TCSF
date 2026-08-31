"""
norm_layers.py

Normalization utilities for ACSF.
"""

import torch.nn as nn


def get_norm_layer(norm_type, num_channels):
    """
    Return normalization layer.

    Args:
        norm_type (str): batch, group, instance, or none
        num_channels (int): Number of channels.

    Returns:
        nn.Module
    """

    norm_type = norm_type.lower()

    if norm_type == "batch":
        return nn.BatchNorm2d(num_channels)

    if norm_type == "instance":
        return nn.InstanceNorm2d(num_channels, affine=True)

    if norm_type == "group":
        num_groups = min(8, num_channels)
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)

    if norm_type == "none":
        return nn.Identity()

    raise ValueError(f"Unsupported normalization type: {norm_type}")


class LayerNorm2d(nn.Module):
    """
    LayerNorm for 2D feature maps.

    Input:
        x: [B, C, H, W]
    """

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()

        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)

        return x