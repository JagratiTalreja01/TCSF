"""
optical_vmamba_encoder.py

Optical encoder for ACSF.

Input:
    Sentinel-2 optical bands: RGB, NIR, SWIR, etc.

Output:
    Multi-scale optical feature maps.
"""

import torch.nn as nn

from models.blocks.conv_block import ConvBNAct, DownBlock
from models.blocks.mamba_block import MambaBlock


class OpticalVMambaEncoder(nn.Module):
    """
    Optical Vision Mamba Encoder.

    Produces four feature levels:
        f1: [B, C, H, W]
        f2: [B, 2C, H/2, W/2]
        f3: [B, 4C, H/4, W/4]
        f4: [B, 8C, H/8, W/8]
    """

    def __init__(self, in_channels=6, base_dim=64):
        super().__init__()

        self.stage1 = nn.Sequential(
            ConvBNAct(in_channels, base_dim),
            MambaBlock(base_dim),
        )

        self.stage2 = nn.Sequential(
            DownBlock(base_dim, base_dim * 2),
            MambaBlock(base_dim * 2),
        )

        self.stage3 = nn.Sequential(
            DownBlock(base_dim * 2, base_dim * 4),
            MambaBlock(base_dim * 4),
        )

        self.stage4 = nn.Sequential(
            DownBlock(base_dim * 4, base_dim * 8),
            MambaBlock(base_dim * 8),
        )

    def forward(self, x):
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)

        return [f1, f2, f3, f4]