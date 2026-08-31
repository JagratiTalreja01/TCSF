"""
mamba_decoder.py

Multi-scale decoder for ACSF.
"""

import torch
import torch.nn as nn

from models.blocks.conv_block import UpBlock, ConvBNAct
from models.blocks.mamba_block import MambaBlock


class MambaDecoder(nn.Module):
    """
    Decoder that upsamples multi-scale fused features.

    Inputs:
        features = [f1, f2, f3, f4]

        f1: [B, C, H, W]
        f2: [B, 2C, H/2, W/2]
        f3: [B, 4C, H/4, W/4]
        f4: [B, 8C, H/8, W/8]

    Output:
        out: [B, C, H, W]
    """

    def __init__(self, base_dim=64):
        super().__init__()

        self.bottleneck = nn.Sequential(
            ConvBNAct(base_dim * 8, base_dim * 8),
            MambaBlock(base_dim * 8),
        )

        self.up3 = UpBlock(
            in_channels=base_dim * 8,
            skip_channels=base_dim * 4,
            out_channels=base_dim * 4,
        )

        self.up2 = UpBlock(
            in_channels=base_dim * 4,
            skip_channels=base_dim * 2,
            out_channels=base_dim * 2,
        )

        self.up1 = UpBlock(
            in_channels=base_dim * 2,
            skip_channels=base_dim,
            out_channels=base_dim,
        )

        self.refine = nn.Sequential(
            ConvBNAct(base_dim, base_dim),
            MambaBlock(base_dim),
        )

    def forward(self, features):
        f1, f2, f3, f4 = features

        x = self.bottleneck(f4)

        x = self.up3(x, f3)
        x = self.up2(x, f2)
        x = self.up1(x, f1)

        x = self.refine(x)

        return x