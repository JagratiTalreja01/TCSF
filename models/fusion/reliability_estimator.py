"""
reliability_estimator.py

Estimate modality reliability maps for SAR and optical inputs.

This module answers:
"Which raw observations should the model trust?"
"""

import torch
import torch.nn as nn

from models.blocks.conv_block import ConvBNAct


class ReliabilityEstimator(nn.Module):
    """
    Produces a pixel-level reliability map.

    Input:
        x: [B, C, H, W]

    Output:
        reliability: [B, 1, H, W]
        values in [0, 1]
    """

    def __init__(self, in_channels, hidden_dim=32):
        super().__init__()

        self.net = nn.Sequential(
            ConvBNAct(in_channels, hidden_dim),
            ConvBNAct(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class DualReliabilityEstimator(nn.Module):
    """
    Estimates SAR and optical reliability separately.
    """

    def __init__(
        self,
        sar_channels=2,
        optical_channels=6,
        hidden_dim=32,
    ):
        super().__init__()

        self.sar_reliability = ReliabilityEstimator(
            in_channels=sar_channels,
            hidden_dim=hidden_dim,
        )

        self.optical_reliability = ReliabilityEstimator(
            in_channels=optical_channels,
            hidden_dim=hidden_dim,
        )

    def forward(self, sar, optical):
        sar_conf = self.sar_reliability(sar)
        optical_conf = self.optical_reliability(optical)

        return sar_conf, optical_conf