"""
adaptive_data_fusion.py

Adaptive data-level fusion for SAR-optical inputs.

This module answers:
"How much should raw SAR and raw optical observations contribute?"
"""

import torch
import torch.nn as nn

from models.blocks.conv_block import ConvBNAct
from models.fusion.reliability_estimator import DualReliabilityEstimator


class AdaptiveDataFusion(nn.Module):
    """
    Reliability-guided data-level fusion.

    Inputs:
        sar:     [B, C_sar, H, W]
        optical: [B, C_opt, H, W]

    Outputs:
        fused:        [B, out_channels, H, W]
        sar_conf:     [B, 1, H, W]
        optical_conf: [B, 1, H, W]
    """

    def __init__(
        self,
        sar_channels=2,
        optical_channels=6,
        out_channels=64,
        hidden_dim=32,
    ):
        super().__init__()

        self.reliability = DualReliabilityEstimator(
            sar_channels=sar_channels,
            optical_channels=optical_channels,
            hidden_dim=hidden_dim,
        )

        self.sar_proj = ConvBNAct(
            in_channels=sar_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.optical_proj = ConvBNAct(
            in_channels=optical_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.fuse = ConvBNAct(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, sar, optical):
        sar_conf, optical_conf = self.reliability(sar, optical)

        sar_feat = self.sar_proj(sar)
        optical_feat = self.optical_proj(optical)

        confidence_sum = sar_conf + optical_conf + 1e-6

        sar_weight = sar_conf / confidence_sum
        optical_weight = optical_conf / confidence_sum

        fused = sar_weight * sar_feat + optical_weight * optical_feat
        fused = self.fuse(fused)

        return fused, sar_conf, optical_conf