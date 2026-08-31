"""
adaptive_decision_fusion.py

Adaptive decision-level fusion for SAR, optical, and fused predictions.

This module answers:
"Which prediction should be trusted most?"
"""

import torch
import torch.nn as nn

from models.blocks.conv_block import ConvBNAct


class AdaptiveDecisionFusion(nn.Module):
    """
    Learn pixel-level weights for three predictions:

        P_sar
        P_optical
        P_fused

    Inputs:
        sar_pred:     [B, 1, H, W]
        optical_pred: [B, 1, H, W]
        fused_pred:   [B, 1, H, W]

    Output:
        final_pred:   [B, 1, H, W]
    """

    def __init__(self, hidden_dim=16):
        super().__init__()

        self.weight_net = nn.Sequential(
            ConvBNAct(3, hidden_dim),
            nn.Conv2d(hidden_dim, 3, kernel_size=1),
            nn.Softmax(dim=1),
        )

    def forward(self, sar_pred, optical_pred, fused_pred):
        preds = torch.cat(
            [sar_pred, optical_pred, fused_pred],
            dim=1,
        )

        weights = self.weight_net(preds)

        sar_w = weights[:, 0:1]
        optical_w = weights[:, 1:2]
        fused_w = weights[:, 2:3]

        final_pred = (
            sar_w * sar_pred
            + optical_w * optical_pred
            + fused_w * fused_pred
        )

        return final_pred, weights