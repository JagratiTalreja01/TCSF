"""
prediction_heads.py

Prediction heads for ACSF.

Each head converts decoder features into task-specific outputs.
"""

import torch.nn as nn

from models.blocks.conv_block import ConvBNAct


class PredictionHead(nn.Module):
    """
    Generic prediction head.

    Input:
        Feature map

    Output:
        Prediction map
    """

    def __init__(
        self,
        in_channels,
        hidden_channels=64,
        out_channels=1,
    ):
        super().__init__()

        self.head = nn.Sequential(
            ConvBNAct(in_channels, hidden_channels),
            nn.Conv2d(
                hidden_channels,
                out_channels,
                kernel_size=1,
            ),
        )

    def forward(self, x):
        return self.head(x)


class SARPredictionHead(PredictionHead):
    """
    SAR-specific flood prediction.
    """

    pass


class OpticalPredictionHead(PredictionHead):
    """
    Optical-specific flood prediction.
    """

    pass


class FusionPredictionHead(PredictionHead):
    """
    Prediction from fused features.
    """

    pass