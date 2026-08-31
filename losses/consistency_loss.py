"""
consistency_loss.py

Prediction consistency loss for ACSF.

Encourages the SAR, Optical, and Fusion branches
to produce mutually consistent flood predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConsistencyLoss(nn.Module):
    """
    Pairwise consistency loss.

    Inputs:
        sar_pred
        optical_pred
        fused_pred

    Returns:
        scalar loss
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        sar_pred,
        optical_pred,
        fused_pred,
    ):
        sar_prob = torch.sigmoid(sar_pred)
        optical_prob = torch.sigmoid(optical_pred)
        fused_prob = torch.sigmoid(fused_pred)

        loss_so = F.mse_loss(sar_prob, optical_prob)
        loss_sf = F.mse_loss(sar_prob, fused_prob)
        loss_of = F.mse_loss(optical_prob, fused_prob)

        loss = (loss_so + loss_sf + loss_of) / 3.0

        return loss