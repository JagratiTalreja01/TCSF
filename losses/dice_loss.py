"""
dice_loss.py

Dice Loss for binary flood segmentation.
"""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Soft Dice Loss.

    Inputs:
        logits : [B,1,H,W]
        targets: [B,1,H,W]
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        probs = probs.contiguous().view(-1)
        targets = targets.float().contiguous().view(-1)

        intersection = (probs * targets).sum()

        dice = (
            (2.0 * intersection + self.smooth)
            /
            (probs.sum() + targets.sum() + self.smooth)
        )

        loss = 1.0 - dice

        return loss