"""
bce_loss.py

Binary Cross Entropy loss for flood segmentation.
"""

import torch.nn as nn


class BCELoss(nn.Module):
    """
    BCEWithLogitsLoss wrapper.

    Inputs:
        logits : [B, 1, H, W]
        targets: [B, 1, H, W]
    """

    def __init__(self):
        super().__init__()

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        targets = targets.float()
        return self.loss(logits, targets)