"""
total_loss.py

Combined ACSF loss function.
"""

import torch.nn as nn

from losses.bce_loss import BCELoss
from losses.dice_loss import DiceLoss
from losses.boundary_loss import BoundaryLoss
from losses.consistency_loss import ConsistencyLoss


class ACSFLoss(nn.Module):
    """
    Total loss for ACSF.

    Supervised loss:
        final_pred vs mask

    Consistency loss:
        sar_pred, optical_pred, fused_pred
    """

    def __init__(
        self,
        bce_weight=1.0,
        dice_weight=1.0,
        boundary_weight=0.2,
        consistency_weight=0.1,
    ):
        super().__init__()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.consistency_weight = consistency_weight

        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.boundary = BoundaryLoss()
        self.consistency = ConsistencyLoss()

    def forward(self, outputs, targets):
        final_pred = outputs["final_pred"]
        sar_pred = outputs["sar_pred"]
        optical_pred = outputs["optical_pred"]
        fused_pred = outputs["fused_pred"]

        loss_bce = self.bce(final_pred, targets)
        loss_dice = self.dice(final_pred, targets)
        loss_boundary = self.boundary(final_pred, targets)

        loss_consistency = self.consistency(
            sar_pred=sar_pred,
            optical_pred=optical_pred,
            fused_pred=fused_pred,
        )

        total_loss = (
            self.bce_weight * loss_bce
            + self.dice_weight * loss_dice
            + self.boundary_weight * loss_boundary
            + self.consistency_weight * loss_consistency
        )

        loss_dict = {
            "total_loss": total_loss,
            "bce_loss": loss_bce,
            "dice_loss": loss_dice,
            "boundary_loss": loss_boundary,
            "consistency_loss": loss_consistency,
        }

        return total_loss, loss_dict


def build_loss(cfg):
    """
    Build ACSF loss from config.
    """

    return ACSFLoss(
        bce_weight=cfg["loss"]["bce_weight"],
        dice_weight=cfg["loss"]["dice_weight"],
        boundary_weight=cfg["loss"]["boundary_weight"],
        consistency_weight=cfg["loss"]["consistency_weight"],
    )