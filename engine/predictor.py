"""
predictor.py

Inference engine for ACSF.
"""

from pathlib import Path

import torch


class Predictor:
    """
    ACSF inference engine.
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device

        self.model.eval()

    @torch.no_grad()
    def predict(self, sar, optical):
        """
        Predict a single batch.

        Args:
            sar:     [B,2,H,W]
            optical: [B,C,H,W]

        Returns:
            Dictionary of predictions.
        """

        sar = sar.to(self.device)
        optical = optical.to(self.device)

        outputs = self.model(sar, optical)

        predictions = {
            "final_prediction": torch.sigmoid(
                outputs["final_pred"]
            ),
            "sar_prediction": torch.sigmoid(
                outputs["sar_pred"]
            ),
            "optical_prediction": torch.sigmoid(
                outputs["optical_pred"]
            ),
            "fusion_prediction": torch.sigmoid(
                outputs["fused_pred"]
            ),
            "sar_reliability": outputs["sar_conf"],
            "optical_reliability": outputs["optical_conf"],
            "decision_weights": outputs["decision_weights"],
        }

        return predictions

    @staticmethod
    def save_tensor(tensor, save_path):
        """
        Save prediction tensor.

        Args:
            tensor: torch.Tensor
            save_path: output file
        """

        save_path = Path(save_path)
        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            tensor.cpu(),
            save_path,
        )