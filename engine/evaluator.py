"""
evaluator.py

Validation and testing engine for ACSF.
"""

import torch
from tqdm import tqdm


class Evaluator:
    """
    Evaluation engine for ACSF.
    """

    def __init__(
        self,
        model,
        dataloader,
        criterion,
        metrics,
        device,
    ):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.metrics = metrics
        self.device = device

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()

        total_loss = 0.0
        metric_sums = {}

        progress_bar = tqdm(
            self.dataloader,
            desc="Evaluating",
        )

        for batch in progress_bar:
            sar = batch["sar"].to(self.device)
            optical = batch["optical"].to(self.device)
            mask = batch["mask"].to(self.device)

            outputs = self.model(sar, optical)

            loss, _ = self.criterion(outputs, mask)
            total_loss += loss.item()

            batch_metrics = self.metrics.evaluate(
                outputs["final_pred"],
                mask,
            )

            for key, value in batch_metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + value

        avg_loss = total_loss / len(self.dataloader)

        avg_metrics = {
            key: value / len(self.dataloader)
            for key, value in metric_sums.items()
        }

        return avg_loss, avg_metrics