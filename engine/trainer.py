"""Training loop for ACSF with checkpointing and TCSF scale logging."""

from pathlib import Path
import csv

import torch
from tqdm import tqdm

from utils.checkpoint import save_checkpoint


TRANSITION_COLUMNS = [
    f"t{level}_{branch}_gamma"
    for level in range(1, 4)
    for branch in ("sar", "optical", "fused")
]


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        cfg,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.cfg = cfg

        self.use_amp = cfg["training"].get("use_amp", False)
        self.grad_clip = cfg["training"].get("grad_clip", 1.0)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.log_dir = (
            Path(cfg["experiment"]["output_dir"])
            / "logs"
            / cfg["experiment"]["name"]
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.log_dir / "metrics.csv"

        if not self.metrics_file.exists():
            with open(self.metrics_file, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        "epoch",
                        "train_loss",
                        "val_loss",
                        "iou",
                        "dice",
                        "precision",
                        "recall",
                        "f1",
                        "pixel_acc",
                        "learning_rate",
                        *TRANSITION_COLUMNS,
                    ]
                )

    def _base_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _transition_scales(self):
        model = self._base_model()
        if hasattr(model, "get_transition_scale_dict"):
            return model.get_transition_scale_dict()
        return {column: float("nan") for column in TRANSITION_COLUMNS}

    def train_one_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")

        for batch in progress_bar:
            sar = batch["sar"].to(self.device)
            optical = batch["optical"].to(self.device)
            mask = batch["mask"].to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(sar, optical)
                loss, loss_dict = self.criterion(outputs, mask)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            epoch_loss += loss.item()

            progress_bar.set_postfix(
                {
                    "loss": loss.item(),
                    "bce": loss_dict["bce_loss"].item(),
                    "dice": loss_dict["dice_loss"].item(),
                }
            )

        return epoch_loss / len(self.train_loader)

    @torch.no_grad()
    def validate_one_epoch(self, epoch, metrics):
        self.model.eval()
        epoch_loss = 0.0
        metric_sums = {}
        progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")

        for batch in progress_bar:
            sar = batch["sar"].to(self.device)
            optical = batch["optical"].to(self.device)
            mask = batch["mask"].to(self.device)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(sar, optical)
                loss, _ = self.criterion(outputs, mask)

            epoch_loss += loss.item()
            batch_metrics = metrics.evaluate(outputs["final_pred"], mask)
            for key, value in batch_metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + value

        avg_loss = epoch_loss / len(self.val_loader)
        avg_metrics = {
            key: value / len(self.val_loader)
            for key, value in metric_sums.items()
        }
        return avg_loss, avg_metrics

    def log_epoch(self, epoch, train_loss, val_loss, val_metrics):
        transition_scales = self._transition_scales()
        learning_rate = self.optimizer.param_groups[0]["lr"]

        with open(self.metrics_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    epoch,
                    train_loss,
                    val_loss,
                    val_metrics["IoU"],
                    val_metrics["Dice"],
                    val_metrics["Precision"],
                    val_metrics["Recall"],
                    val_metrics["F1"],
                    val_metrics["PixelAcc"],
                    learning_rate,
                    *[transition_scales[column] for column in TRANSITION_COLUMNS],
                ]
            )

    def fit(self, metrics):
        best_iou = 0.0
        num_epochs = self.cfg["training"]["epochs"]

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            val_loss, val_metrics = self.validate_one_epoch(epoch, metrics)

            if self.scheduler is not None:
                self.scheduler.step()

            current_iou = val_metrics["IoU"]
            scale_values = self._transition_scales()
            scale_summary = ", ".join(
                f"{key}={value:.4f}"
                for key, value in scale_values.items()
            )

            print(
                f"\nEpoch [{epoch}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"IoU: {current_iou:.4f} | "
                f"Dice: {val_metrics['Dice']:.4f}"
            )
            print(f"[TCSF scales] {scale_summary}")

            self.log_epoch(epoch, train_loss, val_loss, val_metrics)

            if current_iou > best_iou:
                best_iou = current_iou
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    cfg=self.cfg,
                    best=True,
                )

            save_every = self.cfg["checkpoint"]["save_every"]
            if epoch % save_every == 0:
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    cfg=self.cfg,
                    best=False,
                )
