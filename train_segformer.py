"""
train_segformer.py

Training script for the multimodal early-fusion SegFormer baseline.

The script reuses the existing project configuration loader and Sen1Floods11
dataloader, but uses a baseline-specific segmentation loss applied only to
outputs["final_pred"].

Expected model file:
    models/baselines/segformer_baseline.py

Expected config:
    configs/segformer_base.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from datasets.dataloader import build_dataloader
from models.baselines.segformer_baseline import MultimodalSegFormer
from utils.config import load_config


class DiceLoss(nn.Module):
    """Binary soft Dice loss operating on logits."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)

        probabilities = probabilities.reshape(probabilities.shape[0], -1)
        targets = targets.reshape(targets.shape[0], -1)

        intersection = (probabilities * targets).sum(dim=1)
        denominator = probabilities.sum(dim=1) + targets.sum(dim=1)

        dice = (2.0 * intersection + self.eps) / (
            denominator + self.eps
        )

        return 1.0 - dice.mean()


class BaselineSegmentationLoss(nn.Module):
    """
    BCE + Dice segmentation loss for single-output baselines.

    This intentionally excludes ACSF/TCSF consistency, boundary, and
    auxiliary-branch losses so SegFormer is not forced to produce outputs
    it does not have.
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        pos_weight: float | None = None,
    ) -> None:
        super().__init__()

        if bce_weight < 0.0 or dice_weight < 0.0:
            raise ValueError("Loss weights must be non-negative.")

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        if pos_weight is None:
            self.register_buffer("pos_weight", None)
        else:
            self.register_buffer(
                "pos_weight",
                torch.tensor([float(pos_weight)], dtype=torch.float32),
            )

        self.dice_loss = DiceLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
        )
        dice = self.dice_loss(logits, targets)

        total = self.bce_weight * bce + self.dice_weight * dice

        parts = {
            "total": float(total.detach().cpu()),
            "bce": float(bce.detach().cpu()),
            "dice": float(dice.detach().cpu()),
        }

        return total, parts


class ConfusionAccumulator:
    """Accumulate binary segmentation counts over a complete split."""

    def __init__(self, threshold: float = 0.5, eps: float = 1e-6) -> None:
        self.threshold = threshold
        self.eps = eps
        self.reset()

    def reset(self) -> None:
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tn = 0.0

    @torch.no_grad()
    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        predictions = (
            torch.sigmoid(logits) > self.threshold
        ).float()

        predictions = predictions.reshape(-1)
        targets = targets.reshape(-1)

        self.tp += float((predictions * targets).sum().cpu())
        self.fp += float((predictions * (1.0 - targets)).sum().cpu())
        self.fn += float(((1.0 - predictions) * targets).sum().cpu())
        self.tn += float(
            ((1.0 - predictions) * (1.0 - targets)).sum().cpu()
        )

    def compute(self) -> Dict[str, float]:
        iou = (self.tp + self.eps) / (
            self.tp + self.fp + self.fn + self.eps
        )
        dice = (2.0 * self.tp + self.eps) / (
            2.0 * self.tp + self.fp + self.fn + self.eps
        )
        precision = (self.tp + self.eps) / (
            self.tp + self.fp + self.eps
        )
        recall = (self.tp + self.eps) / (
            self.tp + self.fn + self.eps
        )
        pixel_accuracy = (self.tp + self.tn + self.eps) / (
            self.tp + self.fp + self.fn + self.tn + self.eps
        )

        return {
            "IoU": iou,
            "Dice": dice,
            "Precision": precision,
            "Recall": recall,
            "F1": dice,
            "PixelAcc": pixel_accuracy,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the multimodal SegFormer baseline."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/segformer_base.yaml",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Optional output directory override. "
            "Otherwise cfg['output']['checkpoint_dir'] is used."
        ),
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional checkpoint from which to resume training.",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("train_segformer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def build_model(cfg: dict) -> MultimodalSegFormer:
    baseline_cfg = cfg.get("baseline", {})

    return MultimodalSegFormer(
        sar_channels=cfg["dataset"]["sar_channels"],
        optical_channels=cfg["dataset"]["optical_channels"],
        num_classes=cfg["dataset"]["num_classes"],
        embed_dims=tuple(
            int(value)
            for value in baseline_cfg.get(
                "embed_dims",
                [32, 64, 160, 256],
            )
        ),
        depths=tuple(
            int(value)
            for value in baseline_cfg.get(
                "depths",
                [2, 2, 2, 2],
            )
        ),
        num_heads=tuple(
            int(value)
            for value in baseline_cfg.get(
                "num_heads",
                [1, 2, 5, 8],
            )
        ),
        mlp_ratios=tuple(
            float(value)
            for value in baseline_cfg.get(
                "mlp_ratios",
                [4.0, 4.0, 4.0, 4.0],
            )
        ),
        sr_ratios=tuple(
            int(value)
            for value in baseline_cfg.get(
                "sr_ratios",
                [8, 4, 2, 1],
            )
        ),
        decoder_channels=int(
            baseline_cfg.get("decoder_channels", 256)
        ),
        dropout=float(
            baseline_cfg.get("dropout", 0.1)
        ),
        attention_dropout=float(
            baseline_cfg.get("attention_dropout", 0.0)
        ),
        drop_path_rate=float(
            baseline_cfg.get("drop_path_rate", 0.1)
        ),
    )


def build_criterion(cfg: dict) -> BaselineSegmentationLoss:
    loss_cfg = cfg.get("loss", {})

    return BaselineSegmentationLoss(
        bce_weight=float(loss_cfg.get("bce_weight", 1.0)),
        dice_weight=float(loss_cfg.get("dice_weight", 1.0)),
        pos_weight=loss_cfg.get("pos_weight"),
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    epoch: int,
    best_val_iou: float,
    cfg: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_iou": best_val_iou,
            "config": cfg,
        },
        path,
    )


def load_resume_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    device: torch.device,
) -> Tuple[int, float]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint.get("state_dict", checkpoint),
    )
    model.load_state_dict(state_dict)

    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_val_iou = float(checkpoint.get("best_val_iou", -1.0))

    return start_epoch, best_val_iou


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: BaselineSegmentationLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool,
    scaler: torch.cuda.amp.GradScaler,
    grad_clip: float | None,
) -> Dict[str, float]:
    model.train()

    running_total = 0.0
    running_bce = 0.0
    running_dice = 0.0
    sample_count = 0

    progress = tqdm(loader, desc="Train", leave=False)

    for batch in progress:
        sar = batch["sar"].to(device, non_blocking=True)
        optical = batch["optical"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(sar, optical)
            logits = outputs["final_pred"]
            loss, parts = criterion(logits, mask)

        scaler.scale(loss).backward()

        if grad_clip is not None and grad_clip > 0.0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip,
            )

        scaler.step(optimizer)
        scaler.update()

        batch_size = sar.shape[0]
        sample_count += batch_size
        running_total += parts["total"] * batch_size
        running_bce += parts["bce"] * batch_size
        running_dice += parts["dice"] * batch_size

        progress.set_postfix(loss=f"{parts['total']:.4f}")

    denominator = max(sample_count, 1)

    return {
        "loss": running_total / denominator,
        "bce_loss": running_bce / denominator,
        "dice_loss": running_dice / denominator,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: BaselineSegmentationLoss,
    device: torch.device,
    use_amp: bool,
    threshold: float,
) -> Dict[str, float]:
    model.eval()

    running_total = 0.0
    running_bce = 0.0
    running_dice = 0.0
    sample_count = 0

    metrics = ConfusionAccumulator(threshold=threshold)

    progress = tqdm(loader, desc="Val", leave=False)

    for batch in progress:
        sar = batch["sar"].to(device, non_blocking=True)
        optical = batch["optical"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(sar, optical)
            logits = outputs["final_pred"]
            loss, parts = criterion(logits, mask)

        metrics.update(logits, mask)

        batch_size = sar.shape[0]
        sample_count += batch_size
        running_total += parts["total"] * batch_size
        running_bce += parts["bce"] * batch_size
        running_dice += parts["dice"] * batch_size

    denominator = max(sample_count, 1)
    result = metrics.compute()

    result.update(
        {
            "loss": running_total / denominator,
            "bce_loss": running_bce / denominator,
            "dice_loss": running_dice / denominator,
        }
    )

    return result


def append_history(
    csv_path: Path,
    row: Dict[str, float],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))

        if write_header:
            writer.writeheader()

        writer.writerow(row)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    seed = int(cfg["training"].get("seed", 2026))
    set_seed(seed)

    requested_device = cfg["training"].get("device", "cuda")
    device = torch.device(
        "cuda"
        if requested_device == "cuda" and torch.cuda.is_available()
        else "cpu"
    )

    default_output_dir = cfg.get("output", {}).get(
        "checkpoint_dir",
        "outputs/checkpoints/SegFormer_200ep_seed2026",
    )
    output_dir = Path(args.output_dir or default_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(output_dir / "training.log")

    logger.info("Configuration: %s", args.config)
    logger.info("Device: %s", device)
    logger.info("Seed: %d", seed)

    train_loader = build_dataloader(cfg, split="train")
    val_loader = build_dataloader(cfg, split="val")

    model = build_model(cfg).to(device)
    criterion = build_criterion(cfg).to(device)

    learning_rate = float(cfg["training"].get("learning_rate", 1e-4))
    weight_decay = float(cfg["training"].get("weight_decay", 1e-4))

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler_cfg = cfg.get("scheduler", {})
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(scheduler_cfg.get("factor", 0.5)),
        patience=int(scheduler_cfg.get("patience", 3)),
        min_lr=float(scheduler_cfg.get("min_lr", 1e-7)),
    )

    use_amp = bool(cfg["training"].get("amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    epochs = int(cfg["training"].get("epochs", 20))
    threshold = float(cfg.get("evaluation", {}).get("threshold", 0.5))
    grad_clip_value = cfg["training"].get("grad_clip")
    grad_clip = (
        None
        if grad_clip_value is None
        else float(grad_clip_value)
    )

    start_epoch = 1
    best_val_iou = -1.0

    if args.resume is not None:
        start_epoch, best_val_iou = load_resume_checkpoint(
            checkpoint_path=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        logger.info(
            "Resumed from %s at epoch %d.",
            args.resume,
            start_epoch,
        )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    logger.info("Trainable parameters: %d", trainable_parameters)
    logger.info("Training epochs: %d", epochs)
    logger.info("AMP enabled: %s", use_amp)

    history_path = output_dir / "history.csv"

    run_metadata = {
        "model": "MultimodalSegFormer",
        "fusion": "early_channel_concatenation",
        "seed": seed,
        "device": str(device),
        "trainable_parameters": trainable_parameters,
        "config": args.config,
    }

    with (output_dir / "run_metadata.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(run_metadata, file, indent=2)

    for epoch in range(start_epoch, epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            use_amp=use_amp,
            scaler=scaler,
            grad_clip=grad_clip,
        )

        val_stats = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            threshold=threshold,
        )

        scheduler.step(val_stats["IoU"])

        current_lr = optimizer.param_groups[0]["lr"]

        history_row = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "train_loss": train_stats["loss"],
            "train_bce_loss": train_stats["bce_loss"],
            "train_dice_loss": train_stats["dice_loss"],
            "val_loss": val_stats["loss"],
            "val_bce_loss": val_stats["bce_loss"],
            "val_dice_loss": val_stats["dice_loss"],
            "val_iou": val_stats["IoU"],
            "val_dice": val_stats["Dice"],
            "val_precision": val_stats["Precision"],
            "val_recall": val_stats["Recall"],
            "val_pixel_accuracy": val_stats["PixelAcc"],
        }
        append_history(history_path, history_row)

        logger.info(
            "Epoch [%d/%d] "
            "Train Loss: %.4f | "
            "Val Loss: %.4f | "
            "IoU: %.4f | "
            "Dice: %.4f | "
            "Precision: %.4f | "
            "Recall: %.4f | "
            "LR: %.2e",
            epoch,
            epochs,
            train_stats["loss"],
            val_stats["loss"],
            val_stats["IoU"],
            val_stats["Dice"],
            val_stats["Precision"],
            val_stats["Recall"],
            current_lr,
        )

        save_checkpoint(
            path=output_dir / "last_model.pth",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val_iou=best_val_iou,
            cfg=cfg,
        )

        if val_stats["IoU"] > best_val_iou:
            best_val_iou = val_stats["IoU"]

            save_checkpoint(
                path=output_dir / "best_model.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_iou=best_val_iou,
                cfg=cfg,
            )

            logger.info(
                "Saved new best checkpoint with Val IoU %.4f.",
                best_val_iou,
            )

    logger.info("Training completed.")
    logger.info("Best validation IoU: %.4f", best_val_iou)
    logger.info("Best checkpoint: %s", output_dir / "best_model.pth")


if __name__ == "__main__":
    main()