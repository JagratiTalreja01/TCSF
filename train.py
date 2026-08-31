"""
train.py

Main training script for ACSF.
"""

import argparse
import torch

from utils.config import load_config
from utils.seed import seed_everything
from utils.metrics import SegmentationMetrics

from datasets.dataloader import build_dataloader

from models.acsf_net import ACSFNet
from losses.total_loss import build_loss
from engine.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ACSF for SAR-optical flood mapping"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/acsf_base.yaml",
        help="Path to config file",
    )

    return parser.parse_args()


def build_model(cfg):
    model = ACSFNet(
        sar_channels=cfg["dataset"]["sar_channels"],
        optical_channels=cfg["dataset"]["optical_channels"],
        base_dim=cfg["model"]["encoder_dim"],
        num_classes=cfg["dataset"]["num_classes"],
    )

    return model


def build_optimizer(cfg, model):
    if cfg["training"]["optimizer"].lower() == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg["training"]["learning_rate"],
            weight_decay=cfg["training"]["weight_decay"],
        )

    raise NotImplementedError(
        f"Optimizer {cfg['training']['optimizer']} not implemented"
    )


def build_scheduler(cfg, optimizer):
    if cfg["training"]["scheduler"].lower() == "cosineannealinglr":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg["training"]["epochs"],
        )

    return None


def main():
    args = parse_args()

    cfg = load_config(args.config)

    seed_everything(cfg["experiment"]["seed"])

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        and cfg["training"]["device"] == "cuda"
        else "cpu"
    )

    print(f"[INFO] Using device: {device}")

    train_loader = build_dataloader(cfg, split="train")
    val_loader = build_dataloader(cfg, split="val")

    model = build_model(cfg)
    model = model.to(device)

    criterion = build_loss(cfg)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    metrics = SegmentationMetrics()

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        cfg=cfg,
    )

    trainer.fit(metrics)


if __name__ == "__main__":
    main()