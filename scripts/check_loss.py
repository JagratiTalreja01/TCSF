"""
check_loss.py

Sanity-check ACSF loss using real dataset batch and model outputs.

Run:
    python scripts/check_loss.py --config configs/acsf_base.yaml
"""

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from utils.config import load_config
from datasets.dataloader import build_dataloader
from models.acsf_net import ACSFNet
from losses.total_loss import build_loss


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check ACSF loss"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/acsf_base.yaml",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = build_dataloader(cfg, split="train")
    batch = next(iter(loader))

    sar = batch["sar"].to(device)
    optical = batch["optical"].to(device)
    mask = batch["mask"].to(device)

    model = ACSFNet(
        sar_channels=cfg["dataset"]["sar_channels"],
        optical_channels=cfg["dataset"]["optical_channels"],
        base_dim=cfg["model"]["encoder_dim"],
        num_classes=cfg["dataset"]["num_classes"],
    ).to(device)

    criterion = build_loss(cfg)

    outputs = model(sar, optical)

    total_loss, loss_dict = criterion(outputs, mask)

    print("\n[INFO] Loss check")

    for key, value in loss_dict.items():
        print(f"{key}: {value.item():.6f}")

    print("\n[INFO] Loss backward check")

    total_loss.backward()

    print("[INFO] Backward pass successful.")


if __name__ == "__main__":
    main()