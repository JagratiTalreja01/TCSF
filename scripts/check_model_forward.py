"""
check_model_forward.py

Sanity-check ACSF model forward pass using a real Sen1Floods11 batch.

Run:
    python scripts/check_model_forward.py --config configs/acsf_base.yaml
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check ACSF model forward pass"
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

    print(f"[INFO] Using device: {device}")

    loader = build_dataloader(cfg, split="train")
    batch = next(iter(loader))

    sar = batch["sar"].to(device)
    optical = batch["optical"].to(device)

    print(f"[INFO] SAR shape: {tuple(sar.shape)}")
    print(f"[INFO] Optical shape: {tuple(optical.shape)}")

    model = ACSFNet(
        sar_channels=cfg["dataset"]["sar_channels"],
        optical_channels=cfg["dataset"]["optical_channels"],
        base_dim=cfg["model"]["encoder_dim"],
        num_classes=cfg["dataset"]["num_classes"],
    ).to(device)

    model.eval()

    with torch.no_grad():
        outputs = model(sar, optical)

    print("\n[INFO] Model outputs:")

    for key, value in outputs.items():
        print(f"{key}: {tuple(value.shape)}")

    print("\n[INFO] Model forward pass successful.")


if __name__ == "__main__":
    main()