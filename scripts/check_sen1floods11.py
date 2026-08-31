"""
check_sen1floods11.py

Sanity-check script for Sen1Floods11 dataset loading.

Run:
    python scripts/check_sen1floods11.py --config configs/acsf_base.yaml
"""

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from utils.config import load_config
from datasets.dataloader import build_dataloader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check Sen1Floods11 dataloader"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/acsf_base.yaml",
    )

    return parser.parse_args()


def print_tensor_info(name, tensor):
    print(f"\n{name}")
    print(f"  shape: {tuple(tensor.shape)}")
    print(f"  dtype: {tensor.dtype}")
    print(f"  min:   {tensor.min().item():.4f}")
    print(f"  max:   {tensor.max().item():.4f}")
    print(f"  mean:  {tensor.mean().item():.4f}")


def main():
    args = parse_args()
    cfg = load_config(args.config)

    print("[INFO] Building train dataloader...")
    loader = build_dataloader(cfg, split="train")

    print(f"[INFO] Number of batches: {len(loader)}")
    print(f"[INFO] Batch size: {cfg['training']['batch_size']}")

    batch = next(iter(loader))

    sar = batch["sar"]
    optical = batch["optical"]
    mask = batch["mask"]

    print_tensor_info("SAR", sar)
    print_tensor_info("Optical", optical)
    print_tensor_info("Mask", mask)

    print("\nMask unique values:")
    print(torch.unique(mask))

    print("\nMetadata example:")
    for key, value in batch["meta"].items():
        print(f"  {key}: {value[0]}")

    print("\n[INFO] Dataset sanity check completed.")


if __name__ == "__main__":
    main()