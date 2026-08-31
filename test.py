"""
test.py

Testing script for ACSF / TCSF.

This file performs dataset-level quantitative evaluation.
Use infer.py for colored qualitative visualizations.
"""

import argparse

import torch

from utils.config import load_config
from utils.metrics import SegmentationMetrics
from utils.checkpoint import load_model_weights

from datasets.dataloader import build_dataloader
from models.acsf_net import ACSFNet
from losses.total_loss import build_loss
from engine.evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test ACSF / TCSF for SAR-optical flood mapping."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/acsf_base.yaml",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )

    return parser.parse_args()


def build_model(cfg):
    return ACSFNet(
        sar_channels=cfg["dataset"]["sar_channels"],
        optical_channels=cfg["dataset"]["optical_channels"],
        base_dim=cfg["model"]["encoder_dim"],
        num_classes=cfg["dataset"]["num_classes"],
    )


def main():
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        and cfg["training"]["device"] == "cuda"
        else "cpu"
    )

    test_loader = build_dataloader(cfg, split="test")

    model = build_model(cfg).to(device)

    load_model_weights(
        checkpoint_path=args.checkpoint,
        model=model,
        device=device,
    )

    criterion = build_loss(cfg)
    metrics = SegmentationMetrics()

    evaluator = Evaluator(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        metrics=metrics,
        device=device,
    )

    test_loss, test_metrics = evaluator.evaluate()

    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    print(f"Test Loss: {test_loss:.4f}")

    for key, value in test_metrics.items():
        print(f"{key}: {value:.4f}")

    print("=" * 50)


if __name__ == "__main__":
    main()