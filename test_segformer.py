"""
test_segformer.py

Test-set evaluation for the multimodal early-fusion SegFormer baseline.

Reports both:
1. Macro metrics: compute each metric per image, then average.
2. Micro/global metrics: accumulate TP, FP, FN, and TN over all pixels.

The macro metrics should be used for direct comparison with the published
TCSF v3.1 result.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from datasets.dataloader import build_dataloader
from models.baselines.segformer_baseline import MultimodalSegFormer
from utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the multimodal SegFormer baseline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/segformer_base.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="outputs/test/SegFormer_B0_200ep_seed2026",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )
    return parser.parse_args()


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


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    device: torch.device,
) -> None:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint.get("state_dict", checkpoint),
    )

    model.load_state_dict(state_dict)
    print(f"[INFO] Loaded model weights: {checkpoint_path}")


def safe_sample_id(batch: dict, sample_index: int) -> str:
    try:
        value = batch["meta"]["id"][sample_index]
    except (KeyError, TypeError, IndexError):
        value = f"sample_{sample_index:05d}"

    return str(value).replace("/", "_").replace("\\", "_")


def image_metrics_from_counts(
    tp: float,
    fp: float,
    fn: float,
    tn: float,
    eps: float = 1e-6,
) -> Dict[str, float]:
    iou = (tp + eps) / (tp + fp + fn + eps)
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    pixel_accuracy = (tp + tn + eps) / (
        tp + fp + fn + tn + eps
    )

    return {
        "IoU": iou,
        "Dice": dice,
        "Precision": precision,
        "Recall": recall,
        "F1": dice,
        "PixelAcc": pixel_accuracy,
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    threshold: float,
) -> tuple[Dict[str, float], Dict[str, float], List[Dict[str, float]]]:
    model.eval()

    per_image_rows: List[Dict[str, float]] = []

    global_tp = 0.0
    global_fp = 0.0
    global_fn = 0.0
    global_tn = 0.0

    sample_counter = 0

    for batch in tqdm(loader, desc="Testing"):
        sar = batch["sar"].to(device, non_blocking=True)
        optical = batch["optical"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        outputs = model(sar, optical)
        logits = outputs["final_pred"]
        predictions = (torch.sigmoid(logits) > threshold).float()

        batch_size = sar.shape[0]

        for index in range(batch_size):
            pred = predictions[index].reshape(-1)
            target = mask[index].reshape(-1)

            tp = float((pred * target).sum().cpu())
            fp = float((pred * (1.0 - target)).sum().cpu())
            fn = float(((1.0 - pred) * target).sum().cpu())
            tn = float(((1.0 - pred) * (1.0 - target)).sum().cpu())

            metrics = image_metrics_from_counts(tp, fp, fn, tn)

            per_image_rows.append(
                {
                    "sample_id": safe_sample_id(batch, index),
                    **metrics,
                }
            )

            global_tp += tp
            global_fp += fp
            global_fn += fn
            global_tn += tn
            sample_counter += 1

    metric_names = [
        "IoU",
        "Dice",
        "Precision",
        "Recall",
        "F1",
        "PixelAcc",
    ]

    macro = {}
    for name in metric_names:
        values = np.asarray(
            [row[name] for row in per_image_rows],
            dtype=np.float64,
        )
        macro[name] = float(values.mean())
        macro[f"{name}_std"] = float(values.std(ddof=0))

    macro["num_samples"] = sample_counter

    micro = image_metrics_from_counts(
        global_tp,
        global_fp,
        global_fn,
        global_tn,
    )
    micro["num_samples"] = sample_counter

    return macro, micro, per_image_rows


def save_csv(rows: List[Dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must lie between 0 and 1.")

    requested_device = cfg["training"].get("device", "cuda")
    device = torch.device(
        "cuda"
        if requested_device == "cuda" and torch.cuda.is_available()
        else "cpu"
    )

    loader = build_dataloader(cfg, split="test")
    model = build_model(cfg).to(device)

    load_checkpoint(
        checkpoint_path=args.checkpoint,
        model=model,
        device=device,
    )

    macro, micro, rows = evaluate(
        model=model,
        loader=loader,
        device=device,
        threshold=args.threshold,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "model": "MultimodalSegFormer",
        "checkpoint": args.checkpoint,
        "threshold": args.threshold,
        "macro": macro,
        "micro": micro,
    }

    save_json(result, save_dir / "test_metrics.json")
    save_csv(rows, save_dir / "per_image_metrics.csv")

    print("\n" + "=" * 62)
    print("SegFormer TEST RESULTS")
    print("=" * 62)

    print("\nMacro mean ± sample SD")
    print("-" * 62)
    for name in ["IoU", "Dice", "Precision", "Recall", "F1", "PixelAcc"]:
        print(
            f"{name:10s}: "
            f"{macro[name]:.4f} ± {macro[f'{name}_std']:.4f}"
        )

    print("\nMicro/global")
    print("-" * 62)
    for name in ["IoU", "Dice", "Precision", "Recall", "F1", "PixelAcc"]:
        print(f"{name:10s}: {micro[name]:.4f}")

    print("-" * 62)
    print(f"Samples   : {macro['num_samples']}")
    print(f"Results   : {save_dir / 'test_metrics.json'}")
    print(f"Per-image : {save_dir / 'per_image_metrics.csv'}")
    print("=" * 62)


if __name__ == "__main__":
    main()