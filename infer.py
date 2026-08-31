"""
infer.py

Inference script for ACSF / TCSF.

Outputs:
- raw tensors;
- colored probability maps;
- reliability heatmaps;
- decision-weight heatmaps;
- paper-ready qualitative comparison figures;
- per-sample metrics.
"""

import argparse
import csv
from pathlib import Path
from typing import Sequence, Tuple

import torch

from utils.config import load_config
from utils.checkpoint import load_model_weights
from utils.visualization import (
    save_comparison_figure,
    save_decision_weights_figure,
    save_prediction_map,
)

from datasets.dataloader import build_dataloader
from models.acsf_net import ACSFNet
from engine.predictor import Predictor


def parse_rgb_indices(values: Sequence[str]) -> Tuple[int, int, int]:
    """Parse exactly three optical RGB band indices."""
    if len(values) != 3:
        raise argparse.ArgumentTypeError(
            "--rgb_indices requires exactly three integers."
        )

    indices = tuple(int(value) for value in values)
    return indices


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ACSF / TCSF inference and save explanatory figures."
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

    parser.add_argument(
        "--save_dir",
        type=str,
        default="outputs/predictions",
    )

    parser.add_argument(
        "--save_limit",
        type=int,
        default=-1,
        help="Number of samples to save. Use -1 to save all.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold used to convert flood probability to a binary mask.",
    )

    parser.add_argument(
        "--rgb_indices",
        nargs=3,
        default=(3, 2, 1),
        metavar=("R", "G", "B"),
        help=(
            "Optical channel indices used for RGB visualization. "
            "Default: 3 2 1 for B1,B2,B3,B4,... channel order."
        ),
    )

    parser.add_argument(
        "--sar_channel",
        type=int,
        default=0,
        help="SAR channel used for grayscale visualization. Default: 0.",
    )

    return parser.parse_args()


def build_model(cfg):
    return ACSFNet(
        sar_channels=cfg["dataset"]["sar_channels"],
        optical_channels=cfg["dataset"]["optical_channels"],
        base_dim=cfg["model"]["encoder_dim"],
        num_classes=cfg["dataset"]["num_classes"],
    )


def _extract_sample_id(batch, fallback_index: int) -> str:
    """Extract a safe sample ID from the dataloader batch."""
    try:
        sample_id = batch["meta"]["id"][0]
    except (KeyError, TypeError, IndexError):
        sample_id = f"sample_{fallback_index:05d}"

    return str(sample_id).replace("/", "_").replace("\\", "_")


def _save_metrics_csv(metrics_rows, csv_path: Path) -> None:
    """Write per-sample metrics to CSV."""
    if not metrics_rows:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sample_id",
        "IoU",
        "Dice",
        "Precision",
        "Recall",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    rgb_indices = tuple(int(value) for value in args.rgb_indices)

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must lie between 0 and 1.")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        and cfg["training"]["device"] == "cuda"
        else "cpu"
    )

    loader = build_dataloader(cfg, split="test")

    model = build_model(cfg).to(device)

    load_model_weights(
        checkpoint_path=args.checkpoint,
        model=model,
        device=device,
    )

    model.eval()
    predictor = Predictor(model, device)

    save_root = Path(args.save_dir)
    tensor_dir = save_root / "tensors"
    figure_dir = save_root / "figures"

    tensor_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    metrics_rows = []

    for batch_index, batch in enumerate(loader):
        sar = batch["sar"]
        optical = batch["optical"]
        mask = batch["mask"]

        preds = predictor.predict(sar, optical)

        sample_id = _extract_sample_id(batch, batch_index)

        sample_tensor_dir = tensor_dir / sample_id
        sample_figure_dir = figure_dir / sample_id

        sample_tensor_dir.mkdir(parents=True, exist_ok=True)
        sample_figure_dir.mkdir(parents=True, exist_ok=True)

        # Save all prediction tensors.
        for key, value in preds.items():
            if isinstance(value, torch.Tensor):
                torch.save(
                    value.detach().cpu(),
                    sample_tensor_dir / f"{key}.pt",
                )

        # Save colored prediction maps.
        prediction_map_specs = [
            ("final_prediction", "Final Flood Probability", "turbo"),
            ("sar_prediction", "SAR-Only Flood Probability", "turbo"),
            ("optical_prediction", "Optical-Only Flood Probability", "turbo"),
            ("sar_reliability", "SAR Reliability", "magma"),
            ("optical_reliability", "Optical Reliability", "magma"),
        ]

        for key, title, cmap in prediction_map_specs:
            if key in preds and preds[key] is not None:
                save_prediction_map(
                    preds[key],
                    sample_figure_dir / f"{key}.png",
                    title=title,
                    cmap=cmap,
                    add_colorbar=True,
                )

        if "decision_weights" in preds and preds["decision_weights"] is not None:
            save_decision_weights_figure(
                preds["decision_weights"],
                sample_figure_dir / "decision_weights.png",
            )

        sample_metrics = save_comparison_figure(
            sar=sar,
            optical=optical,
            mask=mask,
            predictions=preds,
            save_path=sample_figure_dir / "comparison.png",
            sample_name=sample_id,
            rgb_indices=rgb_indices,
            threshold=args.threshold,
            sar_channel=args.sar_channel,
        )

        metrics_rows.append(
            {
                "sample_id": sample_id,
                **sample_metrics,
            }
        )

        print(
            f"[INFO] Saved {sample_id} | "
            f"IoU={sample_metrics['IoU']:.4f}, "
            f"Dice={sample_metrics['Dice']:.4f}"
        )

        saved_count += 1

        if args.save_limit > 0 and saved_count >= args.save_limit:
            break

    metrics_csv_path = save_root / "sample_metrics.csv"
    _save_metrics_csv(metrics_rows, metrics_csv_path)

    print(f"[INFO] Inference completed. Saved {saved_count} samples.")
    print(f"[INFO] Figures: {figure_dir}")
    print(f"[INFO] Tensors: {tensor_dir}")
    print(f"[INFO] Per-sample metrics: {metrics_csv_path}")


if __name__ == "__main__":
    main()