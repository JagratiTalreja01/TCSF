"""Save Swin-UNet predictions for later multi-model paper panels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from datasets.dataloader import build_dataloader
from models.baselines.swin_unet_baseline import MultimodalSwinUNet
from utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/swin_unet_base.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--save_dir",
        default="outputs/predictions/SwinUNet_200ep_seed2026",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--rgb_indices", nargs=3, type=int, default=(3, 2, 1))
    parser.add_argument("--overlay_alpha", type=float, default=0.45)
    parser.add_argument("--save_limit", type=int, default=-1)
    return parser.parse_args()


def build_model(cfg):
    baseline_cfg = cfg.get("baseline", {})

    return MultimodalSwinUNet(
        sar_channels=cfg["dataset"]["sar_channels"],
        optical_channels=cfg["dataset"]["optical_channels"],
        num_classes=cfg["dataset"]["num_classes"],
        embed_dim=int(
            baseline_cfg.get("embed_dim", 48)
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
                [3, 6, 12, 24],
            )
        ),
        window_size=int(
            baseline_cfg.get("window_size", 8)
        ),
        mlp_ratio=float(
            baseline_cfg.get("mlp_ratio", 4.0)
        ),
        decoder_channels=tuple(
            int(value)
            for value in baseline_cfg.get(
                "decoder_channels",
                [192, 96, 48],
            )
        ),
        patch_size=int(
            baseline_cfg.get("patch_size", 4)
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


def load_checkpoint(path, model, device):
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint.get("state_dict", checkpoint),
    )
    model.load_state_dict(state_dict)
    print(f"[INFO] Loaded model weights: {path}")


def sanitize(value):
    return str(value).replace("/", "_").replace("\\", "_").replace(" ", "_")


def extract_ids(batch, batch_size, start):
    result = []
    for index in range(batch_size):
        try:
            value = batch["meta"]["id"][index]
        except (KeyError, TypeError, IndexError):
            value = f"sample_{start + index:05d}"
        result.append(sanitize(value))
    return result


def percentile_stretch(image, low=2.0, high=98.0):
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    lo = np.percentile(image, low)
    hi = np.percentile(image, high)
    if hi <= lo:
        lo = float(image.min())
        hi = float(image.max())
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def optical_to_rgb(optical: torch.Tensor, indices: Sequence[int]):
    array = optical.detach().cpu().numpy()
    rgb = np.transpose(array[list(indices)], (1, 2, 0))
    return np.stack(
        [percentile_stretch(rgb[..., channel]) for channel in range(3)],
        axis=-1,
    )


def compute_metrics(prediction, target, eps=1e-6) -> Dict[str, float]:
    pred = prediction.astype(np.float32).reshape(-1)
    truth = target.astype(np.float32).reshape(-1)
    tp = float((pred * truth).sum())
    fp = float((pred * (1.0 - truth)).sum())
    fn = float(((1.0 - pred) * truth).sum())
    tn = float(((1.0 - pred) * (1.0 - truth)).sum())
    iou = (tp + eps) / (tp + fp + fn + eps)
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    pixel_acc = (tp + tn + eps) / (tp + fp + fn + tn + eps)
    return {
        "IoU": iou,
        "Dice": dice,
        "Precision": precision,
        "Recall": recall,
        "F1": dice,
        "PixelAcc": pixel_acc,
    }


def save_gray(array, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, array, cmap="gray", vmin=0.0, vmax=1.0)


def save_overlay(rgb, prediction, path, alpha):
    overlay = rgb.copy()
    mask = prediction.astype(bool)
    overlay[mask, 0] = (1.0 - alpha) * overlay[mask, 0] + alpha
    overlay[mask, 1] = (1.0 - alpha) * overlay[mask, 1]
    overlay[mask, 2] = (1.0 - alpha) * overlay[mask, 2]
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, np.clip(overlay, 0.0, 1.0))


def save_preview(rgb, target, probability, prediction, path, sample_id, metrics):
    error = np.zeros((*target.shape, 3), dtype=np.float32)
    tp = (prediction == 1) & (target == 1)
    fp = (prediction == 1) & (target == 0)
    fn = (prediction == 0) & (target == 1)
    error[tp] = [1.0, 1.0, 1.0]
    error[fp] = [1.0, 0.0, 0.0]
    error[fn] = [0.0, 1.0, 1.0]

    figure, axes = plt.subplots(1, 5, figsize=(17, 3.6))
    panels = [
        (rgb, "Optical RGB", None),
        (target, "Ground Truth", "gray"),
        (probability, "Flood Probability", "gray"),
        (prediction, "Swin-UNet Prediction", "gray"),
        (error, "Error Map", None),
    ]
    for axis, (image, title, cmap) in zip(axes, panels):
        axis.imshow(image, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(
        f"{sample_id} | IoU={metrics['IoU']:.4f}, Dice={metrics['Dice']:.4f}",
        fontsize=11,
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def write_csv(rows: List[Dict[str, object]], path: Path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_config(args.config)

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")

    requested = cfg["training"].get("device", "cuda")
    device = torch.device(
        "cuda" if requested == "cuda" and torch.cuda.is_available() else "cpu"
    )

    loader = build_dataloader(cfg, split="test")
    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, device)
    model.eval()

    root = Path(args.save_dir)
    directories = {
        "prob_npy": root / "arrays" / "probability",
        "binary_npy": root / "arrays" / "binary",
        "prob_pt": root / "tensors" / "probability",
        "binary_pt": root / "tensors" / "binary",
        "prob_png": root / "png" / "probability",
        "binary_png": root / "png" / "binary",
        "overlay": root / "png" / "overlay",
        "gt": root / "png" / "ground_truth",
        "preview": root / "preview",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model": "MultimodalSwinUNet",
        "checkpoint": args.checkpoint,
        "config": args.config,
        "threshold": args.threshold,
        "rgb_indices": list(args.rgb_indices),
        "overlay_alpha": args.overlay_alpha,
        "probability_definition": "sigmoid(final_pred)",
        "binary_definition": f"probability > {args.threshold}",
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    rows = []
    saved = 0

    for batch in tqdm(loader, desc="Saving Swin-U-Net predictions"):
        sar = batch["sar"].to(device, non_blocking=True)
        optical = batch["optical"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        logits = model(sar, optical)["final_pred"]
        probabilities = torch.sigmoid(logits)
        binaries = (probabilities > args.threshold).float()
        sample_ids = extract_ids(batch, sar.shape[0], saved)

        for index, sample_id in enumerate(sample_ids):
            probability = probabilities[index, 0].cpu().numpy().astype(np.float32)
            prediction = binaries[index, 0].cpu().numpy().astype(np.uint8)
            target = mask[index, 0].cpu().numpy().astype(np.uint8)
            rgb = optical_to_rgb(optical[index], args.rgb_indices)
            metrics = compute_metrics(prediction, target)

            np.save(directories["prob_npy"] / f"{sample_id}.npy", probability)
            np.save(directories["binary_npy"] / f"{sample_id}.npy", prediction)
            torch.save(probabilities[index, 0].cpu(), directories["prob_pt"] / f"{sample_id}.pt")
            torch.save(binaries[index, 0].cpu(), directories["binary_pt"] / f"{sample_id}.pt")

            save_gray(probability, directories["prob_png"] / f"{sample_id}.png")
            save_gray(prediction, directories["binary_png"] / f"{sample_id}.png")
            save_gray(target, directories["gt"] / f"{sample_id}.png")
            save_overlay(rgb, prediction, directories["overlay"] / f"{sample_id}.png", args.overlay_alpha)
            save_preview(
                rgb,
                target,
                probability,
                prediction,
                directories["preview"] / f"{sample_id}.png",
                sample_id,
                metrics,
            )

            rows.append({"sample_id": sample_id, **metrics})
            saved += 1

            if args.save_limit > 0 and saved >= args.save_limit:
                break

        if args.save_limit > 0 and saved >= args.save_limit:
            break

    write_csv(rows, root / "per_image_metrics.csv")

    print("\n" + "=" * 68)
    print("Swin-U-NET PREDICTION EXPORT COMPLETED")
    print("=" * 68)
    print(f"Saved samples : {saved}")
    print(f"Output root   : {root}")
    print(f"Probabilities : {directories['prob_npy']}")
    print(f"Binary masks  : {directories['binary_npy']}")
    print(f"Overlays      : {directories['overlay']}")
    print(f"Preview panels: {directories['preview']}")
    print(f"Metrics CSV   : {root / 'per_image_metrics.csv'}")
    print("=" * 68)


if __name__ == "__main__":
    main()