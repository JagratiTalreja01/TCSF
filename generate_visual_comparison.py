#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import importlib
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from datasets.dataloader import build_dataloader

MODEL_SPECS = [
    {"name":"U-Net","factory":"test_unet:build_model","config":"configs/unet_base.yaml","checkpoint":"outputs/checkpoints/UNet_200ep_seed2026/best_model.pth"},
    {"name":"DeepLabV3+","factory":"test_deeplabv3plus:build_model","config":"configs/deeplabv3plus_base.yaml","checkpoint":"outputs/checkpoints/DeepLabV3Plus_200ep_seed2026/best_model.pth"},
    {"name":"Swin-UNet","factory":"test_swin_unet:build_model","config":"configs/swin_unet_base.yaml","checkpoint":"outputs/checkpoints/SwinUNet_200ep_seed2026/best_model.pth"},
    {"name":"SegFormer-B0","factory":"test_segformer:build_model","config":"configs/segformer_base.yaml","checkpoint":"outputs/checkpoints/SegFormer_B0_200ep_seed2026/best_model.pth"},
    {"name":"Vision Mamba","factory":"test_vision_mamba:build_model","config":"configs/vision_mamba_base.yaml","checkpoint":"outputs/checkpoints/VisionMamba_200ep_seed2026/best_model.pth"},
    {"name":"TCSF v3.1","factory":"test:build_model","config":"configs/acsf_base.yaml","checkpoint":"outputs/checkpoints/TCSF_v31_final_200ep/best_model.pth"},
]


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def import_attr(path: str):
    module_name, attr_name = path.rsplit(":", 1)
    return getattr(importlib.import_module(module_name), attr_name)


def extract_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        for key in ("model_state_dict", "state_dict", "model", "net"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return value
        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint
    raise TypeError("Could not locate state_dict in checkpoint")


def load_model(spec: Dict[str, str], device: torch.device) -> nn.Module:
    cfg = load_yaml(spec["config"])
    model = import_attr(spec["factory"])(cfg)
    checkpoint = torch.load(spec["checkpoint"], map_location="cpu")
    state = extract_state_dict(checkpoint)
    clean = {}
    for key, value in state.items():
        new_key = key
        for prefix in ("module.", "model.", "net."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        clean[new_key] = value
    missing, unexpected = model.load_state_dict(clean, strict=False)
    print(f"[INFO] Loaded {spec['name']} | missing={len(missing)} unexpected={len(unexpected)}")
    return model.to(device).eval()


def forward_model(model: nn.Module, sar: torch.Tensor, optical: torch.Tensor):
    try:
        return model(sar, optical)
    except TypeError:
        return model(torch.cat([sar, optical], dim=1))


def extract_logits(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, Mapping):
        for key in ("final_pred", "logits", "pred", "prediction", "out", "fused_pred", "mask"):
            value = output.get(key)
            if torch.is_tensor(value):
                return value
        for value in output.values():
            if torch.is_tensor(value) and value.ndim >= 3:
                return value
    if isinstance(output, (tuple, list)):
        for value in output:
            try:
                return extract_logits(value)
            except (TypeError, ValueError):
                pass
    raise TypeError(f"Unsupported output type: {type(output)}")


def parse_batch(batch: Any, device: torch.device):
    metadata = {}
    if isinstance(batch, Mapping):
        sar = next((batch[k] for k in ("sar", "s1", "sentinel1") if k in batch), None)
        optical = next((batch[k] for k in ("optical", "s2", "sentinel2", "image") if k in batch), None)
        mask = next((batch[k] for k in ("mask", "label", "target", "flood_mask") if k in batch), None)
        for key in ("name", "filename", "id", "sample_id", "tile_id"):
            if key in batch:
                metadata[key] = batch[key]
    else:
        sar, optical, mask = batch[:3]
    if sar is None or optical is None or mask is None:
        raise KeyError("Batch must contain SAR, optical, and mask")
    sar = sar.float().to(device)
    optical = optical.float().to(device)
    mask = mask.float().to(device)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    return sar, optical, mask, metadata


def stretch(image: np.ndarray, low: float = 2, high: float = 98) -> np.ndarray:
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros_like(image, dtype=np.float32)
    lo = np.percentile(image[finite], low)
    hi = np.percentile(image[finite], high)
    return np.clip((image - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def render_sar(sar_sample: torch.Tensor) -> np.ndarray:
    return stretch(sar_sample[0].detach().cpu().numpy())


def render_rgb(optical_sample: torch.Tensor, r: int, g: int, b: int, gamma: float) -> np.ndarray:
    optical = optical_sample.detach().cpu().numpy()
    rgb = np.stack([optical[r], optical[g], optical[b]], axis=-1)
    finite = np.isfinite(rgb)
    lo = np.percentile(rgb[finite], 2)
    hi = np.percentile(rgb[finite], 98)
    rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)
    if gamma > 0:
        rgb = np.power(rgb, 1.0 / gamma)
    return rgb.astype(np.float32)


def prediction_from_logits(logits: torch.Tensor, size: Tuple[int, int], threshold: float):
    if logits.ndim == 3:
        logits = logits.unsqueeze(1)
    if logits.shape[-2:] != size:
        logits = F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
    if logits.shape[1] > 1:
        logits = logits[:, 1:2]
    min_v = float(logits.detach().min())
    max_v = float(logits.detach().max())
    probs = logits if min_v >= 0 and max_v <= 1 else torch.sigmoid(logits)
    prob = probs[0, 0].detach().cpu().numpy()
    pred = (prob >= threshold).astype(np.uint8)
    return prob, pred


def iou(pred: np.ndarray, target: np.ndarray) -> float:
    p = pred.astype(bool)
    t = target.astype(bool)
    inter = np.logical_and(p, t).sum()
    union = np.logical_or(p, t).sum()
    return 1.0 if union == 0 else float(inter / union)


def sample_name(metadata: Dict[str, Any], index: int) -> str:
    for key in ("name", "filename", "id", "sample_id", "tile_id"):
        if key in metadata:
            value = metadata[key]
            if isinstance(value, (list, tuple)) and value:
                value = value[0]
            elif torch.is_tensor(value):
                value = value.flatten()[0].item()
            return Path(str(value)).stem
    return f"sample_{index:04d}"


def safe_name(text: str) -> str:
    return text.lower().replace("+", "plus").replace(" ", "_").replace(".", "")


def create_figure(index: int, name: str, sar_img: np.ndarray, rgb: np.ndarray, gt: np.ndarray,
                  predictions: Dict[str, np.ndarray], probabilities: Dict[str, np.ndarray],
                  output_dir: Path, dpi: int, show_iou: bool, save_masks: bool):
    panels = [("SAR (VV)", sar_img, "gray"), ("Optical RGB", rgb, None), ("Ground Truth", gt, "gray")]
    for spec in MODEL_SPECS:
        model_name = spec["name"]
        title = model_name if not show_iou else f"{model_name}\nIoU: {iou(predictions[model_name], gt):.3f}"
        panels.append((title, predictions[model_name], "gray"))

    fig, axes = plt.subplots(3, 3, figsize=(13.5, 13.0))
    for ax, (title, image, cmap) in zip(axes.ravel(), panels):
        if cmap is None:
            ax.imshow(image)
        else:
            ax.imshow(image, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
        ax.axis("off")
    fig.suptitle(f"Flood Segmentation Comparison — {name}", fontsize=17, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975), pad=1.0)
    stem = f"{index:04d}_{name}_comparison"
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

    if save_masks:
        mask_dir = output_dir / f"{index:04d}_{name}_masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        plt.imsave(mask_dir / "ground_truth.png", gt, cmap="gray", vmin=0, vmax=1)
        for model_name, pred in predictions.items():
            key = safe_name(model_name)
            plt.imsave(mask_dir / f"{key}_binary.png", pred, cmap="gray", vmin=0, vmax=1)
            plt.imsave(mask_dir / f"{key}_probability.png", probabilities[model_name], cmap="viridis", vmin=0, vmax=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-indices", type=int, nargs="+", default=[0, 10, 20, 30, 40])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--output-dir", default="outputs/publication/visual_comparison")
    parser.add_argument("--red-index", type=int, default=3)
    parser.add_argument("--green-index", type=int, default=2)
    parser.add_argument("--blue-index", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=1.15)
    parser.add_argument("--no-iou", action="store_true")
    parser.add_argument("--save-masks", action="store_true")
    args = parser.parse_args()

    indices = sorted(set(args.sample_indices))
    if any(i < 0 for i in indices):
        raise ValueError("Sample indices must be non-negative")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    data_cfg = load_yaml("configs/acsf_base.yaml")
    loader = build_dataloader(data_cfg, split="test")
    print(f"[INFO] Test loader: {len(loader)} batches, {len(loader.dataset)} samples")
    if max(indices) >= len(loader.dataset):
        raise IndexError(f"Largest requested index is {max(indices)}, but test set has {len(loader.dataset)} samples")

    models = {spec["name"]: load_model(spec, device) for spec in MODEL_SPECS}

    generated = 0
    global_index = 0
    with torch.no_grad():
        for batch in loader:
            sar, optical, mask, metadata = parse_batch(batch, device)
            batch_size = sar.shape[0]
            for local_index in range(batch_size):
                idx = global_index + local_index
                if idx not in indices:
                    continue
                sar_one = sar[local_index:local_index+1]
                optical_one = optical[local_index:local_index+1]
                gt = (mask[local_index, 0].detach().cpu().numpy() >= 0.5).astype(np.uint8)
                name = sample_name(metadata, idx)
                sar_img = render_sar(sar_one[0])
                rgb = render_rgb(optical_one[0], args.red_index, args.green_index, args.blue_index, args.gamma)

                predictions = {}
                probabilities = {}
                for model_name, model in models.items():
                    with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                        logits = extract_logits(forward_model(model, sar_one, optical_one))
                    prob, pred = prediction_from_logits(logits, gt.shape, args.threshold)
                    probabilities[model_name] = prob
                    predictions[model_name] = pred

                create_figure(idx, name, sar_img, rgb, gt, predictions, probabilities,
                              output_dir, args.dpi, not args.no_iou, args.save_masks)
                generated += 1
                print(f"[INFO] Generated sample {idx}: {name}")

            global_index += batch_size
            if generated == len(indices):
                break

    del models
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"[INFO] Generated {generated} figures in {output_dir}")


if __name__ == "__main__":
    main()