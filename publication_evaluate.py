"""
publication_evaluate.py

Unified publication evaluation for ACSF / TCSF flood segmentation.

Produces:
- parameter counts;
- FLOPs/MACs when fvcore or thop is installed;
- FP32 and optional AMP latency, FPS, throughput, and CUDA memory;
- dataset-level test metrics computed from global confusion counts;
- per-sample metrics;
- transition-scale values;
- raw probability and binary prediction maps;
- automatically selected best, median, and worst qualitative figures;
- JSON, CSV, TXT, and log summaries.

The script does not modify the model or checkpoint.

Example:
    python publication_evaluate.py \
        --config configs/acsf_base.yaml \
        --checkpoint outputs/checkpoints/TCSF_v31_seed2026/best_model.pth \
        --save_dir outputs/publication/TCSF_v31_seed2026
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import platform
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.dataloader import build_dataset
from models.acsf_net import ACSFNet
from utils.checkpoint import load_model_weights
from utils.config import load_config


METRIC_NAMES = ("IoU", "Dice", "Precision", "Recall", "F1", "PixelAcc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure TCSF computational complexity and generate publication-ready "
            "test metrics and qualitative figures."
        )
    )
    parser.add_argument("--config", default="configs/acsf_base.yaml", type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument(
        "--save_dir",
        default="outputs/publication/TCSF_v31_seed2026",
        type=str,
    )
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument(
        "--rgb_indices",
        nargs=3,
        default=(3, 2, 1),
        type=int,
        metavar=("R", "G", "B"),
        help="Zero-based Sentinel-2 channel indices used for RGB display.",
    )
    parser.add_argument("--sar_channel", default=0, type=int)
    parser.add_argument(
        "--qualitative_per_group",
        default=5,
        type=int,
        help="Number of best, median, and worst examples to save.",
    )
    parser.add_argument(
        "--warmup_iterations",
        default=30,
        type=int,
        help="Untimed warm-up forwards used before latency measurement.",
    )
    parser.add_argument(
        "--timed_iterations",
        default=100,
        type=int,
        help="Number of synchronized forwards used for latency measurement.",
    )
    parser.add_argument(
        "--benchmark_batch_size",
        default=1,
        type=int,
        help="Batch size used for latency/FPS benchmarking.",
    )
    parser.add_argument(
        "--throughput_batch_size",
        default=0,
        type=int,
        help=(
            "Batch size for throughput benchmarking. Use 0 to reuse the config "
            "training batch size."
        ),
    )
    parser.add_argument(
        "--num_workers",
        default=-1,
        type=int,
        help="Evaluation DataLoader workers. Use -1 to reuse the config value.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Also benchmark CUDA automatic mixed precision.",
    )
    parser.add_argument(
        "--save_all_maps",
        action="store_true",
        help="Save probability and binary PNG maps for every test image.",
    )
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="Optional runtime seed. Defaults to experiment.seed in the config.",
    )
    return parser.parse_args()


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("publication_evaluate")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(cfg: Mapping[str, Any]) -> ACSFNet:
    model_cfg = cfg.get("model", {})
    dataset_cfg = cfg["dataset"]
    return ACSFNet(
        sar_channels=dataset_cfg["sar_channels"],
        optical_channels=dataset_cfg["optical_channels"],
        base_dim=model_cfg["encoder_dim"],
        num_classes=dataset_cfg["num_classes"],
        use_reliability_guidance=model_cfg.get(
            "use_reliability_guidance", True
        ),
        use_cross_scale_propagation=model_cfg.get(
            "use_cross_scale_propagation", True
        ),
        transition_residual_init=model_cfg.get(
            "transition_residual_init", 1e-3
        ),
    )


def build_eval_loader(
    cfg: Mapping[str, Any],
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = build_dataset(cfg, split="test")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(cfg["dataset"].get("pin_memory", True)),
        drop_last=False,
    )


def safe_sample_ids(batch: Mapping[str, Any], batch_index: int, batch_size: int) -> List[str]:
    ids: List[str] = []
    try:
        raw_ids = batch["meta"]["id"]
        if isinstance(raw_ids, str):
            raw_ids = [raw_ids]
        for raw_id in raw_ids:
            ids.append(str(raw_id).replace("/", "_").replace("\\", "_"))
    except (KeyError, TypeError, IndexError):
        pass

    while len(ids) < batch_size:
        ids.append(f"sample_{batch_index:05d}_{len(ids):02d}")
    return ids[:batch_size]


def logits_from_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if not isinstance(output, Mapping):
        raise TypeError(f"Unsupported model output type: {type(output)!r}")
    if "final_pred" not in output:
        raise KeyError("Model output does not contain 'final_pred'.")
    return output["final_pred"]


def confusion_counts(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
) -> Tuple[int, int, int, int]:
    predictions = torch.sigmoid(logits) >= threshold
    targets_bool = targets >= 0.5

    tp = torch.logical_and(predictions, targets_bool).sum().item()
    fp = torch.logical_and(predictions, ~targets_bool).sum().item()
    fn = torch.logical_and(~predictions, targets_bool).sum().item()
    tn = torch.logical_and(~predictions, ~targets_bool).sum().item()
    return int(tp), int(fp), int(fn), int(tn)


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    # Match utils.metrics.SegmentationMetrics exactly, including its treatment
    # of empty ground-truth / empty-prediction samples.
    eps = 1e-6
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    f1 = (2 * precision * recall) / (precision + recall + eps)
    pixel_accuracy = (tp + tn) / (tp + fp + fn + tn + eps)
    return {
        "IoU": float(iou),
        "Dice": float(dice),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "PixelAcc": float(pixel_accuracy),
    }


def parameter_statistics(model: torch.nn.Module) -> Dict[str, Any]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    buffers = sum(buffer.numel() for buffer in model.buffers())
    model_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    ) + sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())

    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "non_trainable_parameters": int(total - trainable),
        "buffer_elements": int(buffers),
        "model_size_mb": float(model_bytes / (1024**2)),
    }


class _FinalPredictionWrapper(torch.nn.Module):
    """Expose only final logits so FLOP profilers see a tensor output."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, sar: torch.Tensor, optical: torch.Tensor) -> torch.Tensor:
        return logits_from_output(self.model(sar, optical))


def profile_flops(
    model: torch.nn.Module,
    sar: torch.Tensor,
    optical: torch.Tensor,
    logger: logging.Logger,
) -> Dict[str, Any]:
    wrapper = _FinalPredictionWrapper(model).eval()
    result: Dict[str, Any] = {
        "profiler": None,
        "macs": None,
        "gmacs": None,
        "flops": None,
        "gflops": None,
        "note": None,
    }

    # fvcore reports FLOPs. Some custom/state-space operations may be unsupported.
    try:
        from fvcore.nn import FlopCountAnalysis  # type: ignore

        analysis = FlopCountAnalysis(wrapper, (sar, optical))
        analysis.unsupported_ops_warnings(False)
        analysis.uncalled_modules_warnings(False)
        flops = int(analysis.total())
        unsupported = {
            str(key): int(value)
            for key, value in analysis.unsupported_ops().items()
        }
        result.update(
            {
                "profiler": "fvcore",
                "flops": flops,
                "gflops": flops / 1e9,
                "unsupported_operations": unsupported,
                "note": (
                    "FLOPs may be underestimated when custom operations are "
                    "listed as unsupported."
                    if unsupported
                    else None
                ),
            }
        )
        return result
    except ImportError:
        logger.info("fvcore is not installed; trying thop for MACs.")
    except Exception as exc:  # profiler failures must not stop evaluation
        logger.warning("fvcore profiling failed: %s", exc)

    # thop conventionally reports MACs; FLOPs are approximated as 2 * MACs.
    try:
        from thop import profile  # type: ignore

        macs, _ = profile(wrapper, inputs=(sar, optical), verbose=False)
        macs = int(macs)
        result.update(
            {
                "profiler": "thop",
                "macs": macs,
                "gmacs": macs / 1e9,
                "flops": 2 * macs,
                "gflops": 2 * macs / 1e9,
                "note": "FLOPs approximated as 2 × MACs.",
            }
        )
        return result
    except ImportError:
        result["note"] = (
            "Install fvcore or thop to measure operation counts: "
            "pip install fvcore thop"
        )
    except Exception as exc:
        result["note"] = f"thop profiling failed: {exc}"
        logger.warning("thop profiling failed: %s", exc)

    return result


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def benchmark_forward(
    model: torch.nn.Module,
    sar: torch.Tensor,
    optical: torch.Tensor,
    device: torch.device,
    warmup_iterations: int,
    timed_iterations: int,
    use_amp: bool,
) -> Dict[str, Any]:
    model.eval()
    if timed_iterations <= 0:
        raise ValueError("timed_iterations must be greater than zero.")

    with torch.inference_mode():
        for _ in range(max(0, warmup_iterations)):
            with autocast_context(device, use_amp):
                _ = model(sar, optical)
        synchronize(device)

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        latencies_ms: List[float] = []
        for _ in range(timed_iterations):
            synchronize(device)
            start = time.perf_counter()
            with autocast_context(device, use_amp):
                _ = model(sar, optical)
            synchronize(device)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)

    mean_ms = statistics.fmean(latencies_ms)
    median_ms = statistics.median(latencies_ms)
    std_ms = statistics.pstdev(latencies_ms) if len(latencies_ms) > 1 else 0.0
    sorted_times = sorted(latencies_ms)
    p95_index = min(len(sorted_times) - 1, math.ceil(0.95 * len(sorted_times)) - 1)
    p95_ms = sorted_times[p95_index]
    batch_size = int(sar.shape[0])

    result: Dict[str, Any] = {
        "precision": "amp_fp16" if use_amp else "fp32",
        "batch_size": batch_size,
        "warmup_iterations": int(warmup_iterations),
        "timed_iterations": int(timed_iterations),
        "mean_latency_ms_per_batch": float(mean_ms),
        "median_latency_ms_per_batch": float(median_ms),
        "latency_std_ms": float(std_ms),
        "p95_latency_ms_per_batch": float(p95_ms),
        "mean_latency_ms_per_image": float(mean_ms / batch_size),
        "fps": float(batch_size * 1000.0 / mean_ms),
    }

    if device.type == "cuda":
        result.update(
            {
                "peak_allocated_memory_mb": float(
                    torch.cuda.max_memory_allocated(device) / (1024**2)
                ),
                "peak_reserved_memory_mb": float(
                    torch.cuda.max_memory_reserved(device) / (1024**2)
                ),
            }
        )
    else:
        result.update(
            {
                "peak_allocated_memory_mb": None,
                "peak_reserved_memory_mb": None,
            }
        )
    return result


def tensor_to_numpy_2d(value: torch.Tensor) -> np.ndarray:
    array = value.detach().float().cpu().numpy()
    while array.ndim > 2:
        array = array[0]
    return array


def robust_minmax(array: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array, dtype=np.float32)
    lower = np.percentile(finite, low)
    upper = np.percentile(finite, high)
    if upper <= lower:
        lower = float(finite.min())
        upper = float(finite.max())
    if upper <= lower:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - lower) / (upper - lower), 0.0, 1.0).astype(np.float32)


def optical_to_rgb(optical: torch.Tensor, rgb_indices: Sequence[int]) -> np.ndarray:
    optical_np = optical.detach().float().cpu().numpy()
    if optical_np.ndim == 4:
        optical_np = optical_np[0]
    if optical_np.ndim != 3:
        raise ValueError(f"Expected optical [C,H,W], got {optical_np.shape}")
    if max(rgb_indices) >= optical_np.shape[0] or min(rgb_indices) < 0:
        raise IndexError(
            f"RGB indices {tuple(rgb_indices)} are invalid for "
            f"{optical_np.shape[0]} optical channels."
        )
    channels = [robust_minmax(optical_np[index]) for index in rgb_indices]
    rgb = np.stack(channels, axis=-1)
    # Mild gamma adjustment makes dark Sentinel-2 chips readable without inventing data.
    return np.clip(rgb, 0.0, 1.0) ** (1.0 / 1.15)


def error_map(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """RGB error map: TP white, FP red, FN blue, TN black."""
    pred = prediction.astype(bool)
    truth = target.astype(bool)
    image = np.zeros((*truth.shape, 3), dtype=np.float32)
    image[np.logical_and(pred, truth)] = (1.0, 1.0, 1.0)
    image[np.logical_and(pred, ~truth)] = (1.0, 0.0, 0.0)
    image[np.logical_and(~pred, truth)] = (0.0, 0.45, 1.0)
    return image


def save_grayscale_png(array: np.ndarray, path: Path, vmin: float = 0.0, vmax: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, array, cmap="gray", vmin=vmin, vmax=vmax)


def save_probability_png(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, array, cmap="viridis", vmin=0.0, vmax=1.0)


def save_paper_figure(
    sample: Mapping[str, Any],
    save_path: Path,
    rgb_indices: Sequence[int],
    sar_channel: int,
    threshold: float,
    dpi: int = 300,
) -> None:
    sar = sample["sar"]
    optical = sample["optical"]
    target = sample["mask"]
    logits = sample["logits"]

    sar_np = sar.detach().float().cpu().numpy()
    if sar_np.ndim == 4:
        sar_np = sar_np[0]
    if not 0 <= sar_channel < sar_np.shape[0]:
        raise IndexError(
            f"SAR channel {sar_channel} is invalid for {sar_np.shape[0]} channels."
        )
    sar_display = robust_minmax(sar_np[sar_channel])
    rgb = optical_to_rgb(optical, rgb_indices)
    truth = tensor_to_numpy_2d(target) >= 0.5
    probability = torch.sigmoid(logits).detach().float().cpu().numpy()
    while probability.ndim > 2:
        probability = probability[0]
    binary = probability >= threshold
    errors = error_map(binary, truth)

    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.5), constrained_layout=True)
    panels = [
        (sar_display, "SAR", "gray", 0.0, 1.0),
        (rgb, "Optical RGB", None, None, None),
        (truth, "Ground Truth", "gray", 0.0, 1.0),
        (binary, "Final Prediction", "gray", 0.0, 1.0),
        (probability, "Flood Probability", "viridis", 0.0, 1.0),
        (errors, "Error Map", None, None, None),
    ]

    for axis, (image, title, cmap, vmin, vmax) in zip(axes.flat, panels):
        axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title, fontsize=13, fontweight="bold")
        axis.axis("off")

    metrics_text = (
        f"IoU={sample['IoU']:.4f}  |  Dice={sample['Dice']:.4f}  |  "
        f"Precision={sample['Precision']:.4f}  |  Recall={sample['Recall']:.4f}"
    )
    fig.suptitle(f"{sample['sample_id']}\n{metrics_text}", fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        0.005,
        "Error map: white = true positive, red = false positive, blue = false negative, black = true negative",
        ha="center",
        fontsize=9,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def flatten_transition_scales(scales: Any) -> List[Dict[str, Any]]:
    if isinstance(scales, torch.Tensor):
        array = scales.detach().float().cpu().numpy()
    else:
        array = np.asarray(scales, dtype=np.float32)

    if array.ndim == 1 and array.size == 9:
        array = array.reshape(3, 3)
    if array.shape != (3, 3):
        raise ValueError(
            f"Expected transition scales with shape (3, 3), got {array.shape}."
        )

    modalities = ("sar", "optical", "fused")
    rows: List[Dict[str, Any]] = []
    for transition_index in range(3):
        for modality_index, modality in enumerate(modalities):
            rows.append(
                {
                    "transition": transition_index + 1,
                    "modality": modality,
                    "gamma": float(array[transition_index, modality_index]),
                }
            )
    return rows


def get_transition_rows(model: torch.nn.Module) -> List[Dict[str, Any]]:
    if hasattr(model, "tri_level_fusion") and hasattr(
        model.tri_level_fusion, "get_transition_scales"
    ):
        return flatten_transition_scales(
            model.tri_level_fusion.get_transition_scales()
        )
    if hasattr(model, "get_transition_scale_dict"):
        scale_dict = model.get_transition_scale_dict()
        rows = []
        for name, value in scale_dict.items():
            scalar = value.item() if isinstance(value, torch.Tensor) else float(value)
            rows.append({"name": str(name), "gamma": float(scalar)})
        return rows
    return []


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=False)


def select_representative_samples(
    samples: Sequence[Mapping[str, Any]],
    count_per_group: int,
) -> Dict[str, List[Mapping[str, Any]]]:
    if not samples or count_per_group <= 0:
        return {"best": [], "median": [], "worst": []}

    ordered = sorted(samples, key=lambda row: float(row["IoU"]))
    count = min(count_per_group, len(ordered))
    worst = ordered[:count]
    best = list(reversed(ordered[-count:]))

    center = (len(ordered) - 1) / 2.0
    median = sorted(
        ordered,
        key=lambda row: abs(float(row["rank_index"]) - center),
    )[:count]
    median = sorted(median, key=lambda row: float(row["IoU"]), reverse=True)
    return {"best": best, "median": median, "worst": worst}


def evaluate_test_set(
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, Any]],
    device: torch.device,
    threshold: float,
    save_root: Path,
    save_all_maps: bool,
    logger: logging.Logger,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    model.eval()
    total_tp = total_fp = total_fn = total_tn = 0
    per_sample: List[Dict[str, Any]] = []
    cache_dir = save_root / "cache"
    probability_dir = save_root / "raw_predictions" / "probability_maps"
    binary_dir = save_root / "raw_predictions" / "binary_masks"
    cache_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    sample_counter = 0

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            sar = batch["sar"].to(device, non_blocking=True)
            optical = batch["optical"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            output = model(sar, optical)
            logits = logits_from_output(output)

            batch_size = int(logits.shape[0])
            sample_ids = safe_sample_ids(batch, batch_index, batch_size)

            for item_index in range(batch_size):
                item_logits = logits[item_index : item_index + 1]
                item_mask = mask[item_index : item_index + 1]
                tp, fp, fn, tn = confusion_counts(item_logits, item_mask, threshold)
                item_metrics = metrics_from_counts(tp, fp, fn, tn)
                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_tn += tn

                sample_id = sample_ids[item_index]
                cache_path = cache_dir / f"{sample_counter:05d}_{sample_id}.pt"
                torch.save(
                    {
                        "sample_id": sample_id,
                        "sar": sar[item_index : item_index + 1].detach().cpu(),
                        "optical": optical[item_index : item_index + 1].detach().cpu(),
                        "mask": item_mask.detach().cpu(),
                        "logits": item_logits.detach().cpu(),
                        **item_metrics,
                    },
                    cache_path,
                )

                probability = torch.sigmoid(item_logits).detach().float().cpu().numpy()
                while probability.ndim > 2:
                    probability = probability[0]
                binary = (probability >= threshold).astype(np.float32)
                if save_all_maps:
                    save_probability_png(probability, probability_dir / f"{sample_id}.png")
                    save_grayscale_png(binary, binary_dir / f"{sample_id}.png")

                per_sample.append(
                    {
                        "sample_id": sample_id,
                        **item_metrics,
                        "TP": tp,
                        "FP": fp,
                        "FN": fn,
                        "TN": tn,
                        "cache_path": str(cache_path),
                        "rank_index": sample_counter,
                    }
                )
                sample_counter += 1

            if (batch_index + 1) % 10 == 0 or batch_index == 0:
                logger.info(
                    "Evaluated %d batches / %d samples",
                    batch_index + 1,
                    sample_counter,
                )

    synchronize(device)
    elapsed = time.perf_counter() - start_time
    micro_metrics = metrics_from_counts(total_tp, total_fp, total_fn, total_tn)

    macro_metrics: Dict[str, float] = {}
    macro_std: Dict[str, float] = {}
    for metric_name in METRIC_NAMES:
        values = np.asarray(
            [float(row[metric_name]) for row in per_sample],
            dtype=np.float64,
        )
        macro_metrics[metric_name] = float(values.mean()) if values.size else float("nan")
        macro_std[metric_name] = float(values.std(ddof=1)) if values.size > 1 else 0.0

    # Keep macro means at the top level because these are directly comparable
    # with the existing Evaluator results. Micro/global metrics are retained
    # separately as useful pixel-weighted diagnostics.
    test_metrics: Dict[str, Any] = {
        **macro_metrics,
        "macro_mean": macro_metrics,
        "macro_std": macro_std,
        "micro_global": micro_metrics,
        "threshold": threshold,
        "number_of_samples": sample_counter,
        "global_confusion_counts": {
            "TP": total_tp,
            "FP": total_fp,
            "FN": total_fn,
            "TN": total_tn,
        },
        "evaluation_wall_time_seconds": elapsed,
    }
    return test_metrics, per_sample


def environment_information(device: torch.device) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        info.update(
            {
                "gpu_name": properties.name,
                "gpu_total_memory_gb": properties.total_memory / (1024**3),
                "gpu_compute_capability": f"{properties.major}.{properties.minor}",
            }
        )
    return info


def human_int(number: Optional[int]) -> str:
    return "N/A" if number is None else f"{number:,}"


def human_float(number: Optional[float], digits: int = 4) -> str:
    return "N/A" if number is None else f"{number:.{digits}f}"


def write_text_summary(
    path: Path,
    checkpoint: str,
    parameters: Mapping[str, Any],
    operation_counts: Mapping[str, Any],
    benchmarks: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    transition_rows: Sequence[Mapping[str, Any]],
    environment: Mapping[str, Any],
) -> None:
    lines = [
        "TCSF v3.1 Publication Evaluation",
        "=" * 40,
        f"Checkpoint: {checkpoint}",
        f"Device: {environment.get('device')}",
        f"GPU: {environment.get('gpu_name', 'N/A')}",
        "",
        "Model complexity",
        "-" * 40,
        f"Total parameters: {human_int(parameters.get('total_parameters'))}",
        f"Trainable parameters: {human_int(parameters.get('trainable_parameters'))}",
        f"Model size (MB): {human_float(parameters.get('model_size_mb'), 2)}",
        f"Profiler: {operation_counts.get('profiler') or 'N/A'}",
        f"GMACs: {human_float(operation_counts.get('gmacs'), 3)}",
        f"GFLOPs: {human_float(operation_counts.get('gflops'), 3)}",
    ]
    if operation_counts.get("note"):
        lines.append(f"Profiler note: {operation_counts['note']}")

    lines.extend(["", "Runtime", "-" * 40])
    for benchmark in benchmarks:
        precision = benchmark["precision"]
        batch_size = benchmark["batch_size"]
        lines.extend(
            [
                f"{precision}, batch size {batch_size}:",
                f"  Mean latency/batch (ms): {benchmark['mean_latency_ms_per_batch']:.3f}",
                f"  Mean latency/image (ms): {benchmark['mean_latency_ms_per_image']:.3f}",
                f"  FPS: {benchmark['fps']:.3f}",
                f"  Peak allocated GPU memory (MB): {human_float(benchmark.get('peak_allocated_memory_mb'), 2)}",
                f"  Peak reserved GPU memory (MB): {human_float(benchmark.get('peak_reserved_memory_mb'), 2)}",
            ]
        )

    lines.extend(["", "Test-set metrics (macro mean ± sample SD)", "-" * 40])
    macro_std = metrics.get("macro_std", {})
    for metric_name in METRIC_NAMES:
        lines.append(
            f"{metric_name}: {metrics[metric_name]:.4f} ± "
            f"{float(macro_std.get(metric_name, 0.0)):.4f}"
        )

    micro = metrics.get("micro_global", {})
    if micro:
        lines.extend(["", "Test-set metrics (micro/global)", "-" * 40])
        for metric_name in METRIC_NAMES:
            lines.append(f"{metric_name}: {float(micro[metric_name]):.4f}")

    lines.append(f"Number of samples: {metrics['number_of_samples']}")
    lines.append(f"Threshold: {metrics['threshold']}")

    if transition_rows:
        lines.extend(["", "Transition scales", "-" * 40])
        for row in transition_rows:
            if "transition" in row:
                lines.append(
                    f"Transition {row['transition']} {row['modality']}: {row['gamma']:.6f}"
                )
            else:
                lines.append(f"{row['name']}: {row['gamma']:.6f}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must lie between 0 and 1.")
    if args.qualitative_per_group < 0:
        raise ValueError("--qualitative_per_group cannot be negative.")
    if args.benchmark_batch_size <= 0:
        raise ValueError("--benchmark_batch_size must be greater than zero.")

    save_root = Path(args.save_dir)
    metrics_dir = save_root / "metrics"
    complexity_dir = save_root / "complexity"
    qualitative_dir = save_root / "qualitative"
    logs_dir = save_root / "logs"
    for directory in (metrics_dir, complexity_dir, qualitative_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(logs_dir / "evaluation.log")
    cfg = load_config(args.config)
    seed = args.seed
    if seed is None:
        seed = int(cfg.get("experiment", {}).get("seed", 2026))
    set_seed(seed)

    wants_cuda = str(cfg.get("training", {}).get("device", "cuda")).lower() == "cuda"
    device = torch.device("cuda" if wants_cuda and torch.cuda.is_available() else "cpu")
    if wants_cuda and device.type != "cuda":
        logger.warning("CUDA was requested but is unavailable; using CPU.")

    num_workers = args.num_workers
    if num_workers < 0:
        num_workers = int(cfg["dataset"].get("num_workers", 0))

    logger.info("Loading test dataset and final checkpoint.")
    eval_loader = build_eval_loader(cfg, batch_size=1, num_workers=num_workers)
    model = build_model(cfg).to(device)
    load_model_weights(
        checkpoint_path=args.checkpoint,
        model=model,
        device=device,
    )
    model.eval()

    image_size = int(cfg["dataset"]["image_size"])
    sar_channels = int(cfg["dataset"]["sar_channels"])
    optical_channels = int(cfg["dataset"]["optical_channels"])

    example_sar = torch.randn(
        args.benchmark_batch_size,
        sar_channels,
        image_size,
        image_size,
        device=device,
    )
    example_optical = torch.randn(
        args.benchmark_batch_size,
        optical_channels,
        image_size,
        image_size,
        device=device,
    )

    logger.info("Counting parameters and profiling operations.")
    parameters = parameter_statistics(model)
    operation_counts = profile_flops(
        model,
        example_sar[:1],
        example_optical[:1],
        logger,
    )

    logger.info("Benchmarking FP32 inference at batch size %d.", args.benchmark_batch_size)
    benchmarks: List[Dict[str, Any]] = [
        benchmark_forward(
            model=model,
            sar=example_sar,
            optical=example_optical,
            device=device,
            warmup_iterations=args.warmup_iterations,
            timed_iterations=args.timed_iterations,
            use_amp=False,
        )
    ]

    throughput_batch_size = args.throughput_batch_size
    if throughput_batch_size <= 0:
        throughput_batch_size = int(cfg["training"].get("batch_size", 1))
    if throughput_batch_size != args.benchmark_batch_size:
        throughput_sar = torch.randn(
            throughput_batch_size,
            sar_channels,
            image_size,
            image_size,
            device=device,
        )
        throughput_optical = torch.randn(
            throughput_batch_size,
            optical_channels,
            image_size,
            image_size,
            device=device,
        )
        logger.info("Benchmarking FP32 throughput at batch size %d.", throughput_batch_size)
        benchmarks.append(
            benchmark_forward(
                model=model,
                sar=throughput_sar,
                optical=throughput_optical,
                device=device,
                warmup_iterations=args.warmup_iterations,
                timed_iterations=args.timed_iterations,
                use_amp=False,
            )
        )
        del throughput_sar, throughput_optical

    if args.amp:
        if device.type == "cuda":
            logger.info("Benchmarking AMP inference at batch size %d.", args.benchmark_batch_size)
            benchmarks.append(
                benchmark_forward(
                    model=model,
                    sar=example_sar,
                    optical=example_optical,
                    device=device,
                    warmup_iterations=args.warmup_iterations,
                    timed_iterations=args.timed_iterations,
                    use_amp=True,
                )
            )
        else:
            logger.warning("--amp was requested, but AMP benchmarking requires CUDA.")

    del example_sar, example_optical
    if device.type == "cuda":
        torch.cuda.empty_cache()

    logger.info("Running full test-set evaluation.")
    test_metrics, per_sample = evaluate_test_set(
        model=model,
        loader=eval_loader,
        device=device,
        threshold=args.threshold,
        save_root=save_root,
        save_all_maps=args.save_all_maps,
        logger=logger,
    )

    # Rank index must reflect sorted position for correct median selection.
    ordered_indices = sorted(range(len(per_sample)), key=lambda i: per_sample[i]["IoU"])
    for sorted_position, original_index in enumerate(ordered_indices):
        per_sample[original_index]["rank_index"] = sorted_position

    logger.info("Selecting best, median, and worst qualitative examples.")
    selected = select_representative_samples(
        per_sample,
        args.qualitative_per_group,
    )
    selected_rows: List[Dict[str, Any]] = []
    for group_name, rows in selected.items():
        for group_rank, row in enumerate(rows, start=1):
            sample = torch.load(row["cache_path"], map_location="cpu")
            figure_path = (
                qualitative_dir
                / group_name
                / f"{group_rank:02d}_{row['sample_id']}.png"
            )
            save_paper_figure(
                sample=sample,
                save_path=figure_path,
                rgb_indices=args.rgb_indices,
                sar_channel=args.sar_channel,
                threshold=args.threshold,
            )

            probability = torch.sigmoid(sample["logits"]).float().cpu().numpy()
            while probability.ndim > 2:
                probability = probability[0]
            binary = (probability >= args.threshold).astype(np.float32)
            save_probability_png(
                probability,
                qualitative_dir / group_name / "maps" / f"{row['sample_id']}_probability.png",
            )
            save_grayscale_png(
                binary,
                qualitative_dir / group_name / "maps" / f"{row['sample_id']}_binary.png",
            )
            selected_rows.append(
                {
                    "group": group_name,
                    "group_rank": group_rank,
                    "sample_id": row["sample_id"],
                    **{metric: row[metric] for metric in METRIC_NAMES},
                    "figure_path": str(figure_path),
                }
            )

    transition_rows = get_transition_rows(model)
    environment = environment_information(device)

    public_per_sample = [
        {
            key: value
            for key, value in row.items()
            if key not in {"cache_path", "rank_index"}
        }
        for row in per_sample
    ]

    write_json(parameters, complexity_dir / "parameter_counts.json")
    write_json(operation_counts, complexity_dir / "operation_counts.json")
    write_json({"benchmarks": benchmarks}, complexity_dir / "runtime_benchmarks.json")
    write_json(test_metrics, metrics_dir / "test_metrics.json")
    write_json(environment, complexity_dir / "environment.json")
    write_csv(public_per_sample, metrics_dir / "per_image_metrics.csv")
    write_csv(transition_rows, metrics_dir / "transition_scales.csv")
    write_csv(selected_rows, metrics_dir / "selected_qualitative_samples.csv")

    combined_summary = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "seed": seed,
        "parameters": parameters,
        "operation_counts": operation_counts,
        "runtime_benchmarks": benchmarks,
        "test_metrics": test_metrics,
        "transition_scales": transition_rows,
        "environment": environment,
    }
    write_json(combined_summary, save_root / "publication_summary.json")
    write_text_summary(
        path=save_root / "summary.txt",
        checkpoint=args.checkpoint,
        parameters=parameters,
        operation_counts=operation_counts,
        benchmarks=benchmarks,
        metrics=test_metrics,
        transition_rows=transition_rows,
        environment=environment,
    )

    logger.info("Publication evaluation completed.")
    logger.info("Macro mean IoU: %.4f", test_metrics["IoU"])
    logger.info("Macro mean Dice: %.4f", test_metrics["Dice"])
    logger.info(
        "Micro/global IoU: %.4f",
        test_metrics["micro_global"]["IoU"],
    )
    logger.info(
        "Micro/global Dice: %.4f",
        test_metrics["micro_global"]["Dice"],
    )
    logger.info("Summary: %s", save_root / "summary.txt")
    logger.info("Machine-readable summary: %s", save_root / "publication_summary.json")


if __name__ == "__main__":
    main()