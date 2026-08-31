"""
visualization.py

Clean paper-style visualization utilities for ACSF / TCSF.

The main comparison figure follows this layout:

SAR | Optical RGB | Ground Truth | Final Prediction |
SAR Prediction | Optical Prediction | SAR Reliability | Optical Reliability

Predictions use one shared flood-probability color scale.
Reliability uses one shared grayscale color scale.

Important:
The Sentinel-2 RGB visualization uses a shared reflectance scale across
red, green, and blue channels. It does not independently stretch each
channel, because independent stretching often produces unrealistic
magenta or purple optical images.
"""

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_numpy(value: ArrayLike) -> np.ndarray:
    """Convert a tensor or array to NumPy."""
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()

    return np.asarray(value)


def _remove_single_batch(array: np.ndarray) -> np.ndarray:
    """Remove a leading batch dimension when batch size is one."""
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]

    return array


def tensor_to_numpy(tensor: ArrayLike) -> np.ndarray:
    """
    Convert [B,C,H,W], [C,H,W], [B,H,W], or [H,W] to a 2-D array.

    For a multi-channel tensor, the first channel is selected.
    """
    array = _remove_single_batch(_to_numpy(tensor))

    if array.ndim == 3:
        if array.shape[0] == 1:
            array = array[0]
        elif array.shape[-1] == 1:
            array = array[..., 0]
        else:
            array = array[0]

    return np.squeeze(array)


def probability_from_tensor(tensor: ArrayLike) -> np.ndarray:
    """
    Convert logits or probabilities to a probability map in [0, 1].
    """
    array = tensor_to_numpy(tensor).astype(np.float32)

    if np.nanmin(array) < 0.0 or np.nanmax(array) > 1.0:
        array = 1.0 / (1.0 + np.exp(-np.clip(array, -30.0, 30.0)))

    return np.clip(array, 0.0, 1.0)


def percentile_stretch(
    image: np.ndarray,
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
    gamma: float = 0.85,
) -> np.ndarray:
    """
    Apply contrast stretching for grayscale display.

    This is used for SAR imagery only. Optical RGB uses a shared physical
    reflectance scale instead of independent per-channel stretching.
    """
    image = np.asarray(image, dtype=np.float32)

    if image.ndim == 2:
        image = image[..., None]

    output = np.zeros_like(image, dtype=np.float32)

    for channel in range(image.shape[-1]):
        band = image[..., channel]
        valid = np.isfinite(band)

        if not np.any(valid):
            continue

        low = np.percentile(band[valid], lower_percentile)
        high = np.percentile(band[valid], upper_percentile)

        if high <= low:
            high = low + 1e-6

        stretched = (band - low) / (high - low)
        output[..., channel] = np.clip(stretched, 0.0, 1.0)

    output = np.power(output, gamma)

    if output.shape[-1] == 1:
        output = output[..., 0]

    return np.clip(output, 0.0, 1.0)


def prepare_sar_image(
    sar: ArrayLike,
    channel: int = 0,
) -> np.ndarray:
    """Prepare one SAR channel as a readable grayscale image."""
    array = _remove_single_batch(_to_numpy(sar))

    if array.ndim == 3:
        if channel >= array.shape[0]:
            raise ValueError(
                f"SAR channel {channel} is invalid for input shape {array.shape}."
            )
        array = array[channel]

    return percentile_stretch(np.squeeze(array))


def prepare_optical_rgb(
    optical: ArrayLike,
    rgb_indices: Sequence[int] = (3, 2, 1),
    reflectance_max: float = 0.30,
    gamma: float = 0.85,
) -> np.ndarray:
    """
    Prepare Sentinel-2 true-color RGB using a shared reflectance scale.

    Default channel order:
        B1, B2, B3, B4, ..., B12

    True-color indices:
        Red   = B4 -> index 3
        Green = B3 -> index 2
        Blue  = B2 -> index 1

    If the loader begins at B2 instead of B1, use:
        rgb_indices=(2, 1, 0)

    Notes:
    - All RGB channels use the same reflectance range.
    - Raw Sentinel-2 digital numbers are divided by 10000 automatically.
    - This function assumes the optical tensor contains reflectance-like
      values. If the dataset applies mean/std normalization, denormalize
      before calling this function.
    """
    array = _remove_single_batch(_to_numpy(optical)).astype(np.float32)

    if array.ndim != 3:
        raise ValueError(
            f"Expected optical data with shape [C,H,W] or [H,W,C], "
            f"got {array.shape}."
        )

    if array.shape[0] <= 20:
        channels_first = array
    elif array.shape[-1] <= 20:
        channels_first = np.transpose(array, (2, 0, 1))
    else:
        raise ValueError(
            f"Could not determine the optical channel axis for shape {array.shape}."
        )

    if len(rgb_indices) != 3:
        raise ValueError("rgb_indices must contain exactly three values.")

    if min(rgb_indices) < 0 or max(rgb_indices) >= channels_first.shape[0]:
        raise ValueError(
            f"RGB indices {tuple(rgb_indices)} are invalid for "
            f"{channels_first.shape[0]} optical channels."
        )

    if reflectance_max <= 0:
        raise ValueError("reflectance_max must be greater than zero.")

    rgb = channels_first[list(rgb_indices)]
    rgb = np.transpose(rgb, (1, 2, 0))

    rgb = np.nan_to_num(
        rgb,
        nan=0.0,
        posinf=reflectance_max,
        neginf=0.0,
    )

    # Raw Sentinel-2 surface reflectance is commonly stored as 0-10000.
    if np.nanmax(rgb) > 2.0:
        rgb = rgb / 10000.0

    # Preserve the relative balance between RGB bands.
    rgb = np.clip(rgb, 0.0, reflectance_max)
    rgb = rgb / reflectance_max

    # Gamma correction brightens shadows without changing band ordering.
    rgb = np.power(rgb, gamma)

    return np.clip(rgb, 0.0, 1.0)


def prepare_ground_truth(mask: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return a binary ground-truth mask and valid-pixel mask.
    """
    array = tensor_to_numpy(mask).astype(np.float32)
    valid = np.isfinite(array) & (array >= 0)

    if np.nanmax(array) > 1.0:
        flood = array > 127.0
    else:
        flood = array > 0.5

    flood &= valid

    return flood.astype(np.float32), valid


def calculate_sample_metrics(
    probability: np.ndarray,
    ground_truth: np.ndarray,
    valid_mask: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Calculate sample-level segmentation metrics."""
    prediction = probability >= threshold

    pred = prediction[valid_mask]
    target = ground_truth.astype(bool)[valid_mask]

    true_positive = np.logical_and(pred, target).sum()
    false_positive = np.logical_and(pred, ~target).sum()
    false_negative = np.logical_and(~pred, target).sum()

    iou = true_positive / (
        true_positive + false_positive + false_negative + 1e-8
    )
    dice = (2.0 * true_positive) / (
        2.0 * true_positive + false_positive + false_negative + 1e-8
    )
    precision = true_positive / (
        true_positive + false_positive + 1e-8
    )
    recall = true_positive / (
        true_positive + false_negative + 1e-8
    )

    return {
        "IoU": float(iou),
        "Dice": float(dice),
        "Precision": float(precision),
        "Recall": float(recall),
    }


def save_prediction_map(
    tensor: ArrayLike,
    save_path: Union[str, Path],
    title: Optional[str] = None,
    cmap: str = "viridis",
    add_colorbar: bool = True,
) -> None:
    """Save an individual probability or reliability map."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    array = probability_from_tensor(tensor)

    fig, ax = plt.subplots(figsize=(6, 6))
    image = ax.imshow(
        array,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )

    ax.set_title(title or "")
    ax.axis("off")

    if add_colorbar:
        colorbar = fig.colorbar(
            image,
            ax=ax,
            fraction=0.046,
            pad=0.04,
        )
        colorbar.set_ticks([0.0, 0.5, 1.0])

    fig.tight_layout()
    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def save_decision_weights_figure(
    weights: ArrayLike,
    save_path: Union[str, Path],
) -> None:
    """Save SAR, optical, and fusion decision weights."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    array = _remove_single_batch(_to_numpy(weights))

    if array.ndim != 3 or array.shape[0] != 3:
        raise ValueError(
            f"Expected decision weights [3,H,W], got {array.shape}."
        )

    titles = (
        "SAR Weight",
        "Optical Weight",
        "Fusion Weight",
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))

    for index, axis in enumerate(axes):
        axis.imshow(
            np.clip(array[index], 0.0, 1.0),
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        axis.set_title(titles[index], fontsize=12)
        axis.axis("off")

    probability_scalar = ScalarMappable(
        norm=Normalize(vmin=0.0, vmax=1.0),
        cmap="viridis",
    )
    probability_scalar.set_array([])

    colorbar = fig.colorbar(
        probability_scalar,
        ax=axes,
        orientation="horizontal",
        fraction=0.06,
        pad=0.08,
        aspect=45,
    )
    colorbar.set_label("Decision Weight (Low → High)")
    colorbar.set_ticks([0.0, 0.5, 1.0])

    fig.subplots_adjust(
        left=0.02,
        right=0.98,
        top=0.90,
        bottom=0.20,
        wspace=0.06,
    )
    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def save_comparison_figure(
    sar: ArrayLike,
    optical: ArrayLike,
    mask: ArrayLike,
    predictions: Dict[str, ArrayLike],
    save_path: Union[str, Path],
    sample_name: str = "Sample",
    rgb_indices: Sequence[int] = (3, 2, 1),
    threshold: float = 0.5,
    sar_channel: int = 0,
    reflectance_max: float = 0.30,
    optical_gamma: float = 0.85,
) -> Dict[str, float]:
    """
    Save a clean eight-panel qualitative comparison.

    Layout:
        SAR
        Optical RGB
        Ground Truth Flood Mask
        Final Prediction
        SAR-Only Prediction
        Optical-Only Prediction
        SAR Reliability
        Optical Reliability
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    required_keys = (
        "final_prediction",
        "sar_prediction",
        "optical_prediction",
        "sar_reliability",
        "optical_reliability",
    )

    missing_keys = [
        key for key in required_keys
        if key not in predictions or predictions[key] is None
    ]

    if missing_keys:
        raise KeyError(
            "Missing required prediction outputs: "
            + ", ".join(missing_keys)
        )

    sar_image = prepare_sar_image(
        sar,
        channel=sar_channel,
    )

    optical_rgb = prepare_optical_rgb(
        optical,
        rgb_indices=rgb_indices,
        reflectance_max=reflectance_max,
        gamma=optical_gamma,
    )

    ground_truth, valid_mask = prepare_ground_truth(mask)

    final_probability = probability_from_tensor(
        predictions["final_prediction"]
    )
    sar_probability = probability_from_tensor(
        predictions["sar_prediction"]
    )
    optical_probability = probability_from_tensor(
        predictions["optical_prediction"]
    )
    sar_reliability = probability_from_tensor(
        predictions["sar_reliability"]
    )
    optical_reliability = probability_from_tensor(
        predictions["optical_reliability"]
    )

    metrics = calculate_sample_metrics(
        final_probability,
        ground_truth,
        valid_mask,
        threshold=threshold,
    )

    panels = (
        (
            sar_image,
            "SAR (VV)",
            "gray",
            None,
        ),
        (
            optical_rgb,
            "Optical RGB\n(Sentinel-2)",
            None,
            None,
        ),
        (
            ground_truth,
            "Ground Truth\nFlood Mask",
            "gray",
            (0.0, 1.0),
        ),
        (
            final_probability,
            "Final Prediction\n(TCSF v3.1)",
            "viridis",
            (0.0, 1.0),
        ),
        (
            sar_probability,
            "SAR-Only\nPrediction",
            "viridis",
            (0.0, 1.0),
        ),
        (
            optical_probability,
            "Optical-Only\nPrediction",
            "viridis",
            (0.0, 1.0),
        ),
        (
            sar_reliability,
            "SAR Reliability\n(High = Bright)",
            "gray",
            (0.0, 1.0),
        ),
        (
            optical_reliability,
            "Optical Reliability\n(High = Bright)",
            "gray",
            (0.0, 1.0),
        ),
    )

    fig, axes = plt.subplots(
        1,
        7,
        figsize=(21, 4.7),
    )

    for axis, (image, title, cmap, limits) in zip(axes, panels):
        if cmap is None:
            axis.imshow(
                image,
                interpolation="nearest",
            )
        elif limits is None:
            axis.imshow(
                image,
                cmap=cmap,
                interpolation="nearest",
            )
        else:
            axis.imshow(
                image,
                cmap=cmap,
                vmin=limits[0],
                vmax=limits[1],
                interpolation="nearest",
            )

        axis.set_title(
            title,
            fontsize=11,
            fontweight="semibold",
            pad=8,
        )
        axis.axis("off")

    metric_text = (
        f"IoU {metrics['IoU']:.4f}   |   "
        f"Dice {metrics['Dice']:.4f}   |   "
        f"Precision {metrics['Precision']:.4f}   |   "
        f"Recall {metrics['Recall']:.4f}"
    )

    fig.suptitle(
        f"{sample_name}\n{metric_text}",
        fontsize=12,
        y=0.99,
    )

    probability_scalar = ScalarMappable(
        norm=Normalize(vmin=0.0, vmax=1.0),
        cmap="viridis",
    )
    probability_scalar.set_array([])

    # Use dedicated colorbar axes so the two bars cannot overlap.
    # Coordinates are [left, bottom, width, height] in figure space.
    probability_cax = fig.add_axes([0.46, 0.10, 0.25, 0.035])
    probability_colorbar = fig.colorbar(
        probability_scalar,
        cax=probability_cax,
        orientation="horizontal",
    )
    probability_colorbar.set_label(
        "Flood Probability (Low → High)",
        fontsize=10,
        labelpad=5,
    )
    probability_colorbar.set_ticks([0.0, 0.5, 1.0])

    reliability_scalar = ScalarMappable(
        norm=Normalize(vmin=0.0, vmax=1.0),
        cmap="gray",
    )
    reliability_scalar.set_array([])

    reliability_cax = fig.add_axes([0.75, 0.10, 0.16, 0.035])
    reliability_colorbar = fig.colorbar(
        reliability_scalar,
        cax=reliability_cax,
        orientation="horizontal",
    )
    reliability_colorbar.set_label(
        "Reliability (Low → High)",
        fontsize=10,
        labelpad=5,
    )
    reliability_colorbar.set_ticks([0.0, 0.5, 1.0])

    fig.subplots_adjust(
        left=0.01,
        right=0.995,
        top=0.80,
        bottom=0.24,
        wspace=0.055,
    )

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    return metrics