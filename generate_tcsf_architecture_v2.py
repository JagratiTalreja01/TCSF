#!/usr/bin/env python3
"""
generate_tcsf_architecture_v2.py

Publication-ready architecture generator for TCSF v3.1.

Key improvements
----------------
- Clean left-to-right stage layout
- Orthogonal arrows with minimal crossings
- Strong teal / turquoise / cyan palette
- Dark blue, dark red, and deep purple accents
- Optional real SAR, RGB, and prediction images
- Separate Cross-State Fusion inset
- Vector PDF and SVG output
- High-resolution PNG output

Example
-------
python generate_tcsf_architecture_v2.py

With real images:
python generate_tcsf_architecture_v2.py \
    --sar-image path/to/sar.png \
    --optical-image path/to/rgb.png \
    --mask-image path/to/mask.png

Outputs
-------
outputs/publication/architecture_v2/
    tcsf_v31_overall_architecture.pdf
    tcsf_v31_overall_architecture.svg
    tcsf_v31_overall_architecture.png
    tcsf_v31_csf_block.pdf
    tcsf_v31_csf_block.svg
    tcsf_v31_csf_block.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
)


# ---------------------------------------------------------------------
# COLOR PALETTE
# ---------------------------------------------------------------------
COLORS = {
    "navy": "#0B1F3A",
    "dark_blue": "#123B6D",
    "blue": "#1D5FA7",
    "cyan": "#19B5D8",
    "bright_cyan": "#2CD7F2",
    "teal": "#0B7A75",
    "deep_teal": "#075E59",
    "turquoise": "#16A6A1",
    "mint": "#D6F5F1",
    "pale_cyan": "#DDF7FB",
    "light_teal": "#D8F0EE",
    "purple": "#5A2A83",
    "bright_purple": "#7A3DB8",
    "lavender": "#EADCF8",
    "dark_red": "#8E1B2D",
    "red": "#B8324A",
    "rose": "#F8DDE3",
    "orange": "#D86A1A",
    "gold": "#D7A51B",
    "soft_gold": "#F9EDC9",
    "gray": "#5F6B73",
    "light_gray": "#EEF2F4",
    "line_gray": "#AAB4BA",
    "white": "#FFFFFF",
    "black": "#101820",
    "green": "#2F7D4C",
    "light_green": "#DFF2E5",
}


# ---------------------------------------------------------------------
# BASIC DRAWING HELPERS
# ---------------------------------------------------------------------
def rounded_box(
    ax,
    xy: Tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str,
    linewidth: float = 1.8,
    fontsize: float = 10,
    fontweight: str = "normal",
    textcolor: str = COLORS["black"],
    radius: float = 0.03,
    zorder: int = 3,
):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        color=textcolor,
        linespacing=1.12,
        zorder=zorder + 1,
    )
    return patch


def stage_panel(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    *,
    edgecolor: str,
    title_fill: str,
):
    panel = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.03",
        facecolor=COLORS["white"],
        edgecolor=edgecolor,
        linewidth=1.5,
        zorder=0,
    )
    ax.add_patch(panel)

    header_h = 0.42
    header = FancyBboxPatch(
        (x, y + h - header_h),
        w,
        header_h,
        boxstyle="round,pad=0.01,rounding_size=0.03",
        facecolor=title_fill,
        edgecolor=edgecolor,
        linewidth=1.5,
        zorder=1,
    )
    ax.add_patch(header)

    # Hide rounded lower header corners for a cleaner title bar.
    ax.add_patch(
        Rectangle(
            (x, y + h - header_h),
            w,
            header_h * 0.55,
            facecolor=title_fill,
            edgecolor="none",
            zorder=1,
        )
    )

    ax.text(
        x + w / 2,
        y + h - header_h / 2,
        title,
        ha="center",
        va="center",
        fontsize=10.8,
        fontweight="bold",
        color=COLORS["navy"],
        zorder=4,
    )


def orthogonal_arrow(
    ax,
    start: Tuple[float, float],
    end: Tuple[float, float],
    *,
    color: str,
    linewidth: float = 1.8,
    linestyle: str = "-",
    mutation_scale: float = 12,
    via_x: Optional[float] = None,
    via_y: Optional[float] = None,
    zorder: int = 2,
):
    """
    Draw an orthogonal arrow with optional routing through one x or y coordinate.
    """
    x1, y1 = start
    x2, y2 = end

    points = [(x1, y1)]

    if via_x is not None:
        points.extend([(via_x, y1), (via_x, y2)])
    elif via_y is not None:
        points.extend([(x1, via_y), (x2, via_y)])

    points.append((x2, y2))

    for i in range(len(points) - 2):
        ax.plot(
            [points[i][0], points[i + 1][0]],
            [points[i][1], points[i + 1][1]],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            solid_capstyle="round",
            zorder=zorder,
        )

    arrow = FancyArrowPatch(
        points[-2],
        points[-1],
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
        zorder=zorder,
    )
    ax.add_patch(arrow)


def feature_cube(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    depth: float,
    *,
    front_color: str,
    side_color: str,
    top_color: str,
    edgecolor: str,
    label: Optional[str] = None,
    fontsize: float = 8.5,
):
    front = Rectangle(
        (x, y),
        w,
        h,
        facecolor=front_color,
        edgecolor=edgecolor,
        linewidth=1.4,
        zorder=4,
    )
    ax.add_patch(front)

    top = Polygon(
        [
            (x, y + h),
            (x + depth, y + h + depth),
            (x + w + depth, y + h + depth),
            (x + w, y + h),
        ],
        closed=True,
        facecolor=top_color,
        edgecolor=edgecolor,
        linewidth=1.2,
        zorder=3,
    )
    ax.add_patch(top)

    side = Polygon(
        [
            (x + w, y),
            (x + w + depth, y + depth),
            (x + w + depth, y + h + depth),
            (x + w, y + h),
        ],
        closed=True,
        facecolor=side_color,
        edgecolor=edgecolor,
        linewidth=1.2,
        zorder=3,
    )
    ax.add_patch(side)

    # subtle internal grid
    for frac in (0.33, 0.66):
        ax.plot(
            [x + frac * w, x + frac * w],
            [y, y + h],
            color=edgecolor,
            linewidth=0.45,
            alpha=0.6,
            zorder=5,
        )
        ax.plot(
            [x, x + w],
            [y + frac * h, y + frac * h],
            color=edgecolor,
            linewidth=0.45,
            alpha=0.6,
            zorder=5,
        )

    if label:
        ax.text(
            x + w / 2,
            y - 0.18,
            label,
            ha="center",
            va="top",
            fontsize=fontsize,
            color=COLORS["navy"],
            fontweight="bold",
            zorder=6,
        )


def load_image_or_placeholder(
    path: Optional[str],
    *,
    kind: str,
    size: Tuple[int, int] = (256, 256),
) -> np.ndarray:
    if path:
        p = Path(path)
        if p.exists():
            image = mpimg.imread(p)
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            if image.shape[-1] == 4:
                image = image[..., :3]
            return image

    h, w = size

    if kind == "sar":
        rng = np.random.default_rng(10)
        base = rng.normal(0.45, 0.14, (h, w))
        yy, xx = np.mgrid[:h, :w]
        river = np.exp(-((xx - 0.55 * w - 0.15 * yy) ** 2) / (2 * (0.06 * w) ** 2))
        base -= 0.25 * river
        base += 0.08 * np.sin(xx / 7.0) * np.sin(yy / 13.0)
        base = np.clip(base, 0, 1)
        return np.stack([base] * 3, axis=-1)

    if kind == "optical":
        yy, xx = np.mgrid[:h, :w]
        green = 0.35 + 0.25 * np.sin(xx / 30.0) * np.cos(yy / 23.0)
        red = 0.22 + 0.10 * np.cos(xx / 25.0)
        blue = 0.18 + 0.08 * np.sin(yy / 18.0)
        river = np.exp(-((xx - 0.55 * w - 0.15 * yy) ** 2) / (2 * (0.06 * w) ** 2))
        rgb = np.stack([red, green, blue], axis=-1)
        rgb[..., 2] += 0.45 * river
        rgb[..., 1] -= 0.15 * river
        return np.clip(rgb, 0, 1)

    # mask
    yy, xx = np.mgrid[:h, :w]
    river = np.exp(-((xx - 0.55 * w - 0.15 * yy) ** 2) / (2 * (0.06 * w) ** 2))
    mask = (river > 0.35).astype(float)
    return np.stack([mask] * 3, axis=-1)


def image_box(
    ax,
    image: np.ndarray,
    extent: Tuple[float, float, float, float],
    *,
    edgecolor: str,
    title: str,
    subtitle: str,
):
    x1, x2, y1, y2 = extent
    ax.imshow(image, extent=extent, interpolation="nearest", zorder=3)
    ax.add_patch(
        Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor=edgecolor,
            linewidth=1.8,
            zorder=4,
        )
    )
    ax.text(
        (x1 + x2) / 2,
        y2 + 0.18,
        title,
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=edgecolor,
    )
    ax.text(
        (x1 + x2) / 2,
        y1 - 0.16,
        subtitle,
        ha="center",
        va="top",
        fontsize=8.5,
        color=COLORS["gray"],
    )


# ---------------------------------------------------------------------
# OVERALL ARCHITECTURE
# ---------------------------------------------------------------------
def draw_overall_architecture(
    output_dir: Path,
    dpi: int,
    transparent: bool,
    sar_image_path: Optional[str],
    optical_image_path: Optional[str],
    mask_image_path: Optional[str],
):
    fig, ax = plt.subplots(figsize=(22, 11.5))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 11.5)
    ax.axis("off")

    ax.text(
        11,
        11.15,
        "Tri-Level Cross-State Fusion Network (TCSF v3.1)",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color=COLORS["navy"],
    )
    ax.text(
        11,
        10.78,
        "Reliability-aware multimodal flood segmentation using Sentinel-1 SAR and Sentinel-2 optical imagery",
        ha="center",
        va="center",
        fontsize=11.2,
        color=COLORS["gray"],
    )

    # Stage panels
    stage_panel(
        ax, 0.25, 1.15, 2.35, 8.95, "1  INPUTS",
        edgecolor=COLORS["dark_blue"],
        title_fill=COLORS["pale_cyan"],
    )
    stage_panel(
        ax, 2.85, 1.15, 3.45, 8.95, "2  RELIABILITY & FUSION",
        edgecolor=COLORS["teal"],
        title_fill=COLORS["mint"],
    )
    stage_panel(
        ax, 6.55, 1.15, 8.80, 8.95, "3  DUAL ENCODERS & CROSS-STATE FUSION",
        edgecolor=COLORS["purple"],
        title_fill=COLORS["lavender"],
    )
    stage_panel(
        ax, 15.60, 1.15, 2.65, 8.95, "4  MULTI-SCALE DECODER",
        edgecolor=COLORS["deep_teal"],
        title_fill=COLORS["light_teal"],
    )
    stage_panel(
        ax, 18.50, 1.15, 3.20, 8.95, "5  OUTPUTS",
        edgecolor=COLORS["dark_red"],
        title_fill=COLORS["rose"],
    )

    sar_img = load_image_or_placeholder(sar_image_path, kind="sar")
    opt_img = load_image_or_placeholder(optical_image_path, kind="optical")
    mask_img = load_image_or_placeholder(mask_image_path, kind="mask")

    image_box(
        ax, sar_img, (0.63, 2.22, 6.32, 8.20),
        edgecolor=COLORS["dark_blue"],
        title="Sentinel-1 SAR",
        subtitle="VV + VH",
    )
    image_box(
        ax, opt_img, (0.63, 2.22, 2.63, 4.51),
        edgecolor=COLORS["teal"],
        title="Sentinel-2 Optical",
        subtitle="13 spectral bands",
    )

    # Reliability estimators
    rounded_box(
        ax, (3.12, 6.40), 2.88, 1.55,
        "SAR Reliability Estimator\n\nReliability map  $R_{sar}$",
        facecolor=COLORS["soft_gold"],
        edgecolor=COLORS["gold"],
        fontsize=10.2,
        fontweight="bold",
    )
    rounded_box(
        ax, (3.12, 4.22), 2.88, 1.55,
        "Optical Reliability Estimator\n\nReliability map  $R_{opt}$",
        facecolor=COLORS["mint"],
        edgecolor=COLORS["teal"],
        fontsize=10.2,
        fontweight="bold",
    )
    rounded_box(
        ax, (3.12, 2.05), 2.88, 1.38,
        "Adaptive Data Fusion (ADF)\nInitial fused context",
        facecolor=COLORS["lavender"],
        edgecolor=COLORS["purple"],
        fontsize=10.2,
        fontweight="bold",
    )

    orthogonal_arrow(
        ax, (2.22, 7.26), (3.12, 7.26),
        color=COLORS["dark_blue"]
    )
    orthogonal_arrow(
        ax, (2.22, 3.57), (3.12, 4.99),
        color=COLORS["teal"],
        via_x=2.72,
    )
    orthogonal_arrow(
        ax, (2.22, 3.57), (3.12, 2.73),
        color=COLORS["teal"],
        via_x=2.72,
    )

    orthogonal_arrow(
        ax, (4.56, 6.40), (4.56, 3.43),
        color=COLORS["orange"],
        linewidth=1.5,
        linestyle="--",
    )
    orthogonal_arrow(
        ax, (4.56, 4.22), (4.56, 3.43),
        color=COLORS["orange"],
        linewidth=1.5,
        linestyle="--",
    )

    # Encoder labels
    rounded_box(
        ax, (6.92, 8.93), 1.95, 0.68,
        "SAR Encoder\nVision Mamba",
        facecolor=COLORS["pale_cyan"],
        edgecolor=COLORS["dark_blue"],
        fontsize=9.6,
        fontweight="bold",
    )
    rounded_box(
        ax, (9.40, 8.93), 1.95, 0.68,
        "Optical Encoder\nVision Mamba",
        facecolor=COLORS["mint"],
        edgecolor=COLORS["teal"],
        fontsize=9.6,
        fontweight="bold",
    )
    rounded_box(
        ax, (11.92, 8.93), 2.18, 0.68,
        "Cross-State Fusion\nat each level",
        facecolor=COLORS["lavender"],
        edgecolor=COLORS["purple"],
        fontsize=9.6,
        fontweight="bold",
    )

    # Four aligned scales
    ys = [7.62, 5.93, 4.24, 2.55]
    scales = ["L1   1/4", "L2   1/8", "L3   1/16", "L4   1/32"]

    for i, (y, label) in enumerate(zip(ys, scales)):
        ax.text(
            6.76,
            y + 0.38,
            label,
            ha="right",
            va="center",
            fontsize=9.3,
            color=COLORS["navy"],
            fontweight="bold",
        )

        cube_scale = 1.0 - 0.08 * i

        feature_cube(
            ax, 7.10, y, 0.72 * cube_scale, 0.72 * cube_scale, 0.14,
            front_color=COLORS["cyan"],
            side_color=COLORS["blue"],
            top_color=COLORS["bright_cyan"],
            edgecolor=COLORS["dark_blue"],
        )
        feature_cube(
            ax, 9.62, y, 0.72 * cube_scale, 0.72 * cube_scale, 0.14,
            front_color=COLORS["turquoise"],
            side_color=COLORS["teal"],
            top_color=COLORS["mint"],
            edgecolor=COLORS["deep_teal"],
        )

        rounded_box(
            ax, (12.05, y - 0.02), 1.72, 0.78,
            f"CSF Block\nLevel {i + 1}",
            facecolor=COLORS["lavender"],
            edgecolor=COLORS["purple"],
            fontsize=9.0,
            fontweight="bold",
        )

        feature_cube(
            ax, 14.20, y + 0.07, 0.55 * cube_scale, 0.55 * cube_scale, 0.11,
            front_color="#8F63C7",
            side_color=COLORS["purple"],
            top_color="#B999DE",
            edgecolor=COLORS["purple"],
        )

        orthogonal_arrow(
            ax, (7.98, y + 0.36), (9.62, y + 0.36),
            color=COLORS["gray"],
            linewidth=1.4,
        )
        orthogonal_arrow(
            ax, (10.48, y + 0.36), (12.05, y + 0.36),
            color=COLORS["teal"],
            linewidth=1.6,
        )
        orthogonal_arrow(
            ax, (13.77, y + 0.36), (14.20, y + 0.36),
            color=COLORS["purple"],
            linewidth=1.7,
        )

    # Encoder hierarchy arrows
    for x, color in [(7.46, COLORS["dark_blue"]), (9.98, COLORS["teal"])]:
        for i in range(3):
            orthogonal_arrow(
                ax,
                (x, ys[i]),
                (x, ys[i + 1] + 0.72),
                color=color,
                linewidth=1.7,
            )

    # Controlled residual transitions, cleanly positioned
    for i in range(3):
        y_mid = (ys[i] + ys[i + 1]) / 2 + 0.05
        rounded_box(
            ax, (13.78, y_mid - 0.28), 1.18, 0.56,
            f"T{i + 1}\n$x_{{l+1}}+\\gamma_lT(x_l)$",
            facecolor="#FBE4D5",
            edgecolor=COLORS["orange"],
            fontsize=7.7,
            fontweight="bold",
        )
        orthogonal_arrow(
            ax,
            (14.54, ys[i] + 0.05),
            (14.54, ys[i + 1] + 0.68),
            color=COLORS["orange"],
            linewidth=1.6,
        )

    ax.text(
        14.34,
        8.72,
        "Controlled residual\ncross-scale transitions",
        ha="center",
        va="center",
        fontsize=8.6,
        color=COLORS["orange"],
        fontweight="bold",
    )

    # Inputs to encoders, routed with no crossing
    orthogonal_arrow(
        ax, (5.98, 7.17), (7.10, 7.98),
        color=COLORS["dark_blue"],
        via_x=6.55,
    )
    orthogonal_arrow(
        ax, (5.98, 4.99), (9.62, 7.98),
        color=COLORS["teal"],
        via_x=6.35,
    )
    orthogonal_arrow(
        ax, (5.98, 2.73), (12.05, 7.98),
        color=COLORS["purple"],
        linestyle="--",
        via_x=6.18,
    )

    # Reliability guidance arrows only to CSF column
    for y in ys:
        orthogonal_arrow(
            ax,
            (5.98, 7.17),
            (12.05, y + 0.39),
            color=COLORS["orange"],
            linewidth=1.0,
            linestyle="--",
            via_x=11.65,
        )

    # Decoder blocks
    decoder_ys = [7.58, 5.90, 4.22, 2.54]
    for i, y in enumerate(decoder_ys):
        feature_cube(
            ax, 16.30, y, 0.85, 0.72, 0.18,
            front_color=COLORS["turquoise"],
            side_color=COLORS["deep_teal"],
            top_color=COLORS["mint"],
            edgecolor=COLORS["deep_teal"],
        )
        if i > 0:
            orthogonal_arrow(
                ax,
                (16.74, y + 0.72),
                (16.74, decoder_ys[i - 1]),
                color=COLORS["deep_teal"],
                linewidth=1.9,
            )

    for i, y in enumerate(ys):
        orthogonal_arrow(
            ax,
            (14.86, y + 0.36),
            (16.30, decoder_ys[i] + 0.36),
            color=COLORS["purple"],
            linewidth=1.4,
            linestyle="--",
        )

    rounded_box(
        ax, (15.94, 1.55), 1.45, 0.58,
        "Multi-scale\naggregation",
        facecolor=COLORS["light_teal"],
        edgecolor=COLORS["deep_teal"],
        fontsize=8.8,
        fontweight="bold",
    )

    # Outputs
    image_box(
        ax, mask_img, (19.02, 21.20, 6.72, 8.90),
        edgecolor=COLORS["dark_red"],
        title="Final Flood Segmentation",
        subtitle="H × W × 1",
    )

    rounded_box(
        ax, (18.86, 4.83), 2.48, 1.15,
        "Adaptive Decision Fusion\nReliability-weighted prediction",
        facecolor=COLORS["rose"],
        edgecolor=COLORS["dark_red"],
        fontsize=9.4,
        fontweight="bold",
    )

    rounded_box(
        ax, (18.86, 2.12), 2.48, 1.95,
        "Auxiliary Heads\n\nSAR-only\nOptical-only\nFused",
        facecolor=COLORS["light_gray"],
        edgecolor=COLORS["gray"],
        fontsize=9.3,
        fontweight="bold",
    )

    orthogonal_arrow(
        ax,
        (17.18, 8.10),
        (18.86, 5.40),
        color=COLORS["deep_teal"],
        via_x=18.02,
    )
    orthogonal_arrow(
        ax,
        (20.10, 5.98),
        (20.10, 6.72),
        color=COLORS["dark_red"],
        linewidth=1.9,
    )
    orthogonal_arrow(
        ax,
        (14.58, 2.55),
        (18.86, 3.10),
        color=COLORS["purple"],
        via_x=17.85,
    )

    # Bottom legend
    legend_y = 0.48
    legend_items = [
        ("SAR stream", COLORS["dark_blue"], "-"),
        ("Optical stream", COLORS["teal"], "-"),
        ("Fused stream", COLORS["purple"], "-"),
        ("Reliability guidance", COLORS["orange"], "--"),
        ("Decoder skip", COLORS["purple"], "--"),
        ("Decision output", COLORS["dark_red"], "-"),
    ]

    x = 0.55
    for text, color, style in legend_items:
        ax.plot(
            [x, x + 0.46],
            [legend_y, legend_y],
            color=color,
            linewidth=2.1,
            linestyle=style,
        )
        ax.text(
            x + 0.56,
            legend_y,
            text,
            ha="left",
            va="center",
            fontsize=8.5,
            color=COLORS["gray"],
        )
        x += 2.35

    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / "tcsf_v31_overall_architecture"

    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", transparent=transparent)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", transparent=transparent)
    fig.savefig(
        base.with_suffix(".png"),
        dpi=dpi,
        bbox_inches="tight",
        transparent=transparent,
    )
    plt.close(fig)


# ---------------------------------------------------------------------
# CROSS-STATE FUSION BLOCK
# ---------------------------------------------------------------------
def draw_csf_block(output_dir: Path, dpi: int, transparent: bool):
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis("off")

    ax.text(
        8,
        6.62,
        "Cross-State Fusion (CSF) Block",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color=COLORS["navy"],
    )
    ax.text(
        8,
        6.25,
        "Reliability-guided interaction between SAR and optical feature states at one encoder level",
        ha="center",
        va="center",
        fontsize=10.8,
        color=COLORS["gray"],
    )

    # Input features
    feature_cube(
        ax, 0.65, 4.34, 0.88, 0.88, 0.16,
        front_color=COLORS["cyan"],
        side_color=COLORS["blue"],
        top_color=COLORS["bright_cyan"],
        edgecolor=COLORS["dark_blue"],
        label="SAR feature  $F_{sar}$",
    )
    feature_cube(
        ax, 0.65, 1.74, 0.88, 0.88, 0.16,
        front_color=COLORS["turquoise"],
        side_color=COLORS["teal"],
        top_color=COLORS["mint"],
        edgecolor=COLORS["deep_teal"],
        label="Optical feature  $F_{opt}$",
    )

    rounded_box(
        ax, (2.20, 4.20), 1.30, 0.72,
        "LayerNorm\n+ Linear",
        facecolor=COLORS["pale_cyan"],
        edgecolor=COLORS["dark_blue"],
        fontsize=9.2,
        fontweight="bold",
    )
    rounded_box(
        ax, (2.20, 1.60), 1.30, 0.72,
        "LayerNorm\n+ Linear",
        facecolor=COLORS["mint"],
        edgecolor=COLORS["teal"],
        fontsize=9.2,
        fontweight="bold",
    )

    rounded_box(
        ax, (4.20, 3.12), 2.22, 1.34,
        "Bidirectional\nCross-State Interaction\nSAR ↔ Optical",
        facecolor=COLORS["lavender"],
        edgecolor=COLORS["purple"],
        fontsize=10.0,
        fontweight="bold",
    )

    rounded_box(
        ax, (7.05, 3.12), 2.12, 1.34,
        "Reliability Gating\n$R_{sar}, R_{opt}$\nSigmoid weights",
        facecolor=COLORS["soft_gold"],
        edgecolor=COLORS["gold"],
        fontsize=9.5,
        fontweight="bold",
    )

    rounded_box(
        ax, (9.80, 3.12), 2.05, 1.34,
        "Gated Fusion\nElement-wise\nweighted combination",
        facecolor=COLORS["rose"],
        edgecolor=COLORS["dark_red"],
        fontsize=9.5,
        fontweight="bold",
    )

    rounded_box(
        ax, (12.45, 3.12), 1.72, 1.34,
        "Fusion Projection\n1×1 Conv + LN",
        facecolor=COLORS["light_teal"],
        edgecolor=COLORS["deep_teal"],
        fontsize=9.5,
        fontweight="bold",
    )

    feature_cube(
        ax, 14.70, 3.34, 0.92, 0.92, 0.17,
        front_color="#8F63C7",
        side_color=COLORS["purple"],
        top_color="#B999DE",
        edgecolor=COLORS["purple"],
        label="Fused feature  $F_{fused}$",
    )

    orthogonal_arrow(
        ax, (1.68, 4.78), (2.20, 4.56),
        color=COLORS["dark_blue"],
        via_x=1.95,
    )
    orthogonal_arrow(
        ax, (1.68, 2.18), (2.20, 1.96),
        color=COLORS["teal"],
        via_x=1.95,
    )
    orthogonal_arrow(
        ax, (3.50, 4.56), (4.20, 3.98),
        color=COLORS["dark_blue"],
        via_x=3.85,
    )
    orthogonal_arrow(
        ax, (3.50, 1.96), (4.20, 3.58),
        color=COLORS["teal"],
        via_x=3.85,
    )
    orthogonal_arrow(
        ax, (6.42, 3.79), (7.05, 3.79),
        color=COLORS["purple"],
    )
    orthogonal_arrow(
        ax, (9.17, 3.79), (9.80, 3.79),
        color=COLORS["dark_red"],
    )
    orthogonal_arrow(
        ax, (11.85, 3.79), (12.45, 3.79),
        color=COLORS["deep_teal"],
    )
    orthogonal_arrow(
        ax, (14.17, 3.79), (14.70, 3.79),
        color=COLORS["purple"],
    )

    # Residual routes
    orthogonal_arrow(
        ax,
        (3.10, 4.92),
        (13.31, 4.46),
        color=COLORS["dark_blue"],
        linewidth=1.25,
        linestyle="--",
        via_y=5.35,
    )
    orthogonal_arrow(
        ax,
        (3.10, 1.60),
        (13.31, 3.12),
        color=COLORS["teal"],
        linewidth=1.25,
        linestyle="--",
        via_y=1.00,
    )

    ax.text(
        8.0,
        5.55,
        "Residual preservation",
        ha="center",
        va="center",
        fontsize=8.8,
        color=COLORS["dark_blue"],
        fontweight="bold",
    )
    ax.text(
        8.0,
        0.78,
        "Modality-specific information retained",
        ha="center",
        va="center",
        fontsize=8.8,
        color=COLORS["teal"],
        fontweight="bold",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / "tcsf_v31_csf_block"

    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", transparent=transparent)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", transparent=transparent)
    fig.savefig(
        base.with_suffix(".png"),
        dpi=dpi,
        bbox_inches="tight",
        transparent=transparent,
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-ready TCSF v3.1 architecture diagrams."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/publication/architecture_v2",
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--transparent", action="store_true")
    parser.add_argument("--sar-image", default=None)
    parser.add_argument("--optical-image", default=None)
    parser.add_argument("--mask-image", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    draw_overall_architecture(
        output_dir=output_dir,
        dpi=args.dpi,
        transparent=args.transparent,
        sar_image_path=args.sar_image,
        optical_image_path=args.optical_image,
        mask_image_path=args.mask_image,
    )
    draw_csf_block(
        output_dir=output_dir,
        dpi=args.dpi,
        transparent=args.transparent,
    )

    print(f"[INFO] Architecture figures saved to: {output_dir}")
    print("[INFO] Generated overall architecture and CSF block diagrams.")


if __name__ == "__main__":
    main()