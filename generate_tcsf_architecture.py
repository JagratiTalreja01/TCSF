#!/usr/bin/env python3
"""
generate_tcsf_architecture.py

Generate a publication-ready architecture diagram for the TCSF v3.1
multimodal flood-segmentation framework.

Outputs:
    outputs/publication/architecture/tcsf_v31_architecture.pdf
    outputs/publication/architecture/tcsf_v31_architecture.svg
    outputs/publication/architecture/tcsf_v31_architecture.png

The diagram contains:
    - Sentinel-1 SAR input (VV, VH)
    - Sentinel-2 optical input (13 bands)
    - Modality reliability estimation
    - Adaptive data fusion
    - SAR and optical Vision-Mamba encoders
    - Four-scale cross-state fusion
    - Three controlled residual cross-scale transitions
    - Multi-scale decoder
    - SAR-only, optical-only, and fused auxiliary heads
    - Adaptive decision fusion
    - Final flood segmentation output

Run:
    python generate_tcsf_architecture.py

Optional:
    python generate_tcsf_architecture.py \
        --output-dir outputs/publication/architecture \
        --dpi 600 \
        --transparent
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


def add_box(
    ax,
    x,
    y,
    width,
    height,
    text,
    *,
    facecolor,
    edgecolor="#222222",
    linewidth=1.4,
    fontsize=9.5,
    fontweight="normal",
    radius=0.02,
    zorder=3,
):
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        zorder=zorder + 1,
        linespacing=1.15,
    )
    return box


def add_arrow(
    ax,
    start,
    end,
    *,
    color="#333333",
    linewidth=1.5,
    style="-|>",
    mutation_scale=12,
    linestyle="-",
    connectionstyle="arc3",
    zorder=2,
):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        linestyle=linestyle,
        connectionstyle=connectionstyle,
        zorder=zorder,
    )
    ax.add_patch(arrow)
    return arrow


def add_label(ax, x, y, text, *, fontsize=8.5, color="#333333", rotation=0):
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color,
        rotation=rotation,
        zorder=5,
    )


def draw_architecture(output_dir: Path, dpi: int, transparent: bool):
    # Restrained publication palette.
    sar_color = "#DDEBFF"
    optical_color = "#E3F3DF"
    reliability_color = "#FFF1C9"
    fusion_color = "#F4E1FF"
    encoder_color = "#EAF0F8"
    transition_color = "#FFE7D6"
    decoder_color = "#DDF4F2"
    head_color = "#F1F1F1"
    decision_color = "#FFE0EA"
    output_color = "#DCEED1"
    dark = "#222222"
    sar_edge = "#3566A8"
    optical_edge = "#4A8B3A"
    fusion_edge = "#7A4AA8"
    transition_edge = "#C2692D"

    fig, ax = plt.subplots(figsize=(18, 11))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 11)
    ax.axis("off")

    # ------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------
    ax.text(
        9,
        10.72,
        "Tri-Level Cross-State Fusion Network (TCSF v3.1)",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color=dark,
    )
    ax.text(
        9,
        10.38,
        "Reliability-aware multimodal flood segmentation from Sentinel-1 SAR and Sentinel-2 optical imagery",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#4A4A4A",
    )

    # ------------------------------------------------------------------
    # Input and reliability stage
    # ------------------------------------------------------------------
    add_box(
        ax, 0.45, 8.95, 2.05, 0.82,
        "Sentinel-1 SAR\nVV + VH",
        facecolor=sar_color,
        edgecolor=sar_edge,
        fontweight="bold",
    )
    add_box(
        ax, 0.45, 7.42, 2.05, 0.82,
        "Sentinel-2 Optical\n13 spectral bands",
        facecolor=optical_color,
        edgecolor=optical_edge,
        fontweight="bold",
    )

    add_box(
        ax, 3.15, 8.95, 2.05, 0.82,
        "SAR Reliability\nEstimator",
        facecolor=reliability_color,
        edgecolor="#B38A24",
    )
    add_box(
        ax, 3.15, 7.42, 2.05, 0.82,
        "Optical Reliability\nEstimator",
        facecolor=reliability_color,
        edgecolor="#B38A24",
    )

    add_arrow(ax, (2.50, 9.36), (3.15, 9.36), color=sar_edge)
    add_arrow(ax, (2.50, 7.83), (3.15, 7.83), color=optical_edge)

    add_box(
        ax, 5.95, 8.18, 2.25, 1.05,
        "Adaptive Data Fusion\n(ADF)",
        facecolor=fusion_color,
        edgecolor=fusion_edge,
        fontweight="bold",
    )
    add_arrow(
        ax, (5.20, 9.36), (5.95, 8.86),
        color=sar_edge, connectionstyle="arc3,rad=-0.10"
    )
    add_arrow(
        ax, (5.20, 7.83), (5.95, 8.54),
        color=optical_edge, connectionstyle="arc3,rad=0.10"
    )
    add_label(ax, 5.57, 9.22, r"$R_{sar}$", color=sar_edge)
    add_label(ax, 5.57, 7.98, r"$R_{opt}$", color=optical_edge)

    # ------------------------------------------------------------------
    # Dual encoders
    # ------------------------------------------------------------------
    add_box(
        ax, 0.55, 5.55, 3.15, 0.82,
        "SAR Vision-Mamba Encoder",
        facecolor=encoder_color,
        edgecolor=sar_edge,
        fontweight="bold",
    )
    add_box(
        ax, 0.55, 4.35, 3.15, 0.82,
        "Optical Vision-Mamba Encoder",
        facecolor=encoder_color,
        edgecolor=optical_edge,
        fontweight="bold",
    )

    add_arrow(
        ax, (1.48, 8.95), (1.48, 6.37),
        color=sar_edge
    )
    add_arrow(
        ax, (1.48, 7.42), (1.48, 5.17),
        color=optical_edge
    )

    # ADF feeds fused initial context into all scales.
    add_arrow(
        ax, (7.08, 8.18), (7.08, 6.95),
        color=fusion_edge
    )
    add_label(ax, 7.62, 7.54, "Initial fused context", color=fusion_edge)

    # ------------------------------------------------------------------
    # Four-scale feature hierarchy
    # ------------------------------------------------------------------
    scale_y = [6.25, 4.95, 3.65, 2.35]
    scale_names = ["Scale 1", "Scale 2", "Scale 3", "Scale 4"]
    spatial_labels = ["1/4", "1/8", "1/16", "1/32"]

    sar_x = 4.35
    opt_x = 7.05
    fused_x = 9.75
    box_w = 2.05
    box_h = 0.76

    for i, (y, scale, spatial) in enumerate(zip(scale_y, scale_names, spatial_labels)):
        add_box(
            ax, sar_x, y, box_w, box_h,
            f"{scale} SAR state\n({spatial} resolution)",
            facecolor=sar_color,
            edgecolor=sar_edge,
            fontsize=8.8,
        )
        add_box(
            ax, opt_x, y, box_w, box_h,
            f"{scale} Optical state\n({spatial} resolution)",
            facecolor=optical_color,
            edgecolor=optical_edge,
            fontsize=8.8,
        )
        add_box(
            ax, fused_x, y, box_w, box_h,
            f"{scale} Fused state\n({spatial} resolution)",
            facecolor=fusion_color,
            edgecolor=fusion_edge,
            fontsize=8.8,
            fontweight="bold",
        )

        # Cross-state interactions at each scale.
        add_arrow(
            ax, (sar_x + box_w, y + box_h / 2),
            (fused_x, y + box_h / 2),
            color=sar_edge,
            linewidth=1.3,
        )
        add_arrow(
            ax, (opt_x + box_w, y + box_h / 2),
            (fused_x, y + box_h / 2),
            color=optical_edge,
            linewidth=1.3,
        )

        # Reliability guidance.
        add_arrow(
            ax, (5.20, 9.10),
            (sar_x + box_w / 2, y + box_h),
            color="#B38A24",
            linewidth=0.9,
            linestyle="--",
            mutation_scale=9,
            connectionstyle=f"arc3,rad={0.08 + 0.03*i}",
        )
        add_arrow(
            ax, (5.20, 7.60),
            (opt_x + box_w / 2, y + box_h),
            color="#B38A24",
            linewidth=0.9,
            linestyle="--",
            mutation_scale=9,
            connectionstyle=f"arc3,rad={-0.08 - 0.03*i}",
        )

    # Encoder to first scale.
    add_arrow(ax, (3.70, 5.96), (4.35, 6.63), color=sar_edge)
    add_arrow(ax, (3.70, 4.76), (7.05, 6.63), color=optical_edge)

    # Initial fused context to first fused state.
    add_arrow(
        ax, (7.08, 6.95), (10.78, 7.01),
        color=fusion_edge,
        connectionstyle="arc3,rad=-0.08"
    )
    add_arrow(ax, (10.78, 7.01), (10.78, 7.01 - 0.01), color=fusion_edge)

    # ------------------------------------------------------------------
    # Controlled residual cross-scale propagation
    # ------------------------------------------------------------------
    for i in range(3):
        y_top = scale_y[i]
        y_bottom = scale_y[i + 1]

        # SAR transition
        add_arrow(
            ax,
            (sar_x + box_w / 2, y_top),
            (sar_x + box_w / 2, y_bottom + box_h),
            color=transition_edge,
            linewidth=1.5,
        )

        # Optical transition
        add_arrow(
            ax,
            (opt_x + box_w / 2, y_top),
            (opt_x + box_w / 2, y_bottom + box_h),
            color=transition_edge,
            linewidth=1.5,
        )

        # Fused transition
        add_arrow(
            ax,
            (fused_x + box_w / 2, y_top),
            (fused_x + box_w / 2, y_bottom + box_h),
            color=transition_edge,
            linewidth=1.8,
        )

        add_box(
            ax,
            12.25,
            (y_top + y_bottom) / 2 + 0.10,
            1.65,
            0.56,
            rf"Transition {i+1}" + "\n" + r"$x_{l+1}+\gamma_l\,T(x_l)$",
            facecolor=transition_color,
            edgecolor=transition_edge,
            fontsize=8.2,
        )
        add_arrow(
            ax,
            (fused_x + box_w, (y_top + y_bottom) / 2 + 0.39),
            (12.25, (y_top + y_bottom) / 2 + 0.39),
            color=transition_edge,
            linewidth=1.0,
            linestyle="--",
            mutation_scale=9,
        )

    add_label(
        ax, 13.05, 6.93,
        "Controlled residual\ncross-scale propagation",
        fontsize=8.5,
        color=transition_edge,
    )

    # ------------------------------------------------------------------
    # Multi-scale decoder
    # ------------------------------------------------------------------
    decoder_x = 14.25
    add_box(
        ax, decoder_x, 3.10, 2.55, 2.55,
        "Multi-Scale Decoder\n\n• Feature aggregation\n• Progressive upsampling\n• Boundary refinement",
        facecolor=decoder_color,
        edgecolor="#2C8580",
        fontweight="bold",
        fontsize=9.2,
    )

    for y in scale_y:
        add_arrow(
            ax,
            (fused_x + box_w, y + box_h / 2),
            (decoder_x, 4.38),
            color=fusion_edge,
            linewidth=1.1,
            connectionstyle="arc3,rad=0.0",
        )

    # ------------------------------------------------------------------
    # Auxiliary heads and adaptive decision fusion
    # ------------------------------------------------------------------
    add_box(
        ax, 3.90, 0.62, 2.25, 0.75,
        "SAR Auxiliary Head",
        facecolor=head_color,
        edgecolor=sar_edge,
    )
    add_box(
        ax, 6.70, 0.62, 2.25, 0.75,
        "Optical Auxiliary Head",
        facecolor=head_color,
        edgecolor=optical_edge,
    )
    add_box(
        ax, 9.50, 0.62, 2.25, 0.75,
        "Fused Auxiliary Head",
        facecolor=head_color,
        edgecolor=fusion_edge,
    )

    add_arrow(
        ax,
        (sar_x + box_w / 2, scale_y[-1]),
        (5.03, 1.37),
        color=sar_edge,
        connectionstyle="arc3,rad=0.08",
    )
    add_arrow(
        ax,
        (opt_x + box_w / 2, scale_y[-1]),
        (7.83, 1.37),
        color=optical_edge,
        connectionstyle="arc3,rad=0.03",
    )
    add_arrow(
        ax,
        (decoder_x + 0.75, 3.10),
        (10.63, 1.37),
        color=fusion_edge,
        connectionstyle="arc3,rad=-0.12",
    )

    add_box(
        ax, 12.35, 0.52, 2.40, 0.95,
        "Adaptive Decision Fusion\nReliability-weighted prediction",
        facecolor=decision_color,
        edgecolor="#B74269",
        fontweight="bold",
        fontsize=8.9,
    )

    add_arrow(ax, (6.15, 0.99), (12.35, 1.15), color=sar_edge)
    add_arrow(ax, (8.95, 0.99), (12.35, 0.99), color=optical_edge)
    add_arrow(ax, (11.75, 0.99), (12.35, 0.83), color=fusion_edge)

    add_box(
        ax, 15.35, 0.52, 2.15, 0.95,
        "Final Flood\nSegmentation Map",
        facecolor=output_color,
        edgecolor="#4E7C36",
        fontweight="bold",
    )
    add_arrow(
        ax, (14.75, 0.995), (15.35, 0.995),
        color="#4E7C36",
        linewidth=1.8,
    )

    # Direct decoder contribution to final fusion.
    add_arrow(
        ax,
        (15.52, 3.10),
        (13.55, 1.47),
        color="#2C8580",
        linewidth=1.3,
        connectionstyle="arc3,rad=0.14",
    )

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------
    legend_y = 9.92
    legend_items = [
        ("SAR stream", sar_color, sar_edge),
        ("Optical stream", optical_color, optical_edge),
        ("Reliability guidance", reliability_color, "#B38A24"),
        ("Cross-state fusion", fusion_color, fusion_edge),
        ("Residual transition", transition_color, transition_edge),
    ]
    x = 9.55
    for label, face, edge in legend_items:
        add_box(
            ax,
            x,
            legend_y,
            1.43,
            0.30,
            label,
            facecolor=face,
            edgecolor=edge,
            linewidth=1.0,
            fontsize=7.0,
            radius=0.01,
        )
        x += 1.57

    # Dashed-line annotation.
    add_label(
        ax,
        6.0,
        2.05,
        "Dashed arrows: reliability guidance",
        fontsize=8.0,
        color="#6A6A6A",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / "tcsf_v31_architecture"

    fig.savefig(
        base.with_suffix(".pdf"),
        bbox_inches="tight",
        transparent=transparent,
    )
    fig.savefig(
        base.with_suffix(".svg"),
        bbox_inches="tight",
        transparent=transparent,
    )
    fig.savefig(
        base.with_suffix(".png"),
        dpi=dpi,
        bbox_inches="tight",
        transparent=transparent,
    )
    plt.close(fig)

    print(f"[INFO] Saved: {base.with_suffix('.pdf')}")
    print(f"[INFO] Saved: {base.with_suffix('.svg')}")
    print(f"[INFO] Saved: {base.with_suffix('.png')}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate the TCSF v3.1 architecture diagram."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/publication/architecture",
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--transparent", action="store_true")
    args = parser.parse_args()

    draw_architecture(
        output_dir=Path(args.output_dir),
        dpi=args.dpi,
        transparent=args.transparent,
    )


if __name__ == "__main__":
    main()