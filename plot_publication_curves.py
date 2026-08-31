#!/usr/bin/env python3
"""
Generate publication-quality training and transition-scale figures.

Examples
--------
1) Plot one experiment:
python plot_publication_curves.py \
    --experiments "TCSF v3.1=outputs/checkpoints/TCSF_v31_seed2026"

2) Compare several experiments:
python plot_publication_curves.py \
    --experiments \
      "ACSF v2=outputs/checkpoints/ACSF_v2" \
      "TCSF v3=outputs/checkpoints/TCSF_v3" \
      "TCSF v3.1=outputs/checkpoints/TCSF_v31_seed2026" \
      "RU-TCSF=outputs/checkpoints/RU_TCSF" \
      "TCSF 200 ep=outputs/checkpoints/TCSF_v31_final_200ep" \
    --output_dir outputs/publication/curves

The script recursively searches each experiment directory for CSV files and
automatically detects common metric column names.
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Column aliases
# ---------------------------------------------------------------------

ALIASES: Mapping[str, Sequence[str]] = {
    "epoch": (
        "epoch", "epochs", "step", "global_step"
    ),
    "train_loss": (
        "train_loss", "training_loss", "loss_train", "train total loss",
        "trainloss", "train/loss", "loss"
    ),
    "val_loss": (
        "val_loss", "validation_loss", "valid_loss", "loss_val",
        "valloss", "validation/loss"
    ),
    "val_iou": (
        "val_iou", "validation_iou", "valid_iou", "iou", "mean_iou",
        "miou", "val/ iou", "val/iou"
    ),
    "val_dice": (
        "val_dice", "validation_dice", "valid_dice", "dice", "f1",
        "f1_score", "val_f1", "val/f1", "val/dice"
    ),
    "precision": (
        "precision", "val_precision", "validation_precision"
    ),
    "recall": (
        "recall", "val_recall", "validation_recall", "sensitivity"
    ),
    "pixel_accuracy": (
        "pixel_accuracy", "pixel_acc", "pixelacc", "accuracy",
        "val_pixel_accuracy"
    ),
    "learning_rate": (
        "learning_rate", "lr", "learning rate", "optimizer_lr"
    ),
}

TRANSITION_ALIASES: Mapping[str, Sequence[str]] = {
    "t1_sar_gamma": (
        "t1_sar_gamma", "transition1_sar_gamma", "transition_1_sar_gamma",
        "t1 sar gamma"
    ),
    "t1_optical_gamma": (
        "t1_optical_gamma", "transition1_optical_gamma",
        "transition_1_optical_gamma", "t1 opt gamma", "t1_opt_gamma"
    ),
    "t1_fused_gamma": (
        "t1_fused_gamma", "transition1_fused_gamma",
        "transition_1_fused_gamma", "t1 fusion gamma", "t1_fusion_gamma"
    ),
    "t2_sar_gamma": (
        "t2_sar_gamma", "transition2_sar_gamma", "transition_2_sar_gamma",
        "t2 sar gamma"
    ),
    "t2_optical_gamma": (
        "t2_optical_gamma", "transition2_optical_gamma",
        "transition_2_optical_gamma", "t2 opt gamma", "t2_opt_gamma"
    ),
    "t2_fused_gamma": (
        "t2_fused_gamma", "transition2_fused_gamma",
        "transition_2_fused_gamma", "t2 fusion gamma", "t2_fusion_gamma"
    ),
    "t3_sar_gamma": (
        "t3_sar_gamma", "transition3_sar_gamma", "transition_3_sar_gamma",
        "t3 sar gamma"
    ),
    "t3_optical_gamma": (
        "t3_optical_gamma", "transition3_optical_gamma",
        "transition_3_optical_gamma", "t3 opt gamma", "t3_opt_gamma"
    ),
    "t3_fused_gamma": (
        "t3_fused_gamma", "transition3_fused_gamma",
        "transition_3_fused_gamma", "t3 fusion gamma", "t3_fusion_gamma"
    ),
}


def normalize_name(name: str) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"[\s/\\\-]+", "_", text)
    text = re.sub(r"[^a-z0-9_]+", "", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def resolve_column(columns: Iterable[str], aliases: Sequence[str]) -> Optional[str]:
    normalized = {normalize_name(c): c for c in columns}
    for alias in aliases:
        key = normalize_name(alias)
        if key in normalized:
            return normalized[key]
    return None


@dataclass
class Experiment:
    label: str
    directory: Path
    csv_path: Path
    frame: pd.DataFrame
    columns: Dict[str, str]


def csv_score(path: Path) -> Tuple[int, int, int]:
    """
    Prefer files named metrics/history/log, then files with more useful columns.
    """
    name = path.name.lower()
    name_score = 0
    for token, score in (
        ("metrics", 6),
        ("history", 5),
        ("training", 4),
        ("train", 3),
        ("log", 2),
    ):
        if token in name:
            name_score = max(name_score, score)

    try:
        df = pd.read_csv(path, nrows=5)
        metric_score = sum(
            resolve_column(df.columns, aliases) is not None
            for aliases in list(ALIASES.values()) + list(TRANSITION_ALIASES.values())
        )
        rows_hint = sum(1 for _ in open(path, "r", encoding="utf-8", errors="ignore"))
    except Exception:
        metric_score = -1
        rows_hint = -1

    return name_score, metric_score, rows_hint


def find_best_csv(directory: Path) -> Path:
    if directory.is_file() and directory.suffix.lower() == ".csv":
        return directory

    candidates = sorted(directory.rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV file found under: {directory}")

    ranked = sorted(candidates, key=csv_score, reverse=True)
    best = ranked[0]

    if csv_score(best)[1] <= 0:
        raise ValueError(
            f"CSV files were found under {directory}, but none contained recognizable "
            f"training metric columns. Candidates: {[str(p) for p in candidates[:10]]}"
        )
    return best


def load_experiment(label: str, directory: Path) -> Experiment:
    csv_path = find_best_csv(directory)
    frame = pd.read_csv(csv_path)

    if frame.empty:
        raise ValueError(f"CSV is empty: {csv_path}")

    columns: Dict[str, str] = {}
    for key, aliases in {**ALIASES, **TRANSITION_ALIASES}.items():
        found = resolve_column(frame.columns, aliases)
        if found is not None:
            columns[key] = found

    if "epoch" not in columns:
        frame = frame.copy()
        frame["_generated_epoch"] = np.arange(1, len(frame) + 1)
        columns["epoch"] = "_generated_epoch"

    epoch_col = columns["epoch"]
    frame[epoch_col] = pd.to_numeric(frame[epoch_col], errors="coerce")
    frame = frame.dropna(subset=[epoch_col]).sort_values(epoch_col)

    for key, col in columns.items():
        if key != "epoch":
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    return Experiment(
        label=label,
        directory=directory,
        csv_path=csv_path,
        frame=frame,
        columns=columns,
    )


def save_figure(fig: plt.Figure, base_path: Path, dpi: int = 600) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def apply_publication_format(ax: plt.Axes, xlabel: str = "Epoch") -> None:
    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.tick_params(direction="out")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_metric_comparison(
    experiments: Sequence[Experiment],
    metric_key: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> bool:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    plotted = False

    for exp in experiments:
        metric_col = exp.columns.get(metric_key)
        if metric_col is None:
            continue
        epoch_col = exp.columns["epoch"]
        subset = exp.frame[[epoch_col, metric_col]].dropna()
        if subset.empty:
            continue
        ax.plot(
            subset[epoch_col],
            subset[metric_col],
            linewidth=1.9,
            label=exp.label,
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    apply_publication_format(ax)
    if len(experiments) > 1:
        ax.legend(frameon=False, loc="best")
    save_figure(fig, output_path)
    return True


def plot_loss_panel(experiments: Sequence[Experiment], output_path: Path) -> bool:
    """
    One panel per experiment because train/validation losses may have very
    different ranges between models.
    """
    valid = [
        exp for exp in experiments
        if "train_loss" in exp.columns or "val_loss" in exp.columns
    ]
    if not valid:
        return False

    n = len(valid)
    fig, axes = plt.subplots(
        nrows=n,
        ncols=1,
        figsize=(7.2, max(4.0, 3.15 * n)),
        squeeze=False,
    )

    for row, exp in enumerate(valid):
        ax = axes[row, 0]
        epoch_col = exp.columns["epoch"]

        if "train_loss" in exp.columns:
            col = exp.columns["train_loss"]
            subset = exp.frame[[epoch_col, col]].dropna()
            ax.plot(subset[epoch_col], subset[col], linewidth=1.8, label="Training")

        if "val_loss" in exp.columns:
            col = exp.columns["val_loss"]
            subset = exp.frame[[epoch_col, col]].dropna()
            ax.plot(subset[epoch_col], subset[col], linewidth=1.8, label="Validation")

        ax.set_ylabel("Loss")
        ax.set_title(exp.label)
        apply_publication_format(ax)
        ax.legend(frameon=False)

    fig.tight_layout()
    save_figure(fig, output_path)
    return True


def plot_transition_grid(exp: Experiment, output_path: Path) -> bool:
    keys = list(TRANSITION_ALIASES.keys())
    available = [key for key in keys if key in exp.columns]
    if not available:
        return False

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.8), sharex=True)
    epoch_col = exp.columns["epoch"]

    for transition_index in range(1, 4):
        ax = axes[transition_index - 1]
        plotted = False
        for branch, label in (
            ("sar", "SAR"),
            ("optical", "Optical"),
            ("fused", "Fused"),
        ):
            key = f"t{transition_index}_{branch}_gamma"
            col = exp.columns.get(key)
            if col is None:
                continue
            subset = exp.frame[[epoch_col, col]].dropna()
            if subset.empty:
                continue
            ax.plot(
                subset[epoch_col],
                subset[col],
                linewidth=1.9,
                label=label,
            )
            plotted = True

        ax.axhline(0.0, linewidth=0.8, alpha=0.5)
        ax.set_ylabel(r"$\gamma$")
        ax.set_title(f"Transition {transition_index}")
        apply_publication_format(ax)
        if plotted:
            ax.legend(frameon=False, ncol=3, loc="best")

    axes[-1].set_xlabel("Epoch")
    fig.suptitle(f"{exp.label}: Learned Transition Scales", y=1.01)
    fig.tight_layout()
    save_figure(fig, output_path)
    return True


def plot_gamma_vs_iou(exp: Experiment, output_path: Path) -> bool:
    """
    Exploratory diagnostic: mean absolute transition coefficient and validation IoU.
    Useful for the 200-epoch overfitting analysis.
    """
    gamma_cols = [
        exp.columns[key]
        for key in TRANSITION_ALIASES
        if key in exp.columns
    ]
    iou_col = exp.columns.get("val_iou")
    if not gamma_cols or iou_col is None:
        return False

    epoch_col = exp.columns["epoch"]
    data = exp.frame[[epoch_col, iou_col] + gamma_cols].copy()
    data["mean_abs_gamma"] = data[gamma_cols].abs().mean(axis=1)
    data = data.dropna(subset=[epoch_col, iou_col, "mean_abs_gamma"])
    if data.empty:
        return False

    fig, ax1 = plt.subplots(figsize=(7.2, 4.6))
    ax2 = ax1.twinx()

    line1 = ax1.plot(
        data[epoch_col],
        data[iou_col],
        linewidth=2.0,
        label="Validation IoU",
    )
    line2 = ax2.plot(
        data[epoch_col],
        data["mean_abs_gamma"],
        linewidth=2.0,
        linestyle="--",
        label=r"Mean $|\gamma|$",
    )

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation IoU")
    ax2.set_ylabel(r"Mean absolute transition scale $|\gamma|$")
    ax1.set_title(f"{exp.label}: Transition Strength vs Validation IoU")
    ax1.grid(True, alpha=0.25, linewidth=0.7)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, frameon=False, loc="best")

    save_figure(fig, output_path)
    return True


def parse_experiment(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        path = Path(value).expanduser()
        return path.name, path
    label, raw_path = value.split("=", 1)
    label = label.strip()
    path = Path(raw_path.strip()).expanduser()
    if not label:
        raise ValueError(f"Experiment label is empty: {value}")
    return label, path


def write_manifest(experiments: Sequence[Experiment], output_dir: Path) -> None:
    manifest = {
        "experiments": [
            {
                "label": exp.label,
                "directory": str(exp.directory),
                "csv_path": str(exp.csv_path),
                "detected_columns": exp.columns,
                "rows": int(len(exp.frame)),
            }
            for exp in experiments
        ]
    }
    with open(output_dir / "curve_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate publication-quality training curves."
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        help='Entries formatted as "Label=path/to/experiment" or just a path.',
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/publication/curves"),
        help="Directory for PNG/PDF figures.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    experiments: List[Experiment] = []
    for item in args.experiments:
        label, directory = parse_experiment(item)
        if not directory.exists():
            warnings.warn(f"Skipping missing path: {directory}")
            continue
        try:
            experiment = load_experiment(label, directory)
            experiments.append(experiment)
            print(f"[OK] {label}: {experiment.csv_path}")
            print(f"     detected: {experiment.columns}")
        except Exception as exc:
            warnings.warn(f"Skipping {label}: {exc}")

    if not experiments:
        raise SystemExit("No valid experiments could be loaded.")

    write_manifest(experiments, args.output_dir)

    generated: List[str] = []

    if plot_loss_panel(
        experiments,
        args.output_dir / "training_validation_loss",
    ):
        generated.append("training_validation_loss")

    metric_specs = [
        ("val_iou", "Validation IoU", "Validation IoU", "validation_iou"),
        ("val_dice", "Validation Dice", "Validation Dice", "validation_dice"),
        ("precision", "Validation Precision", "Validation Precision", "validation_precision"),
        ("recall", "Validation Recall", "Validation Recall", "validation_recall"),
        (
            "pixel_accuracy",
            "Pixel Accuracy",
            "Validation Pixel Accuracy",
            "validation_pixel_accuracy",
        ),
        (
            "learning_rate",
            "Learning Rate",
            "Learning-Rate Schedule",
            "learning_rate",
        ),
    ]

    for key, ylabel, title, filename in metric_specs:
        if plot_metric_comparison(
            experiments,
            key,
            ylabel,
            title,
            args.output_dir / filename,
        ):
            generated.append(filename)

    for exp in experiments:
        safe_label = re.sub(r"[^A-Za-z0-9._-]+", "_", exp.label).strip("_")
        if plot_transition_grid(
            exp,
            args.output_dir / f"{safe_label}_transition_scales",
        ):
            generated.append(f"{safe_label}_transition_scales")

        if plot_gamma_vs_iou(
            exp,
            args.output_dir / f"{safe_label}_gamma_vs_iou",
        ):
            generated.append(f"{safe_label}_gamma_vs_iou")

    print("\nGenerated figures:")
    for item in generated:
        print(f"  - {args.output_dir / item}.png")
        print(f"  - {args.output_dir / item}.pdf")

    if not generated:
        print(
            "No figures were generated. Inspect curve_manifest.json and verify "
            "that the CSV contains recognized metric columns."
        )


if __name__ == "__main__":
    main()