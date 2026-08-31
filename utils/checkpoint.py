"""
checkpoint.py

Checkpoint utilities for ACSF.
"""

from pathlib import Path
import torch


def save_checkpoint(
    model,
    optimizer,
    epoch,
    cfg,
    best=False,
    scheduler=None,
):
    """
    Save training checkpoint.

    Args:
        model
        optimizer
        epoch
        cfg
        best (bool)
        scheduler
    """

    checkpoint_dir = (
        Path(cfg["experiment"]["output_dir"])
        / "checkpoints"
        / cfg["experiment"]["name"]
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        "best_model.pth"
        if best
        else f"epoch_{epoch}.pth"
    )

    save_path = checkpoint_dir / filename

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = (
            scheduler.state_dict()
        )

    torch.save(
        checkpoint,
        save_path,
    )

    print(f"[INFO] Checkpoint saved: {save_path}")


def load_checkpoint(
    checkpoint_path,
    model,
    optimizer=None,
    scheduler=None,
    device="cpu",
):
    """
    Load checkpoint.

    Returns:
        epoch
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    if (
        scheduler is not None
        and "scheduler_state_dict" in checkpoint
    ):
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

    epoch = checkpoint.get("epoch", 0)

    print(f"[INFO] Loaded checkpoint: {checkpoint_path}")

    return epoch


def load_model_weights(
    checkpoint_path,
    model,
    device="cpu",
):
    """
    Load only model weights for inference.
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"[INFO] Loaded model weights: {checkpoint_path}"
    )