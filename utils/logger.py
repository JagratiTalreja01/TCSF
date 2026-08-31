"""
logger.py

Logging utilities for ACSF.
Handles console logging, file logging, and TensorBoard logging.
"""

import logging
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def setup_logger(cfg):
    """
    Create logger and TensorBoard writer.

    Returns:
        logger
        writer
    """

    exp_name = cfg["experiment"]["name"]
    output_dir = Path(cfg["experiment"]["output_dir"])

    log_dir = output_dir / "logs" / exp_name
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "train.log"

    logger = logging.getLogger(exp_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    writer = None

    if cfg["logging"]["tensorboard"]:
        tb_dir = log_dir / "tensorboard"
        writer = SummaryWriter(log_dir=str(tb_dir))

    logger.info(f"Logger initialized: {log_file}")

    return logger, writer


def log_metrics(logger, metrics, epoch, prefix="Train"):
    """
    Log metrics dictionary.

    Args:
        logger: Python logger
        metrics: dict
        epoch: int
        prefix: Train / Val / Test
    """

    metric_str = " | ".join(
        [f"{key}: {value:.4f}" for key, value in metrics.items()]
    )

    logger.info(f"Epoch {epoch} [{prefix}] {metric_str}")


def write_tensorboard(writer, metrics, epoch, prefix="Train"):
    """
    Write metrics to TensorBoard.
    """

    if writer is None:
        return

    for key, value in metrics.items():
        writer.add_scalar(f"{prefix}/{key}", value, epoch)