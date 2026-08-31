"""
dataloader.py

Factory functions for creating PyTorch DataLoaders.
"""

from torch.utils.data import DataLoader

from datasets.sen1floods11 import Sen1Floods11Dataset
from datasets.transforms import FloodTransforms


def build_dataset(cfg, split="train"):
    """
    Build dataset based on configuration.

    Args:
        cfg (dict): Configuration dictionary.
        split (str): train / val / test

    Returns:
        torch.utils.data.Dataset
    """

    dataset_name = cfg["dataset"]["name"].lower()

    image_size = cfg["dataset"]["image_size"]

    if split == "train":
        transform = FloodTransforms.get_train_transforms(image_size)
    else:
        transform = FloodTransforms.get_val_transforms(image_size)

    if dataset_name == "sen1floods11":
        dataset = Sen1Floods11Dataset(
            root=cfg["dataset"]["root"],
            split=split,
            transform=transform,
        )

    else:
        raise NotImplementedError(
            f"Dataset '{dataset_name}' is not implemented."
        )

    return dataset


def build_dataloader(cfg, split="train"):
    """
    Build DataLoader.

    Args:
        cfg (dict)
        split (str)

    Returns:
        torch.utils.data.DataLoader
    """

    dataset = build_dataset(cfg, split)

    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=(split == "train"),
        num_workers=cfg["dataset"]["num_workers"],
        pin_memory=cfg["dataset"]["pin_memory"],
        drop_last=(split == "train"),
    )

    return loader