"""
base_dataset.py

Base dataset class for SAR-optical flood mapping datasets.
All dataset-specific loaders should inherit from this class.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch.utils.data import Dataset


class BaseFloodDataset(Dataset, ABC):
    """
    Abstract base class for flood mapping datasets.

    Expected output from every dataset:
        sample = {
            "sar": Tensor [C_sar, H, W],
            "optical": Tensor [C_optical, H, W],
            "mask": Tensor [1, H, W],
            "meta": dict
        }
    """

    def __init__(self, root, split="train", transform=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        self.samples = self._load_samples()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples found for split='{split}' in {self.root}"
            )

    @abstractmethod
    def _load_samples(self):
        """
        Load dataset file paths.

        Returns:
            list[dict]: Each item should contain paths for SAR, optical, mask, etc.
        """
        pass

    @abstractmethod
    def _read_sample(self, sample_info):
        """
        Read one sample from disk.

        Args:
            sample_info (dict): File paths and metadata for one sample.

        Returns:
            dict: Raw sample containing SAR, optical, mask, and meta.
        """
        pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        sample = self._read_sample(sample_info)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    @staticmethod
    def to_tensor(array):
        """
        Convert numpy array to torch tensor.

        Accepts:
            [H, W]
            [H, W, C]
            [C, H, W]

        Returns:
            torch.FloatTensor
        """

        tensor = torch.from_numpy(array).float()

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)

        elif tensor.ndim == 3:
            # If channel-last, convert to channel-first
            if tensor.shape[-1] <= 20:
                tensor = tensor.permute(2, 0, 1)

        return tensor