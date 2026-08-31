"""
sen1floods11.py

Dataset loader for SEN1FLOODS11 v1.1 official structure.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from datasets.base_dataset import BaseFloodDataset


class Sen1Floods11Dataset(BaseFloodDataset):
    def __init__(self, root, split="train", transform=None):
        super().__init__(root=root, split=split, transform=transform)

    def _load_samples(self):
        hand_root = self.root / "data" / "flood_events" / "HandLabeled"

        s1_dir = hand_root / "S1Hand"
        s2_dir = hand_root / "S2Hand"
        label_dir = hand_root / "LabelHand"

        split_dir = self.root / "splits" / "flood_handlabeled"

        split_map = {
            "train": "flood_train_data.csv",
            "val": "flood_valid_data.csv",
            "valid": "flood_valid_data.csv",
            "test": "flood_test_data.csv",
        }

        split_file = split_dir / split_map[self.split]

        if not split_file.exists():
            raise FileNotFoundError(f"Missing split file: {split_file}")

        df = pd.read_csv(split_file, header=None)

        samples = []

        for value in df.iloc[:, 0].tolist():
            sample_id = Path(str(value)).stem

            sample_id = sample_id.replace("_S1Hand", "")
            sample_id = sample_id.replace("_S2Hand", "")
            sample_id = sample_id.replace("_LabelHand", "")

            s1_path = s1_dir / f"{sample_id}_S1Hand.tif"
            s2_path = s2_dir / f"{sample_id}_S2Hand.tif"
            label_path = label_dir / f"{sample_id}_LabelHand.tif"

            if s1_path.exists() and s2_path.exists() and label_path.exists():
                samples.append(
                    {
                        "id": sample_id,
                        "sar": s1_path,
                        "optical": s2_path,
                        "mask": label_path,
                    }
                )

        if len(samples) == 0:
            raise RuntimeError(
                f"No valid samples found for split={self.split}. "
                f"Check paths under {hand_root}"
            )

        return samples

    @staticmethod
    def _read_tif(path):
        with rasterio.open(path) as src:
            arr = src.read()
        return arr.astype(np.float32)

    @staticmethod
    def _normalize_sar(sar):
        sar = np.nan_to_num(sar, nan=0.0, posinf=0.0, neginf=0.0)

        mean = sar.mean(axis=(1, 2), keepdims=True)
        std = sar.std(axis=(1, 2), keepdims=True) + 1e-6

        return (sar - mean) / std

    @staticmethod
    def _normalize_optical(optical):
        optical = np.nan_to_num(optical, nan=0.0, posinf=0.0, neginf=0.0)
        optical = np.clip(optical, 0, 10000)
        optical = optical / 10000.0
        return optical

    @staticmethod
    def _process_mask(mask):
        mask = mask[0:1]
        mask = (mask > 0).astype(np.float32)
        return mask

    def _read_sample(self, sample_info):
        sar = self._read_tif(sample_info["sar"])
        optical = self._read_tif(sample_info["optical"])
        mask = self._read_tif(sample_info["mask"])

        sar = self._normalize_sar(sar)
        optical = self._normalize_optical(optical)
        mask = self._process_mask(mask)

        return {
            "sar": self.to_tensor(sar),
            "optical": self.to_tensor(optical),
            "mask": self.to_tensor(mask),
            "meta": {
                "id": sample_info["id"],
                "sar_path": str(sample_info["sar"]),
                "optical_path": str(sample_info["optical"]),
                "mask_path": str(sample_info["mask"]),
            },
        }