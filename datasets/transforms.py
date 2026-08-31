"""
transforms.py

Custom multimodal transforms for SAR-optical flood mapping.

Works with samples shaped as:
{
    "sar": Tensor [C_sar, H, W],
    "optical": Tensor [C_opt, H, W],
    "mask": Tensor [1, H, W],
    "meta": dict
}
"""

import random
import torch
import torch.nn.functional as F


class Resize:
    def __init__(self, size):
        self.size = size

    def _resize_image(self, x, mode="bilinear"):
        x = x.unsqueeze(0)

        x = F.interpolate(
            x,
            size=(self.size, self.size),
            mode=mode,
            align_corners=False if mode == "bilinear" else None,
        )

        return x.squeeze(0)

    def __call__(self, sample):
        sample["sar"] = self._resize_image(sample["sar"], mode="bilinear")
        sample["optical"] = self._resize_image(sample["optical"], mode="bilinear")
        sample["mask"] = self._resize_image(sample["mask"], mode="nearest")

        return sample


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            sample["sar"] = torch.flip(sample["sar"], dims=[2])
            sample["optical"] = torch.flip(sample["optical"], dims=[2])
            sample["mask"] = torch.flip(sample["mask"], dims=[2])

        return sample


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            sample["sar"] = torch.flip(sample["sar"], dims=[1])
            sample["optical"] = torch.flip(sample["optical"], dims=[1])
            sample["mask"] = torch.flip(sample["mask"], dims=[1])

        return sample


class RandomRotate90:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            k = random.randint(1, 3)

            sample["sar"] = torch.rot90(sample["sar"], k, dims=[1, 2])
            sample["optical"] = torch.rot90(sample["optical"], k, dims=[1, 2])
            sample["mask"] = torch.rot90(sample["mask"], k, dims=[1, 2])

        return sample


class OpticalBrightnessNoise:
    """
    Apply brightness noise only to optical bands.
    Do not apply this to SAR or mask.
    """

    def __init__(self, p=0.3, scale=0.05):
        self.p = p
        self.scale = scale

    def __call__(self, sample):
        if random.random() < self.p:
            noise = torch.randn_like(sample["optical"]) * self.scale
            sample["optical"] = torch.clamp(sample["optical"] + noise, 0.0, 1.0)

        return sample


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)

        return sample


class FloodTransforms:
    @staticmethod
    def get_train_transforms(image_size=256):
        return Compose(
            [
                Resize(image_size),
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                RandomRotate90(p=0.5),
                OpticalBrightnessNoise(p=0.3, scale=0.05),
            ]
        )

    @staticmethod
    def get_val_transforms(image_size=256):
        return Compose(
            [
                Resize(image_size),
            ]
        )

    @staticmethod
    def get_test_transforms(image_size=256):
        return Compose(
            [
                Resize(image_size),
            ]
        )