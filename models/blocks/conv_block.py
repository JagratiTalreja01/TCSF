"""
conv_block.py

Reusable convolution blocks for ACSF.
Used in encoders, decoders, prediction heads, and baseline models.
"""

import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    """
    Convolution + BatchNorm + Activation block.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        activation=True,
    ):
        super().__init__()

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]

        if activation:
            layers.append(nn.SiLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DoubleConv(nn.Module):
    """
    Two ConvBNAct blocks.
    Commonly used in segmentation models.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            ConvBNAct(in_channels, out_channels),
            ConvBNAct(out_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    """
    Downsampling block using stride-2 convolution.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            ConvBNAct(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            ConvBNAct(out_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """
    Upsampling block using bilinear interpolation + convolution.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.conv = DoubleConv(
            in_channels + skip_channels,
            out_channels,
        )

    def forward(self, x, skip=None):
        x = torch.nn.functional.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        if skip is not None:
            x = torch.cat([x, skip], dim=1)

        return self.conv(x)