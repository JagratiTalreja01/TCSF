"""
Multimodal U-Net baseline for SAR-optical flood segmentation.

Inputs
------
sar:
    Tensor [B, sar_channels, H, W]
optical:
    Tensor [B, optical_channels, H, W]

Output
------
A dictionary compatible with the ACSF/TCSF evaluation interface:
{
    "final_pred": Tensor [B, num_classes, H, W],
    "sar_pred": None,
    "optical_pred": None,
    "fused_pred": None,
    "sar_conf": None,
    "optical_conf": None,
    "decision_weights": None,
    "initial_fused": None,
    "transition_scales": None,
}

The model performs early/data-level fusion by concatenating SAR and optical
channels before the first U-Net encoder block.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two 3x3 convolution blocks with BatchNorm and ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]

        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))

        layers.extend(
            [
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
        )

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """Spatial downsampling followed by a DoubleConv block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(
                in_channels=in_channels,
                out_channels=out_channels,
                dropout=dropout,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Upsampling, skip concatenation, and DoubleConv refinement."""

    def __init__(
        self,
        decoder_channels: int,
        skip_channels: int,
        out_channels: int,
        bilinear: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
            )
            upsampled_channels = decoder_channels
        else:
            self.up = nn.ConvTranspose2d(
                decoder_channels,
                decoder_channels // 2,
                kernel_size=2,
                stride=2,
            )
            upsampled_channels = decoder_channels // 2

        self.conv = DoubleConv(
            in_channels=upsampled_channels + skip_channels,
            out_channels=out_channels,
            dropout=dropout,
        )

    @staticmethod
    def _match_spatial_size(
        x: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """
        Resize x to match the skip tensor if odd image dimensions produce
        a one-pixel mismatch.
        """
        if x.shape[-2:] != reference.shape[-2:]:
            x = F.interpolate(
                x,
                size=reference.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        return x

    def forward(
        self,
        decoder_feature: torch.Tensor,
        skip_feature: torch.Tensor,
    ) -> torch.Tensor:
        decoder_feature = self.up(decoder_feature)
        decoder_feature = self._match_spatial_size(
            decoder_feature,
            skip_feature,
        )

        x = torch.cat([skip_feature, decoder_feature], dim=1)
        return self.conv(x)


class MultimodalUNet(nn.Module):
    """
    Standard U-Net using early fusion of SAR and optical channels.

    Parameters
    ----------
    sar_channels:
        Number of Sentinel-1 channels.
    optical_channels:
        Number of Sentinel-2 channels.
    base_channels:
        Width of the first encoder stage.
    num_classes:
        Number of output segmentation channels.
    bilinear:
        Use bilinear interpolation instead of transposed convolutions.
    dropout:
        Dropout probability used in deeper encoder/decoder blocks.
    """

    def __init__(
        self,
        sar_channels: int = 2,
        optical_channels: int = 13,
        base_channels: int = 32,
        num_classes: int = 1,
        bilinear: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if sar_channels <= 0:
            raise ValueError("sar_channels must be positive.")
        if optical_channels <= 0:
            raise ValueError("optical_channels must be positive.")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive.")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1).")

        self.sar_channels = sar_channels
        self.optical_channels = optical_channels
        self.input_channels = sar_channels + optical_channels

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        self.encoder1 = DoubleConv(
            self.input_channels,
            c1,
            dropout=0.0,
        )
        self.encoder2 = DownBlock(
            c1,
            c2,
            dropout=0.0,
        )
        self.encoder3 = DownBlock(
            c2,
            c3,
            dropout=dropout,
        )
        self.encoder4 = DownBlock(
            c3,
            c4,
            dropout=dropout,
        )
        self.bottleneck = DownBlock(
            c4,
            c5,
            dropout=dropout,
        )

        self.decoder4 = UpBlock(
            decoder_channels=c5,
            skip_channels=c4,
            out_channels=c4,
            bilinear=bilinear,
            dropout=dropout,
        )
        self.decoder3 = UpBlock(
            decoder_channels=c4,
            skip_channels=c3,
            out_channels=c3,
            bilinear=bilinear,
            dropout=dropout,
        )
        self.decoder2 = UpBlock(
            decoder_channels=c3,
            skip_channels=c2,
            out_channels=c2,
            bilinear=bilinear,
            dropout=0.0,
        )
        self.decoder1 = UpBlock(
            decoder_channels=c2,
            skip_channels=c1,
            out_channels=c1,
            bilinear=bilinear,
            dropout=0.0,
        )

        self.segmentation_head = nn.Conv2d(
            c1,
            num_classes,
            kernel_size=1,
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Kaiming initialization for convolutional layers."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _validate_inputs(
        self,
        sar: torch.Tensor,
        optical: torch.Tensor,
    ) -> None:
        if sar.ndim != 4 or optical.ndim != 4:
            raise ValueError(
                "SAR and optical inputs must both have shape [B, C, H, W]."
            )

        if sar.shape[0] != optical.shape[0]:
            raise ValueError("SAR and optical batch sizes must match.")

        if sar.shape[-2:] != optical.shape[-2:]:
            raise ValueError("SAR and optical spatial dimensions must match.")

        if sar.shape[1] != self.sar_channels:
            raise ValueError(
                f"Expected {self.sar_channels} SAR channels, "
                f"received {sar.shape[1]}."
            )

        if optical.shape[1] != self.optical_channels:
            raise ValueError(
                f"Expected {self.optical_channels} optical channels, "
                f"received {optical.shape[1]}."
            )

    def forward(
        self,
        sar: torch.Tensor,
        optical: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        self._validate_inputs(sar, optical)

        x = torch.cat([sar, optical], dim=1)

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        bottleneck = self.bottleneck(e4)

        d4 = self.decoder4(bottleneck, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1)

        final_pred = self.segmentation_head(d1)

        return {
            "final_pred": final_pred,
            "sar_pred": None,
            "optical_pred": None,
            "fused_pred": None,
            "sar_conf": None,
            "optical_conf": None,
            "decision_weights": None,
            "initial_fused": None,
            "transition_scales": None,
        }


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


if __name__ == "__main__":
    model = MultimodalUNet(
        sar_channels=2,
        optical_channels=13,
        base_channels=32,
        num_classes=1,
    )

    sar = torch.randn(2, 2, 256, 256)
    optical = torch.randn(2, 13, 256, 256)

    outputs = model(sar, optical)

    print("final_pred:", tuple(outputs["final_pred"].shape))
    print("trainable_parameters:", count_parameters(model))