"""
deeplabv3plus_baseline.py

Multimodal DeepLabV3+ baseline for SAR-optical flood segmentation.

Compatible with:
- torch 1.11 / torchvision 0.12
- newer torchvision versions using the weights= API

Architecture
------------
- Early fusion of 2-channel SAR and 13-channel optical input
- ResNet-50 encoder
- Output stride 16
- Atrous Spatial Pyramid Pooling (ASPP)
- DeepLabV3+ low-level feature decoder
- Binary flood-segmentation output
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

try:
    from torchvision.models import ResNet50_Weights
except ImportError:
    ResNet50_Weights = None


def build_resnet50_backbone(
    pretrained: bool,
    replace_stride_with_dilation,
):
    """
    Build ResNet-50 using the appropriate torchvision API.

    torchvision <= 0.12:
        resnet50(pretrained=...)

    newer torchvision:
        resnet50(weights=...)
    """
    if ResNet50_Weights is not None:
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        return resnet50(
            weights=weights,
            replace_stride_with_dilation=replace_stride_with_dilation,
        )

    return resnet50(
        pretrained=pretrained,
        replace_stride_with_dilation=replace_stride_with_dilation,
    )


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ASPPConv(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int,
    ) -> None:
        super().__init__(
            ConvBNReLU(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            )
        )


class ASPPPooling(nn.Module):
    """
    Global image-pooling branch.

    GroupNorm is used instead of BatchNorm because the pooled feature has
    spatial size 1x1. BatchNorm fails during training when batch size is 1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=32,
                num_channels=out_channels,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_size = x.shape[-2:]

        pooled = self.pool(x)
        pooled = self.projection(pooled)

        return F.interpolate(
            pooled,
            size=spatial_size,
            mode="bilinear",
            align_corners=False,
        )


class ASPP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        atrous_rates: Tuple[int, int, int] = (6, 12, 18),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        branches = [
            ConvBNReLU(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )
        ]

        for rate in atrous_rates:
            branches.append(
                ASPPConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dilation=rate,
                )
            )

        branches.append(
            ASPPPooling(
                in_channels=in_channels,
                out_channels=out_channels,
            )
        )

        self.branches = nn.ModuleList(branches)

        self.project = nn.Sequential(
            ConvBNReLU(
                in_channels=out_channels * len(branches),
                out_channels=out_channels,
                kernel_size=1,
            ),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [branch(x) for branch in self.branches]
        x = torch.cat(features, dim=1)
        return self.project(x)


class ResNet50Encoder(nn.Module):
    """
    ResNet-50 encoder modified for arbitrary multimodal input channels.

    Output stride 16 is obtained by replacing the final stage stride
    with dilation.
    """

    def __init__(
        self,
        input_channels: int,
        pretrained: bool = False,
    ) -> None:
        super().__init__()

        backbone = build_resnet50_backbone(
            pretrained=pretrained,
            replace_stride_with_dilation=[False, False, True],
        )

        original_conv = backbone.conv1

        backbone.conv1 = nn.Conv2d(
            input_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )

        if pretrained:
            self._initialize_multimodal_conv(
                new_conv=backbone.conv1,
                original_weight=original_conv.weight.detach(),
                input_channels=input_channels,
            )
        else:
            nn.init.kaiming_normal_(
                backbone.conv1.weight,
                mode="fan_out",
                nonlinearity="relu",
            )

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    @staticmethod
    def _initialize_multimodal_conv(
        new_conv: nn.Conv2d,
        original_weight: torch.Tensor,
        input_channels: int,
    ) -> None:
        channel_mean = original_weight.mean(dim=1, keepdim=True)

        expanded = channel_mean.repeat(
            1,
            input_channels,
            1,
            1,
        )

        expanded *= 3.0 / float(input_channels)

        with torch.no_grad():
            new_conv.weight.copy_(expanded)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)

        low_level = self.layer1(x)
        x = self.layer2(low_level)
        x = self.layer3(x)
        high_level = self.layer4(x)

        return low_level, high_level


class DeepLabV3PlusDecoder(nn.Module):
    def __init__(
        self,
        low_level_channels: int = 256,
        high_level_channels: int = 2048,
        aspp_channels: int = 256,
        low_level_projection_channels: int = 48,
        decoder_channels: int = 256,
        num_classes: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.aspp = ASPP(
            in_channels=high_level_channels,
            out_channels=aspp_channels,
            dropout=dropout,
        )

        self.low_level_projection = ConvBNReLU(
            in_channels=low_level_channels,
            out_channels=low_level_projection_channels,
            kernel_size=1,
        )

        fused_channels = (
            aspp_channels
            + low_level_projection_channels
        )

        self.refine = nn.Sequential(
            ConvBNReLU(
                in_channels=fused_channels,
                out_channels=decoder_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.Dropout(dropout),
            ConvBNReLU(
                in_channels=decoder_channels,
                out_channels=decoder_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Conv2d(
            decoder_channels,
            num_classes,
            kernel_size=1,
        )

    def forward(
        self,
        low_level: torch.Tensor,
        high_level: torch.Tensor,
        output_size: Tuple[int, int],
    ) -> torch.Tensor:
        high_level = self.aspp(high_level)
        low_level = self.low_level_projection(low_level)

        high_level = F.interpolate(
            high_level,
            size=low_level.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        x = torch.cat(
            [low_level, high_level],
            dim=1,
        )

        x = self.refine(x)
        x = self.classifier(x)

        return F.interpolate(
            x,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )


class MultimodalDeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ using early fusion of SAR and optical channels.
    """

    def __init__(
        self,
        sar_channels: int = 2,
        optical_channels: int = 13,
        num_classes: int = 1,
        pretrained_backbone: bool = False,
        aspp_channels: int = 256,
        decoder_channels: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if sar_channels <= 0:
            raise ValueError("sar_channels must be positive.")
        if optical_channels <= 0:
            raise ValueError("optical_channels must be positive.")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")

        self.sar_channels = sar_channels
        self.optical_channels = optical_channels
        self.input_channels = sar_channels + optical_channels

        self.encoder = ResNet50Encoder(
            input_channels=self.input_channels,
            pretrained=pretrained_backbone,
        )

        self.decoder = DeepLabV3PlusDecoder(
            low_level_channels=256,
            high_level_channels=2048,
            aspp_channels=aspp_channels,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
            dropout=dropout,
        )

        self._initialize_decoder_weights()

    def _initialize_decoder_weights(self) -> None:
        for module in self.decoder.modules():
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
                "SAR and optical tensors must have shape [B,C,H,W]."
            )

        if sar.shape[0] != optical.shape[0]:
            raise ValueError(
                "SAR and optical batch sizes must match."
            )

        if sar.shape[-2:] != optical.shape[-2:]:
            raise ValueError(
                "SAR and optical spatial dimensions must match."
            )

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

        output_size = sar.shape[-2:]

        x = torch.cat(
            [sar, optical],
            dim=1,
        )

        low_level, high_level = self.encoder(x)

        final_pred = self.decoder(
            low_level=low_level,
            high_level=high_level,
            output_size=output_size,
        )

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
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


if __name__ == "__main__":
    model = MultimodalDeepLabV3Plus(
        sar_channels=2,
        optical_channels=13,
        num_classes=1,
        pretrained_backbone=False,
    )

    sar = torch.randn(2, 2, 256, 256)
    optical = torch.randn(2, 13, 256, 256)

    outputs = model(sar, optical)

    print("final_pred:", tuple(outputs["final_pred"].shape))
    print("trainable_parameters:", count_parameters(model))