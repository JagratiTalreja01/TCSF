"""
vision_mamba_baseline.py

Hierarchical Mamba-only baseline for multimodal SAR-optical flood
segmentation.

Important
---------
This baseline deliberately reuses the project's existing lightweight
Vision-Mamba-style ``MambaBlock`` from:

    models/blocks/mamba_block.py

It is therefore not the official VMamba implementation. Its purpose is to
provide a controlled Mamba-only comparison against ACSF/TCSF using the same
state-space-inspired building block already used in the project.

Architecture
------------
- Early fusion of SAR and optical channels
- Convolutional stem
- Four-stage hierarchical Mamba encoder
- Strided-convolution downsampling
- U-Net-style multi-scale decoder
- Binary flood-segmentation output

Inputs
------
sar:
    Tensor [B, sar_channels, H, W]

optical:
    Tensor [B, optical_channels, H, W]

Output
------
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
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from models.blocks.mamba_block import MambaBlock


def valid_group_count(
    num_channels: int,
    max_groups: int = 32,
) -> int:
    """Return the largest GroupNorm group count dividing num_channels."""
    for groups in range(
        min(max_groups, num_channels),
        0,
        -1,
    ):
        if num_channels % groups == 0:
            return groups

    return 1


class ConvNormActivation(nn.Sequential):
    """Convolution followed by GroupNorm and GELU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=valid_group_count(out_channels),
                num_channels=out_channels,
            ),
            nn.GELU(),
        )


class MambaStage(nn.Module):
    """A stack of lightweight 2D Mamba-style blocks."""

    def __init__(
        self,
        dim: int,
        depth: int,
        expansion: int = 2,
        kernel_size: int = 7,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if depth <= 0:
            raise ValueError("depth must be positive.")

        self.blocks = nn.Sequential(
            *[
                MambaBlock(
                    dim=dim,
                    expansion=expansion,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class DownsampleBlock(nn.Module):
    """Learned spatial downsampling between encoder stages."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.block = ConvNormActivation(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class HierarchicalMambaEncoder(nn.Module):
    """Four-stage hierarchical encoder built only from Mamba blocks."""

    def __init__(
        self,
        input_channels: int,
        stage_channels: Sequence[int] = (32, 64, 128, 256),
        depths: Sequence[int] = (2, 2, 3, 2),
        expansion: int = 2,
        kernel_size: int = 7,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if len(stage_channels) != 4:
            raise ValueError(
                "stage_channels must contain four values."
            )

        if len(depths) != 4:
            raise ValueError("depths must contain four values.")

        self.stem = nn.Sequential(
            ConvNormActivation(
                in_channels=input_channels,
                out_channels=stage_channels[0],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            ConvNormActivation(
                in_channels=stage_channels[0],
                out_channels=stage_channels[0],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

        self.stage1 = MambaStage(
            dim=stage_channels[0],
            depth=depths[0],
            expansion=expansion,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        self.down1 = DownsampleBlock(
            stage_channels[0],
            stage_channels[1],
        )

        self.stage2 = MambaStage(
            dim=stage_channels[1],
            depth=depths[1],
            expansion=expansion,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        self.down2 = DownsampleBlock(
            stage_channels[1],
            stage_channels[2],
        )

        self.stage3 = MambaStage(
            dim=stage_channels[2],
            depth=depths[2],
            expansion=expansion,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        self.down3 = DownsampleBlock(
            stage_channels[2],
            stage_channels[3],
        )

        self.stage4 = MambaStage(
            dim=stage_channels[3],
            depth=depths[3],
            expansion=expansion,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feature1 = self.stage1(self.stem(x))
        feature2 = self.stage2(self.down1(feature1))
        feature3 = self.stage3(self.down2(feature2))
        feature4 = self.stage4(self.down3(feature3))

        return [
            feature1,
            feature2,
            feature3,
            feature4,
        ]


class DecoderBlock(nn.Module):
    """Upsample, fuse a skip feature, and refine with Mamba blocks."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        mamba_depth: int = 1,
        expansion: int = 2,
        kernel_size: int = 7,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_projection = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )

        self.skip_projection = nn.Conv2d(
            skip_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )

        self.fusion = nn.Sequential(
            ConvNormActivation(
                in_channels=out_channels * 2,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            MambaStage(
                dim=out_channels,
                depth=mamba_depth,
                expansion=expansion,
                kernel_size=kernel_size,
                dropout=dropout,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        x = self.input_projection(x)
        skip = self.skip_projection(skip)

        x = torch.cat([x, skip], dim=1)
        return self.fusion(x)


class MambaUNetDecoder(nn.Module):
    """Multi-scale U-Net decoder with Mamba refinement."""

    def __init__(
        self,
        encoder_channels: Sequence[int],
        decoder_channels: Sequence[int] = (128, 64, 32),
        decoder_depths: Sequence[int] = (1, 1, 1),
        num_classes: int = 1,
        expansion: int = 2,
        kernel_size: int = 7,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if len(encoder_channels) != 4:
            raise ValueError(
                "encoder_channels must contain four values."
            )

        if len(decoder_channels) != 3:
            raise ValueError(
                "decoder_channels must contain three values."
            )

        if len(decoder_depths) != 3:
            raise ValueError(
                "decoder_depths must contain three values."
            )

        self.decoder3 = DecoderBlock(
            in_channels=encoder_channels[3],
            skip_channels=encoder_channels[2],
            out_channels=decoder_channels[0],
            mamba_depth=decoder_depths[0],
            expansion=expansion,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        self.decoder2 = DecoderBlock(
            in_channels=decoder_channels[0],
            skip_channels=encoder_channels[1],
            out_channels=decoder_channels[1],
            mamba_depth=decoder_depths[1],
            expansion=expansion,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        self.decoder1 = DecoderBlock(
            in_channels=decoder_channels[1],
            skip_channels=encoder_channels[0],
            out_channels=decoder_channels[2],
            mamba_depth=decoder_depths[2],
            expansion=expansion,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        self.segmentation_head = nn.Sequential(
            ConvNormActivation(
                in_channels=decoder_channels[2],
                out_channels=decoder_channels[2],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Conv2d(
                decoder_channels[2],
                num_classes,
                kernel_size=1,
            ),
        )

    def forward(
        self,
        features: Sequence[torch.Tensor],
        output_size: Tuple[int, int],
    ) -> torch.Tensor:
        if len(features) != 4:
            raise ValueError(
                "MambaUNetDecoder expects four feature maps."
            )

        feature1, feature2, feature3, feature4 = features

        x = self.decoder3(feature4, feature3)
        x = self.decoder2(x, feature2)
        x = self.decoder1(x, feature1)

        logits = self.segmentation_head(x)

        if logits.shape[-2:] != output_size:
            logits = F.interpolate(
                logits,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )

        return logits


class MultimodalVisionMamba(nn.Module):
    """
    Early-fusion hierarchical Mamba-only segmentation baseline.

    The same project-level ``MambaBlock`` is used in both the encoder and
    decoder so the experiment isolates hierarchical Mamba processing without
    reliability estimation, cross-state fusion, auxiliary heads, or adaptive
    decision fusion.
    """

    def __init__(
        self,
        sar_channels: int = 2,
        optical_channels: int = 13,
        num_classes: int = 1,
        stage_channels: Sequence[int] = (32, 64, 128, 256),
        depths: Sequence[int] = (2, 2, 3, 2),
        decoder_channels: Sequence[int] = (128, 64, 32),
        decoder_depths: Sequence[int] = (1, 1, 1),
        expansion: int = 2,
        kernel_size: int = 7,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if sar_channels <= 0:
            raise ValueError("sar_channels must be positive.")
        if optical_channels <= 0:
            raise ValueError("optical_channels must be positive.")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if expansion <= 0:
            raise ValueError("expansion must be positive.")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                "kernel_size must be a positive odd integer."
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1).")

        self.sar_channels = sar_channels
        self.optical_channels = optical_channels
        self.input_channels = sar_channels + optical_channels

        self.encoder = HierarchicalMambaEncoder(
            input_channels=self.input_channels,
            stage_channels=stage_channels,
            depths=depths,
            expansion=expansion,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        self.decoder = MambaUNetDecoder(
            encoder_channels=stage_channels,
            decoder_channels=decoder_channels,
            decoder_depths=decoder_depths,
            num_classes=num_classes,
            expansion=expansion,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    def forward(
        self,
        sar: torch.Tensor,
        optical: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if sar.ndim != 4:
            raise ValueError(
                f"sar must be 4D, received {tuple(sar.shape)}."
            )

        if optical.ndim != 4:
            raise ValueError(
                "optical must be 4D, "
                f"received {tuple(optical.shape)}."
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

        output_size = sar.shape[-2:]
        fused_input = torch.cat([sar, optical], dim=1)

        features = self.encoder(fused_input)
        logits = self.decoder(
            features,
            output_size=output_size,
        )

        return {
            "final_pred": logits,
            "sar_pred": None,
            "optical_pred": None,
            "fused_pred": None,
            "sar_conf": None,
            "optical_conf": None,
            "decision_weights": None,
            "initial_fused": None,
            "transition_scales": None,
        }


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


if __name__ == "__main__":
    model = MultimodalVisionMamba(
        sar_channels=2,
        optical_channels=13,
        num_classes=1,
    )

    sar_tensor = torch.randn(1, 2, 256, 256)
    optical_tensor = torch.randn(1, 13, 256, 256)

    with torch.no_grad():
        outputs = model(sar_tensor, optical_tensor)

    print(
        "final_pred:",
        tuple(outputs["final_pred"].shape),
    )
    print(
        "Trainable parameters:",
        f"{count_trainable_parameters(model):,}",
    )