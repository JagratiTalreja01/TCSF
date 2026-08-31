"""
segformer_baseline.py

Native PyTorch SegFormer-B0-style baseline for multimodal SAR-optical
flood segmentation.

Compatibility
-------------
- PyTorch 1.11+
- No Hugging Face dependency
- No timm dependency
- Batch size 1 compatible
- Arbitrary multimodal input channels

Architecture
------------
- Early fusion of SAR and optical channels
- Four-stage Mix Transformer encoder
- Overlapping patch embeddings
- Efficient spatial-reduction self-attention
- Mix-FFN blocks with depthwise convolution
- Lightweight SegFormer MLP decoder
- Binary flood-segmentation output

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
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """Stochastic depth applied independently to residual branches."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()

        if not 0.0 <= drop_prob < 1.0:
            raise ValueError("drop_prob must lie in [0, 1).")

        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)

        random_tensor = keep_prob + torch.rand(
            shape,
            dtype=x.dtype,
            device=x.device,
        )
        random_tensor.floor_()

        return x.div(keep_prob) * random_tensor


class OverlapPatchEmbedding(nn.Module):
    """
    Convert an image or feature map into overlapping patch tokens.

    Returns
    -------
    tokens:
        Tensor [B, H*W, C]
    height:
        Token-grid height
    width:
        Token-grid width
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> None:
        super().__init__()

        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, int, int]:
        x = self.projection(x)
        height, width = x.shape[-2:]

        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, height, width


class EfficientSelfAttention(nn.Module):
    """
    Multi-head self-attention with spatial reduction for keys and values.

    Spatial reduction substantially lowers the attention cost in early,
    high-resolution encoder stages.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        sr_ratio: int = 1,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})."
            )
        if sr_ratio < 1:
            raise ValueError("sr_ratio must be at least 1.")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sr_ratio = sr_ratio

        self.query = nn.Linear(dim, dim, bias=qkv_bias)
        self.key_value = nn.Linear(dim, dim * 2, bias=qkv_bias)

        if sr_ratio > 1:
            self.spatial_reduction = nn.Conv2d(
                dim,
                dim,
                kernel_size=sr_ratio,
                stride=sr_ratio,
            )
            self.sr_norm = nn.LayerNorm(dim)
        else:
            self.spatial_reduction = None
            self.sr_norm = None

        self.attention_dropout = nn.Dropout(attention_dropout)
        self.projection = nn.Linear(dim, dim)
        self.projection_dropout = nn.Dropout(projection_dropout)

    def forward(
        self,
        x: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        batch_size, num_tokens, channels = x.shape

        query = self.query(x)
        query = query.reshape(
            batch_size,
            num_tokens,
            self.num_heads,
            self.head_dim,
        )
        query = query.permute(0, 2, 1, 3)

        if self.spatial_reduction is not None:
            reduced = x.transpose(1, 2).reshape(
                batch_size,
                channels,
                height,
                width,
            )
            reduced = self.spatial_reduction(reduced)
            reduced = reduced.reshape(batch_size, channels, -1)
            reduced = reduced.transpose(1, 2)
            reduced = self.sr_norm(reduced)
        else:
            reduced = x

        key_value = self.key_value(reduced)
        key_value = key_value.reshape(
            batch_size,
            -1,
            2,
            self.num_heads,
            self.head_dim,
        )
        key_value = key_value.permute(2, 0, 3, 1, 4)

        key = key_value[0]
        value = key_value[1]

        attention = torch.matmul(query, key.transpose(-2, -1))
        attention = attention * self.scale
        attention = torch.softmax(attention, dim=-1)
        attention = self.attention_dropout(attention)

        output = torch.matmul(attention, value)
        output = output.transpose(1, 2).reshape(
            batch_size,
            num_tokens,
            channels,
        )

        output = self.projection(output)
        output = self.projection_dropout(output)

        return output


class MixFeedForward(nn.Module):
    """
    SegFormer Mix-FFN.

    A depthwise 3x3 convolution injects local spatial information between
    the two linear projections.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.fc1 = nn.Linear(dim, hidden_dim)

        self.depthwise_conv = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_dim,
            bias=True,
        )

        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        batch_size, _, _ = x.shape

        x = self.fc1(x)

        x = x.transpose(1, 2).reshape(
            batch_size,
            -1,
            height,
            width,
        )
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2)

        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)

        return x


class MixTransformerBlock(nn.Module):
    """Transformer block used in each MiT encoder stage."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        sr_ratio: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()

        hidden_dim = int(dim * mlp_ratio)

        self.norm1 = nn.LayerNorm(dim)
        self.attention = EfficientSelfAttention(
            dim=dim,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
            attention_dropout=attention_dropout,
            projection_dropout=dropout,
        )

        self.norm2 = nn.LayerNorm(dim)
        self.feed_forward = MixFeedForward(
            dim=dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.drop_path = (
            DropPath(drop_path)
            if drop_path > 0.0
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        x = x + self.drop_path(
            self.attention(
                self.norm1(x),
                height,
                width,
            )
        )

        x = x + self.drop_path(
            self.feed_forward(
                self.norm2(x),
                height,
                width,
            )
        )

        return x


class MixTransformerStage(nn.Module):
    """One hierarchical stage of the MiT encoder."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        sr_ratio: int,
        patch_kernel: int,
        patch_stride: int,
        patch_padding: int,
        dropout: float,
        attention_dropout: float,
        drop_path_rates: Sequence[float],
    ) -> None:
        super().__init__()

        if len(drop_path_rates) != depth:
            raise ValueError(
                "Number of drop-path rates must equal stage depth."
            )

        self.patch_embedding = OverlapPatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            kernel_size=patch_kernel,
            stride=patch_stride,
            padding=patch_padding,
        )

        self.blocks = nn.ModuleList(
            [
                MixTransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    sr_ratio=sr_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    drop_path=drop_path_rates[index],
                )
                for index in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens, height, width = self.patch_embedding(x)

        for block in self.blocks:
            tokens = block(tokens, height, width)

        tokens = self.norm(tokens)

        feature_map = tokens.transpose(1, 2).reshape(
            tokens.shape[0],
            tokens.shape[2],
            height,
            width,
        )

        return feature_map


class MixTransformerEncoder(nn.Module):
    """
    Four-stage hierarchical MiT-B0-style encoder.

    Default B0 dimensions:
        embed_dims = (32, 64, 160, 256)
        depths = (2, 2, 2, 2)
        heads = (1, 2, 5, 8)
        sr_ratios = (8, 4, 2, 1)
    """

    def __init__(
        self,
        input_channels: int,
        embed_dims: Sequence[int] = (32, 64, 160, 256),
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (1, 2, 5, 8),
        mlp_ratios: Sequence[float] = (4.0, 4.0, 4.0, 4.0),
        sr_ratios: Sequence[int] = (8, 4, 2, 1),
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()

        sequences = [
            embed_dims,
            depths,
            num_heads,
            mlp_ratios,
            sr_ratios,
        ]

        if any(len(values) != 4 for values in sequences):
            raise ValueError(
                "embed_dims, depths, num_heads, mlp_ratios, and "
                "sr_ratios must each contain four values."
            )

        total_depth = int(sum(depths))
        all_drop_path_rates = torch.linspace(
            0.0,
            drop_path_rate,
            total_depth,
        ).tolist()

        stage_definitions = [
            {
                "in_channels": input_channels,
                "embed_dim": embed_dims[0],
                "depth": depths[0],
                "num_heads": num_heads[0],
                "mlp_ratio": mlp_ratios[0],
                "sr_ratio": sr_ratios[0],
                "patch_kernel": 7,
                "patch_stride": 4,
                "patch_padding": 3,
            },
            {
                "in_channels": embed_dims[0],
                "embed_dim": embed_dims[1],
                "depth": depths[1],
                "num_heads": num_heads[1],
                "mlp_ratio": mlp_ratios[1],
                "sr_ratio": sr_ratios[1],
                "patch_kernel": 3,
                "patch_stride": 2,
                "patch_padding": 1,
            },
            {
                "in_channels": embed_dims[1],
                "embed_dim": embed_dims[2],
                "depth": depths[2],
                "num_heads": num_heads[2],
                "mlp_ratio": mlp_ratios[2],
                "sr_ratio": sr_ratios[2],
                "patch_kernel": 3,
                "patch_stride": 2,
                "patch_padding": 1,
            },
            {
                "in_channels": embed_dims[2],
                "embed_dim": embed_dims[3],
                "depth": depths[3],
                "num_heads": num_heads[3],
                "mlp_ratio": mlp_ratios[3],
                "sr_ratio": sr_ratios[3],
                "patch_kernel": 3,
                "patch_stride": 2,
                "patch_padding": 1,
            },
        ]

        stages: List[nn.Module] = []
        depth_offset = 0

        for definition in stage_definitions:
            depth = int(definition["depth"])
            stage_rates = all_drop_path_rates[
                depth_offset : depth_offset + depth
            ]
            depth_offset += depth

            stage = MixTransformerStage(
                in_channels=int(definition["in_channels"]),
                embed_dim=int(definition["embed_dim"]),
                depth=depth,
                num_heads=int(definition["num_heads"]),
                mlp_ratio=float(definition["mlp_ratio"]),
                sr_ratio=int(definition["sr_ratio"]),
                patch_kernel=int(definition["patch_kernel"]),
                patch_stride=int(definition["patch_stride"]),
                patch_padding=int(definition["patch_padding"]),
                dropout=dropout,
                attention_dropout=attention_dropout,
                drop_path_rates=stage_rates,
            )
            stages.append(stage)

        self.stages = nn.ModuleList(stages)

        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(
                module.weight,
                mode="fan_out",
                nonlinearity="relu",
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features: List[torch.Tensor] = []

        for stage in self.stages:
            x = stage(x)
            features.append(x)

        return features


class SegFormerDecoder(nn.Module):
    """
    Lightweight all-MLP decoder from SegFormer.

    Each encoder feature is projected to a common channel dimension,
    resized to the first-stage resolution, concatenated, fused, and
    classified.
    """

    def __init__(
        self,
        encoder_channels: Sequence[int],
        decoder_channels: int = 256,
        num_classes: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if len(encoder_channels) != 4:
            raise ValueError("encoder_channels must contain four values.")

        self.projections = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels,
                    decoder_channels,
                    kernel_size=1,
                    bias=True,
                )
                for in_channels in encoder_channels
            ]
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(
                decoder_channels * 4,
                decoder_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=32,
                num_channels=decoder_channels,
            ),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )

        self.classifier = nn.Conv2d(
            decoder_channels,
            num_classes,
            kernel_size=1,
        )

    def forward(
        self,
        features: Sequence[torch.Tensor],
        output_size: Tuple[int, int],
    ) -> torch.Tensor:
        if len(features) != 4:
            raise ValueError("SegFormerDecoder expects four feature maps.")

        target_size = features[0].shape[-2:]
        projected_features: List[torch.Tensor] = []

        for feature, projection in zip(features, self.projections):
            feature = projection(feature)

            if feature.shape[-2:] != target_size:
                feature = F.interpolate(
                    feature,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )

            projected_features.append(feature)

        fused = torch.cat(projected_features, dim=1)
        fused = self.fusion(fused)
        logits = self.classifier(fused)

        logits = F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        return logits


class MultimodalSegFormer(nn.Module):
    """
    Early-fusion multimodal SegFormer-B0-style baseline.

    Parameters
    ----------
    sar_channels:
        Number of Sentinel-1 input channels.

    optical_channels:
        Number of Sentinel-2 input channels.

    num_classes:
        Number of segmentation output channels.

    embed_dims:
        Channel dimensions for the four MiT encoder stages.

    depths:
        Number of transformer blocks per encoder stage.

    num_heads:
        Attention heads per encoder stage.

    mlp_ratios:
        Mix-FFN expansion ratios.

    sr_ratios:
        Spatial-reduction ratios for efficient attention.

    decoder_channels:
        Common projection width in the SegFormer decoder.

    dropout:
        Dropout probability for transformer projections and decoder.

    attention_dropout:
        Dropout probability applied to attention weights.

    drop_path_rate:
        Maximum stochastic-depth probability.
    """

    def __init__(
        self,
        sar_channels: int = 2,
        optical_channels: int = 13,
        num_classes: int = 1,
        embed_dims: Sequence[int] = (32, 64, 160, 256),
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (1, 2, 5, 8),
        mlp_ratios: Sequence[float] = (4.0, 4.0, 4.0, 4.0),
        sr_ratios: Sequence[int] = (8, 4, 2, 1),
        decoder_channels: int = 256,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()

        if sar_channels <= 0:
            raise ValueError("sar_channels must be positive.")
        if optical_channels <= 0:
            raise ValueError("optical_channels must be positive.")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if decoder_channels <= 0:
            raise ValueError("decoder_channels must be positive.")
        if decoder_channels % 32 != 0:
            raise ValueError(
                "decoder_channels must be divisible by 32 for GroupNorm."
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1).")
        if not 0.0 <= attention_dropout < 1.0:
            raise ValueError(
                "attention_dropout must lie in [0, 1)."
            )
        if not 0.0 <= drop_path_rate < 1.0:
            raise ValueError("drop_path_rate must lie in [0, 1).")

        self.sar_channels = sar_channels
        self.optical_channels = optical_channels
        self.input_channels = sar_channels + optical_channels

        self.encoder = MixTransformerEncoder(
            input_channels=self.input_channels,
            embed_dims=embed_dims,
            depths=depths,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            sr_ratios=sr_ratios,
            dropout=dropout,
            attention_dropout=attention_dropout,
            drop_path_rate=drop_path_rate,
        )

        self.decoder = SegFormerDecoder(
            encoder_channels=embed_dims,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(
        self,
        sar: torch.Tensor,
        optical: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if sar.ndim != 4:
            raise ValueError(
                f"sar must be a 4D tensor, received shape {tuple(sar.shape)}."
            )

        if optical.ndim != 4:
            raise ValueError(
                "optical must be a 4D tensor, "
                f"received shape {tuple(optical.shape)}."
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
        logits = self.decoder(features, output_size=output_size)

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
    """Return the number of trainable model parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


if __name__ == "__main__":
    model = MultimodalSegFormer(
        sar_channels=2,
        optical_channels=13,
        num_classes=1,
    )

    sar_tensor = torch.randn(1, 2, 256, 256)
    optical_tensor = torch.randn(1, 13, 256, 256)

    with torch.no_grad():
        output = model(sar_tensor, optical_tensor)

    print("final_pred:", tuple(output["final_pred"].shape))
    print(
        "Trainable parameters:",
        f"{count_trainable_parameters(model):,}",
    )