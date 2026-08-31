"""
swin_unet_baseline.py

Native PyTorch Swin-UNet-style baseline for multimodal SAR-optical
flood segmentation.

Compatibility
-------------
- PyTorch 1.11+
- No timm dependency
- No Hugging Face dependency
- Batch size 1 compatible
- Arbitrary multimodal input channels

Architecture
------------
- Early fusion of SAR and optical channels
- Hierarchical Swin Transformer encoder
- Shifted-window self-attention
- Patch merging between encoder stages
- Lightweight U-Net-style decoder with skip connections
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

import torch
import torch.nn as nn
import torch.nn.functional as F


def valid_group_count(num_channels: int, max_groups: int = 32) -> int:
    """Return the largest GroupNorm group count that divides num_channels."""
    for groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


class DropPath(nn.Module):
    """Stochastic depth for residual branches."""

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


def window_partition(
    x: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    """
    Partition BHWC feature maps into non-overlapping windows.

    Returns
    -------
    Tensor:
        [num_windows * B, window_size, window_size, C]
    """
    batch_size, height, width, channels = x.shape

    x = x.view(
        batch_size,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        channels,
    )

    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()

    return windows.view(
        -1,
        window_size,
        window_size,
        channels,
    )


def window_reverse(
    windows: torch.Tensor,
    window_size: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Reverse window partitioning into a BHWC feature map."""
    windows_per_image = (
        height // window_size
    ) * (
        width // window_size
    )

    batch_size = windows.shape[0] // windows_per_image

    x = windows.view(
        batch_size,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        -1,
    )

    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()

    return x.view(
        batch_size,
        height,
        width,
        -1,
    )


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with relative position bias."""

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})."
            )

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        relative_size = 2 * window_size - 1
        table_size = relative_size * relative_size

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(table_size, num_heads)
        )

        coordinates_h = torch.arange(window_size)
        coordinates_w = torch.arange(window_size)
        coordinates = torch.stack(
            torch.meshgrid(coordinates_h, coordinates_w, indexing="ij")
        )
        coordinates_flat = torch.flatten(coordinates, 1)

        relative_coordinates = (
            coordinates_flat[:, :, None]
            - coordinates_flat[:, None, :]
        )
        relative_coordinates = relative_coordinates.permute(
            1,
            2,
            0,
        ).contiguous()

        relative_coordinates[:, :, 0] += window_size - 1
        relative_coordinates[:, :, 1] += window_size - 1
        relative_coordinates[:, :, 0] *= relative_size

        relative_position_index = relative_coordinates.sum(-1)

        self.register_buffer(
            "relative_position_index",
            relative_position_index,
        )

        self.qkv = nn.Linear(
            dim,
            dim * 3,
            bias=qkv_bias,
        )

        self.attention_dropout = nn.Dropout(attention_dropout)
        self.projection = nn.Linear(dim, dim)
        self.projection_dropout = nn.Dropout(projection_dropout)

        nn.init.trunc_normal_(
            self.relative_position_bias_table,
            std=0.02,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_windows, num_tokens, channels = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(
            batch_windows,
            num_tokens,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)

        query, key, value = qkv[0], qkv[1], qkv[2]
        query = query * self.scale

        attention = query @ key.transpose(-2, -1)

        relative_position_bias = (
            self.relative_position_bias_table[
                self.relative_position_index.reshape(-1)
            ]
            .view(
                num_tokens,
                num_tokens,
                -1,
            )
            .permute(2, 0, 1)
            .contiguous()
        )

        attention = attention + relative_position_bias.unsqueeze(0)

        if mask is not None:
            num_windows = mask.shape[0]

            attention = attention.view(
                batch_windows // num_windows,
                num_windows,
                self.num_heads,
                num_tokens,
                num_tokens,
            )

            attention = attention + mask.unsqueeze(0).unsqueeze(2)
            attention = attention.view(
                -1,
                self.num_heads,
                num_tokens,
                num_tokens,
            )

        attention = torch.softmax(attention, dim=-1)
        attention = self.attention_dropout(attention)

        output = attention @ value
        output = output.transpose(1, 2).reshape(
            batch_windows,
            num_tokens,
            channels,
        )

        output = self.projection(output)
        output = self.projection_dropout(output)

        return output


class MLP(nn.Module):
    """Feed-forward network used inside Swin Transformer blocks."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)

        return x


class SwinTransformerBlock(nn.Module):
    """Standard or shifted-window Swin Transformer block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 8,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()

        if shift_size < 0 or shift_size >= window_size:
            raise ValueError(
                "shift_size must satisfy 0 <= shift_size < window_size."
            )

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)

        self.attention = WindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            projection_dropout=dropout,
        )

        self.drop_path = (
            DropPath(drop_path)
            if drop_path > 0.0
            else nn.Identity()
        )

        self.norm2 = nn.LayerNorm(dim)

        self.mlp = MLP(
            dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            dropout=dropout,
        )

    def _build_attention_mask(
        self,
        padded_height: int,
        padded_width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if self.shift_size == 0:
            return None

        image_mask = torch.zeros(
            (1, padded_height, padded_width, 1),
            device=device,
            dtype=dtype,
        )

        height_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )

        width_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )

        region_id = 0

        for height_slice in height_slices:
            for width_slice in width_slices:
                image_mask[
                    :,
                    height_slice,
                    width_slice,
                    :,
                ] = region_id
                region_id += 1

        mask_windows = window_partition(
            image_mask,
            self.window_size,
        )
        mask_windows = mask_windows.view(
            -1,
            self.window_size * self.window_size,
        )

        attention_mask = (
            mask_windows.unsqueeze(1)
            - mask_windows.unsqueeze(2)
        )

        attention_mask = attention_mask.masked_fill(
            attention_mask != 0,
            float(-100.0),
        )
        attention_mask = attention_mask.masked_fill(
            attention_mask == 0,
            float(0.0),
        )

        return attention_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, height, width, channels = x.shape
        shortcut = x

        x = self.norm1(x)

        pad_bottom = (
            self.window_size - height % self.window_size
        ) % self.window_size
        pad_right = (
            self.window_size - width % self.window_size
        ) % self.window_size

        if pad_bottom > 0 or pad_right > 0:
            x = F.pad(
                x,
                (0, 0, 0, pad_right, 0, pad_bottom),
            )

        padded_height = height + pad_bottom
        padded_width = width + pad_right

        attention_mask = self._build_attention_mask(
            padded_height=padded_height,
            padded_width=padded_width,
            device=x.device,
            dtype=x.dtype,
        )

        if self.shift_size > 0:
            shifted = torch.roll(
                x,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2),
            )
        else:
            shifted = x

        windows = window_partition(
            shifted,
            self.window_size,
        )
        windows = windows.view(
            -1,
            self.window_size * self.window_size,
            channels,
        )

        attended_windows = self.attention(
            windows,
            mask=attention_mask,
        )
        attended_windows = attended_windows.view(
            -1,
            self.window_size,
            self.window_size,
            channels,
        )

        shifted = window_reverse(
            attended_windows,
            self.window_size,
            padded_height,
            padded_width,
        )

        if self.shift_size > 0:
            x = torch.roll(
                shifted,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            )
        else:
            x = shifted

        if pad_bottom > 0 or pad_right > 0:
            x = x[:, :height, :width, :].contiguous()

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchEmbedding(nn.Module):
    """Initial non-overlapping patch embedding."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int = 4,
    ) -> None:
        super().__init__()

        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)

        return x


class PatchMerging(nn.Module):
    """Downsample a BHWC feature map by 2 while doubling channels."""

    def __init__(self, dim: int) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(dim * 4)
        self.reduction = nn.Linear(
            dim * 4,
            dim * 2,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, height, width, channels = x.shape

        pad_bottom = height % 2
        pad_right = width % 2

        if pad_bottom > 0 or pad_right > 0:
            x = F.pad(
                x,
                (0, 0, 0, pad_right, 0, pad_bottom),
            )

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = self.norm(x)
        x = self.reduction(x)

        return x


class SwinStage(nn.Module):
    """One hierarchical Swin encoder stage."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        dropout: float,
        attention_dropout: float,
        drop_path_rates: Sequence[float],
        downsample: bool,
    ) -> None:
        super().__init__()

        if len(drop_path_rates) != depth:
            raise ValueError(
                "Number of drop-path rates must equal stage depth."
            )

        blocks: List[nn.Module] = []

        for block_index in range(depth):
            shift_size = (
                0
                if block_index % 2 == 0
                else window_size // 2
            )

            blocks.append(
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift_size,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    drop_path=drop_path_rates[block_index],
                )
            )

        self.blocks = nn.ModuleList(blocks)
        self.downsample = (
            PatchMerging(dim)
            if downsample
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            x = block(x)

        skip = x

        if self.downsample is not None:
            x = self.downsample(x)

        return skip, x


class SwinEncoder(nn.Module):
    """Four-stage hierarchical Swin Transformer encoder."""

    def __init__(
        self,
        input_channels: int,
        embed_dim: int = 48,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        patch_size: int = 4,
    ) -> None:
        super().__init__()

        if len(depths) != 4 or len(num_heads) != 4:
            raise ValueError(
                "depths and num_heads must each contain four values."
            )

        self.patch_embedding = PatchEmbedding(
            in_channels=input_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )

        stage_dims = [
            embed_dim,
            embed_dim * 2,
            embed_dim * 4,
            embed_dim * 8,
        ]

        total_depth = int(sum(depths))
        all_drop_path_rates = torch.linspace(
            0.0,
            drop_path_rate,
            total_depth,
        ).tolist()

        stages: List[nn.Module] = []
        depth_offset = 0

        for stage_index in range(4):
            depth = int(depths[stage_index])

            stage_rates = all_drop_path_rates[
                depth_offset : depth_offset + depth
            ]
            depth_offset += depth

            stages.append(
                SwinStage(
                    dim=stage_dims[stage_index],
                    depth=depth,
                    num_heads=int(num_heads[stage_index]),
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    drop_path_rates=stage_rates,
                    downsample=stage_index < 3,
                )
            )

        self.stages = nn.ModuleList(stages)
        self.stage_dims = stage_dims

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
        x = self.patch_embedding(x)
        features: List[torch.Tensor] = []

        for stage in self.stages:
            skip, x = stage(x)
            features.append(
                skip.permute(0, 3, 1, 2).contiguous()
            )

        return features


class DecoderBlock(nn.Module):
    """Upsampling and skip-feature fusion."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.projection = nn.Conv2d(
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

        self.refine = nn.Sequential(
            nn.Conv2d(
                out_channels * 2,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=valid_group_count(out_channels),
                num_channels=out_channels,
            ),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=valid_group_count(out_channels),
                num_channels=out_channels,
            ),
            nn.GELU(),
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

        x = self.projection(x)
        skip = self.skip_projection(skip)

        x = torch.cat([x, skip], dim=1)
        x = self.refine(x)

        return x


class SwinUNetDecoder(nn.Module):
    """U-Net-style decoder for hierarchical Swin features."""

    def __init__(
        self,
        encoder_channels: Sequence[int],
        decoder_channels: Sequence[int] = (192, 96, 48),
        num_classes: int = 1,
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

        self.decoder3 = DecoderBlock(
            in_channels=encoder_channels[3],
            skip_channels=encoder_channels[2],
            out_channels=decoder_channels[0],
            dropout=dropout,
        )

        self.decoder2 = DecoderBlock(
            in_channels=decoder_channels[0],
            skip_channels=encoder_channels[1],
            out_channels=decoder_channels[1],
            dropout=dropout,
        )

        self.decoder1 = DecoderBlock(
            in_channels=decoder_channels[1],
            skip_channels=encoder_channels[0],
            out_channels=decoder_channels[2],
            dropout=dropout,
        )

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(
                decoder_channels[2],
                decoder_channels[2],
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=valid_group_count(decoder_channels[2]),
                num_channels=decoder_channels[2],
            ),
            nn.GELU(),
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
                "SwinUNetDecoder expects four feature maps."
            )

        feature1, feature2, feature3, feature4 = features

        x = self.decoder3(feature4, feature3)
        x = self.decoder2(x, feature2)
        x = self.decoder1(x, feature1)

        logits = self.segmentation_head(x)

        logits = F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        return logits


class MultimodalSwinUNet(nn.Module):
    """
    Early-fusion multimodal Swin-UNet baseline.

    Default settings are intentionally smaller than canonical Swin-T to keep
    training practical on 256x256 images with the existing experimental setup.
    """

    def __init__(
        self,
        sar_channels: int = 2,
        optical_channels: int = 13,
        num_classes: int = 1,
        embed_dim: int = 48,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        decoder_channels: Sequence[int] = (192, 96, 48),
        patch_size: int = 4,
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
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")
        if window_size <= 0:
            raise ValueError("window_size must be positive.")
        if patch_size <= 0:
            raise ValueError("patch_size must be positive.")
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

        self.encoder = SwinEncoder(
            input_channels=self.input_channels,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            drop_path_rate=drop_path_rate,
            patch_size=patch_size,
        )

        encoder_channels = (
            embed_dim,
            embed_dim * 2,
            embed_dim * 4,
            embed_dim * 8,
        )

        self.decoder = SwinUNetDecoder(
            encoder_channels=encoder_channels,
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
                f"sar must be a 4D tensor, received {tuple(sar.shape)}."
            )

        if optical.ndim != 4:
            raise ValueError(
                "optical must be a 4D tensor, "
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
    """Return the number of trainable model parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


if __name__ == "__main__":
    model = MultimodalSwinUNet(
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