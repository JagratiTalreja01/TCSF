"""
mamba_block.py

Lightweight Vision-Mamba-style block for ACSF v2.

This is still not the official VMamba implementation. It is a stable,
trainable state-space-inspired 2D block for early SAR-optical fusion
experiments.

Main v2 changes:
    - separate horizontal and vertical state scans
    - learnable state mixing gate
    - lightweight channel mixing branch
    - residual layer scaling for stable training
"""

import torch
import torch.nn as nn

from models.blocks.norm_layers import LayerNorm2d


class MambaBlock(nn.Module):
    """
    Lightweight 2D Vision-Mamba-style block.

    Args:
        dim: number of input/output channels.
        expansion: hidden channel expansion ratio.
        kernel_size: scan kernel size for horizontal/vertical depthwise filters.
        dropout: spatial dropout probability.

    Input:
        x: [B, C, H, W]

    Output:
        out: [B, C, H, W]
    """

    def __init__(
        self,
        dim,
        expansion=2,
        kernel_size=7,
        dropout=0.0,
    ):
        super().__init__()

        hidden_dim = dim * expansion

        self.norm = LayerNorm2d(dim)

        # Project into state features and gates.
        self.in_proj = nn.Conv2d(
            dim,
            hidden_dim * 2,
            kernel_size=1,
            bias=False,
        )

        # Directional state propagation. These are cheap approximations of
        # horizontal and vertical selective scans.
        self.dwconv_h = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=(1, kernel_size),
            padding=(0, kernel_size // 2),
            groups=hidden_dim,
            bias=False,
        )

        self.dwconv_v = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=(kernel_size, 1),
            padding=(kernel_size // 2, 0),
            groups=hidden_dim,
            bias=False,
        )

        # Learns how much horizontal vs vertical state information to retain.
        self.state_gate = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # Local channel mixing after directional state propagation.
        self.channel_mixer = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
        )

        self.act = nn.SiLU(inplace=True)

        self.out_proj = nn.Conv2d(
            hidden_dim,
            dim,
            kernel_size=1,
            bias=False,
        )

        self.dropout = nn.Dropout2d(dropout)

        # Small initialization keeps early training stable.
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1) * 1e-3)

    def forward(self, x):
        residual = x

        x = self.norm(x)
        x_state, x_gate = self.in_proj(x).chunk(2, dim=1)

        h_state = self.dwconv_h(x_state)
        v_state = self.dwconv_v(x_state)

        mix_gate = self.state_gate(torch.cat([h_state, v_state], dim=1))
        scan_state = mix_gate * h_state + (1.0 - mix_gate) * v_state
        scan_state = self.act(scan_state)

        input_gate = torch.sigmoid(x_gate)
        scan_state = scan_state * input_gate

        mixed_state = self.channel_mixer(scan_state)
        scan_state = scan_state + mixed_state

        out = self.out_proj(scan_state)
        out = self.dropout(out)

        return residual + self.gamma * out
