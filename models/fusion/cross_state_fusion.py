"""
cross_state_fusion.py

Reliability-guided local Cross-State Fusion and controlled Tri-Level
Cross-State Fusion (TCSF v3.1) for hierarchical SAR-optical interaction.

TCSF uses four encoder feature scales and three learned cross-scale state
transitions. Each transition starts close to an identity mapping through
near-zero learnable residual scales, allowing the network to gradually learn
whether shallow states should influence deeper semantic features.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks.conv_block import ConvBNAct
from models.blocks.mamba_block import MambaBlock


class CrossStateFusionBlock(nn.Module):
    """Bidirectional reliability-guided fusion at one feature scale."""

    def __init__(self, channels: int, use_reliability: bool = True):
        super().__init__()
        self.use_reliability = use_reliability

        self.sar_state = MambaBlock(channels)
        self.optical_state = MambaBlock(channels)

        self.sar_to_optical_gate = nn.Sequential(
            ConvBNAct(channels * 2, channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.optical_to_sar_gate = nn.Sequential(
            ConvBNAct(channels * 2, channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

        self.sar_reliability_calibrator = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.optical_reliability_calibrator = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.sar_update = ConvBNAct(channels * 2, channels)
        self.optical_update = ConvBNAct(channels * 2, channels)
        self.fuse = ConvBNAct(channels * 2, channels)

    @staticmethod
    def _prepare_reliability(
        confidence: Optional[torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, height, width = target.shape

        if confidence is None:
            return target.new_ones((batch_size, 1, height, width))

        if confidence.dim() == 2:
            confidence = confidence[:, :, None, None]
        elif confidence.dim() == 3:
            confidence = confidence[:, None, :, :]
        elif confidence.dim() != 4:
            raise ValueError(
                "Confidence must have 2, 3, or 4 dimensions; "
                f"received shape {tuple(confidence.shape)}."
            )

        confidence = confidence.to(device=target.device, dtype=target.dtype)

        if confidence.shape[1] != 1:
            confidence = confidence.mean(dim=1, keepdim=True)

        if confidence.shape[-2:] != (height, width):
            confidence = F.interpolate(
                confidence,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )

        # Never use an in-place operation here: this tensor may be the direct
        # output of a sigmoid reliability head and is required by autograd.
        return confidence.clamp(0.0, 1.0)

    def forward(
        self,
        sar_feat: torch.Tensor,
        optical_feat: torch.Tensor,
        sar_conf: Optional[torch.Tensor] = None,
        optical_conf: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if sar_feat.shape != optical_feat.shape:
            raise ValueError(
                "SAR and optical features must have identical shapes at a "
                f"fusion scale, got {tuple(sar_feat.shape)} and "
                f"{tuple(optical_feat.shape)}."
            )

        sar_state = self.sar_state(sar_feat)
        optical_state = self.optical_state(optical_feat)
        state_pair = torch.cat([sar_state, optical_state], dim=1)

        sar_to_optical = self.sar_to_optical_gate(state_pair)
        optical_to_sar = self.optical_to_sar_gate(state_pair)

        if self.use_reliability:
            sar_reliability = self.sar_reliability_calibrator(
                self._prepare_reliability(sar_conf, sar_state)
            )
            optical_reliability = self.optical_reliability_calibrator(
                self._prepare_reliability(optical_conf, optical_state)
            )
        else:
            sar_reliability = torch.ones_like(sar_state[:, :1])
            optical_reliability = torch.ones_like(optical_state[:, :1])

        optical_injection = sar_to_optical * sar_reliability * sar_state
        sar_injection = optical_to_sar * optical_reliability * optical_state

        sar_out = self.sar_update(torch.cat([sar_state, sar_injection], dim=1))
        optical_out = self.optical_update(
            torch.cat([optical_state, optical_injection], dim=1)
        )
        fused_out = self.fuse(torch.cat([sar_out, optical_out], dim=1))

        return sar_out, optical_out, fused_out


class CrossScaleStateTransition(nn.Module):
    """Controlled residual propagation from one encoder scale to the next."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        residual_init: float = 1e-3,
    ):
        super().__init__()

        self.sar_projection = nn.Sequential(
            ConvBNAct(in_channels, out_channels),
            nn.GroupNorm(1, out_channels),
        )
        self.optical_projection = nn.Sequential(
            ConvBNAct(in_channels, out_channels),
            nn.GroupNorm(1, out_channels),
        )
        self.fused_projection = nn.Sequential(
            ConvBNAct(in_channels, out_channels),
            nn.GroupNorm(1, out_channels),
        )

        gate_channels = out_channels * 4
        self.sar_transition_gate = nn.Sequential(
            ConvBNAct(gate_channels, out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.optical_transition_gate = nn.Sequential(
            ConvBNAct(gate_channels, out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        self.sar_private_refine = ConvBNAct(out_channels, out_channels)
        self.optical_private_refine = ConvBNAct(out_channels, out_channels)
        self.fused_to_sar_refine = ConvBNAct(out_channels, out_channels)
        self.fused_to_optical_refine = ConvBNAct(out_channels, out_channels)

        # Separate, interpretable residual strengths. tanh keeps them bounded.
        self.gamma_sar = nn.Parameter(torch.tensor(float(residual_init)))
        self.gamma_optical = nn.Parameter(torch.tensor(float(residual_init)))
        self.gamma_fused = nn.Parameter(torch.tensor(float(residual_init)))

    @staticmethod
    def _resize(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == reference.shape[-2:]:
            return x
        return F.interpolate(
            x,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def effective_scales(self) -> torch.Tensor:
        """Return bounded effective scales as [sar, optical, fused]."""
        return torch.stack(
            [
                torch.tanh(self.gamma_sar),
                torch.tanh(self.gamma_optical),
                torch.tanh(self.gamma_fused),
            ]
        )

    def forward(
        self,
        current_sar: torch.Tensor,
        current_optical: torch.Tensor,
        previous_sar: torch.Tensor,
        previous_optical: torch.Tensor,
        previous_fused: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        previous_sar = self.sar_projection(
            self._resize(previous_sar, current_sar)
        )
        previous_optical = self.optical_projection(
            self._resize(previous_optical, current_optical)
        )
        previous_fused = self.fused_projection(
            self._resize(previous_fused, current_sar)
        )

        transition_context = torch.cat(
            [current_sar, current_optical, previous_sar, previous_optical],
            dim=1,
        )
        sar_gate = self.sar_transition_gate(transition_context)
        optical_gate = self.optical_transition_gate(transition_context)

        sar_private_delta = self.sar_private_refine(sar_gate * previous_sar)
        optical_private_delta = self.optical_private_refine(
            optical_gate * previous_optical
        )
        sar_fused_delta = self.fused_to_sar_refine(
            sar_gate * previous_fused
        )
        optical_fused_delta = self.fused_to_optical_refine(
            optical_gate * previous_fused
        )

        gamma_sar, gamma_optical, gamma_fused = self.effective_scales()

        current_sar = (
            current_sar
            + gamma_sar * sar_private_delta
            + gamma_fused * sar_fused_delta
        )
        current_optical = (
            current_optical
            + gamma_optical * optical_private_delta
            + gamma_fused * optical_fused_delta
        )

        return current_sar, current_optical


class TriLevelCrossStateFusion(nn.Module):
    """Four local fusion scales connected by three controlled transitions."""

    def __init__(
        self,
        dims: Sequence[int],
        use_reliability: bool = True,
        use_cross_scale_propagation: bool = True,
        transition_residual_init: float = 1e-3,
    ):
        super().__init__()

        if len(dims) != 4:
            raise ValueError(
                "TriLevelCrossStateFusion expects four encoder scales so that "
                "three cross-scale transitions can be formed."
            )

        self.dims = list(dims)
        self.use_cross_scale_propagation = use_cross_scale_propagation

        self.local_fusion_blocks = nn.ModuleList(
            [
                CrossStateFusionBlock(
                    channels=channels,
                    use_reliability=use_reliability,
                )
                for channels in self.dims
            ]
        )
        self.state_transitions = nn.ModuleList(
            [
                CrossScaleStateTransition(
                    in_channels=self.dims[index],
                    out_channels=self.dims[index + 1],
                    residual_init=transition_residual_init,
                )
                for index in range(len(self.dims) - 1)
            ]
        )

    def get_transition_scales(self) -> torch.Tensor:
        """Return transition scales with shape [3 transitions, 3 branches]."""
        if not self.state_transitions:
            return torch.empty(0, 3)
        return torch.stack(
            [transition.effective_scales() for transition in self.state_transitions],
            dim=0,
        )

    def transition_scale_dict(self) -> Dict[str, float]:
        scales = self.get_transition_scales().detach().cpu()
        result: Dict[str, float] = {}
        names = ("sar", "optical", "fused")
        for level_index in range(scales.shape[0]):
            for branch_index, branch_name in enumerate(names):
                result[f"t{level_index + 1}_{branch_name}_gamma"] = float(
                    scales[level_index, branch_index]
                )
        return result

    def forward(
        self,
        sar_features: Sequence[torch.Tensor],
        optical_features: Sequence[torch.Tensor],
        sar_conf: Optional[torch.Tensor] = None,
        optical_conf: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        if len(sar_features) != 4 or len(optical_features) != 4:
            raise ValueError(
                "TCSF requires exactly four SAR and four optical feature maps."
            )

        sar_outputs: List[torch.Tensor] = []
        optical_outputs: List[torch.Tensor] = []
        fused_outputs: List[torch.Tensor] = []

        previous_sar = previous_optical = previous_fused = None

        for scale_index, (sar_feat, optical_feat, fusion_block) in enumerate(
            zip(sar_features, optical_features, self.local_fusion_blocks)
        ):
            if scale_index > 0 and self.use_cross_scale_propagation:
                transition = self.state_transitions[scale_index - 1]
                sar_feat, optical_feat = transition(
                    current_sar=sar_feat,
                    current_optical=optical_feat,
                    previous_sar=previous_sar,
                    previous_optical=previous_optical,
                    previous_fused=previous_fused,
                )

            sar_out, optical_out, fused_out = fusion_block(
                sar_feat=sar_feat,
                optical_feat=optical_feat,
                sar_conf=sar_conf,
                optical_conf=optical_conf,
            )

            sar_outputs.append(sar_out)
            optical_outputs.append(optical_out)
            fused_outputs.append(fused_out)

            previous_sar = sar_out
            previous_optical = optical_out
            previous_fused = fused_out

        return sar_outputs, optical_outputs, fused_outputs
