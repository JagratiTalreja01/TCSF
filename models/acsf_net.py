"""Adaptive Cross-State Fusion Network with controlled TCSF v3.1."""

import torch
import torch.nn as nn

from models.fusion.adaptive_data_fusion import AdaptiveDataFusion
from models.fusion.cross_state_fusion import TriLevelCrossStateFusion
from models.fusion.adaptive_decision_fusion import AdaptiveDecisionFusion
from models.encoders.sar_vmamba_encoder import SARVMambaEncoder
from models.encoders.optical_vmamba_encoder import OpticalVMambaEncoder
from models.decoders.mamba_decoder import MambaDecoder
from models.decoders.prediction_heads import (
    SARPredictionHead,
    OpticalPredictionHead,
    FusionPredictionHead,
)


class ACSFNet(nn.Module):
    def __init__(
        self,
        sar_channels=2,
        optical_channels=13,
        base_dim=64,
        num_classes=1,
        use_reliability_guidance=True,
        use_cross_scale_propagation=True,
        transition_residual_init=1e-3,
    ):
        super().__init__()

        self.data_fusion = AdaptiveDataFusion(
            sar_channels=sar_channels,
            optical_channels=optical_channels,
            out_channels=base_dim,
        )
        self.sar_encoder = SARVMambaEncoder(
            in_channels=sar_channels,
            base_dim=base_dim,
        )
        self.optical_encoder = OpticalVMambaEncoder(
            in_channels=optical_channels,
            base_dim=base_dim,
        )

        dims = [base_dim, base_dim * 2, base_dim * 4, base_dim * 8]
        self.tri_level_fusion = TriLevelCrossStateFusion(
            dims=dims,
            use_reliability=use_reliability_guidance,
            use_cross_scale_propagation=use_cross_scale_propagation,
            transition_residual_init=transition_residual_init,
        )

        self.sar_decoder = MambaDecoder(base_dim=base_dim)
        self.optical_decoder = MambaDecoder(base_dim=base_dim)
        self.fusion_decoder = MambaDecoder(base_dim=base_dim)

        self.sar_head = SARPredictionHead(base_dim, base_dim, num_classes)
        self.optical_head = OpticalPredictionHead(base_dim, base_dim, num_classes)
        self.fusion_head = FusionPredictionHead(base_dim, base_dim, num_classes)
        self.decision_fusion = AdaptiveDecisionFusion()

    def get_transition_scale_dict(self):
        return self.tri_level_fusion.transition_scale_dict()

    def forward(self, sar, optical):
        initial_fused, sar_conf, optical_conf = self.data_fusion(sar, optical)
        sar_features = self.sar_encoder(sar)
        optical_features = self.optical_encoder(optical)

        sar_fused_features, optical_fused_features, fused_features = (
            self.tri_level_fusion(
                sar_features=sar_features,
                optical_features=optical_features,
                sar_conf=sar_conf,
                optical_conf=optical_conf,
            )
        )

        sar_pred = self.sar_head(self.sar_decoder(sar_fused_features))
        optical_pred = self.optical_head(
            self.optical_decoder(optical_fused_features)
        )
        fused_pred = self.fusion_head(self.fusion_decoder(fused_features))

        final_pred, decision_weights = self.decision_fusion(
            sar_pred,
            optical_pred,
            fused_pred,
        )

        return {
            "final_pred": final_pred,
            "sar_pred": sar_pred,
            "optical_pred": optical_pred,
            "fused_pred": fused_pred,
            "sar_conf": sar_conf,
            "optical_conf": optical_conf,
            "decision_weights": decision_weights,
            "initial_fused": initial_fused,
            "transition_scales": self.tri_level_fusion.get_transition_scales(),
        }


if __name__ == "__main__":
    model = ACSFNet(base_dim=24)
    outputs = model(
        torch.randn(2, 2, 256, 256),
        torch.randn(2, 13, 256, 256),
    )
    for key, value in outputs.items():
        print(key, tuple(value.shape))
