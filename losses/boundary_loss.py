"""
boundary_loss.py

Boundary-aware loss for flood segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()

        sobel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _edge_map(self, x):
        sobel_x = self.sobel_x.to(device=x.device, dtype=x.dtype)
        sobel_y = self.sobel_y.to(device=x.device, dtype=x.dtype)

        edge_x = F.conv2d(x, sobel_x, padding=1)
        edge_y = F.conv2d(x, sobel_y, padding=1)

        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)

        return edge

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()

        pred_edges = self._edge_map(probs)
        target_edges = self._edge_map(targets)

        return F.l1_loss(pred_edges, target_edges)