from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PreEmphasisFilter(nn.Module):
    """
    First-stage perceptual pre-emphasis filter (BasicTuning):
    y[n] = x[n] - 0.7225 * x[n-2]
    Implemented via tensor slicing along the time dimension:
      x[..., 2:] - 0.7225 * x[..., :-2]
    """
    def __init__(self, alpha: float = 0.7225):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (B, 1, T) or (B, T)
        returns filtered tensor of shape (B, 1, T - 2) or (B, T - 2)
        """
        return x[..., 2:] - self.alpha * x[..., :-2]


class SpectralPreEmphasisFilter(nn.Module):
    """
    Second-order perceptual pre-emphasis filter (FineTuning):
    H(z) = (1 - 0.85 * z^-1)^2 = 1 - 1.7 * z^-1 + 0.7225 * z^-2
    y[n] = x[n] - 1.7 * x[n-1] + 0.7225 * x[n-2]
    Implemented via tensor slicing along the time dimension:
      x[..., 2:] - 1.7 * x[..., 1:-1] + 0.7225 * x[..., :-2]
    """
    def __init__(self, c1: float = 1.7, c2: float = 0.7225):
        super().__init__()
        self.c1 = c1
        self.c2 = c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (B, 1, T) or (B, T)
        returns filtered tensor of shape (B, 1, T - 2) or (B, T - 2)
        """
        return x[..., 2:] - self.c1 * x[..., 1:-1] + self.c2 * x[..., :-2]

