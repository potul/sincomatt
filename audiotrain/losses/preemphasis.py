from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PreEmphasisFilter(nn.Module):
    """
    Perceptual pre-emphasis filter matching FUN_14003f1a0 in AudioTrainHHAG.exe:
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
