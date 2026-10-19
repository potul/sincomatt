from __future__ import annotations

import torch
import torch.nn as nn
from audiotrain.losses.preemphasis import SpectralPreEmphasisFilter


class SpectralESRLoss(nn.Module):
    """
    Pre-emphasized Spectral / Cepstral Loss (FineTuning stage):
      1. 2nd-order pre-emphasis: y[n] = x[n] - 1.7*x[n-1] + 0.7225*x[n-2]
      2. Zero-phase spectral transform: z = irfft(|rfft(y)|)
      3. ESR: sum((z_target - z_pred)^2) / (sum(z_target^2) + eps)
    """
    def __init__(self, use_preemphasis: bool = True, eps: float = 1e-20):
        super().__init__()
        self.use_preemphasis = use_preemphasis
        self.preemphasis = SpectralPreEmphasisFilter() if use_preemphasis else None
        self.eps = eps

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.preemphasis is not None:
            x_f = self.preemphasis(x)
        else:
            x_f = x
        x_mag = torch.abs(torch.fft.rfft(x_f, dim=-1))
        return torch.fft.irfft(x_mag, dim=-1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_z = self.transform(pred)
        target_z = self.transform(target)

        diff = target_z - pred_z
        loss = torch.sum(diff ** 2) / (torch.sum(target_z ** 2) + self.eps)
        return loss


