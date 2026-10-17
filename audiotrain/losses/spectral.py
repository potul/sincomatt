from __future__ import annotations

import torch
import torch.nn as nn
from audiotrain.losses.preemphasis import SpectralPreEmphasisFilter


class SpectralESRLoss(nn.Module):
    """
    Pre-emphasized Spectral / Cepstral Loss (FineTuning stage, FUN_14003f9f0):
      1. 2nd-order pre-emphasis (FUN_14003eb70): y[n] = x[n] - 1.7*x[n-1] + 0.7225*x[n-2]
      2. Zero-phase spectral transform (FUN_14003f560): z = irfft(|rfft(y)|)
      3. ESR (FUN_14003f9f0): sum((z_target - z_pred)^2) / (sum(z_target^2) + eps)
    """
    def __init__(self, use_preemphasis: bool = True, eps: float = 1e-20):
        super().__init__()
        self.use_preemphasis = use_preemphasis
        self.preemphasis = SpectralPreEmphasisFilter() if use_preemphasis else None
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.preemphasis is not None:
            pred_f = self.preemphasis(pred)
            target_f = self.preemphasis(target)
        else:
            pred_f = pred
            target_f = target

        # Zero-phase magnitude spectrum via rfft -> abs -> irfft (FUN_14003f560)
        pred_mag = torch.abs(torch.fft.rfft(pred_f, dim=-1))
        target_mag = torch.abs(torch.fft.rfft(target_f, dim=-1))

        pred_z = torch.fft.irfft(pred_mag, dim=-1)
        target_z = torch.fft.irfft(target_mag, dim=-1)

        diff = target_z - pred_z
        loss = torch.sum(diff ** 2) / (torch.sum(target_z ** 2) + self.eps)
        return loss

