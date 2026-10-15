from __future__ import annotations

import torch
import torch.nn as nn
from audiotrain.losses.preemphasis import PreEmphasisFilter


class SpectralESRLoss(nn.Module):
    """
    Pre-emphasized Spectral Magnitude Loss (FineTuning stage):
    Loss = sum((|rfft(y_filt)| - |rfft(pred_filt)|)^2) / (sum(|rfft(y_filt)|^2) + eps)
    """
    def __init__(self, use_preemphasis: bool = True, eps: float = 1e-8):
        super().__init__()
        self.use_preemphasis = use_preemphasis
        self.preemphasis = PreEmphasisFilter() if use_preemphasis else None
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.preemphasis is not None:
            pred_f = self.preemphasis(pred)
            target_f = self.preemphasis(target)
        else:
            pred_f = pred
            target_f = target

        # Real FFT along time dimension
        pred_fft = torch.fft.rfft(pred_f, dim=-1)
        target_fft = torch.fft.rfft(target_f, dim=-1)

        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)

        diff = target_mag - pred_mag
        loss = torch.sum(diff ** 2) / (torch.sum(target_mag ** 2) + self.eps)
        return loss
