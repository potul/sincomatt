from __future__ import annotations

import torch
import torch.nn as nn
from audiotrain.losses.preemphasis import PreEmphasisFilter


class ESRLoss(nn.Module):
    """
    Pre-emphasized Error-to-Signal Ratio (ESR) Loss:
    Loss = sum((y_filt - pred_filt)^2) / (sum(y_filt^2) + eps)
    with numerical stability constant eps = 1e-20.
    """
    def __init__(self, use_preemphasis: bool = True, eps: float = 1e-20):
        super().__init__()
        self.use_preemphasis = use_preemphasis
        self.preemphasis = PreEmphasisFilter() if use_preemphasis else None
        self.eps = eps

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.preemphasis is not None:
            return self.preemphasis(x)
        return x

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_f = self.transform(pred)
        target_f = self.transform(target)

        error = target_f - pred_f
        loss = torch.sum(error ** 2) / (torch.sum(target_f ** 2) + self.eps)
        return loss

