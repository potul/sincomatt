from __future__ import annotations

import torch
import torch.nn as nn
from audiotrain.losses.esr import ESRLoss
from audiotrain.losses.spectral import SpectralESRLoss


class CurriculumLoss(nn.Module):
    """
    Two-stage curriculum training loss:
      - Epoch < fine_tuning_epoch: BasicTuning Stage (Time-domain ESR)
      - Epoch >= fine_tuning_epoch: FineTuning Stage (Spectral ESR)
    """
    def __init__(self, fine_tuning_epoch: int = 700):
        super().__init__()
        self.fine_tuning_epoch = fine_tuning_epoch
        self.esr_loss = ESRLoss(use_preemphasis=True)
        self.spectral_loss = SpectralESRLoss(use_preemphasis=True)
        self.current_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch

    @property
    def is_fine_tuning(self) -> bool:
        return self.current_epoch >= self.fine_tuning_epoch

    @property
    def eps(self) -> float:
        return self.spectral_loss.eps if self.is_fine_tuning else self.esr_loss.eps

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_fine_tuning:
            return self.spectral_loss.transform(x)
        return self.esr_loss.transform(x)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.is_fine_tuning:
            return self.spectral_loss(pred, target)
        return self.esr_loss(pred, target)

