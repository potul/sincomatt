from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from audiotrain.models.base import BaseAmpModel


class IRModel(BaseAmpModel):
    """
    Linear Impulse Response (IR) Model:
      - fir: Conv1d(1, 1, kernel_size=1024 or 2048, bias=False, causal left-padding=filter_size-1)
    """
    def __init__(self, filter_size: int = 1024):
        super().__init__()
        self.filter_size = filter_size
        self.fir = nn.Conv1d(1, 1, kernel_size=filter_size, bias=False, padding=0)
        self.reset_parameters()

    @property
    def model_tag(self) -> str:
        return f"IR{self.filter_size}"

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.fir.weight)
        with torch.no_grad():
            self.fir.weight[0, 0, self.filter_size - 1] = 1.0

    def forward(self, x: torch.Tensor, target_len: int = 512) -> torch.Tensor:
        x_pad = F.pad(x, (self.filter_size - 1, 0))
        out = self.fir(x_pad)
        if out.shape[-1] > target_len:
            out = out[:, :, -target_len:]
        return out

    def export_weights_array(self) -> torch.Tensor:
        with torch.no_grad():
            return self.fir.weight.detach().cpu().view(-1).to(torch.float32)

    def load_weights_array(self, weights: torch.Tensor) -> None:
        w = weights.view(-1).cpu()
        self.fir.weight.data.copy_(w[: self.filter_size].view_as(self.fir.weight))
