from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from audiotrain.models.base import BaseAmpModel


class AM4Model(BaseAmpModel):
    """
    AM4 Architecture:
      - fir1: Conv1d(1, 1, kernel_size=512, bias=False, causal left-padding=511)
      - rnn:  GRU(input_size=1, hidden_size=16, num_layers=1, batch_first=True, bias=True)
      - ln:   Linear(in_features=16, out_features=1, bias=False)
      - fir2: Conv1d(1, 1, kernel_size=512, bias=False, causal left-padding=511)
    """
    def __init__(self, fir1_size: int = 512, fir2_size: int = 512, hidden_size: int = 16):
        super().__init__()
        self.fir1_size = fir1_size
        self.fir2_size = fir2_size
        self.hidden_size = hidden_size

        self.fir1 = nn.Conv1d(1, 1, kernel_size=fir1_size, bias=False, padding=0)
        self.rnn = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bias=True,
        )
        self.ln = nn.Linear(in_features=hidden_size, out_features=1, bias=False)
        self.fir2 = nn.Conv1d(1, 1, kernel_size=fir2_size, bias=False, padding=0)

        self.reset_parameters()

    @property
    def model_tag(self) -> str:
        return "AM4"

    @property
    def receptive_field_pad(self) -> int:
        return (self.fir1_size - 1) + (self.fir2_size - 1)

    def reset_parameters(self) -> None:
        # Initialize FIR1 and FIR2 close to causal identity (delta impulse at causal end tap K-1)
        nn.init.zeros_(self.fir1.weight)
        with torch.no_grad():
            self.fir1.weight[0, 0, self.fir1_size - 1] = 1.0

        nn.init.zeros_(self.fir2.weight)
        with torch.no_grad():
            self.fir2.weight[0, 0, self.fir2_size - 1] = 1.0

        # GRU initialization
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        # Linear initialization
        nn.init.xavier_uniform_(self.ln.weight)

    def forward(self, x: torch.Tensor, target_len: int = 512) -> torch.Tensor:
        """
        Causal forward pass:
        When x has shape (B, 1, target_len + receptive_field_pad) [e.g. 512 + 1022 = 1534],
        fir1 and fir2 execute pure valid convolution (padding=0) over real historical audio:
          1534 -> fir1 -> 1023 -> GRU -> Linear -> 1023 -> fir2 -> 512.
        If x is shorter than target_len + receptive_field_pad, left-pads with zeros to maintain causality.
        """
        needed_len = target_len + self.receptive_field_pad
        if x.shape[-1] < needed_len:
            x = F.pad(x, (needed_len - x.shape[-1], 0))

        # Valid input FIR filtering: (B, 1, T) -> (B, 1, T - 511)
        x = self.fir1(x)

        # Transpose to (B, T - 511, 1) for GRU (warms up GRU state over 511 real samples!)
        x = x.transpose(1, 2)
        x, _ = self.rnn(x)

        # Linear projection
        x = self.ln(x)

        # Transpose back to (B, 1, T - 511)
        x = x.transpose(1, 2)

        # Valid output FIR filtering: (B, 1, T - 511) -> (B, 1, T - 1022)
        x = self.fir2(x)

        if x.shape[-1] > target_len:
            x = x[:, :, -target_len:]
        return x

    def export_weights_array(self) -> torch.Tensor:
        """
        Exports weights in contiguous order:
          fir1 (512)
          rnn.weight_ih_l0 (48)
          rnn.weight_hh_l0 (768)
          rnn.bias_ih_l0 (48)
          rnn.bias_hh_l0 (48)
          ln.weight (16)
          fir2 (512)
        Total floats = 1952
        """
        with torch.no_grad():
            fir1_w = self.fir1.weight.detach().cpu().view(-1)
            w_ih = self.rnn.weight_ih_l0.detach().cpu().view(-1)
            w_hh = self.rnn.weight_hh_l0.detach().cpu().view(-1)
            b_ih = self.rnn.bias_ih_l0.detach().cpu().view(-1)
            b_hh = self.rnn.bias_hh_l0.detach().cpu().view(-1)
            ln_w = self.ln.weight.detach().cpu().view(-1)
            fir2_w = self.fir2.weight.detach().cpu().view(-1)

            tensors = [fir1_w, w_ih, w_hh, b_ih, b_hh, ln_w, fir2_w]
            return torch.cat(tensors, dim=0).to(torch.float32)

    def load_weights_array(self, weights: torch.Tensor) -> None:
        """Loads weights from contiguous 1D float32 array."""
        w = weights.view(-1).cpu()
        idx = 0

        # fir1
        f1_num = self.fir1.weight.numel()
        self.fir1.weight.data.copy_(w[idx : idx + f1_num].view_as(self.fir1.weight))
        idx += f1_num

        # rnn
        w_ih_num = self.rnn.weight_ih_l0.numel()
        self.rnn.weight_ih_l0.data.copy_(w[idx : idx + w_ih_num].view_as(self.rnn.weight_ih_l0))
        idx += w_ih_num

        w_hh_num = self.rnn.weight_hh_l0.numel()
        self.rnn.weight_hh_l0.data.copy_(w[idx : idx + w_hh_num].view_as(self.rnn.weight_hh_l0))
        idx += w_hh_num

        b_ih_num = self.rnn.bias_ih_l0.numel()
        self.rnn.bias_ih_l0.data.copy_(w[idx : idx + b_ih_num].view_as(self.rnn.bias_ih_l0))
        idx += b_ih_num

        b_hh_num = self.rnn.bias_hh_l0.numel()
        self.rnn.bias_hh_l0.data.copy_(w[idx : idx + b_hh_num].view_as(self.rnn.bias_hh_l0))
        idx += b_hh_num

        # ln
        ln_num = self.ln.weight.numel()
        self.ln.weight.data.copy_(w[idx : idx + ln_num].view_as(self.ln.weight))
        idx += ln_num

        # fir2
        f2_num = self.fir2.weight.numel()
        self.fir2.weight.data.copy_(w[idx : idx + f2_num].view_as(self.fir2.weight))
        idx += f2_num
