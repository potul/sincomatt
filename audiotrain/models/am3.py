from __future__ import annotations

import torch
import torch.nn as nn
from audiotrain.models.base import BaseAmpModel


class AM3Model(BaseAmpModel):
    """
    AM3 Architecture:
      - cnn:  Conv1d(1, 3, kernel_size=32, bias=False, padding='same')
      - fir1: Conv1d(1, 1, kernel_size=256, bias=False, padding='same')
      - arsinh non-linear saturation: asinh(x) = ln(x + sqrt(x^2 + 1))
      - rnn:  GRU(input_size=1, hidden_size=16, num_layers=1, batch_first=True, bias=True)
      - ln:   Linear(in_features=16, out_features=1, bias=False)
      - fir2: Conv1d(1, 1, kernel_size=256, bias=False, padding='same')
    """
    def __init__(
        self,
        fir1_size: int = 256,
        fir2_size: int = 256,
        cnn_channels: int = 3,
        cnn_kernel_size: int = 32,
        hidden_size: int = 16,
    ):
        super().__init__()
        self.fir1_size = fir1_size
        self.fir2_size = fir2_size
        self.cnn_channels = cnn_channels
        self.cnn_kernel_size = cnn_kernel_size
        self.hidden_size = hidden_size

        self.cnn = nn.Conv1d(1, cnn_channels, kernel_size=cnn_kernel_size, bias=False, padding="same")
        self.fir1 = nn.Conv1d(1, 1, kernel_size=fir1_size, bias=False, padding="same")
        self.rnn = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bias=True,
        )
        self.ln = nn.Linear(in_features=hidden_size, out_features=1, bias=False)
        self.fir2 = nn.Conv1d(1, 1, kernel_size=fir2_size, bias=False, padding="same")

        self.reset_parameters()

    @property
    def model_tag(self) -> str:
        return "AM3"

    def reset_parameters(self) -> None:
        # Initializing CNN
        nn.init.xavier_uniform_(self.cnn.weight)

        # FIR initialization close to identity
        nn.init.zeros_(self.fir1.weight)
        center1 = self.fir1_size // 2
        with torch.no_grad():
            self.fir1.weight[0, 0, center1] = 1.0

        nn.init.zeros_(self.fir2.weight)
        center2 = self.fir2_size // 2
        with torch.no_grad():
            self.fir2.weight[0, 0, center2] = 1.0

        # GRU initialization
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        nn.init.xavier_uniform_(self.ln.weight)

    def forward(self, x: torch.Tensor, target_len: int = 512) -> torch.Tensor:
        """
        x shape: (B, 1, T_in)
        returns: (B, 1, target_len)
        """
        # Multi-channel CNN filter bank
        c = self.cnn(x)
        # Sum over channels to form input for FIR1
        x_cnn = torch.sum(c, dim=1, keepdim=True)

        # Input FIR filtering
        x = self.fir1(x_cnn)

        # Non-linear saturation: arsinh(x) = ln(x + sqrt(x^2 + 1))
        x = torch.asinh(x)

        # Transpose to (B, T, 1) for GRU
        x = x.transpose(1, 2)
        x, _ = self.rnn(x)

        # Linear projection
        x = self.ln(x)

        # Transpose back to (B, 1, T)
        x = x.transpose(1, 2)

        # Output FIR filtering
        x = self.fir2(x)

        # Crop to target length if delay padding was added
        if x.shape[-1] > target_len:
            x = x[:, :, -target_len:]
        return x

    def export_weights_array(self) -> torch.Tensor:
        """
        Exact binary layout verified against kpa-blue.am3Data:
          fir1 (256)
          rnn.weight_ih_l0 (48)
          rnn.weight_hh_l0 (768)
          rnn.bias_ih_l0 (48)
          rnn.bias_hh_l0 (48)
          ln.weight (16)
          cnn.weight (96)
          fir2 (256)
        Total floats = 1536
        """
        with torch.no_grad():
            fir1_w = self.fir1.weight.detach().cpu().view(-1)
            w_ih = self.rnn.weight_ih_l0.detach().cpu().view(-1)
            w_hh = self.rnn.weight_hh_l0.detach().cpu().view(-1)
            b_ih = self.rnn.bias_ih_l0.detach().cpu().view(-1)
            b_hh = self.rnn.bias_hh_l0.detach().cpu().view(-1)
            ln_w = self.ln.weight.detach().cpu().view(-1)
            cnn_w = self.cnn.weight.detach().cpu().view(-1)
            fir2_w = self.fir2.weight.detach().cpu().view(-1)

            tensors = [fir1_w, w_ih, w_hh, b_ih, b_hh, ln_w, cnn_w, fir2_w]
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

        # cnn
        cnn_num = self.cnn.weight.numel()
        self.cnn.weight.data.copy_(w[idx : idx + cnn_num].view_as(self.cnn.weight))
        idx += cnn_num

        # fir2
        f2_num = self.fir2.weight.numel()
        self.fir2.weight.data.copy_(w[idx : idx + f2_num].view_as(self.fir2.weight))
        idx += f2_num
