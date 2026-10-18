from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from audiotrain.models.base import BaseAmpModel


class AM3Model(BaseAmpModel):
    """
    AM3 Architecture:
      - cnn:  Conv1d(1, 3, kernel_size=64, bias=False, causal left-padding=63)
      - fir1: Conv1d(1, 1, kernel_size=256, bias=False, causal left-padding=255)
      - arsinh non-linear saturation: asinh(x) = ln(x + sqrt(x^2 + 1))
      - rnn:  GRU(input_size=1, hidden_size=16, num_layers=1, batch_first=True, bias=False)
      - ln:   Linear(in_features=16, out_features=1, bias=False)
      - fir2: Conv1d(1, 1, kernel_size=256, bias=False, causal left-padding=255)
    """
    def __init__(
        self,
        fir1_size: int = 256,
        fir2_size: int = 256,
        cnn_channels: int = 3,
        cnn_kernel_size: int = 64,
        hidden_size: int = 16,
    ):
        super().__init__()
        self.fir1_size = fir1_size
        self.fir2_size = fir2_size
        self.cnn_channels = cnn_channels
        self.cnn_kernel_size = cnn_kernel_size
        self.hidden_size = hidden_size

        self.cnn = nn.Conv1d(1, cnn_channels, kernel_size=cnn_kernel_size, bias=False, padding=0)
        self.fir1 = nn.Conv1d(1, 1, kernel_size=fir1_size, bias=False, padding=0)
        self.rnn = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bias=False,
        )
        self.ln = nn.Linear(in_features=hidden_size, out_features=1, bias=False)
        self.fir2 = nn.Conv1d(1, 1, kernel_size=fir2_size, bias=False, padding=0)

        self.reset_parameters()

    @property
    def model_tag(self) -> str:
        return "AM3"

    @property
    def receptive_field_pad(self) -> int:
        return (self.cnn_kernel_size - 1) + (self.fir1_size - 1) + (self.fir2_size - 1)

    def reset_parameters(self) -> None:
        # Initializing CNN
        nn.init.xavier_uniform_(self.cnn.weight)

        # FIR initialization close to causal identity (delta impulse at causal end tap K-1)
        nn.init.zeros_(self.fir1.weight)
        with torch.no_grad():
            self.fir1.weight[0, 0, self.fir1_size - 1] = 1.0

        nn.init.zeros_(self.fir2.weight)
        with torch.no_grad():
            self.fir2.weight[0, 0, self.fir2_size - 1] = 1.0

        # GRU initialization (bias=False in AM3)
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)

        nn.init.xavier_uniform_(self.ln.weight)

    def forward(self, x: torch.Tensor, target_len: int = 512) -> torch.Tensor:
        """
        Matches FUN_14003dec0 in AudioTrainHHAG.exe:
        When x has shape (B, 1, target_len + 573) [e.g. 512 + 573 = 1085],
        cnn, fir1, and fir2 execute pure valid convolution (padding=0) over real historical audio:
          1085 -> cnn -> 1022 -> fir1 -> 767 -> asinh -> GRU -> Linear -> 767 -> fir2 -> 512.
        """
        needed_len = target_len + self.receptive_field_pad
        if x.shape[-1] < needed_len:
            x = F.pad(x, (needed_len - x.shape[-1], 0))

        # Multi-channel valid CNN filter bank: (B, 1, T) -> (B, 3, T - 63)
        c = self.cnn(x)
        # Hardware AM3 quadratic Volterra / gated product (FUN_14003dec0 at 0x14003e43b-0x14003e464):
        # x_cnn = c0 * c1 + c2
        x_cnn = c[:, 0:1, :] * c[:, 1:2, :] + c[:, 2:3, :]

        # Valid input FIR filtering: (B, 1, T - 63) -> (B, 1, T - 318)
        x = self.fir1(x_cnn)

        # Non-linear saturation: arsinh(x) = ln(x + sqrt(x^2 + 1))
        x = torch.asinh(x)

        # Transpose to (B, T - 318, 1) for GRU
        x = x.transpose(1, 2)
        x, _ = self.rnn(x)

        # Linear projection
        x = self.ln(x)

        # Transpose back to (B, 1, T - 318)
        x = x.transpose(1, 2)

        # Valid output FIR filtering: (B, 1, T - 318) -> (B, 1, T - 573)
        x = self.fir2(x)

        # Crop to target length if extra delay padding was present
        if x.shape[-1] > target_len:
            x = x[:, :, -target_len:]
        return x

    def export_weights_array(self) -> torch.Tensor:
        """
        Exact binary layout verified against AudioTrainHHAG.exe and kpa-blue.am3Data:
          1. fir1 (256)
          2. rnn.weight_ih_l0 (48)
          3. rnn.weight_hh_l0 (768)
          4. ln.weight (16)
          5. fir2 (256)
          6. cnn.weight (192)
        Total floats = 1536 (6144 bytes)
        """
        with torch.no_grad():
            fir1_w = self.fir1.weight.detach().cpu().view(-1)
            w_ih = self.rnn.weight_ih_l0.detach().cpu().view(-1)
            w_hh = self.rnn.weight_hh_l0.detach().cpu().view(-1)
            ln_w = self.ln.weight.detach().cpu().view(-1)
            fir2_w = self.fir2.weight.detach().cpu().view(-1)
            cnn_w = self.cnn.weight.detach().cpu().view(-1)

            tensors = [fir1_w, w_ih, w_hh, ln_w, fir2_w, cnn_w]
            return torch.cat(tensors, dim=0).to(torch.float32)

    def load_weights_array(self, weights: torch.Tensor) -> None:
        """Loads weights from contiguous 1D float32 array matching AudioTrainHHAG.exe."""
        w = weights.view(-1).cpu()
        idx = 0

        # fir1
        f1_num = self.fir1.weight.numel()
        self.fir1.weight.data.copy_(w[idx : idx + f1_num].view_as(self.fir1.weight))
        idx += f1_num

        # rnn (bias=False)
        w_ih_num = self.rnn.weight_ih_l0.numel()
        self.rnn.weight_ih_l0.data.copy_(w[idx : idx + w_ih_num].view_as(self.rnn.weight_ih_l0))
        idx += w_ih_num

        w_hh_num = self.rnn.weight_hh_l0.numel()
        self.rnn.weight_hh_l0.data.copy_(w[idx : idx + w_hh_num].view_as(self.rnn.weight_hh_l0))
        idx += w_hh_num

        # ln
        ln_num = self.ln.weight.numel()
        self.ln.weight.data.copy_(w[idx : idx + ln_num].view_as(self.ln.weight))
        idx += ln_num

        # fir2
        f2_num = self.fir2.weight.numel()
        self.fir2.weight.data.copy_(w[idx : idx + f2_num].view_as(self.fir2.weight))
        idx += f2_num

        # cnn
        cnn_num = self.cnn.weight.numel()
        self.cnn.weight.data.copy_(w[idx : idx + cnn_num].view_as(self.cnn.weight))
        idx += cnn_num
