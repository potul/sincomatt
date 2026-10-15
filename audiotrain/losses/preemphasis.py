from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PreEmphasisFilter(nn.Module):
    """
    Perceptual 2nd-order pre-emphasis high-pass filter:
    H(z) = (1 - 0.85 z^-1)^2 = 1 - 1.7 z^-1 + 0.7225 z^-2

    Difference equation:
    y_filt[n] = x[n] - 1.7 * x[n-1] + 0.7225 * x[n-2]
    """
    def __init__(self):
        super().__init__()
        # PyTorch Conv1d cross-correlation kernel:
        # conv1d([x0, x1, x2], [k0, k1, k2]) computes k0*x0 + k1*x1 + k2*x2.
        # To compute 1.0 * x[t] - 1.7 * x[t-1] + 0.7225 * x[t-2],
        # with causal left padding of 2 samples:
        # kernel = [0.7225, -1.7, 1.0]
        kernel = torch.tensor([0.7225, -1.7, 1.0], dtype=torch.float32).view(1, 1, 3)
        self.register_buffer("kernel", kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (B, 1, T) or (B, T)
        returns filtered tensor of same shape
        """
        orig_dim = x.dim()
        if orig_dim == 2:
            x = x.unsqueeze(1)

        # Causal padding on left
        x_pad = F.pad(x, (2, 0), mode="replicate")
        kernel = self.kernel.to(device=x.device, dtype=x.dtype)
        y = F.conv1d(x_pad, kernel)

        if orig_dim == 2:
            y = y.squeeze(1)
        return y
