from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from audiotrain.models.base import BaseAmpModel
from audiotrain.models.am3 import AM3Model
from audiotrain.models.am4 import AM4Model

logger = logging.getLogger(__name__)


class NAMDilatedConvBlock(nn.Module):
    """Single WaveNet layer block from NAM A1 specification."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        gated: bool = False,
    ):
        super().__init__()
        self.gated = gated
        out_mult = 2 if gated else 1
        self.conv = nn.Conv1d(
            in_channels,
            out_channels * out_mult,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=True,
        )
        self.dilation = dilation
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_len = (self.kernel_size - 1) * self.dilation
        x_pad = F.pad(x, (pad_len, 0))
        h = self.conv(x_pad)
        if self.gated:
            a, b = torch.chunk(h, 2, dim=1)
            return torch.tanh(a) * torch.sigmoid(b)
        return torch.tanh(h)


class NAMWaveNetHead(nn.Module):
    """Standard 1x1 projection head for A1 WaveNet."""
    def __init__(self, in_channels: int, out_channels: int = 1, bias: bool = True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class NAMA1Model(nn.Module):
    """
    Pure PyTorch implementation of NAM Architecture 1 (A1) WaveNet models
    (Standard, Lite, Feather, Nano).
    """
    def __init__(self, config: Dict[str, Any], weights: List[float]):
        super().__init__()
        self.architecture = config.get("architecture", "WaveNet")
        self.architecture_variant = "A1-WaveNet"
        self.layers_config = config.get("config", {}).get("layers", [])
        self.head_scale = config.get("config", {}).get("head_scale", 1.0)
        self.sample_rate = float(config.get("sample_rate", 44100.0) or 44100.0)
        self.metadata = config.get("metadata", {})

        self.blocks = nn.ModuleList()
        self.heads = nn.ModuleList()

        for layer_cfg in self.layers_config:
            in_ch = layer_cfg.get("input_size", 1)
            ch = layer_cfg.get("channels", 16)
            k = layer_cfg.get("kernel_size", 3)
            dilations = layer_cfg.get("dilations", [1])
            gated = layer_cfg.get("gated", False)
            head_bias = layer_cfg.get("head_bias", True)

            layer_blocks = nn.ModuleList()
            for d in dilations:
                block = NAMDilatedConvBlock(in_ch, ch, kernel_size=k, dilation=d, gated=gated)
                layer_blocks.append(block)
                in_ch = ch
            self.blocks.append(layer_blocks)

            head = NAMWaveNetHead(ch, 1, bias=head_bias)
            self.heads.append(head)

        self._load_flat_weights(weights)

    def _load_flat_weights(self, weights: List[float]) -> None:
        """Loads weights from contiguous list in NAM A1 JSON format."""
        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        offset = 0
        for p in self.parameters():
            num = p.numel()
            if offset + num <= len(weight_tensor):
                chunk = weight_tensor[offset : offset + num].view_as(p)
                p.data.copy_(chunk)
                offset += num

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, T)
        returns: (B, 1, T)
        """
        out = torch.zeros_like(x)
        current_x = x

        for layer_blocks, head in zip(self.blocks, self.heads):
            h = current_x
            for block in layer_blocks:
                h = block(h)
            out = out + head(h)
            current_x = h

        return out * self.head_scale


class NAMA2Model(nn.Module):
    """
    Pure PyTorch implementation of NAM Architecture 2 (A2) WaveNet models
    (A2-Full, A2-Lite, and custom A2 variants).

    A2 Specifications:
    - Single layer array with 23 dilated layers (receptive field ~6,350 samples).
    - Channels: 8 for A2-Full, 3 for A2-Lite.
    - LeakyReLU(0.01) activation on every layer (no gating).
    - Input mixin conditioning on every layer.
    - Residual layer 1x1 projection.
    - 16-tap FIR head convolution filter.
    - Trailing head_scale factor.
    """
    def __init__(
        self,
        channels: int,
        kernel_sizes: List[int],
        dilations: List[int],
        weights: List[float],
        sample_rate: float = 48000.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.architecture = "WaveNet"
        self.architecture_variant = f"A2-{'Full' if channels == 8 else 'Lite' if channels == 3 else f'{channels}ch'}"
        self.sample_rate = float(sample_rate or 48000.0)
        self.metadata = metadata or {}
        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.dilations = dilations
        self.num_layers = len(kernel_sizes)

        # 1. Rechannel Conv1x1: 1 -> Channels (no bias)
        self.rechannel = nn.Conv1d(1, channels, kernel_size=1, bias=False)

        # 2. Layer array convolutions
        self.convs = nn.ModuleList()
        self.mixins = nn.ModuleList()
        self.layer1x1s = nn.ModuleList()

        for k, d in zip(kernel_sizes, dilations):
            self.convs.append(nn.Conv1d(channels, channels, kernel_size=k, dilation=d, bias=True))
            self.mixins.append(nn.Conv1d(1, channels, kernel_size=1, bias=False))
            self.layer1x1s.append(nn.Conv1d(channels, channels, kernel_size=1, bias=True))

        # 3. Head FIR filter: Channels -> 1, kernel_size=16 (with bias)
        self.head = nn.Conv1d(channels, 1, kernel_size=16, bias=True)
        self.head_scale: float = 1.0

        self._load_weights(weights)

    def _load_weights(self, weights: List[float]) -> None:
        """Loads weights exactly matching NeuralAmpModelerCore A2 memory order."""
        w = torch.tensor(weights, dtype=torch.float32)
        idx = 0

        def take(n: int) -> torch.Tensor:
            nonlocal idx
            res = w[idx : idx + n]
            idx += n
            return res

        # Rechannel
        self.rechannel.weight.data.copy_(take(self.channels).view(self.channels, 1, 1))

        # 23 Layers
        for i in range(self.num_layers):
            k = self.kernel_sizes[i]
            # Conv1D: (channels, channels, k)
            self.convs[i].weight.data.copy_(
                take(self.channels * self.channels * k).view(self.channels, self.channels, k)
            )
            self.convs[i].bias.data.copy_(take(self.channels))

            # Input mixin: (channels, 1, 1)
            self.mixins[i].weight.data.copy_(take(self.channels).view(self.channels, 1, 1))

            # Layer 1x1: (channels, channels, 1)
            self.layer1x1s[i].weight.data.copy_(
                take(self.channels * self.channels).view(self.channels, self.channels, 1)
            )
            self.layer1x1s[i].bias.data.copy_(take(self.channels))

        # Head rechannel: (1, channels, 16)
        self.head.weight.data.copy_(take(self.channels * 16).view(1, self.channels, 16))
        self.head.bias.data.copy_(take(1))

        # Head scale: trailing float
        if idx < len(w):
            self.head_scale = float(take(1).detach()[0])

        if idx != len(w):
            logger.warning(f"A2 Model: consumed {idx} of {len(w)} weights ({len(w) - idx} trailing).")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pure PyTorch causal forward pass matching A2 fast path.
        x: (B, 1, T)
        returns: (B, 1, T)
        """
        cond = x
        layer_in = self.rechannel(cond)
        head_sum = torch.zeros_like(layer_in)

        for i in range(self.num_layers):
            k = self.kernel_sizes[i]
            d = self.dilations[i]
            pad = (k - 1) * d
            x_pad = F.pad(layer_in, (pad, 0))
            conv_out = self.convs[i](x_pad)
            mixin_out = self.mixins[i](cond)
            act = F.leaky_relu(conv_out + mixin_out, negative_slope=0.01)
            head_sum = head_sum + act
            layer_in = layer_in + self.layer1x1s[i](act)

        # Head 16-tap FIR causal convolution
        head_pad = 16 - 1
        head_sum_pad = F.pad(head_sum, (head_pad, 0))
        out = self.head(head_sum_pad) * self.head_scale
        return out


class NAMModel:
    """
    Unified factory loader and router for Neural Amp Modeler (.nam) files.
    Supports both Architecture 1 (A1 WaveNet) and Architecture 2 (A2-Full, A2-Lite,
    and SlimmableContainer).
    """
    @classmethod
    def from_file(
        cls,
        nam_file: Union[str, Path],
        submodel_index: int = -1,
    ) -> Union[NAMA1Model, NAMA2Model]:
        """
        Loads a NAM model from a .nam JSON file.

        Args:
            nam_file: Path to the .nam file.
            submodel_index: For SlimmableContainer models, which submodel to use.
                           Defaults to -1 (highest fidelity, e.g. A2-Full).
                           Use 0 for low-latency/embedded (e.g. A2-Lite).
        """
        nam_path = Path(nam_file)
        if not nam_path.exists():
            raise FileNotFoundError(f"NAM model file not found: {nam_path}")

        with open(nam_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        architecture = data.get("architecture", "WaveNet")
        sample_rate = data.get("sample_rate", 48000.0) or 48000.0
        metadata = data.get("metadata", {})

        # Handle SlimmableContainer (standard A2 packaging)
        if architecture == "SlimmableContainer":
            submodels = data.get("config", {}).get("submodels", [])
            if not submodels:
                raise ValueError(f"SlimmableContainer in {nam_path.name} contains no submodels.")

            # Select requested submodel (default last = highest fidelity A2-Full)
            chosen_entry = submodels[submodel_index]
            target_model_data = chosen_entry.get("model", {})
            max_val = chosen_entry.get("max_value", 1.0)
            logger.info(
                f"Loaded SlimmableContainer from {nam_path.name}: selecting submodel "
                f"[{submodel_index}] (max_value={max_val}) out of {len(submodels)} submodels."
            )
            # Recursively build from inner submodel dict, preserving container sample_rate & metadata
            target_model_data.setdefault("sample_rate", sample_rate)
            if not target_model_data.get("metadata"):
                target_model_data["metadata"] = metadata
            return cls._from_dict(target_model_data)

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> Union[NAMA1Model, NAMA2Model]:
        architecture = data.get("architecture", "WaveNet")
        weights = data.get("weights", [])
        cfg = data.get("config", {})
        layers = cfg.get("layers", [])
        sample_rate = data.get("sample_rate", 48000.0) or 48000.0
        metadata = data.get("metadata", {})

        if architecture != "WaveNet":
            logger.warning(
                f"Unknown NAM architecture '{architecture}'. Falling back to WaveNet parser."
            )

        if not layers:
            raise ValueError("Invalid NAM model: no layers found in model config.")

        la0 = layers[0]
        # Detect Architecture 2 (A2)
        is_a2 = False
        if "kernel_sizes" in la0:
            is_a2 = True
        elif len(la0.get("dilations", [])) == 23 and la0.get("channels") in (3, 8):
            is_a2 = True
        elif isinstance(la0.get("head"), dict) and la0.get("head", {}).get("kernel_size") == 16:
            is_a2 = True

        if is_a2:
            channels = la0.get("channels", 8)
            kernel_sizes = la0.get("kernel_sizes", [6] * 14 + [15, 15] + [6] * 7)
            dilations = la0.get(
                "dilations",
                [1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13, 1, 3, 7, 17, 41, 101, 239]
            )
            model = NAMA2Model(
                channels=channels,
                kernel_sizes=kernel_sizes,
                dilations=dilations,
                weights=weights,
                sample_rate=sample_rate,
                metadata=metadata,
            )
            logger.info(
                f"Constructed NAM A2 Model ({model.architecture_variant}): "
                f"{len(kernel_sizes)} layers, {len(weights)} weights, {sample_rate} Hz."
            )
            return model
        else:
            model = NAMA1Model(data, weights)
            logger.info(
                f"Constructed NAM A1 Model: {len(layers)} layer blocks, "
                f"{len(weights)} weights, {sample_rate} Hz."
            )
            return model


def create_distillation_pair(
    nam_path: Union[str, Path],
    target_model_type: int = 1,
    submodel_index: int = -1,
) -> Tuple[Union[NAMA1Model, NAMA2Model], BaseAmpModel]:
    """
    Creates a teacher NAM model (A1 or A2) and a student (AM3 or AM4) model for distillation.
    """
    teacher = NAMModel.from_file(nam_path, submodel_index=submodel_index)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    if target_model_type in (0, 4):
        student = AM3Model()
    else:
        student = AM4Model()

    return teacher, student
