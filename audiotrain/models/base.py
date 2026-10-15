from __future__ import annotations

import abc
from typing import Dict, Any, Union
from pathlib import Path
import torch
import torch.nn as nn


class BaseAmpModel(nn.Module, abc.ABC):
    """Abstract base class for all Sinco neural amp models."""

    @property
    @abc.abstractmethod
    def model_tag(self) -> str:
        """Returns magic tag, e.g. 'AM3', 'AM4', 'IR'."""
        pass

    @abc.abstractmethod
    def export_weights_array(self) -> torch.Tensor:
        """Returns contiguous 1D float32 tensor of all weights in binary serialization order."""
        pass

    @abc.abstractmethod
    def load_weights_array(self, weights: torch.Tensor) -> None:
        """Loads weights from contiguous 1D float32 tensor."""
        pass

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_checkpoint(self, path: Union[str, Path], metadata: Dict[str, Any] = None) -> None:
        payload = {
            "model_tag": self.model_tag,
            "state_dict": self.state_dict(),
            "metadata": metadata or {},
        }
        torch.save(payload, str(path))

    def load_checkpoint(self, path: Union[str, Path]) -> Dict[str, Any]:
        payload = torch.load(str(path), map_location="cpu")
        self.load_state_dict(payload["state_dict"])
        return payload.get("metadata", {})
