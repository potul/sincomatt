from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

import numpy as np
import torch

from audiotrain.models.base import BaseAmpModel


HEADER_SIZE = 60
CONTROL_WORD_DEFAULT = 0x1E3C  # Control word observed across Sinco binary weights


def build_header(
    magic_tag: str,
    model_name: str,
    control_word: int = CONTROL_WORD_DEFAULT,
    bass_freq: float = 100.0,
    bass_min: float = -6.0,
    bass_max: float = 6.0,
    mid_freq: float = 2000.0,
    mid_min: float = -6.0,
    mid_max: float = 6.0,
    treble_freq: float = 4000.0,
    treble_min: float = -6.0,
    treble_max: float = 6.0,
) -> bytes:
    """
    Constructs the exact 60-byte binary header required by SincoANN hardware & VSTs:
      0..3:   Magic (4 bytes, e.g. b'AM3\x00' or b'AM4\x00')
      4..19:  Model name (16 bytes, ASCII null-padded)
      20..23: Control word (uint32, 0x1E3C)
      24..35: Bass EQ band (freq, min_gain, max_gain as 3 x float32)
      36..47: Mid EQ band (freq, min_gain, max_gain as 3 x float32)
      48..59: Treble EQ band (freq, min_gain, max_gain as 3 x float32)
    Total = 60 bytes (0x3C).
    """
    # Magic tag: 4 bytes
    magic_bytes = magic_tag.encode("ascii")[:3] + b"\x00"
    magic_bytes = magic_bytes.ljust(4, b"\x00")

    # Model name: 16 bytes
    name_bytes = model_name.encode("ascii", errors="replace")[:16].ljust(16, b"\x00")

    # Header payload (1 uint32 + 9 float32 = 10 words = 40 bytes)
    eq_payload = struct.pack(
        "<I9f",
        control_word,
        bass_freq,
        bass_min,
        bass_max,
        mid_freq,
        mid_min,
        mid_max,
        treble_freq,
        treble_min,
        treble_max,
    )

    header = magic_bytes + name_bytes + eq_payload
    assert len(header) == HEADER_SIZE, f"Header size must be 60 bytes, got {len(header)}"
    return header


def parse_header(header_bytes: bytes) -> Dict[str, Any]:
    if len(header_bytes) < HEADER_SIZE:
        raise ValueError(f"Header must be at least {HEADER_SIZE} bytes, got {len(header_bytes)}")

    magic = header_bytes[:4].split(b"\x00")[0].decode("ascii", errors="ignore")
    name = header_bytes[4:20].split(b"\x00")[0].decode("ascii", errors="ignore")
    words = struct.unpack("<I9f", header_bytes[20:60])

    return {
        "magic": magic,
        "name": name,
        "control_word": words[0],
        "eq_bass": (words[1], words[2], words[3]),
        "eq_mid": (words[4], words[5], words[6]),
        "eq_treble": (words[7], words[8], words[9]),
    }


def pack_am_data(
    model: BaseAmpModel,
    model_name: str = "sinco_model",
    eq_settings: Optional[Dict[str, float]] = None,
) -> bytes:
    eq = eq_settings or {}
    header = build_header(
        magic_tag=model.model_tag,
        model_name=model_name,
        bass_freq=eq.get("bass_freq", 100.0),
        bass_min=eq.get("bass_min", -6.0),
        bass_max=eq.get("bass_max", 6.0),
        mid_freq=eq.get("mid_freq", 2000.0),
        mid_min=eq.get("mid_min", -6.0),
        mid_max=eq.get("mid_max", 6.0),
        treble_freq=eq.get("treble_freq", 4000.0),
        treble_min=eq.get("treble_min", -6.0),
        treble_max=eq.get("treble_max", 6.0),
    )

    weights_tensor = model.export_weights_array()
    weights_bytes = weights_tensor.detach().cpu().numpy().astype(np.float32).tobytes()
    return header + weights_bytes


def unpack_am_data(
    data: bytes,
    model: Optional[BaseAmpModel] = None,
) -> Tuple[Dict[str, Any], torch.Tensor]:
    if len(data) < HEADER_SIZE:
        raise ValueError("Data too short for header")

    header_info = parse_header(data[:HEADER_SIZE])
    weights_np = np.frombuffer(data[HEADER_SIZE:], dtype=np.float32)
    weights_tensor = torch.from_numpy(weights_np.copy())

    if model is not None:
        model.load_weights_array(weights_tensor)

    return header_info, weights_tensor


def save_am_data(
    file_path: Union[str, Path],
    model: BaseAmpModel,
    model_name: str = "sinco_model",
    eq_settings: Optional[Dict[str, float]] = None,
) -> None:
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    binary_content = pack_am_data(model, model_name=model_name, eq_settings=eq_settings)
    with open(p, "wb") as f:
        f.write(binary_content)


def load_am_data(
    file_path: Union[str, Path],
    model: Optional[BaseAmpModel] = None,
) -> Tuple[Dict[str, Any], torch.Tensor]:
    with open(file_path, "rb") as f:
        data = f.read()
    return unpack_am_data(data, model=model)


def save_lossrec(file_path: Union[str, Path], best_loss: float) -> None:
    """Writes the .am4lossrec or .am3lossrec file."""
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(f"loss : {best_loss:.7f}\n")


def load_lossrec(file_path: Union[str, Path]) -> float:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    # Format: "loss : 0.0947242"
    val_str = content.split(":")[-1].strip()
    return float(val_str)
