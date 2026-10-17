from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

import numpy as np
import torch

from audiotrain.models.base import BaseAmpModel


HEADER_SIZE = 60
SIGNATURE_DEFAULT = 0x6464      # Hardware validation signature (100.100, uint16 0x6464 = b'dd')
CONTROL_WORD_DEFAULT = 0x1E3C   # Control word (uint16 0x1E3C = 7740)
RESERVED_DEFAULT = 0x0000       # Reserved padding (uint16 0x0000)


def build_header(
    magic_tag: str,
    model_name: str,
    version_sig: int = SIGNATURE_DEFAULT,
    control_word: int = CONTROL_WORD_DEFAULT,
    reserved: int = RESERVED_DEFAULT,
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
      4..17:  Model name (14 bytes, ASCII null-padded)
      18..19: Hardware validation signature (uint16, 0x6464)
      20..21: Control word (uint16, 0x1E3C)
      22..23: Reserved padding (uint16, 0x0000)
      24..35: Bass EQ band (freq, min_gain, max_gain as 3 x float32)
      36..47: Mid EQ band (freq, min_gain, max_gain as 3 x float32)
      48..59: Treble EQ band (freq, min_gain, max_gain as 3 x float32)
    Total = 60 bytes (0x3C).
    """
    # Magic tag: 4 bytes
    magic_bytes = magic_tag.encode("ascii")[:3] + b"\x00"
    magic_bytes = magic_bytes.ljust(4, b"\x00")

    # Model name: strictly 14 bytes (offset 4..17)
    name_bytes = model_name.encode("ascii", errors="replace")[:14].ljust(14, b"\x00")

    # Header metadata & EQ payload (3 x uint16 + 9 x float32 = 6 + 36 = 42 bytes)
    meta_eq_payload = struct.pack(
        "<HHH9f",
        version_sig,
        control_word,
        reserved,
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

    header = magic_bytes + name_bytes + meta_eq_payload
    assert len(header) == HEADER_SIZE, f"Header size must be 60 bytes, got {len(header)}"
    return header


def parse_header(header_bytes: bytes) -> Dict[str, Any]:
    if len(header_bytes) < HEADER_SIZE:
        raise ValueError(f"Header must be at least {HEADER_SIZE} bytes, got {len(header_bytes)}")

    magic = header_bytes[:4].split(b"\x00")[0].decode("ascii", errors="ignore")
    name = header_bytes[4:18].split(b"\x00")[0].decode("ascii", errors="ignore")
    version_sig, control_word, reserved = struct.unpack("<HHH", header_bytes[18:24])
    words = struct.unpack("<9f", header_bytes[24:60])

    return {
        "magic": magic,
        "name": name,
        "version_sig": version_sig,
        "control_word": control_word,
        "reserved": reserved,
        "eq_bass": (words[0], words[1], words[2]),
        "eq_mid": (words[3], words[4], words[5]),
        "eq_treble": (words[6], words[7], words[8]),
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
        version_sig=int(eq.get("version_sig", SIGNATURE_DEFAULT)),
        control_word=int(eq.get("control_word", CONTROL_WORD_DEFAULT)),
        reserved=int(eq.get("reserved", RESERVED_DEFAULT)),
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
