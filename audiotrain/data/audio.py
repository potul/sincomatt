from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import scipy.signal
import soundfile as sf
import torch


@dataclass
class LoadedAudio:
    dry: np.ndarray             # 1D float32 array normalized to [-1.0, 1.0]
    target: np.ndarray          # 1D float32 array normalized to [-1.0, 1.0]
    sample_rate: int            # Standard 44100 Hz
    dry_loudness: float         # Original peak amplitude before normalization
    target_loudness: float      # Original target peak amplitude before normalization
    total_samples: int


def resolve_audio_path(path: Union[str, Path]) -> Path:
    p = Path(path)
    if p.is_file():
        return p
    # If path provided without extension, check for .wav
    wav_p = p.with_suffix(".wav")
    if wav_p.is_file():
        return wav_p
    # Also check if it's a directory with a single wav file
    if p.is_dir():
        wavs = list(p.glob("*.wav"))
        if wavs:
            return wavs[0]
    raise FileNotFoundError(f"Audio file not found: {path} (also checked {wav_p})")


def resample_signal(signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """High-fidelity polyphase resampling matching libsamplerate sinc interpolation."""
    if orig_sr == target_sr:
        return signal
    # Use gcd for polyphase rational factor
    gcd = np.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    resampled = scipy.signal.resample_poly(signal, up, down, axis=0)
    return resampled.astype(np.float32)


def load_and_preprocess_audio(
    file_path: Union[str, Path],
    target_sr: int = 44100,
    normalize: bool = True,
) -> LoadedAudio:
    """
    Loads stereo audio file (Ch0 = Dry Input, Ch1 = Target Amp Output),
    resamples to 44.1 kHz, and calculates peak loudness factors for normalization.
    """
    resolved_path = resolve_audio_path(file_path)
    audio_data, orig_sr = sf.read(str(resolved_path), dtype="float32")

    if audio_data.ndim == 1:
        raise ValueError(
            f"Expected stereo WAV (Ch0: Dry, Ch1: Target). Got mono audio from {resolved_path}"
        )
    if audio_data.shape[1] < 2:
        raise ValueError(
            f"Expected stereo audio with at least 2 channels, got {audio_data.shape[1]}"
        )

    dry = audio_data[:, 0]
    target = audio_data[:, 1]

    # Resample to target_sr if required
    if orig_sr != target_sr:
        dry = resample_signal(dry, orig_sr, target_sr)
        target = resample_signal(target, orig_sr, target_sr)

    # Ensure length consistency
    min_len = min(len(dry), len(target))
    dry = dry[:min_len]
    target = target[:min_len]

    # Peak normalization (FUN_14003bfd0 in binary)
    dry_loudness = float(np.max(np.abs(dry))) if len(dry) > 0 else 1.0
    target_loudness = float(np.max(np.abs(target))) if len(target) > 0 else 1.0

    if normalize:
        if dry_loudness > 1e-8:
            dry = dry / dry_loudness
        if target_loudness > 1e-8:
            target = target / target_loudness

    return LoadedAudio(
        dry=dry.astype(np.float32),
        target=target.astype(np.float32),
        sample_rate=target_sr,
        dry_loudness=dry_loudness,
        target_loudness=target_loudness,
        total_samples=min_len,
    )


def save_stereo_wav(
    output_path: Union[str, Path],
    dry: np.ndarray,
    target: np.ndarray,
    sample_rate: int = 44100,
    target_loudness: Optional[float] = None,
) -> None:
    """
    Saves stereo audio where:
    Ch0 = Dry input
    Ch1 = Model prediction / target restored with target_loudness
    """
    os.makedirs(Path(output_path).parent, exist_ok=True)
    min_len = min(len(dry), len(target))
    dry = dry[:min_len].copy()
    target = target[:min_len].copy()

    # Restore target loudness if provided
    if target_loudness is not None and target_loudness > 0:
        target = target * target_loudness

    stereo = np.column_stack([dry, target]).astype(np.float32)
    # Clip softly to avoid extreme wrap-around distortion if any
    stereo = np.clip(stereo, -1.0, 1.0)
    sf.write(str(output_path), stereo, sample_rate, subtype="FLOAT")
