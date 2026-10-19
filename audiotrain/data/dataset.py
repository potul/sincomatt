from __future__ import annotations

from typing import Tuple, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from audiotrain.data.audio import LoadedAudio


class AudioFrameDataset(Dataset):
    """
    Framed audio dataset for causal time-series training.
    Slices continuous audio into target frames of size 512 samples, while providing
    the preceding `receptive_field_pad` samples of real continuous dry audio so that
    valid Conv1d layers (and GRU state warmup) operate on real historical context:
      - Target chunk: target[i * 512 + delay : (i + 1) * 512 + delay] (length 512)
      - Input chunk:  dry[i * 512 - receptive_field_pad : (i + 1) * 512] (length 512 + receptive_field_pad)
        (left-padded with zeros only for the very first frame(s) where i * 512 < receptive_field_pad)
    """
    def __init__(
        self,
        dry: np.ndarray,
        target: np.ndarray,
        frame_size: int = 512,
        delay: int = 0,
        receptive_field_pad: int = 1022,
        start_frame: int = 0,
        num_frames: Optional[int] = None,
    ):
        super().__init__()
        self.frame_size = frame_size
        self.delay = max(0, delay)
        self.receptive_field_pad = max(0, receptive_field_pad)
        self.dry = dry
        self.target = target

        total_available = len(dry) - self.delay
        max_possible_frames = max(0, total_available // frame_size)

        self.start_frame = start_frame
        if num_frames is None:
            self.num_frames = max(0, max_possible_frames - start_frame)
        else:
            self.num_frames = min(num_frames, max(0, max_possible_frames - start_frame))

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_idx = self.start_frame + idx
        frame_end = (frame_idx + 1) * self.frame_size
        in_start = frame_idx * self.frame_size - self.receptive_field_pad

        if in_start < 0:
            pad_zeros = -in_start
            in_chunk = np.pad(self.dry[0:frame_end], (pad_zeros, 0), mode="constant")
        else:
            in_chunk = self.dry[in_start:frame_end]

        tgt_start = frame_idx * self.frame_size + self.delay
        tgt_end = tgt_start + self.frame_size
        tgt_chunk = self.target[tgt_start:tgt_end]

        # Shape: (1, seq_len)
        x = torch.from_numpy(in_chunk).unsqueeze(0).to(torch.float32)
        y = torch.from_numpy(tgt_chunk).unsqueeze(0).to(torch.float32)
        return x, y


def detect_reamp_profile(dry: np.ndarray, sr: int) -> str:
    """
    Auto-detects whether the dry excitation signal (Channel 0) is a standardized
    Neural Amp Modeler (NAM) reamp input file ('NAM_V3', 'NAM_V2', 'NAM_V1', 'NAM_PROTEUS')
    or an original ANN / custom reamp file ('LEGACY_60_40').

    Detection uses sample-rate-, bit-depth-, and gain-invariant acoustic signatures.
    """
    if sr <= 0 or len(dry) == 0:
        return "LEGACY_60_40"

    duration = len(dry) / float(sr)
    global_peak = float(np.max(np.abs(dry)))
    if global_peak < 1e-7:
        return "LEGACY_60_40"

    def _seg(t0: float, t1: float) -> np.ndarray:
        i0 = max(0, min(len(dry), int(round(t0 * sr))))
        i1 = max(i0, min(len(dry), int(round(t1 * sr))))
        return dry[i0:i1]

    def _rms(arr: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(arr, dtype=np.float64)))) if len(arr) > 0 else 0.0

    def _crest(arr: np.ndarray) -> float:
        if len(arr) == 0:
            return 0.0
        pk = float(np.max(np.abs(arr)))
        return pk / (_rms(arr) + 1e-12)

    # 1. NAM v3_0_0.wav / input.wav (~190.0s):
    #    [0..9s] Val1, [9..10s] Silence, [10..12s] Blips, [12..17s] Chirp+Noise,
    #    [180.5..181s] Silence, [181..190s] Val2 (identical to Val1)
    if 189.0 <= duration <= 192.0:
        sil1_rms = _rms(_seg(9.1, 9.9))
        sil2_rms = _rms(_seg(180.55, 180.95))
        chirp_rms = _rms(_seg(12.2, 14.8))
        blips_crest = _crest(_seg(10.0, 12.0))
        val1 = _seg(0.0, 9.0)
        val2 = _seg(181.0, 190.0)
        if (
            chirp_rms > 0.02 * global_peak
            and sil1_rms < 0.05 * chirp_rms
            and sil2_rms < 0.05 * chirp_rms
            and blips_crest > 15.0
            and len(val1) == len(val2)
            and len(val1) > 0
        ):
            corr = float(np.corrcoef(val1, val2)[0, 1])
            if corr > 0.98:
                return "NAM_V3"

    # 2. NAM v2_0_0.wav (~191.0s):
    #    [0..2s] Blips, [2..7s] Chirp+Noise, [171..180s] Val1, [180..189s] Val2, [189..191s] Blips
    if 190.0 <= duration <= 193.0:
        start_crest = _crest(_seg(0.0, 2.0))
        end_crest = _crest(_seg(189.0, 191.0))
        val1 = _seg(171.0, 180.0)
        val2 = _seg(180.0, 189.0)
        if start_crest > 15.0 and end_crest > 15.0 and len(val1) == len(val2) and len(val1) > 0:
            corr = float(np.corrcoef(val1, val2)[0, 1])
            if corr > 0.98:
                return "NAM_V2"

    # 3. NAM v1_1_1.wav / v1.wav (~180.0s):
    #    [0..1s] Blips at 0.25s & 0.75s, [1..7s] Sine sweep + white noise, [171..180s] Val
    if 179.0 <= duration <= 182.0:
        start_crest = _crest(_seg(0.0, 1.0))
        sweep_rms = _rms(_seg(1.2, 6.8))
        if start_crest > 15.0 and sweep_rms > 0.05 * global_peak:
            return "NAM_V1"

    # 4. Proteus_Capture.wav (~210.86s):
    #    [0..1s] Blips at 0.0s & 0.5s, [1..17s] Sine sweeps + white noise, last 10s Val
    if 209.5 <= duration <= 212.5:
        blip_crest = _crest(_seg(0.1, 0.9))
        sweep_rms = _rms(_seg(1.5, 16.5))
        if blip_crest > 15.0 and sweep_rms > 0.05 * global_peak:
            return "NAM_PROTEUS"

    return "LEGACY_60_40"


def create_train_val_datasets(
    loaded_audio: LoadedAudio,
    frame_size: int = 512,
    delay: int = 0,
    receptive_field_pad: int = 1022,
    train_ratio: float = 0.6,
) -> Tuple[AudioFrameDataset, AudioFrameDataset]:
    """
    Auto-detects whether the audio is a NAM reamp file or an original ANN / custom reamp:
      - If NAM ('NAM_V3', 'NAM_V2', 'NAM_V1', 'NAM_PROTEUS'): applies NAM's exact
        training and validation segmentation boundaries.
      - If original ANN or custom ('LEGACY_60_40'): applies the 60% train / 40% validation split.
    """
    sr = loaded_audio.sample_rate
    total_available = loaded_audio.total_samples - delay
    total_frames = max(0, total_available // frame_size)

    profile = detect_reamp_profile(loaded_audio.dry, sr)

    if profile == "NAM_V3":
        # Skip [0..10s] (Validation 1 + 1s silence); Train on [10s..181s]; Validate on [181s..190s] (9s)
        train_start_frame = int(round(10.0 * sr)) // frame_size
        val_start_frame = int(round(181.0 * sr)) // frame_size
        val_end_frame = min(total_frames, int(round(190.0 * sr)) // frame_size)
        train_frames = max(0, val_start_frame - train_start_frame)
        val_frames = max(0, val_end_frame - val_start_frame)
    elif profile == "NAM_V2":
        # Train on [0..171s]; Validate on [171s..180s] (9s guitar validation clip)
        train_start_frame = 0
        val_start_frame = int(round(171.0 * sr)) // frame_size
        val_end_frame = min(total_frames, int(round(180.0 * sr)) // frame_size)
        train_frames = max(0, val_start_frame - train_start_frame)
        val_frames = max(0, val_end_frame - val_start_frame)
    elif profile == "NAM_V1":
        # Train on [0..171s]; Validate on [171s..180s] (9s guitar validation clip)
        train_start_frame = 0
        val_start_frame = int(round(171.0 * sr)) // frame_size
        val_end_frame = min(total_frames, int(round(180.0 * sr)) // frame_size)
        train_frames = max(0, val_start_frame - train_start_frame)
        val_frames = max(0, val_end_frame - val_start_frame)
    elif profile == "NAM_PROTEUS":
        # Skip first 1s blips; Validate on final 10s
        train_start_frame = int(round(1.0 * sr)) // frame_size
        val_frames = min(total_frames, int(round(10.0 * sr)) // frame_size)
        val_start_frame = max(train_start_frame, total_frames - val_frames)
        train_frames = max(0, val_start_frame - train_start_frame)
    else:
        # Original ANN 60% train / 40% validation split
        train_start_frame = 0
        train_frames = int(total_frames * train_ratio)
        val_start_frame = train_frames
        val_frames = total_frames - train_frames

    train_dataset = AudioFrameDataset(
        dry=loaded_audio.dry,
        target=loaded_audio.target,
        frame_size=frame_size,
        delay=delay,
        receptive_field_pad=receptive_field_pad,
        start_frame=train_start_frame,
        num_frames=train_frames,
    )
    train_dataset.reamp_profile = profile

    val_dataset = AudioFrameDataset(
        dry=loaded_audio.dry,
        target=loaded_audio.target,
        frame_size=frame_size,
        delay=delay,
        receptive_field_pad=receptive_field_pad,
        start_frame=val_start_frame,
        num_frames=val_frames,
    )
    val_dataset.reamp_profile = profile

    return train_dataset, val_dataset


def get_dataloaders(
    train_dataset: AudioFrameDataset,
    val_dataset: AudioFrameDataset,
    batch_size: int = 1024,
    shuffle_train: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(batch_size, len(train_dataset)) if len(train_dataset) > 0 else 1,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    # Process full batches for validation loss averaging,
    # dropping trailing fractional frames (fading silence) that distort unweighted batch averages.
    val_drop_last = len(val_dataset) >= batch_size
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(batch_size, len(val_dataset)) if len(val_dataset) > 0 else 1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=val_drop_last,
    )

    return train_loader, val_loader
