from __future__ import annotations

from typing import Tuple, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from audiotrain.data.audio import LoadedAudio


class AudioFrameDataset(Dataset):
    """
    Framed dataset matching FUN_14003ad60 in AudioTrainHHAG.exe.
    Slices continuous audio into frames of size 512 samples.
    With input lookahead delay:
      - Input chunk:  dry[i * 512 : i * 512 + 512 + delay] (length 512 + delay)
      - Target chunk: target[i * 512 + delay : i * 512 + delay + 512] (length 512)
    """
    def __init__(
        self,
        dry: np.ndarray,
        target: np.ndarray,
        frame_size: int = 512,
        delay: int = 0,
        start_frame: int = 0,
        num_frames: Optional[int] = None,
    ):
        super().__init__()
        self.frame_size = frame_size
        self.delay = max(0, delay)
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
        in_start = frame_idx * self.frame_size
        in_end = in_start + self.frame_size + self.delay

        tgt_start = in_start + self.delay
        tgt_end = tgt_start + self.frame_size

        in_chunk = self.dry[in_start:in_end]
        tgt_chunk = self.target[tgt_start:tgt_end]

        # Shape: (1, seq_len)
        x = torch.from_numpy(in_chunk).unsqueeze(0).to(torch.float32)
        y = torch.from_numpy(tgt_chunk).unsqueeze(0).to(torch.float32)
        return x, y


def create_train_val_datasets(
    loaded_audio: LoadedAudio,
    frame_size: int = 512,
    delay: int = 0,
    train_ratio: float = 0.6,
) -> Tuple[AudioFrameDataset, AudioFrameDataset]:
    """
    Splits frames according to the reverse-engineered 60% train / 40% validation ratio.
    """
    total_available = loaded_audio.total_samples - delay
    total_frames = max(0, total_available // frame_size)

    train_frames = int(total_frames * train_ratio)
    val_frames = total_frames - train_frames

    train_dataset = AudioFrameDataset(
        dry=loaded_audio.dry,
        target=loaded_audio.target,
        frame_size=frame_size,
        delay=delay,
        start_frame=0,
        num_frames=train_frames,
    )

    val_dataset = AudioFrameDataset(
        dry=loaded_audio.dry,
        target=loaded_audio.target,
        frame_size=frame_size,
        delay=delay,
        start_frame=train_frames,
        num_frames=val_frames,
    )

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

    # AudioTrainHHAG.exe processes full batches for validation loss averaging,
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
