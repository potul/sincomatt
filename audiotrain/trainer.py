from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
import warnings

# Suppress benign PyTorch UserWarning regarding even-length kernels in Conv1d
warnings.filterwarnings("ignore", message=".*Using padding='same' with even kernel lengths.*")

import soundfile as sf

from audiotrain.config import AppConfig, ModelType
from audiotrain.data.audio import (
    LoadedAudio,
    load_and_preprocess_audio,
    resample_signal,
    save_stereo_wav,
)
from audiotrain.data.dataset import create_train_val_datasets, get_dataloaders
from audiotrain.losses.curriculum import CurriculumLoss
from audiotrain.losses.esr import ESRLoss
from audiotrain.losses.spectral import SpectralESRLoss
from audiotrain.models.base import BaseAmpModel
from audiotrain.models.am3 import AM3Model
from audiotrain.models.am4 import AM4Model
from audiotrain.models.ir import IRModel
from audiotrain.models.nam import NAMModel, create_distillation_pair
from audiotrain.serialization.binary_format import save_am_data, save_lossrec

logger = logging.getLogger("AudioTrain")


class AudioTrainer:
    """
    Main neural amplifier training engine:
      - Sinc resampling & peak normalization
      - 60% Train / 40% Validation partition
      - Dual-phase curriculum: BasicTuning (ESR) -> FineTuning (Spectral)
      - Checkpoint export: .am3Data/.am4Data binary, .am4lossrec, and rendered WAV
    """
    def __init__(self, config: AppConfig):
        self.config = config
        self._setup_device()
        self._setup_seeds()

        self.model: BaseAmpModel = self._create_model()
        self.teacher_model: Optional[NAMModel] = None

        if self.config.model.model_type in (ModelType.NAM_TO_AM3, ModelType.NAM_TO_AM4):
            self._setup_nam_distillation()

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

        fine_tuning_epoch = int(self.config.training.epochs * self.config.training.fine_tuning_ratio) + 1
        self.criterion = CurriculumLoss(fine_tuning_epoch=fine_tuning_epoch).to(self.device)
        self.val_esr = ESRLoss(use_preemphasis=True).to(self.device)
        self.val_spectral = SpectralESRLoss(use_preemphasis=True).to(self.device)

        self.audio_data: Optional[LoadedAudio] = None
        self.train_loader = None
        self.val_loader = None

        self.best_val_loss = float("inf")
        self.history: Dict[str, list] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_esr": [],
            "val_spectral": [],
            "stage": [],
        }

    def _setup_device(self) -> None:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        desired = self.config.training.device.lower()
        if "cuda" in desired and torch.cuda.is_available():
            self.device = torch.device("cuda")
            # Configure cuDNN for deterministic high-throughput execution
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device("cpu")
        logger.info(f"Using device: {self.device}")

    def _setup_seeds(self) -> None:
        seed = self.config.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _create_model(self) -> BaseAmpModel:
        mt = self.config.model.model_type
        m_cfg = self.config.model
        if mt in (ModelType.AM4, ModelType.NAM_TO_AM4):
            model = AM4Model(
                fir1_size=m_cfg.fir1_size,
                fir2_size=m_cfg.fir2_size,
                hidden_size=m_cfg.rnn_hidden_size,
            )
        elif mt in (ModelType.AM3, ModelType.NAM_TO_AM3):
            model = AM3Model(
                fir1_size=m_cfg.fir1_size,
                fir2_size=m_cfg.fir2_size,
                cnn_channels=m_cfg.cnn_channels,
                cnn_kernel_size=m_cfg.cnn_kernel_size,
                hidden_size=m_cfg.rnn_hidden_size,
            )
        elif mt == ModelType.IR1024:
            model = IRModel(filter_size=1024)
        elif mt == ModelType.IR2048:
            model = IRModel(filter_size=2048)
        else:
            raise ValueError(f"Unsupported model type: {mt}")

        model = model.to(self.device)
        logger.info(f"Initialized {model.model_tag} with {model.count_parameters():,} trainable parameters")
        return model

    def _setup_nam_distillation(self) -> None:
        nam_p = self.config.nam_path
        if not nam_p or not Path(nam_p).is_file():
            if self.config.samples_path and Path(self.config.samples_path).suffix.lower() == ".nam":
                nam_p = self.config.samples_path
            else:
                # If nam_path not specified, look for .nam in samples directory
                samples_dir = Path(self.config.samples_path).parent
                candidate_nams = list(samples_dir.glob("*.nam"))
                if candidate_nams:
                    nam_p = str(candidate_nams[0])
                else:
                    raise FileNotFoundError(
                        f"Distillation requested ({self.config.model.model_type.name}) but no .nam file found."
                    )

        logger.info(f"Loading teacher NAM model from {nam_p} (variant: {getattr(self.config, 'nam_variant', 'full')})")
        submodel_idx = 0 if getattr(self.config, "nam_variant", "full").lower() == "lite" else -1
        self.teacher_model = NAMModel.from_file(nam_p, submodel_index=submodel_idx).to(self.device)
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False

    def _find_default_dry_audio(self) -> Path:
        """Finds the bundled NAM v3_0_0.wav dry guitar excitation track for NAM distillation."""
        default_asset = Path(__file__).parent / "assets" / "default_dry.wav"
        if default_asset.is_file():
            return default_asset
        raise FileNotFoundError(
            "No dry guitar audio found for NAM distillation. Please specify --dry-wav <path_to_dry.wav>"
        )

    def prepare_data(self) -> None:
        """Loads audio or synthesizes distillation targets from teacher NAM model."""
        is_nam_input = (
            Path(self.config.samples_path).suffix.lower() == ".nam"
            or self.config.model.model_type in (ModelType.NAM_TO_AM3, ModelType.NAM_TO_AM4)
        )

        if is_nam_input and self.teacher_model is not None:
            # Determine dry excitation source
            if self.config.dry_path and Path(self.config.dry_path).is_file():
                dry_file = Path(self.config.dry_path)
            elif Path(self.config.samples_path).suffix.lower() == ".wav":
                dry_file = Path(self.config.samples_path)
            else:
                dry_file = self._find_default_dry_audio()

            logger.info(f"NAM Distillation: Using dry guitar excitation from {dry_file}")
            raw_audio, sr = sf.read(str(dry_file), dtype="float32")
            dry = raw_audio[:, 0] if raw_audio.ndim > 1 else raw_audio

            # Run teacher NAM model at its native sample rate (typically 48 kHz) before resampling to 44.1 kHz
            teacher_sr = int(round(getattr(self.teacher_model, "sample_rate", sr) or sr))
            if sr != teacher_sr:
                dry_teacher = resample_signal(dry, sr, teacher_sr)
            else:
                dry_teacher = dry

            # Render target audio through teacher NAM model in chunks with causal overlap
            logger.info(
                f"NAM Distillation: Generating ground-truth target audio via teacher NAM model at {teacher_sr} Hz..."
            )
            dry_t = torch.from_numpy(dry_teacher).unsqueeze(0).unsqueeze(0).to(self.device)
            target_chunks = []
            chunk_len = 32768
            overlap = 8192
            total_t = dry_t.shape[-1]
            with torch.no_grad():
                for start in range(0, total_t, chunk_len):
                    end = min(start + chunk_len, total_t)
                    in_start = max(0, start - overlap)
                    chunk = dry_t[:, :, in_start:end]
                    chunk_out = self.teacher_model(chunk)
                    valid_len = end - start
                    target_chunks.append(chunk_out[:, :, -valid_len:].squeeze(0).squeeze(0).cpu().numpy())

            target = np.concatenate(target_chunks)
            dry_final = dry_teacher

            # Resample both dry and target to the hardware target sample rate (44.1 kHz)
            target_sr = self.config.data.sample_rate
            if teacher_sr != target_sr:
                dry_final = resample_signal(dry_final, teacher_sr, target_sr)
                target = resample_signal(target, teacher_sr, target_sr)

            min_len = min(len(dry_final), len(target))
            dry_final = dry_final[:min_len]
            target = target[:min_len]

            # Peak normalize dry and target
            dry_loudness = float(np.max(np.abs(dry_final))) if len(dry_final) > 0 else 1.0
            if self.config.data.normalize_peak and dry_loudness > 1e-8:
                dry_final = dry_final / dry_loudness

            target_loudness = float(np.max(np.abs(target))) if len(target) > 0 else 1.0
            if self.config.data.normalize_peak and target_loudness > 1e-8:
                target = target / target_loudness

            self.audio_data = LoadedAudio(
                dry=dry_final.astype(np.float32),
                target=target.astype(np.float32),
                sample_rate=target_sr,
                dry_loudness=dry_loudness,
                target_loudness=target_loudness,
                total_samples=min_len,
            )
        else:
            logger.info(f"Loading and preprocessing audio from {self.config.samples_path}")
            self.audio_data = load_and_preprocess_audio(
                file_path=self.config.samples_path,
                target_sr=self.config.data.sample_rate,
                normalize=self.config.data.normalize_peak,
            )

        train_ds, val_ds = create_train_val_datasets(
            loaded_audio=self.audio_data,
            frame_size=self.config.data.frame_size,
            delay=self.config.data.delay,
            receptive_field_pad=self.model.receptive_field_pad,
            train_ratio=self.config.data.train_ratio,
        )

        profile = getattr(train_ds, "reamp_profile", "LEGACY_60_40")
        active_frames = max(1, len(train_ds) + len(val_ds))
        train_pct = 100.0 * len(train_ds) / active_frames
        val_pct = 100.0 * len(val_ds) / active_frames
        logger.info(
            f"Dataset split [{profile}]: {len(train_ds):,} train frames ({train_pct:.1f}%), "
            f"{len(val_ds):,} val frames ({val_pct:.1f}%) [Active: {len(train_ds) + len(val_ds):,}, "
            f"lookback={self.model.receptive_field_pad} samples]"
        )

        self.train_loader, self.val_loader = get_dataloaders(
            train_dataset=train_ds,
            val_dataset=val_ds,
            batch_size=self.config.training.batch_size,
            shuffle_train=True,
        )

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        self.criterion.set_epoch(epoch)
        total_loss = 0.0
        num_batches = 0

        # Exact micro-batch size for GPU execution so cuDNN workspace never exceeds ~280 MiB
        # on 6 GB GPUs while keeping batch_size=1024 loss & gradients 100% mathematically identical.
        micro_batch = 256 if self.device.type == "cuda" else self.config.training.batch_size

        total_batches = len(self.train_loader)
        for b_idx, (x, y) in enumerate(self.train_loader, 1):
            self.optimizer.zero_grad(set_to_none=True)
            batch_len = x.shape[0]

            if batch_len <= micro_batch:
                x_dev = x.to(self.device, non_blocking=True)
                y_dev = y.to(self.device, non_blocking=True)
                pred = self.model(x_dev, target_len=self.config.data.frame_size)
                loss = self.criterion(pred, y_dev)
                loss.backward()
                batch_loss_val = loss.item()
                del x_dev, y_dev, pred, loss
            else:
                y_dev = y.to(self.device, non_blocking=True)
                with torch.no_grad():
                    y_trans = self.criterion.transform(y_dev)
                    denom = torch.sum(y_trans ** 2) + self.criterion.eps
                del y_dev

                batch_loss_val = 0.0
                for start in range(0, batch_len, micro_batch):
                    end = min(start + micro_batch, batch_len)
                    x_chunk = x[start:end].to(self.device, non_blocking=True)
                    y_chunk_trans = y_trans[start:end]

                    pred_chunk = self.model(x_chunk, target_len=self.config.data.frame_size)
                    pred_chunk_trans = self.criterion.transform(pred_chunk)

                    chunk_loss = torch.sum((y_chunk_trans - pred_chunk_trans) ** 2) / denom
                    chunk_loss.backward()
                    batch_loss_val += chunk_loss.item()

                    del x_chunk, pred_chunk, pred_chunk_trans, chunk_loss

                del y_trans, denom

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0, error_if_nonfinite=False
            )
            if torch.isfinite(grad_norm) and math.isfinite(batch_loss_val):
                self.optimizer.step()
                total_loss += batch_loss_val
                num_batches += 1
            else:
                self.optimizer.zero_grad(set_to_none=True)
                logger.warning(
                    f"Epoch {epoch} Batch {b_idx}: non-finite gradient/loss detected "
                    f"(loss={batch_loss_val}, grad_norm={grad_norm}). Skipping step."
                )

            if self.config.training.show_batch_progress and total_batches > 1:
                if b_idx % max(1, total_batches // 4) == 0 or b_idx == total_batches:
                    pct = int(b_idx * 100 / total_batches)
                    logger.info(f"  -> Batch [{b_idx}/{total_batches}] ({pct}%) | Batch Loss: {batch_loss_val:.4f}")

        return total_loss / max(1, num_batches)

    @torch.no_grad()
    def validate(self) -> Tuple[float, float, float]:
        """
        Computes validation metrics across the validation dataset:
        Evaluates each batch through the active loss function:
          - BasicTuning: 1st-order 2-tap pre-emphasis + time-domain ESR
          - FineTuning:  2nd-order 3-tap pre-emphasis + rfft/abs/irfft ESR
        and computes the arithmetic mean across batches:
          Val Loss = (1 / N) * sum(batch_loss)
        with eps = 1e-20.
        """
        self.model.eval()
        batch_losses = []
        batch_esr = []
        batch_spectral = []

        micro_batch = 256 if self.device.type == "cuda" else self.config.training.batch_size

        for x, y in self.val_loader:
            batch_len = x.shape[0]
            if batch_len <= micro_batch:
                x_dev = x.to(self.device, non_blocking=True)
                pred = self.model(x_dev, target_len=self.config.data.frame_size)
                del x_dev
            else:
                pred_chunks = []
                for start in range(0, batch_len, micro_batch):
                    end = min(start + micro_batch, batch_len)
                    x_chunk = x[start:end].to(self.device, non_blocking=True)
                    pred_chunks.append(self.model(x_chunk, target_len=self.config.data.frame_size))
                    del x_chunk
                pred = torch.cat(pred_chunks, dim=0)
                del pred_chunks

            y_dev = y.to(self.device, non_blocking=True)

            # 1. Time-domain ESR for this batch
            b_esr = self.val_esr(pred, y_dev).item()
            batch_esr.append(b_esr)

            # 2. Spectral / cepstral ESR for this batch
            b_spec = self.val_spectral(pred, y_dev).item()
            batch_spectral.append(b_spec)

            # Active loss for this batch
            b_loss = b_spec if self.criterion.is_fine_tuning else b_esr
            batch_losses.append(b_loss)

            del pred, y_dev

        avg_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        avg_esr = float(np.mean(batch_esr)) if batch_esr else 0.0
        avg_spectral = float(np.mean(batch_spectral)) if batch_spectral else 0.0
        return avg_loss, avg_esr, avg_spectral

    @torch.no_grad()
    def render_full_prediction(self) -> np.ndarray:
        """
        Renders the entire dry audio signal through the trained model with continuous
        receptive-field overlap between chunks.
        """
        self.model.eval()
        dry_tensor = torch.from_numpy(self.audio_data.dry).unsqueeze(0).unsqueeze(0).to(self.device)

        # Process in chunks with receptive_field_pad overlap so chunk boundaries are seamless
        chunk_len = 44100
        rf_pad = self.model.receptive_field_pad
        total_samples = dry_tensor.shape[-1]
        preds = []

        for start in range(0, total_samples, chunk_len):
            end = min(start + chunk_len, total_samples)
            target_len = end - start
            in_start = start - rf_pad
            if in_start < 0:
                chunk = torch.nn.functional.pad(dry_tensor[:, :, 0:end], (-in_start, 0))
            else:
                chunk = dry_tensor[:, :, in_start:end]
            chunk_pred = self.model(chunk, target_len=target_len)
            preds.append(chunk_pred.squeeze().cpu().numpy())

        return np.concatenate(preds)

    def save_artifacts(self, val_loss: float, epoch: int) -> None:
        """
        Saves all training artifacts:
          1. <model_name>.am3Data or .am4Data (hardware binary format)
          2. <model_name>.am4lossrec (float loss tracking text file)
          3. <model_name>_am4.wav (rendered stereo audio with restored target loudness)
          4. <model_name>.pt (modern PyTorch checkpoint)
        """
        if self.config.output_dir:
            out_dir = Path(self.config.output_dir)
        elif self.config.samples_path:
            out_dir = Path(self.config.samples_path).parent
        else:
            out_dir = Path(".")
        out_dir.mkdir(parents=True, exist_ok=True)
        name = self.config.model.model_name

        tag = self.model.model_tag.lower()
        if "am3" in tag:
            bin_ext = ".am3Data"
            loss_ext = ".am3lossrec"
            wav_name = f"{name}_am3.wav"
        elif "am4" in tag:
            bin_ext = ".am4Data"
            loss_ext = ".am4lossrec"
            wav_name = f"{name}_am4.wav"
        else:
            bin_ext = ".irData"
            loss_ext = ".am4lossrec"
            wav_name = f"{name}_ir.wav"

        # 1. Save binary weights
        bin_path = out_dir / f"{name}{bin_ext}"
        eq_settings = {
            "bass_freq": self.config.model.eq_bass_freq,
            "bass_min": self.config.model.eq_bass_gain_min,
            "bass_max": self.config.model.eq_bass_gain_max,
            "mid_freq": self.config.model.eq_mid_freq,
            "mid_min": self.config.model.eq_mid_gain_min,
            "mid_max": self.config.model.eq_mid_gain_max,
            "treble_freq": self.config.model.eq_treble_freq,
            "treble_min": self.config.model.eq_treble_gain_min,
            "treble_max": self.config.model.eq_treble_gain_max,
        }
        save_am_data(bin_path, self.model, model_name=name, eq_settings=eq_settings)

        # 2. Save lossrec
        loss_path = out_dir / f"{name}{loss_ext}"
        save_lossrec(loss_path, val_loss)

        # 3. Save PyTorch checkpoint
        pt_path = out_dir / f"{name}.pt"
        self.model.save_checkpoint(pt_path, metadata={"epoch": epoch, "val_loss": val_loss})

        # 4. Render and export audio preview (optional)
        if self.config.render_audio and self.audio_data is not None:
            logger.info(f"Rendering audio verification WAV ({wav_name})...")
            pred_audio = self.render_full_prediction()
            wav_path = out_dir / wav_name
            save_stereo_wav(
                output_path=wav_path,
                dry=self.audio_data.dry,
                target=pred_audio,
                sample_rate=self.audio_data.sample_rate,
                target_loudness=self.audio_data.target_loudness,
            )

        logger.info(f"-> Checkpoint saved to {bin_path} (best val loss: {val_loss:.6f})")

    def train(self) -> BaseAmpModel:
        if self.train_loader is None or self.val_loader is None:
            self.prepare_data()

        epochs = self.config.training.epochs
        patience = self.config.training.patience
        patience_counter = 0

        batches_count = len(self.train_loader) if self.train_loader else 0
        device_name = self.device.type.upper()
        logger.info(
            f"Training started on {device_name} ({epochs} epochs, {batches_count} batches/epoch, lr={self.config.training.learning_rate})..."
        )

        start_time = time.time()
        for epoch in range(1, epochs + 1):
            if epoch == self.criterion.fine_tuning_epoch:
                # Reset best loss tracker when transitioning to FineTuning spectral loss scale
                self.best_val_loss = float("inf")
                patience_counter = 0
                logger.info(
                    f"-> Transitioning to FineTuning Stage [Spectral]. "
                    f"Resetting best loss tracker..."
                )
            elif not self.config.training.show_batch_progress and epoch == 1:
                stage_hint = "FineTuning [Spectral]" if self.criterion.is_fine_tuning else "BasicTuning [ESR]"
                logger.info(f"-> Processing Epoch {epoch}/{epochs} ({stage_hint})...")

            ep_start = time.time()
            train_loss = self.train_epoch(epoch)
            val_loss, val_esr, val_spectral = self.validate()
            ep_time = time.time() - ep_start

            stage_name = "FineTuning [Spectral]" if self.criterion.is_fine_tuning else "BasicTuning [ESR]"

            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_esr"].append(val_esr)
            self.history["val_spectral"].append(val_spectral)
            self.history["stage"].append(stage_name)

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                patience_counter = 0
                self.save_artifacts(val_loss, epoch)
            else:
                patience_counter += 1

            if epoch % 5 == 0 or epoch == 1 or epoch == epochs or is_best:
                best_marker = " *" if is_best else ""
                logger.info(
                    f"Epoch {epoch:4d}/{epochs} | {stage_name:20s} | "
                    f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} (ESR: {val_esr:.6f}, Spec: {val_spectral:.6f}) | "
                    f"{ep_time:.2f}s{best_marker}"
                )

            if patience_counter >= patience and epoch >= self.criterion.fine_tuning_epoch:
                logger.info(f"Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs)")
                break

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s. Best validation loss: {self.best_val_loss:.6f}")
        return self.model
