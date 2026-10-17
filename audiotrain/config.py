from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml


class ModelType(IntEnum):
    """Matches the integer model selector in AudioTrainHHAG.exe."""
    AM3 = 0
    AM4 = 1
    IR1024 = 2
    IR2048 = 3
    NAM_TO_AM3 = 4
    NAM_TO_AM4 = 5

    @classmethod
    def from_str_or_int(cls, val: Union[str, int]) -> ModelType:
        if isinstance(val, int):
            return cls(val)
        val_str = str(val).strip().upper()
        mapping = {
            "0": cls.AM3,
            "1": cls.AM4,
            "2": cls.IR1024,
            "3": cls.IR2048,
            "4": cls.NAM_TO_AM3,
            "5": cls.NAM_TO_AM4,
            "AM3": cls.AM3,
            "AM4": cls.AM4,
            "IR1024": cls.IR1024,
            "IR_1024": cls.IR1024,
            "IR2048": cls.IR2048,
            "IR_2048": cls.IR2048,
            "NAM_TO_AM3": cls.NAM_TO_AM3,
            "NAM_TO_AM4": cls.NAM_TO_AM4,
            "NAM3": cls.NAM_TO_AM3,
            "NAM4": cls.NAM_TO_AM4,
        }
        if val_str in mapping:
            return mapping[val_str]
        raise ValueError(f"Unknown ModelType: {val}. Valid options: {list(mapping.keys())}")


@dataclass
class DataConfig:
    sample_rate: int = 44100
    frame_size: int = 512
    delay: int = 0
    train_ratio: float = 0.6
    normalize_peak: bool = True


@dataclass
class ModelConfig:
    model_type: ModelType = ModelType.AM4
    model_name: str = "custom_model"
    # FIR Filter sizes
    fir1_size: int = 512       # 256 for AM3, 512 for AM4, 1024/2048 for IR
    fir2_size: int = 512       # 256 for AM3, 512 for AM4
    # RNN settings
    rnn_hidden_size: int = 16
    rnn_num_layers: int = 1
    # AM3 specific
    cnn_kernel_size: int = 32
    cnn_channels: int = 3
    # Equalizer metadata preserved in binary export
    eq_bass_freq: float = 100.0
    eq_bass_gain_min: float = -6.0
    eq_bass_gain_max: float = 6.0
    eq_mid_freq: float = 2000.0
    eq_mid_gain_min: float = -6.0
    eq_mid_gain_max: float = 6.0
    eq_treble_freq: float = 4000.0
    eq_treble_gain_min: float = -6.0
    eq_treble_gain_max: float = 6.0


@dataclass
class TrainingConfig:
    batch_size: int = 1024
    epochs: int = 1000
    learning_rate: float = 0.0008
    weight_decay: float = 1e-4
    # Epoch fraction where training switches from BasicTuning (ESR) to FineTuning (Spectral)
    fine_tuning_ratio: float = 0.80
    patience: int = 50
    device: str = "cuda"  # Auto falls back to cpu if cuda unavailable
    seed: int = 42
    show_batch_progress: bool = False


@dataclass
class AppConfig:
    samples_path: str = ""
    samples_paths: List[str] = field(default_factory=list)
    output_dir: str = ""
    nam_path: Optional[str] = None
    dry_path: Optional[str] = None
    nam_variant: str = "full"  # 'full' (A2-Full, default) or 'lite' (A2-Lite)
    skip_existing: bool = False
    render_audio: bool = False
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_legacy_string(cls, legacy_cmd: str) -> AppConfig:
        """
        Parses legacy AudioTrainHHAG.exe argument format:
        "<SamplesPath>?<BatchSize>?<Times>?<Delay>?<LRate>?<Model>"
        """
        parts = legacy_cmd.split("?")
        if len(parts) != 6:
            raise ValueError(
                f"Expected 6 '?' delimited parts in legacy argument, got {len(parts)}: '{legacy_cmd}'"
            )

        samples_path = parts[0].strip()
        batch_size = int(parts[1].strip())
        times = int(parts[2].strip())
        delay = int(parts[3].strip())
        lrate = float(parts[4].strip())
        model_int = int(parts[5].strip())

        model_type = ModelType(model_int)

        config = cls()
        config.samples_path = samples_path
        config.data.delay = delay
        config.training.batch_size = batch_size
        config.training.epochs = times
        config.training.learning_rate = lrate
        config.model.model_type = model_type

        # Adjust default FIR lengths based on model type
        config.apply_model_defaults()
        config.render_audio = True  # Legacy AudioTrainHHAG.exe always emitted preview audio
        return config

    def apply_model_defaults(self) -> None:
        """Synchronize layer dimensions to match model type specifications."""
        mt = self.model.model_type
        if mt == ModelType.AM3 or mt == ModelType.NAM_TO_AM3:
            self.model.fir1_size = 256
            self.model.fir2_size = 256
        elif mt == ModelType.AM4 or mt == ModelType.NAM_TO_AM4:
            self.model.fir1_size = 512
            self.model.fir2_size = 512
        elif mt == ModelType.IR1024:
            self.model.fir1_size = 1024
            self.model.fir2_size = 0
        elif mt == ModelType.IR2048:
            self.model.fir1_size = 2048
            self.model.fir2_size = 0

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> AppConfig:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        config = cls()
        if "samples_path" in data:
            config.samples_path = data["samples_path"]
        if "samples_paths" in data:
            config.samples_paths = data["samples_paths"]
        if "output_dir" in data:
            config.output_dir = data["output_dir"]
        if "nam_path" in data:
            config.nam_path = data["nam_path"]
        if "dry_path" in data:
            config.dry_path = data["dry_path"]
        if "skip_existing" in data:
            config.skip_existing = bool(data["skip_existing"])
        if "render_audio" in data:
            config.render_audio = bool(data["render_audio"])

        if "data" in data:
            for k, v in data["data"].items():
                if hasattr(config.data, k):
                    setattr(config.data, k, v)

        if "model" in data:
            for k, v in data["model"].items():
                if k == "model_type":
                    config.model.model_type = ModelType.from_str_or_int(v)
                elif hasattr(config.model, k):
                    setattr(config.model, k, v)

        if "training" in data:
            for k, v in data["training"].items():
                if hasattr(config.training, k):
                    setattr(config.training, k, v)

        config.apply_model_defaults()
        return config

    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        data = {
            "samples_path": self.samples_path,
            "samples_paths": self.samples_paths,
            "output_dir": self.output_dir,
            "nam_path": self.nam_path,
            "dry_path": self.dry_path,
            "skip_existing": self.skip_existing,
            "render_audio": self.render_audio,
            "data": asdict(self.data),
            "model": {
                **asdict(self.model),
                "model_type": int(self.model.model_type),
            },
            "training": asdict(self.training),
        }
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
