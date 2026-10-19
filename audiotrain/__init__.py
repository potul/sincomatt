"""
AudioTrain: Modular PyTorch neural amplifier modeling and training pipeline.
Supports AM3, AM4, IR1024, IR2048 architectures and NAM model import/distillation.
"""

__version__ = "1.0.0"

from audiotrain.config import AppConfig, ModelType, TrainingConfig
from audiotrain.trainer import AudioTrainer
from audiotrain.batch import BatchRunner, JobResult, resolve_audio_targets

__all__ = [
    "AppConfig",
    "ModelType",
    "TrainingConfig",
    "AudioTrainer",
    "BatchRunner",
    "JobResult",
    "resolve_audio_targets",
    "__version__",
]
