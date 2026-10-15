from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, List

from audiotrain.config import AppConfig, ModelType
from audiotrain.trainer import AudioTrainer
from audiotrain.batch import BatchRunner, resolve_audio_targets


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args(args: Optional[List[str]] = None) -> tuple[AppConfig, List[Path]]:
    raw_args = args if args is not None else sys.argv[1:]

    # Check for legacy single-argument format: "<SamplesPath>?<BatchSize>?<Times>?<Delay>?<LRate>?<Model>"
    if len(raw_args) == 1 and "?" in raw_args[0]:
        legacy_str = raw_args[0]
        logging.info(f"Detected legacy invocation parameter: {legacy_str}")
        config = AppConfig.from_legacy_string(legacy_str)
        # Resolve target files (could be a directory or single file)
        resolved = resolve_audio_targets(config.samples_path)
        if not config.output_dir:
            sample_p = Path(config.samples_path)
            config.output_dir = str(sample_p if sample_p.is_dir() else sample_p.parent)
            if resolved:
                config.model.model_name = resolved[0].stem
        return config, resolved

    parser = argparse.ArgumentParser(
        description="AudioTrain: Neural Amp Modeler & Batch Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to YAML configuration file (e.g. configs/am4.yaml)",
    )
    parser.add_argument(
        "--samples", "-s",
        type=str,
        nargs="*",
        default=None,
        help="Path(s) to stereo WAV file(s), .nam model file(s) for distillation, directory, or glob pattern ('samples/*.wav', 'models/*.nam')",
    )
    parser.add_argument(
        "--dry-wav",
        type=str,
        default=None,
        help="Optional custom dry guitar audio WAV file used as excitation during NAM distillation",
    )
    parser.add_argument(
        "--batch-file",
        type=str,
        default=None,
        help="Text file listing audio paths to process in batch (one per line)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Directory where exported models, lossrec, and audio are saved (default: same folder as input file)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="AM4",
        help="Model architecture: AM3 (0), AM4 (1), IR1024 (2), IR2048 (3), NAM_TO_AM3 (4), NAM_TO_AM4 (5)",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=1024,
        help="Batch size (number of 512-sample frames per batch)",
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=1000,
        help="Total training epochs",
    )
    parser.add_argument(
        "--delay", "-d",
        type=int,
        default=0,
        help="Input lookahead delay in samples",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0008,
        help="AdamW learning rate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Execution device: 'cuda' or 'cpu'",
    )
    parser.add_argument(
        "--name", "-n",
        type=str,
        default=None,
        help="Model name string stored in binary header (defaults to sample filename stem)",
    )
    parser.add_argument(
        "--nam-path",
        type=str,
        default=None,
        help="Path to .nam file for distillation",
    )
    parser.add_argument(
        "--nam-variant",
        type=str,
        choices=["full", "lite"],
        default="full",
        help="Submodel to distill from NAM A2 SlimmableContainer files: 'full' (8-channel high-fidelity, default) or 'lite' (3-channel low-compute)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already have exported binary profiles in output directory",
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Recursively search directories for WAV files in batch mode",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        default=False,
        help="Show granular batch-level progress during epochs (default: off)",
    )
    parser.add_argument(
        "--render-audio",
        action="store_true",
        default=False,
        help="Render and export stereo audio verification WAV (default: off to maximize speed)",
    )
    parser.add_argument(
        "--save-config",
        type=str,
        default=None,
        help="Optionally dump the resolved config to a YAML file before training",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )

    parsed = parser.parse_args(raw_args)

    # Base configuration: either from YAML or defaults
    if parsed.config:
        config = AppConfig.from_yaml(parsed.config)
    else:
        config = AppConfig()

    # Collect target paths from CLI
    targets: List[str] = []
    if parsed.samples:
        targets.extend(parsed.samples)
    if parsed.batch_file:
        targets.append(parsed.batch_file)
    if not targets and config.samples_paths:
        targets.extend(config.samples_paths)
    if not targets and config.samples_path:
        targets.append(config.samples_path)

    if not targets:
        parser.error(
            "No audio samples path provided. Please specify --samples <path/dir/*.wav> or define samples_path in a YAML configuration file."
        )

    resolved_files = resolve_audio_targets(targets, recursive=parsed.recursive)
    if not resolved_files:
        parser.error(f"No valid audio (.wav) or model (.nam) files found matching targets: {targets}")

    # Set primary samples_path from first resolved target
    config.samples_path = str(resolved_files[0])
    config.samples_paths = [str(p) for p in resolved_files]

    # Auto-detect NAM distillation model if target is a .nam model file
    is_nam_input = resolved_files[0].suffix.lower() == ".nam"
    if is_nam_input:
        config.nam_path = str(resolved_files[0])
        if not parsed.model or parsed.model.upper() == "AM4":
            config.model.model_type = ModelType.NAM_TO_AM4
        elif parsed.model.upper() == "AM3":
            config.model.model_type = ModelType.NAM_TO_AM3
        else:
            config.model.model_type = ModelType.from_str_or_int(parsed.model)
    elif parsed.model:
        config.model.model_type = ModelType.from_str_or_int(parsed.model)

    # Override options
    if parsed.output_dir:
        config.output_dir = parsed.output_dir
    if parsed.dry_wav:
        config.dry_path = parsed.dry_wav
    if parsed.batch_size:
        config.training.batch_size = parsed.batch_size
    if parsed.epochs:
        config.training.epochs = parsed.epochs
    if parsed.delay is not None:
        config.data.delay = parsed.delay
    if parsed.lr:
        config.training.learning_rate = parsed.lr
    if parsed.device:
        config.training.device = parsed.device
    if parsed.nam_path:
        config.nam_path = parsed.nam_path
    if parsed.nam_variant:
        config.nam_variant = parsed.nam_variant
    if parsed.skip_existing:
        config.skip_existing = True
    if parsed.show_progress:
        config.training.show_batch_progress = True
    if parsed.render_audio:
        config.render_audio = True

    # Model name
    if parsed.name:
        config.model.model_name = parsed.name
    elif not config.model.model_name or config.model.model_name == "custom_model":
        config.model.model_name = resolved_files[0].stem

    config.apply_model_defaults()

    if parsed.save_config:
        config.to_yaml(parsed.save_config)
        logging.info(f"Saved configuration to {parsed.save_config}")

    return config, resolved_files


def main(args: Optional[List[str]] = None) -> int:
    setup_logging()
    try:
        config, resolved_files = parse_args(args)

        if len(resolved_files) > 1 or config.skip_existing:
            # Batch execution mode
            batch_runner = BatchRunner(
                base_config=config,
                audio_files=resolved_files,
                skip_existing=config.skip_existing,
            )
            results = batch_runner.run()
            has_failures = any(r.status == "FAILED" for r in results)
            return 1 if has_failures else 0
        else:
            # Single-file execution mode
            trainer = AudioTrainer(config)
            trainer.train()
            return 0
    except Exception as e:
        logging.exception(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
