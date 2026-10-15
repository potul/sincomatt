from __future__ import annotations

import copy
import gc
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from audiotrain.config import AppConfig, ModelType
from audiotrain.trainer import AudioTrainer

logger = logging.getLogger("AudioTrain.Batch")


@dataclass
class JobResult:
    audio_path: str
    model_name: str
    model_type: str
    status: str  # "SUCCESS", "FAILED", "SKIPPED"
    val_loss: Optional[float] = None
    duration_seconds: float = 0.0
    error_message: Optional[str] = None
    output_bin: Optional[str] = None
    output_wav: Optional[str] = None


def resolve_audio_targets(
    targets: Union[str, Path, List[Union[str, Path]]],
    recursive: bool = False,
    valid_extensions: Tuple[str, ...] = (".wav", ".nam"),
) -> List[Path]:
    """
    Expands file paths, directories, wildcards/globs, or text list files into a sorted list of audio/model files.
    Supports both stereo .wav captures and .nam neural amp model files for distillation.
    """
    if isinstance(targets, (str, Path)):
        targets = [targets]

    resolved_files: List[Path] = []
    seen = set()

    for item in targets:
        p = Path(item)
        p_str = str(item)

        # 1. Text file list of paths (e.g. batch_files.txt)
        if p.is_file() and p.suffix.lower() in (".txt", ".lst"):
            with open(p, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
            sub_resolved = resolve_audio_targets(lines, recursive=recursive, valid_extensions=valid_extensions)
            for sub_p in sub_resolved:
                if sub_p not in seen:
                    seen.add(sub_p)
                    resolved_files.append(sub_p)
            continue

        # 2. Glob pattern (e.g. "samples/*.wav", "models/*.nam")
        if any(c in p_str for c in ("*", "?", "[")):
            parent_dir = Path(p_str.split("*")[0].split("?")[0].split("[")[0] or ".")
            pattern = p_str[len(str(parent_dir)):].lstrip("\\/")
            if not parent_dir.is_dir():
                parent_dir = Path(".")
                pattern = p_str

            for match in (parent_dir.rglob(pattern) if recursive else parent_dir.glob(pattern)):
                if match.is_file() and match.suffix.lower() in valid_extensions and match not in seen:
                    seen.add(match)
                    resolved_files.append(match)
            continue

        # 3. Directory
        if p.is_dir():
            matches = []
            for ext in valid_extensions:
                pattern = f"**/*{ext}" if recursive else f"*{ext}"
                matches.extend(p.glob(pattern))
            for match in sorted(matches):
                if match.is_file() and match not in seen:
                    seen.add(match)
                    resolved_files.append(match)
            continue

        # 4. Single file
        if p.is_file():
            if p.suffix.lower() in valid_extensions and p not in seen:
                seen.add(p)
                resolved_files.append(p)
            continue

        # 5. File without extension (check .wav then .nam)
        found = False
        for ext in valid_extensions:
            cand_p = p.with_suffix(ext)
            if cand_p.is_file() and cand_p not in seen:
                seen.add(cand_p)
                resolved_files.append(cand_p)
                found = True
                break
        if found:
            continue

        logger.warning(f"Target file not found or unrecognized: {item}")

    return sorted(resolved_files)


class BatchRunner:
    """
    Executes multi-file batch training jobs with isolated environments,
    automatic VRAM cleanup, fault tolerance, and comprehensive summary reporting.
    """
    def __init__(
        self,
        base_config: AppConfig,
        audio_files: Optional[List[Union[str, Path]]] = None,
        skip_existing: bool = False,
    ):
        self.base_config = base_config
        self.skip_existing = skip_existing

        # Resolve target files
        input_targets = audio_files or self.base_config.samples_paths or [self.base_config.samples_path]
        self.audio_files = resolve_audio_targets(input_targets)
        self.results: List[JobResult] = []

    def _get_expected_bin_name(self, model_name: str, model_type: ModelType) -> str:
        if model_type in (ModelType.AM3, ModelType.NAM_TO_AM3):
            return f"{model_name}.am3Data"
        elif model_type in (ModelType.AM4, ModelType.NAM_TO_AM4):
            return f"{model_name}.am4Data"
        return f"{model_name}.irData"

    def _get_expected_wav_name(self, model_name: str, model_type: ModelType) -> str:
        if model_type in (ModelType.AM3, ModelType.NAM_TO_AM3):
            return f"{model_name}_am3.wav"
        elif model_type in (ModelType.AM4, ModelType.NAM_TO_AM4):
            return f"{model_name}_am4.wav"
        return f"{model_name}_ir.wav"

    def run(self) -> List[JobResult]:
        total_jobs = len(self.audio_files)
        if total_jobs == 0:
            logger.error("No audio files found to process.")
            return []

        # Determine summary directory
        if self.base_config.output_dir:
            summary_dir = Path(self.base_config.output_dir)
            summary_dir.mkdir(parents=True, exist_ok=True)
        elif self.audio_files:
            summary_dir = self.audio_files[0].parent
        else:
            summary_dir = Path(".")

        logger.info(
            f"=== Starting Batch Run: {total_jobs} audio target(s) | Model: {self.base_config.model.model_type.name} ==="
        )

        batch_start_time = time.time()
        for idx, audio_path in enumerate(self.audio_files, 1):
            model_name = audio_path.stem
            is_nam = audio_path.suffix.lower() == ".nam"

            # Auto-detect NAM distillation model type if target is a .nam file
            if is_nam:
                if self.base_config.model.model_type in (ModelType.AM3, ModelType.NAM_TO_AM3):
                    model_type = ModelType.NAM_TO_AM3
                else:
                    model_type = ModelType.NAM_TO_AM4
            else:
                model_type = self.base_config.model.model_type

            bin_name = self._get_expected_bin_name(model_name, model_type)
            wav_name = self._get_expected_wav_name(model_name, model_type)

            # Default to the same folder as the input audio file
            if self.base_config.output_dir:
                job_out_dir = Path(self.base_config.output_dir)
            else:
                job_out_dir = audio_path.parent
            job_out_dir.mkdir(parents=True, exist_ok=True)

            bin_path = job_out_dir / bin_name
            wav_path = job_out_dir / wav_name

            # Check if skip_existing applies
            if self.skip_existing and bin_path.is_file() and bin_path.stat().st_size > 60:
                logger.info(
                    f"[{idx}/{total_jobs}] SKIPPED: '{model_name}' (Target exists at {bin_path})"
                )
                self.results.append(
                    JobResult(
                        audio_path=str(audio_path),
                        model_name=model_name,
                        model_type=model_type.name,
                        status="SKIPPED",
                        output_bin=str(bin_path),
                        output_wav=str(wav_path) if wav_path.is_file() else None,
                    )
                )
                continue

            logger.info(
                f"\n--- [{idx}/{total_jobs}] Processing '{model_name}' ({audio_path}) ---"
            )

            job_config = copy.deepcopy(self.base_config)
            job_config.samples_path = str(audio_path)
            if is_nam:
                job_config.nam_path = str(audio_path)
            job_config.model.model_type = model_type
            job_config.model.model_name = model_name
            job_config.output_dir = str(job_out_dir)
            job_config.apply_model_defaults()

            job_start = time.time()
            trainer = None
            try:
                trainer = AudioTrainer(job_config)
                trainer.train()
                duration = time.time() - job_start

                self.results.append(
                    JobResult(
                        audio_path=str(audio_path),
                        model_name=model_name,
                        model_type=model_type.name,
                        status="SUCCESS",
                        val_loss=trainer.best_val_loss,
                        duration_seconds=duration,
                        output_bin=str(bin_path) if bin_path.is_file() else None,
                        output_wav=str(wav_path) if wav_path.is_file() else None,
                    )
                )
                logger.info(
                    f"-> [{idx}/{total_jobs}] SUCCESS: '{model_name}' trained in {duration:.1f}s (val loss: {trainer.best_val_loss:.6f})"
                )

            except Exception as e:
                duration = time.time() - job_start
                err_msg = str(e)
                logger.error(f"-> [{idx}/{total_jobs}] FAILED on '{model_name}': {err_msg}")
                self.results.append(
                    JobResult(
                        audio_path=str(audio_path),
                        model_name=model_name,
                        model_type=model_type.name,
                        status="FAILED",
                        duration_seconds=duration,
                        error_message=err_msg,
                    )
                )

            finally:
                # Thorough garbage collection and VRAM release between jobs
                if trainer is not None:
                    del trainer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        total_duration = time.time() - batch_start_time
        self._write_summary(summary_dir, total_duration)
        self._print_summary(total_duration)
        return self.results

    def _write_summary(self, out_dir: Path, total_duration: float) -> None:
        summary_path = out_dir / "batch_summary.json"
        data = {
            "total_duration_seconds": round(total_duration, 2),
            "total_jobs": len(self.results),
            "success_count": sum(1 for r in self.results if r.status == "SUCCESS"),
            "failed_count": sum(1 for r in self.results if r.status == "FAILED"),
            "skipped_count": sum(1 for r in self.results if r.status == "SKIPPED"),
            "results": [asdict(r) for r in self.results],
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved batch execution summary to {summary_path}")

    def _print_summary(self, total_duration: float) -> None:
        successes = sum(1 for r in self.results if r.status == "SUCCESS")
        failures = sum(1 for r in self.results if r.status == "FAILED")
        skipped = sum(1 for r in self.results if r.status == "SKIPPED")

        print("\n" + "=" * 80)
        print(f"BATCH PROCESSING SUMMARY ({total_duration:.1f}s total)")
        print(f"Total: {len(self.results)} | Success: {successes} | Failed: {failures} | Skipped: {skipped}")
        print("-" * 80)
        print(f"{'Model Name':<28} | {'Type':<6} | {'Status':<8} | {'Val Loss':<10} | {'Time (s)':<8}")
        print("-" * 80)
        for r in self.results:
            loss_str = f"{r.val_loss:.6f}" if r.val_loss is not None else "-"
            print(f"{r.model_name:<28} | {r.model_type:<6} | {r.status:<8} | {loss_str:<10} | {r.duration_seconds:<8.1f}")
        print("=" * 80 + "\n")
