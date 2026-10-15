#!/usr/bin/env python3
"""
Top-level entrypoint for training Sinco neural amp models.
Usage:
    python train.py --samples path/to/stereo_sample.wav --model AM4 --epochs 1000
    python train.py "path/to/stereo_sample.wav?1024?1000?0?0.0008?1"
"""
import os
import subprocess
import sys


def _ensure_venv() -> None:
    """Automatically redirects to .venv if the current interpreter lacks required packages."""
    root = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(root, ".venv", "Scripts", "python.exe"),
        os.path.join(root, ".venv", "bin", "python"),
        os.path.join(root, "venv", "Scripts", "python.exe"),
        os.path.join(root, "venv", "bin", "python"),
    ]
    venv_python = next((c for c in candidates if os.path.isfile(c)), None)
    if venv_python and os.path.abspath(sys.executable).lower() != os.path.abspath(venv_python).lower():
        try:
            import yaml
            import torch
        except ImportError:
            res = subprocess.call([venv_python] + sys.argv)
            sys.exit(res)


_ensure_venv()

from audiotrain.cli import main

if __name__ == "__main__":
    sys.exit(main())
