#!/usr/bin/env python3
"""
Main entrypoint for the AudioTrain package.
"""
import os
import subprocess
import sys


def _ensure_venv() -> None:
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
