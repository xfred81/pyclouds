#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

from pathlib import Path
from urllib.request import urlretrieve


PYTORCH_INDEXES = {
    "cpu": "https://download.pytorch.org/whl/cpu",
    "gpu126": "https://download.pytorch.org/whl/cu126",
    "gpu128": "https://download.pytorch.org/whl/cu128",
}

DEFAULT_MODEL_URL = (
    "https://github.com/xfred81/pyclouds/"
    "releases/download/v0.1.0/default_model.pth"
)

DEFAULT_MODEL_PATH = Path("share/model/default_model.pth")


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def detect_nvidia() -> bool:
    return shutil.which("nvidia-smi") is not None


def resolve_mode(mode: str) -> str:
    if mode != "gpu":
        return mode

    if not detect_nvidia():
        print("[WARN] nvidia-smi not found; falling back to CPU.")
        return "cpu"

    return "gpu126"


def download_default_model() -> None:
    if DEFAULT_MODEL_PATH.exists():
        print(f"[OK] Default model already present: {DEFAULT_MODEL_PATH}")
        return

    print()
    print("[INFO] Downloading default model...")

    DEFAULT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    tmp_file = DEFAULT_MODEL_PATH.with_suffix(".tmp")

    try:
        urlretrieve(DEFAULT_MODEL_URL, tmp_file)
        tmp_file.replace(DEFAULT_MODEL_PATH)

    except Exception:
        if tmp_file.exists():
            tmp_file.unlink()
        raise

    print(f"[OK] Default model downloaded: {DEFAULT_MODEL_PATH}")


def install_pytorch(python: str, mode: str) -> None:
    print()
    print("[INFO] Removing existing PyTorch packages...")

    run([
        python,
        "-m",
        "pip",
        "uninstall",
        "-y",
        "torch",
        "torchvision",
        "torchaudio",
    ])

    print()
    print(f"[INFO] Installing PyTorch mode: {mode}")

    run([
        python,
        "-m",
        "pip",
        "install",
        "torch",
        "torchvision",
        "--index-url",
        PYTORCH_INDEXES[mode],
    ])


def check_torch_installation(python: str) -> None:
    print()
    print("[INFO] Checking PyTorch installation...")

    run([
        python,
        "-c",
        (
            "import torch; "
            "print('torch:', torch.__version__); "
            "print('cuda build:', torch.version.cuda); "
            "print('cuda available:', torch.cuda.is_available()); "
            "print('device count:', torch.cuda.device_count()); "
            "print('device:', torch.cuda.get_device_name(0) "
            "if torch.cuda.is_available() else 'none')"
        ),
    ])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install PyClouds dependencies.",
    )

    parser.add_argument(
        "--mode",
        default="cpu",
        choices=["cpu", "gpu", "gpu126", "gpu128"],
        help=(
            "Installation mode. "
            "cpu = PyTorch CPU wheels, "
            "gpu = auto/conservative GPU mode, "
            "gpu126/gpu128 = explicit CUDA wheels."
        ),
    )

    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="Install core dependencies only, without PyQt6.",
    )

    parser.add_argument(
        "--no-model",
        action="store_true",
        help="Do not download the default model.",
    )

    parser.add_argument(
        "--no-upgrade-pip",
        action="store_true",
        help="Do not upgrade pip before installing.",
    )

    args = parser.parse_args()

    python = sys.executable
    mode = resolve_mode(args.mode)

    if mode not in PYTORCH_INDEXES:
        raise SystemExit(f"Unsupported mode: {mode}")

    if not args.no_upgrade_pip:
        run([
            python,
            "-m",
            "pip",
            "install",
            "-U",
            "pip",
        ])

    install_pytorch(
        python=python,
        mode=mode,
    )

    requirements_file = (
        "requirements.txt"
        if args.no_ui
        else "requirements-ui.txt"
    )

    run([
        python,
        "-m",
        "pip",
        "install",
        "-r",
        requirements_file,
    ])

    check_torch_installation(python)

    if not args.no_model:
        download_default_model()

    print()
    print("[OK] Installation complete.")
    print(f"[OK] Mode: {mode}")
    print(f"[OK] UI: {'no' if args.no_ui else 'yes'}")
    print(f"[OK] Default model: {'no' if args.no_model else 'yes'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())