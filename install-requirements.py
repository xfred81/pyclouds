#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys


PYTORCH_INDEXES = {
    "cpu": "https://download.pytorch.org/whl/cpu",
    "gpu126": "https://download.pytorch.org/whl/cu126",
    "gpu128": "https://download.pytorch.org/whl/cu128",
}


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

    # Conservative default. User can explicitly choose gpu126/gpu128.
    return "gpu126"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install pyclouds dependencies.",
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

    print()
    print("[OK] Installation complete.")
    print(f"[OK] Mode: {mode}")
    print(f"[OK] UI: {'no' if args.no_ui else 'yes'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
