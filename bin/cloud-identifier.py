#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from cloud.identifier import Identifier


DEFAULT_MODEL = "share/model/default_model.pth"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run cloud detection on an image using a trained ML model.",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Path to the trained model checkpoint. Default: {DEFAULT_MODEL}",
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input image path.",
    )

    parser.add_argument(
        "--out",
        default=None,
        help="Optional output overlay image path.",
    )

    parser.add_argument(
        "--mask-out",
        default=None,
        help="Optional output binary cloud mask path.",
    )

    parser.add_argument(
        "--valid-mask",
        default=None,
        help=(
            "Optional valid sky mask. "
            "White pixels are analyzed, black pixels are ignored."
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override model checkpoint threshold.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Overlay opacity for detected clouds. Default: 0.45",
    )

    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Disable CUDA even if it is available.",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only the cloud coverage percentage.",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    identifier = Identifier(
        model_path=Path(args.model),
        threshold=args.threshold,
        alpha=args.alpha,
        enable_cuda=not args.no_cuda,
    )

    result = identifier.predict_from_file(
        image_path=Path(args.input),
        valid_mask_path=Path(args.valid_mask) if args.valid_mask else None,
    )

    if args.out:
        Identifier.save_overlay(
            result.overlay_bgr,
            Path(args.out),
        )

    if args.mask_out:
        Identifier.save_mask(
            result.cloud_mask,
            Path(args.mask_out),
        )

    if args.quiet:
        print(f"{result.cloud_percent:.2f}")
        return 0

    print()
    print(f"Device: {identifier.device}")
    print(f"Model: {identifier.model_path}")
    print(f"Threshold: {identifier.threshold:.3f}")
    print(f"Alpha: {identifier.alpha:.3f}")
    print()
    print(f"Cloud coverage: {result.cloud_percent:.2f}%")
    print(f"Cloud pixels: {result.cloud_pixels}")
    print(f"Analyzed pixels: {result.analyzed_pixels}")

    if args.out:
        print(f"Overlay: {args.out}")
    else:
        print("Overlay: not saved")

    if args.mask_out:
        print(f"Mask: {args.mask_out}")
    else:
        print("Mask: not saved")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
