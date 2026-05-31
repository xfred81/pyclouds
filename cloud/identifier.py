# cloud/identifier.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

import warnings


@dataclass
class IdentifierResult:
    """Result returned by the cloud identifier."""

    cloud_mask: np.ndarray          # Boolean cloud mask, after valid-mask filtering
    raw_cloud_mask: np.ndarray      # Boolean cloud mask before valid-mask filtering
    valid_mask: np.ndarray          # Boolean mask of pixels included in analysis
    probability_map: np.ndarray     # Float map in [0, 1]
    overlay_bgr: np.ndarray         # Original image with cloud overlay
    cloud_percent: float
    cloud_pixels: int
    analyzed_pixels: int


class Identifier:
    """
    Generic cloud identifier based on a trained segmentation model.

    This class is independent from any CLI or UI code.
    It can be used from:
      - a command-line executable
      - a Qt6 GUI
      - a batch script
      - a Python API
    """

    def __init__(
        self,
        model_path: str | Path,
        threshold: Optional[float] = None,
        alpha: float = 0.45,
        enable_cuda: bool = True,
    ):
        self.model_path = Path(model_path)
        self.alpha = alpha

        self.device_warning: str | None = None

        if enable_cuda:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")

                cuda_available = torch.cuda.is_available()

                if caught:
                    self.device_warning = "\n".join(
                        str(w.message) for w in caught
                    )

            self.device = "cuda" if cuda_available else "cpu"
        else:
            self.device = "cpu"

        self.model, self.img_size, saved_threshold, self.mean, self.std = (
            self._load_model(self.model_path)
        )

        self.threshold = (
            float(threshold)
            if threshold is not None
            else float(saved_threshold)
        )

    def _load_model(self, model_path: Path):
        """Load a segmentation model checkpoint."""

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        checkpoint = torch.load(
            model_path,
            map_location=self.device,
        )

        encoder_name = checkpoint.get("encoder_name", "resnet18")
        img_size = int(checkpoint.get("img_size", 512))
        threshold = float(checkpoint.get("threshold", 0.5))

        # Backward compatibility with older checkpoints.
        mean = np.array(
            checkpoint.get("mean", [0.0, 0.0, 0.0]),
            dtype=np.float32,
        )

        std = np.array(
            checkpoint.get("std", [1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        ).to(self.device)

        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        return model, img_size, threshold, mean, std

    def predict_from_file(
        self,
        image_path: str | Path,
        valid_mask_path: str | Path | None = None,
    ) -> IdentifierResult:
        """Load an image from disk and run cloud detection."""

        image_path = Path(image_path)

        image_bgr = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image_bgr is None:
            raise ValueError(f"Could not read image: {image_path}")

        valid_mask_bgr = None

        if valid_mask_path is not None:
            valid_mask_path = Path(valid_mask_path)

            valid_mask_bgr = cv2.imread(
                str(valid_mask_path),
                cv2.IMREAD_GRAYSCALE,
            )

            if valid_mask_bgr is None:
                raise ValueError(f"Could not read valid mask: {valid_mask_path}")

        return self.predict(
            image_bgr=image_bgr,
            valid_mask_gray=valid_mask_bgr,
        )

    def predict(
        self,
        image_bgr: np.ndarray,
        valid_mask_gray: np.ndarray | None = None,
    ) -> IdentifierResult:
        """
        Run cloud detection on an already loaded BGR image.

        This is the most useful method for a Qt UI, because the UI can load,
        display, cache or convert images before calling the identifier.
        """

        if image_bgr is None or image_bgr.ndim != 3:
            raise ValueError("image_bgr must be a valid BGR image")

        raw_cloud_mask, probability_map = self._predict_mask(image_bgr)

        valid_mask = self._prepare_valid_mask(
            valid_mask_gray=valid_mask_gray,
            target_shape=raw_cloud_mask.shape,
        )

        cloud_mask = raw_cloud_mask & valid_mask

        analyzed_pixels = int(valid_mask.sum())
        cloud_pixels = int(cloud_mask.sum())

        cloud_percent = (
            100.0 * cloud_pixels / analyzed_pixels
            if analyzed_pixels > 0
            else 0.0
        )

        overlay_bgr = self.make_overlay(
            image_bgr=image_bgr,
            cloud_mask=cloud_mask,
            alpha=self.alpha,
        )

        return IdentifierResult(
            cloud_mask=cloud_mask,
            raw_cloud_mask=raw_cloud_mask,
            valid_mask=valid_mask,
            probability_map=probability_map,
            overlay_bgr=overlay_bgr,
            cloud_percent=cloud_percent,
            cloud_pixels=cloud_pixels,
            analyzed_pixels=analyzed_pixels,
        )

    def _predict_mask(
        self,
        image_bgr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the binary cloud mask and the probability map."""

        height, width = image_bgr.shape[:2]

        rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB,
        )

        small = cv2.resize(
            rgb,
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_AREA,
        )

        x = small.astype(np.float32) / 255.0

        # Must match the preprocessing used during training.
        x = (x - self.mean) / self.std

        x = np.transpose(x, (2, 0, 1))

        tensor = torch.tensor(
            x,
            dtype=torch.float32,
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits)[0, 0]
            probs = probs.cpu().numpy()

        probability_map = cv2.resize(
            probs,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )

        cloud_mask = probability_map >= self.threshold

        return cloud_mask, probability_map

    @staticmethod
    def _prepare_valid_mask(
        valid_mask_gray: np.ndarray | None,
        target_shape: tuple[int, int],
    ) -> np.ndarray:
        """Prepare a boolean valid-sky mask."""

        if valid_mask_gray is None:
            return np.ones(
                target_shape,
                dtype=bool,
            )

        resized = cv2.resize(
            valid_mask_gray,
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        return resized > 0

    @staticmethod
    def make_overlay(
        image_bgr: np.ndarray,
        cloud_mask: np.ndarray,
        alpha: float = 0.45,
    ) -> np.ndarray:
        """Create a red overlay on detected cloud pixels."""

        overlay = image_bgr.copy()

        red = np.zeros_like(image_bgr)
        red[:, :] = (0, 0, 255)  # BGR red

        overlay[cloud_mask] = cv2.addWeighted(
            image_bgr[cloud_mask],
            1.0 - alpha,
            red[cloud_mask],
            alpha,
            0,
        )

        return overlay

    @staticmethod
    def save_mask(
        cloud_mask: np.ndarray,
        output_path: str | Path,
    ) -> None:
        """Save a boolean cloud mask as a PNG image."""

        mask_png = cloud_mask.astype(np.uint8) * 255
        cv2.imwrite(str(output_path), mask_png)

    @staticmethod
    def save_overlay(
        overlay_bgr: np.ndarray,
        output_path: str | Path,
    ) -> None:
        """Save an overlay image."""

        cv2.imwrite(str(output_path), overlay_bgr)
