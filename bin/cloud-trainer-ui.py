#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
import warnings
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QPlainTextEdit,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)


DEFAULT_IMG_SIZE = 512


def find_pairs(data_dir: Path):
    jpgs = sorted(data_dir.glob("*.jpg"))
    pairs = []

    for jpg in jpgs:
        stem = jpg.stem

        candidates = [
            data_dir / f"{stem}_mask.png",
            data_dir / f"{stem}mask.png",
            data_dir / f"{stem}.png",
        ]

        mask = next((p for p in candidates if p.exists()), None)

        if mask is not None:
            pairs.append((jpg, mask))

    return pairs


def split_pairs_by_day(pairs, val_ratio=0.2, seed=42):
    groups = defaultdict(list)

    for img, mask in pairs:
        day = img.stem[:10]
        groups[day].append((img, mask))

    days = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(days)

    val_target = max(1, int(len(pairs) * val_ratio))

    val_pairs = []
    train_pairs = []

    for day in days:
        if len(val_pairs) < val_target:
            val_pairs.extend(groups[day])
        else:
            train_pairs.extend(groups[day])

    return train_pairs, val_pairs


class CloudDataset(Dataset):
    def __init__(self, pairs, img_size=512):
        self.pairs = pairs
        self.img_size = img_size

        self.mean = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32,
        )

        self.std = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32,
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

        if image is None:
            raise RuntimeError(f"Could not read image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(
            image,
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_AREA,
        )

        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        image = np.transpose(image, (2, 0, 1))

        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

        if mask is None:
            raise RuntimeError(f"Could not read mask: {mask_path}")

        if mask.ndim == 3:
            if mask.shape[2] == 4:
                alpha = mask[:, :, 3]
                rgb = mask[:, :, :3]
                mask = ((alpha > 0) & (rgb.sum(axis=2) > 0)).astype(np.uint8) * 255
            else:
                mask = (mask.sum(axis=2) > 0).astype(np.uint8) * 255

        mask = cv2.resize(
            mask,
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_NEAREST,
        )

        mask = (mask > 0).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)

        return (
            torch.tensor(image, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
        )


def dice_loss_with_logits(logits, targets, eps=1e-7):
    probs = torch.sigmoid(logits)

    dims = (1, 2, 3)

    intersection = torch.sum(probs * targets, dims)
    cardinality = torch.sum(probs + targets, dims)

    dice = (2.0 * intersection + eps) / (cardinality + eps)

    return 1.0 - dice.mean()


def dice_score_binary(preds, targets, eps=1e-7):
    preds = preds.float()
    targets = targets.float()

    dims = (1, 2, 3)

    intersection = torch.sum(preds * targets, dims)
    cardinality = torch.sum(preds + targets, dims)

    dice = (2.0 * intersection + eps) / (cardinality + eps)

    return dice.mean().item()


def find_best_threshold(model, val_loader, device):
    model.eval()

    thresholds = [i / 100 for i in range(10, 91, 5)]
    scores = {t: [] for t in thresholds}

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)

            for t in thresholds:
                preds = (probs >= t).float()
                score = dice_score_binary(preds, masks)
                scores[t].append(score)

    mean_scores = {
        t: sum(vals) / len(vals)
        for t, vals in scores.items()
        if vals
    }

    best_threshold = max(mean_scores, key=mean_scores.get)
    best_dice = mean_scores[best_threshold]

    return best_threshold, best_dice


class TrainingWorker(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int)
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(
        self,
        data_dir: Path,
        out_path: Path,
        encoder: str,
        epochs: int,
        batch_size: int,
        lr: float,
        img_size: int,
        seed: int,
        patience: int,
        min_delta: float,
        enable_cuda: bool,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.out_path = out_path
        self.encoder = encoder
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.img_size = img_size
        self.seed = seed
        self.patience = patience
        self.min_delta = min_delta
        self.enable_cuda = enable_cuda

        self.stop_requested = False

    def request_stop(self):
        self.stop_requested = True

    def run(self):
        try:
            self._train()
        except Exception as exc:
            self.failed.emit(str(exc))

    def _select_device(self):
        if not self.enable_cuda:
            return "cpu"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cuda_available = torch.cuda.is_available()

        for warning in caught:
            self.log.emit(f"[CUDA WARNING] {warning.message}")

        return "cuda" if cuda_available else "cpu"

    def _train(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        pairs = find_pairs(self.data_dir)

        if len(pairs) < 2:
            raise RuntimeError("Not enough image/mask pairs found.")

        self.log.emit(f"[INFO] Pairs found: {len(pairs)}")

        for img, mask in pairs[:10]:
            self.log.emit(f"  {img.name} -> {mask.name}")

        train_pairs, val_pairs = split_pairs_by_day(
            pairs,
            val_ratio=0.2,
            seed=self.seed,
        )

        if len(train_pairs) < 1 or len(val_pairs) < 1:
            raise RuntimeError("Need at least one training and one validation image.")

        self.log.emit(f"[INFO] Train pairs: {len(train_pairs)}")
        self.log.emit(f"[INFO] Validation pairs: {len(val_pairs)}")

        train_ds = CloudDataset(
            train_pairs,
            img_size=self.img_size,
        )

        val_ds = CloudDataset(
            val_pairs,
            img_size=self.img_size,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        device = self._select_device()

        self.log.emit(f"[INFO] Device: {device}")
        self.log.emit(f"[INFO] Encoder: {self.encoder}")
        self.log.emit(f"[INFO] Image size: {self.img_size}")

        model = smp.Unet(
            encoder_name=self.encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None,
        ).to(device)

        bce = torch.nn.BCEWithLogitsLoss()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.lr,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
        )

        best_val_dice = 0.0
        best_threshold = 0.5
        bad_epochs = 0

        for epoch in range(1, self.epochs + 1):
            if self.stop_requested:
                self.log.emit("[INFO] Training stopped by user.")
                break

            model.train()
            train_loss = 0.0

            for batch_index, (images, masks) in enumerate(train_loader, start=1):
                if self.stop_requested:
                    break

                images = images.to(device)
                masks = masks.to(device)

                logits = model(images)
                loss = bce(logits, masks) + dice_loss_with_logits(logits, masks)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)

            train_loss /= len(train_ds)

            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for images, masks in val_loader:
                    if self.stop_requested:
                        break

                    images = images.to(device)
                    masks = masks.to(device)

                    logits = model(images)
                    loss = bce(logits, masks) + dice_loss_with_logits(logits, masks)

                    val_loss += loss.item() * images.size(0)

            val_loss /= len(val_ds)

            threshold, val_dice = find_best_threshold(
                model,
                val_loader,
                device,
            )

            scheduler.step(val_loss)

            current_lr = optimizer.param_groups[0]["lr"]

            self.log.emit(
                f"[INFO] epoch={epoch:03d} "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_dice={val_dice:.4f} "
                f"best_threshold={threshold:.2f} "
                f"lr={current_lr:.2e}"
            )

            if val_dice > best_val_dice + self.min_delta:
                best_val_dice = val_dice
                best_threshold = threshold
                bad_epochs = 0

                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "encoder_name": self.encoder,
                        "img_size": self.img_size,
                        "threshold": best_threshold,
                        "val_dice": best_val_dice,
                        "val_loss": val_loss,
                        "mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225],
                    },
                    self.out_path,
                )

                self.log.emit(
                    f"[INFO] New best model saved: {self.out_path} "
                    f"(dice={best_val_dice:.4f}, threshold={best_threshold:.2f})"
                )

            else:
                bad_epochs += 1
                self.log.emit(
                    f"[INFO] No Dice improvement: {bad_epochs}/{self.patience}"
                )

                if bad_epochs >= self.patience:
                    self.log.emit("[INFO] Early stopping")
                    break

            progress = int(100 * epoch / self.epochs)
            self.progress.emit(progress)

        self.finished_ok.emit(
            f"Training finished. Best model: {self.out_path}"
        )


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("PyClouds Trainer")

        self.worker: TrainingWorker | None = None

        self._build_ui()
        self.resize(1000, 750)

    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)

        config_box = QGroupBox("Training configuration")
        form = QFormLayout(config_box)

        self.data_edit = QLineEdit()
        self.out_edit = QLineEdit("cloud_unet.pth")

        data_btn = QPushButton("Browse")
        out_btn = QPushButton("Browse")

        data_btn.clicked.connect(self.browse_data_dir)
        out_btn.clicked.connect(self.browse_output_model)

        data_row = QHBoxLayout()
        data_row.addWidget(self.data_edit)
        data_row.addWidget(data_btn)

        out_row = QHBoxLayout()
        out_row.addWidget(self.out_edit)
        out_row.addWidget(out_btn)

        self.encoder_combo = QComboBox()
        self.encoder_combo.addItems(
            [
                "resnet18",
                "resnet34",
                "mobilenet_v2",
                "efficientnet-b0",
                "efficientnet-b1",
            ]
        )

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(100)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(4)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(8)
        self.lr_spin.setRange(0.00000001, 1.0)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setValue(0.0001)

        self.img_size_spin = QSpinBox()
        self.img_size_spin.setRange(64, 4096)
        self.img_size_spin.setSingleStep(64)
        self.img_size_spin.setValue(DEFAULT_IMG_SIZE)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(42)

        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 1000)
        self.patience_spin.setValue(12)

        self.min_delta_spin = QDoubleSpinBox()
        self.min_delta_spin.setDecimals(6)
        self.min_delta_spin.setRange(0.0, 1.0)
        self.min_delta_spin.setSingleStep(0.0001)
        self.min_delta_spin.setValue(0.0001)

        self.cuda_checkbox = QCheckBox("Enable CUDA")
        self.cuda_checkbox.setChecked(True)

        form.addRow("Training pairs directory:", data_row)
        form.addRow("Output model:", out_row)
        form.addRow("Encoder:", self.encoder_combo)
        form.addRow("Epochs:", self.epochs_spin)
        form.addRow("Batch size:", self.batch_spin)
        form.addRow("Learning rate:", self.lr_spin)
        form.addRow("Image size:", self.img_size_spin)
        form.addRow("Seed:", self.seed_spin)
        form.addRow("Patience:", self.patience_spin)
        form.addRow("Min delta:", self.min_delta_spin)
        form.addRow("", self.cuda_checkbox)

        root.addWidget(config_box)

        buttons = QHBoxLayout()

        self.scan_btn = QPushButton("Scan pairs")
        self.start_btn = QPushButton("Start training")
        self.stop_btn = QPushButton("Stop")

        self.stop_btn.setEnabled(False)

        self.scan_btn.clicked.connect(self.scan_pairs)
        self.start_btn.clicked.connect(self.start_training)
        self.stop_btn.clicked.connect(self.stop_training)

        buttons.addWidget(self.scan_btn)
        buttons.addWidget(self.start_btn)
        buttons.addWidget(self.stop_btn)

        root.addLayout(buttons)

        self.status_label = QLabel("Ready.")
        root.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        root.addWidget(self.progress_bar)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        root.addWidget(self.log_view, stretch=1)

        self.setCentralWidget(central)

    def browse_data_dir(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select training pairs directory",
        )

        if directory:
            self.data_edit.setText(directory)
            self.scan_pairs()

    def browse_output_model(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Select output model",
            "cloud_unet.pth",
            "PyTorch checkpoint (*.pth);;All files (*)",
        )

        if path:
            self.out_edit.setText(path)

    def scan_pairs(self):
        data_dir = Path(self.data_edit.text().strip())

        if not data_dir.exists():
            self.status_label.setText("Training directory does not exist.")
            return

        pairs = find_pairs(data_dir)

        self.log_view.appendPlainText("")
        self.log_view.appendPlainText(f"[INFO] Directory: {data_dir}")
        self.log_view.appendPlainText(f"[INFO] Pairs found: {len(pairs)}")

        for img, mask in pairs[:20]:
            self.log_view.appendPlainText(f"  {img.name} -> {mask.name}")

        if len(pairs) > 20:
            self.log_view.appendPlainText(f"  ... {len(pairs) - 20} more")

        self.status_label.setText(f"{len(pairs)} pairs found.")

    def start_training(self):
        data_dir = Path(self.data_edit.text().strip())
        out_path = Path(self.out_edit.text().strip())

        if not data_dir.exists():
            self.error("Training directory does not exist.")
            return

        if not out_path.name:
            self.error("Invalid output model path.")
            return

        pairs = find_pairs(data_dir)

        if len(pairs) < 2:
            self.error("Not enough image/mask pairs found.")
            return

        self.progress_bar.setValue(0)
        self.log_view.appendPlainText("")
        self.log_view.appendPlainText("[INFO] Starting training...")

        self.worker = TrainingWorker(
            data_dir=data_dir,
            out_path=out_path,
            encoder=self.encoder_combo.currentText(),
            epochs=self.epochs_spin.value(),
            batch_size=self.batch_spin.value(),
            lr=self.lr_spin.value(),
            img_size=self.img_size_spin.value(),
            seed=self.seed_spin.value(),
            patience=self.patience_spin.value(),
            min_delta=self.min_delta_spin.value(),
            enable_cuda=self.cuda_checkbox.isChecked(),
        )

        self.worker.log.connect(self.log_view.appendPlainText)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished_ok.connect(self.training_finished)
        self.worker.failed.connect(self.training_failed)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.scan_btn.setEnabled(False)

        self.worker.start()

    def stop_training(self):
        if self.worker is not None:
            self.worker.request_stop()
            self.log_view.appendPlainText("[INFO] Stop requested...")

    def training_finished(self, message: str):
        self.status_label.setText(message)
        self.log_view.appendPlainText(f"[OK] {message}")

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.scan_btn.setEnabled(True)

    def training_failed(self, message: str):
        self.status_label.setText("Training failed.")
        self.log_view.appendPlainText(f"[ERROR] {message}")

        QMessageBox.critical(
            self,
            "Training failed",
            message,
        )

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.scan_btn.setEnabled(True)

    def error(self, message: str):
        QMessageBox.critical(
            self,
            "Error",
            message,
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Qt6 UI for training the cloud segmentation model.",
    )

    parser.add_argument(
        "data",
        nargs="?",
        default=None,
        help="Optional training pairs directory.",
    )

    parser.add_argument(
        "--out",
        default=None,
        help="Optional output model path.",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    app = QApplication([])

    window = MainWindow()

    if args.data:
        window.data_edit.setText(args.data)
        window.scan_pairs()

    if args.out:
        window.out_edit.setText(args.out)

    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
