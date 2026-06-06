#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QActionGroup, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from cloud.identifier import Identifier
from cloud.ui.mask_editor import MaskEditor


DEFAULT_MODEL = "share/model/default_model.pth"
MANUAL_EDIT_WIDTH = 400

def compute_cloud_stats(
    cloud_mask: np.ndarray,
    valid_zone_mask: np.ndarray | None = None,
) -> tuple[float, int, int]:
    """Return cloud coverage percent, cloud pixels and valid sky pixels."""
    mask = cloud_mask.astype(bool)

    if valid_zone_mask is not None:
        analyzed = valid_zone_mask.astype(bool)
        cloud = mask & analyzed
    else:
        analyzed = np.ones(mask.shape, dtype=bool)
        cloud = mask

    valid_pixels = int(analyzed.sum())
    cloud_pixels = int(cloud.sum())
    cloud_percent = (
        100.0 * cloud_pixels / valid_pixels
        if valid_pixels > 0
        else 0.0
    )

    return cloud_percent, cloud_pixels, valid_pixels


def load_valid_zone_mask_for_image(
    path: Path,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Load a valid sky mask and resize it to image_shape=(height, width)."""
    vm = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if vm is None:
        raise ValueError(f"Could not read valid zone mask: {path}")

    h, w = image_shape
    vm = cv2.resize(vm, (w, h), interpolation=cv2.INTER_NEAREST)
    return vm > 0


def predict_cloud_mask(
    image_bgr: np.ndarray,
    model_path: str | Path,
    threshold: float | None,
    alpha: float,
    enable_cuda: bool,
    valid_zone_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, Identifier]:
    """Run the cloud detector and return the predicted mask plus the Identifier."""
    identifier = Identifier(
        model_path=Path(model_path),
        threshold=threshold,
        alpha=alpha,
        enable_cuda=enable_cuda,
    )

    valid_mask_gray = None
    if valid_zone_mask is not None:
        valid_mask_gray = valid_zone_mask.astype(np.uint8) * 255

    result = identifier.predict(
        image_bgr=image_bgr,
        valid_mask_gray=valid_mask_gray,
    )

    return result.cloud_mask, identifier


class MainWindow(QMainWindow):
    def __init__(
        self,
        model_path: str,
        input_path: str | None,
        threshold: float | None,
        alpha: float,
        enable_cuda: bool,
    ):
        super().__init__()

        self.setWindowTitle("PyClouds 0.1.4")

        self.model_path = Path(model_path)
        self.threshold_override = threshold
        self.alpha = alpha
        self.enable_cuda = enable_cuda

        self.identifier: Identifier | None = None

        self.image_path: Path | None = None
        self.image_bgr: np.ndarray | None = None
        self.valid_zone_mask: np.ndarray | None = None

        self.cloud_percent = 0.0
        self.cloud_pixels = 0
        self.analyzed_pixels = 0
        self.detection_running = False

        self._build_ui()
        self._build_menu()
        self._build_shortcuts()
        self._update_actions()

        if input_path:
            self.load_cloudy_image(Path(input_path))

    def _build_ui(self):
        central = QWidget()
        root = QHBoxLayout(central)

        root.addWidget(self._build_left_panel())
        root.addWidget(self._build_center_panel(), stretch=1)
        root.addWidget(self._build_right_panel())

        self.setCentralWidget(central)
        self.resize(1400, 850)

    def _separator(self) -> QFrame:
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        return sep

    def _section_title(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("font-weight: bold;")
        return label

    def _build_left_panel(self) -> QWidget:
        panel = QFrame()
        self.left_panel = panel
        panel.setFrameShape(QFrame.Shape.StyledPanel)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.tool_value_label = QLabel("Brush size / magic threshold: 25")
        self.tool_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.tool_value_slider = QSlider(Qt.Orientation.Horizontal)
        self.tool_value_slider.setRange(1, 200)
        self.tool_value_slider.setValue(25)
        self.tool_value_slider.setMaximumHeight(40)
        self.tool_value_slider.valueChanged.connect(self.on_tool_value_changed)

        self.overlay_btn = QPushButton("Sh&ow cloud overlay")
        self.overlay_btn.setCheckable(True)
        self.overlay_btn.setChecked(True)
        self.overlay_btn.setToolTip("Show or hide the red cloud mask overlay.")
        self.overlay_btn.clicked.connect(self.on_overlay_toggled)

        self.run_btn = QPushButton("Run cloud detection")
        self.run_btn.setToolTip("Run the model on the currently loaded image.")
        self.run_btn.clicked.connect(self.run_identification)

        self.brush_btn = QPushButton("&Brush")
        self.fill_btn = QPushButton("&Fill closed zone")
        self.magic_btn = QPushButton("M&agic select")

        for btn in (self.brush_btn, self.fill_btn, self.magic_btn):
            btn.setCheckable(True)

        self.brush_btn.setChecked(True)

        self.brush_btn.clicked.connect(lambda: self.set_tool_mode("brush"))
        self.fill_btn.clicked.connect(lambda: self.set_tool_mode("fill"))
        self.magic_btn.clicked.connect(lambda: self.set_tool_mode("magic"))

        self.undo_btn = QPushButton("&Undo")
        self.undo_btn.clicked.connect(self.mask_editor_undo)

        self.reset_btn = QPushButton("&Reset mask")
        self.reset_btn.clicked.connect(self.reset_mask)

        detect_section = QFrame()
        detect_section.setFrameShape(QFrame.Shape.StyledPanel)
        detect_layout = QVBoxLayout(detect_section)
        detect_layout.setContentsMargins(8, 8, 8, 8)
        detect_layout.addWidget(self._section_title("Detect clouds"))
        detect_layout.addWidget(self.run_btn)
        detect_layout.addWidget(self.overlay_btn)

        self.edit_toggle = QPushButton("Manual mask editing")
        self.edit_toggle.setFixedWidth(MANUAL_EDIT_WIDTH)
        self.edit_toggle.setCheckable(True)
        self.edit_toggle.setChecked(False)
        self.edit_toggle.setStyleSheet("QPushButton { text-align: center; }")
        self.edit_toggle.clicked.connect(self.on_edit_tools_toggled)

        self.edit_tools_widget = QFrame()
        self.edit_tools_widget.setFixedWidth(MANUAL_EDIT_WIDTH)
        self.edit_tools_widget.setFrameShape(QFrame.Shape.StyledPanel)
        edit_layout = QVBoxLayout(self.edit_tools_widget)
        edit_layout.setContentsMargins(8, 8, 8, 8)

        note = QLabel(
            "Manual edits are mainly useful to create clean training pairs: "
            "a reference sky image plus its cloud mask. These pairs can then "
            "be used to improve the model."
        )
        note.setWordWrap(True)
        note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        note.setStyleSheet("color: #666; font-size: 11px;")

        tool_value_title = QLabel("Tool value")
        tool_value_title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        edit_layout.addWidget(note)
        edit_layout.addWidget(tool_value_title)
        edit_layout.addWidget(self.tool_value_slider)
        edit_layout.addWidget(self.tool_value_label)
        edit_layout.addSpacing(6)
        edit_layout.addWidget(self.brush_btn)
        edit_layout.addWidget(self.fill_btn)
        edit_layout.addWidget(self.magic_btn)
        edit_layout.addWidget(self.undo_btn)
        edit_layout.addWidget(self.reset_btn)

        self.edit_tools_widget.setVisible(False)

        layout.addWidget(detect_section)
        layout.addSpacing(8)
        layout.addWidget(self.edit_toggle)
        layout.addWidget(self.edit_tools_widget)
        layout.addStretch()

        return panel

    def on_edit_tools_toggled(self, checked: bool):
        self.edit_tools_widget.setVisible(checked)

    def _build_center_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self.mask_editor = MaskEditor()
        self.mask_editor.set_overlay_enabled(True)
        self.mask_editor.set_overlay_alpha(self.alpha)
        self.mask_editor.set_tool_value(self.tool_value_slider.value())
        self.mask_editor.mask_changed.connect(self.recompute_stats_from_editor)

        layout.addWidget(self.mask_editor)

        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QFrame()
        panel.setFrameShape(QFrame.Shape.StyledPanel)
        panel.setMaximumWidth(280)

        layout = QVBoxLayout(panel)

        title = QLabel("Statistics")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-weight: bold;")

        self.image_label = QLabel("Image: none")
        self.model_label = QLabel(f"Model: {self.model_path}")
        self.device_label = QLabel("Device: not loaded")
        self.valid_zone_label = QLabel("Valid zone: none")
        self.coverage_label = QLabel("Cloud coverage: —")
        self.cloud_pixels_label = QLabel("Cloud pixels: —")
        self.analyzed_pixels_label = QLabel("Analyzed pixels: —")
        self.threshold_label = QLabel("Model threshold: —")

        for label in (
            self.image_label,
            self.model_label,
            self.device_label,
            self.valid_zone_label,
            self.coverage_label,
            self.cloud_pixels_label,
            self.analyzed_pixels_label,
            self.threshold_label,
        ):
            label.setWordWrap(True)

        layout.addWidget(title)
        layout.addSpacing(10)
        layout.addWidget(self.image_label)
        layout.addWidget(self.model_label)
        layout.addWidget(self.device_label)
        layout.addWidget(self.valid_zone_label)
        layout.addSpacing(10)
        layout.addWidget(self.coverage_label)
        layout.addWidget(self.cloud_pixels_label)
        layout.addWidget(self.analyzed_pixels_label)
        layout.addWidget(self.threshold_label)
        layout.addStretch()

        return panel

    def _build_menu(self):
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("&File")

        self.load_image_action = QAction("&Load cloudy image", self)
        self.load_image_action.setShortcut(QKeySequence("Ctrl+L"))

        self.save_mask_action = QAction("&Save cloud overlay", self)
        self.save_mask_action.setShortcut(QKeySequence("Ctrl+S"))

        self.save_pair_action = QAction("Save input image and mask", self)
        self.load_valid_zone_action = QAction("Load and apply valid zone mask", self)
        self.quit_action = QAction("&Quit", self)

        self.load_image_action.triggered.connect(self.on_load_cloudy_image)
        self.save_mask_action.triggered.connect(self.save_cloud_mask)
        self.save_pair_action.triggered.connect(self.save_input_image_and_mask)
        self.load_valid_zone_action.triggered.connect(self.on_load_valid_zone_mask)
        self.quit_action.triggered.connect(self.close)

        file_menu.addAction(self.load_image_action)
        file_menu.addSeparator()
        file_menu.addAction(self.save_mask_action)
        file_menu.addAction(self.save_pair_action)
        file_menu.addSeparator()
        file_menu.addAction(self.load_valid_zone_action)
        file_menu.addSeparator()
        file_menu.addAction(self.quit_action)

        preferences_menu = menu_bar.addMenu("&Preferences")

        self.enable_cuda_action = QAction("Enable CUDA", self)
        self.enable_cuda_action.setCheckable(True)
        self.enable_cuda_action.setChecked(self.enable_cuda)
        self.enable_cuda_action.triggered.connect(self.on_enable_cuda_changed)

        preferences_menu.addAction(self.enable_cuda_action)

        tools_menu = menu_bar.addMenu("&Tools")

        self.brush_action = QAction("&Brush", self)
        self.fill_action = QAction("&Fill closed zone", self)
        self.magic_action = QAction("M&agic select", self)
        self.undo_action = QAction("&Undo", self)
        self.reset_action = QAction("&Reset", self)
        self.overlay_action = QAction("Sh&ow cloud overlay", self)
        self.overlay_action.setCheckable(True)
        self.overlay_action.setChecked(True)

        for action in (self.brush_action, self.fill_action, self.magic_action):
            action.setCheckable(True)

        self.tool_action_group = QActionGroup(self)
        self.tool_action_group.setExclusive(True)
        self.tool_action_group.addAction(self.brush_action)
        self.tool_action_group.addAction(self.fill_action)
        self.tool_action_group.addAction(self.magic_action)

        self.brush_action.setChecked(True)

        self.brush_action.triggered.connect(lambda: self.set_tool_mode("brush"))
        self.fill_action.triggered.connect(lambda: self.set_tool_mode("fill"))
        self.magic_action.triggered.connect(lambda: self.set_tool_mode("magic"))
        self.undo_action.triggered.connect(self.mask_editor_undo)
        self.reset_action.triggered.connect(self.reset_mask)
        self.overlay_action.triggered.connect(self.set_overlay_visible)

        tools_menu.addAction(self.brush_action)
        tools_menu.addAction(self.fill_action)
        tools_menu.addAction(self.magic_action)
        tools_menu.addSeparator()
        tools_menu.addAction(self.undo_action)
        tools_menu.addAction(self.reset_action)
        tools_menu.addSeparator()
        tools_menu.addAction(self.overlay_action)

    def _build_shortcuts(self):
        QShortcut(QKeySequence("L"), self, activated=self.on_load_cloudy_image)
        QShortcut(QKeySequence("S"), self, activated=self.save_cloud_mask)

        QShortcut(QKeySequence("U"), self, activated=self.mask_editor_undo)
        QShortcut(QKeySequence("R"), self, activated=self.reset_mask)

        QShortcut(QKeySequence("O"), self, activated=self.toggle_overlay)

        QShortcut(QKeySequence("B"), self, activated=lambda: self.set_tool_mode("brush"))
        QShortcut(QKeySequence("F"), self, activated=lambda: self.set_tool_mode("fill"))
        QShortcut(QKeySequence("A"), self, activated=lambda: self.set_tool_mode("magic"))

    def _update_actions(self):
        has_image = self.image_bgr is not None

        self.save_mask_action.setEnabled(has_image)
        self.save_pair_action.setEnabled(has_image)
        self.load_valid_zone_action.setEnabled(has_image)

        self.brush_action.setEnabled(has_image)
        self.fill_action.setEnabled(has_image)
        self.magic_action.setEnabled(has_image)
        self.undo_action.setEnabled(has_image)
        self.reset_action.setEnabled(has_image)
        self.overlay_action.setEnabled(has_image)

        self.run_btn.setEnabled(has_image and not self.detection_running)
        self.undo_btn.setEnabled(has_image)
        self.reset_btn.setEnabled(has_image)
        self.overlay_btn.setEnabled(has_image)
        self.tool_value_slider.setEnabled(has_image)

        self.brush_btn.setEnabled(has_image)
        self.fill_btn.setEnabled(has_image)
        self.magic_btn.setEnabled(has_image)

    def on_load_cloudy_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load cloudy image",
            "",
            "Images (*.jpg *.jpeg *.png *.tif *.tiff);;All files (*)",
        )

        if path:
            self.load_cloudy_image(Path(path))

    def load_cloudy_image(self, path: Path):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)

        if image is None:
            self.error(f"Could not read image: {path}")
            return

        self.image_path = path
        self.image_bgr = image

        self.valid_zone_mask = None
        self.identifier = None

        empty_mask = np.zeros(image.shape[:2], dtype=bool)

        self.mask_editor.set_mask(
            empty_mask,
            base_image_bgr=image,
        )

        self.image_label.setText(
            f"Image: {path.name}\n{image.shape[1]}x{image.shape[0]}"
        )
        self.device_label.setText("Device: not loaded")
        self.valid_zone_label.setText("Valid zone: none")
        self.threshold_label.setText("Model threshold: —")

        self.recompute_stats_from_editor()
        self._update_actions()

    def on_load_valid_zone_mask(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load valid zone mask",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff);;All files (*)",
        )

        if path:
            self.load_and_apply_valid_zone_mask(Path(path))

    def load_and_apply_valid_zone_mask(self, path: Path):
        if self.image_bgr is None:
            self.error("Load an image first")
            return

        try:
            self.valid_zone_mask = load_valid_zone_mask_for_image(
                path,
                self.image_bgr.shape[:2],
            )
        except ValueError as exc:
            self.error(str(exc))
            return

        current_mask = self.mask_editor.get_mask_bool()
        current_mask = current_mask & self.valid_zone_mask

        self.mask_editor.set_mask(
            current_mask,
            base_image_bgr=self.image_bgr,
            reset_zoom=False,
        )

        self.valid_zone_label.setText(f"Valid zone: {path.name}")
        self.recompute_stats_from_editor()

    def set_detection_running(self, running: bool):
        self.detection_running = running
        self.run_btn.setEnabled((self.image_bgr is not None) and not running)
        self.run_btn.setText(
            "Detecting clouds..."
            if running
            else "Run cloud detection"
        )
        QApplication.processEvents()

    def run_identification(self):
        if self.detection_running:
            return

        try:
            if self.image_bgr is None:
                raise ValueError("Load an image first")

            self.set_detection_running(True)

            cloud_mask, self.identifier = predict_cloud_mask(
                image_bgr=self.image_bgr,
                model_path=self.model_path,
                threshold=self.threshold_override,
                alpha=self.alpha,
                enable_cuda=self.enable_cuda_action.isChecked(),
                valid_zone_mask=self.valid_zone_mask,
            )

            self.mask_editor.set_mask(
                cloud_mask,
                base_image_bgr=self.image_bgr,
                reset_zoom=False,
            )

            self.threshold_label.setText(
                f"Model threshold: {self.identifier.threshold:.3f}"
            )

            if self.identifier.device_warning:
                self.device_label.setText(
                    f"Device: {self.identifier.device}\n\n"
                    f"CUDA warning:\n{self.identifier.device_warning}"
                )
            else:
                self.device_label.setText(
                    f"Device: {self.identifier.device}"
                )

            self.recompute_stats_from_editor()

        except Exception as exc:
            self.error(str(exc))
        finally:
            self.set_detection_running(False)

    def save_cloud_mask(self):
        if self.image_bgr is None:
            return
        
        if self.image_path is not None:
            default_name = (
                f"{self.image_path.stem}_mask.png"
            )
        else:
            default_name = "cloud_mask.png"

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save cloud mask",
            default_name,
            "PNG image (*.png);;All files (*)",
        )

        if path:
            editable_mask = self.mask_editor.get_mask_bool()
            Identifier.save_mask(editable_mask, path)

    def save_input_image_and_mask(self):
        if self.image_path is None or self.image_bgr is None:
            self.error("No image loaded")
            return

        directory = QFileDialog.getExistingDirectory(
            self,
            "Select output training directory",
        )

        if not directory:
            return

        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        stem = self.image_path.stem

        image_out = out_dir / f"{stem}.jpg"
        mask_out = out_dir / f"{stem}_mask.png"

        shutil.copy2(self.image_path, image_out)

        editable_mask = self.mask_editor.get_mask_bool()
        Identifier.save_mask(editable_mask, mask_out)

        QMessageBox.information(
            self,
            "Training pair saved",
            f"Image: {image_out}\nMask: {mask_out}",
        )

    def reset_mask(self):
        if self.image_bgr is None:
            return

        empty_mask = np.zeros(
            self.image_bgr.shape[:2],
            dtype=bool,
        )

        self.mask_editor.set_mask(
            empty_mask,
            base_image_bgr=self.image_bgr,
            reset_zoom=False,
        )

        self.recompute_stats_from_editor()

    def recompute_stats_from_editor(self):
        if self.image_bgr is None:
            self.coverage_label.setText("Cloud coverage: —")
            self.cloud_pixels_label.setText("Cloud pixels: —")
            self.analyzed_pixels_label.setText("Analyzed pixels: —")
            return

        mask = self.mask_editor.get_mask_bool()
        (
            self.cloud_percent,
            self.cloud_pixels,
            self.analyzed_pixels,
        ) = compute_cloud_stats(mask, self.valid_zone_mask)

        self.coverage_label.setText(f"Cloud coverage: {self.cloud_percent:.2f}%")
        self.cloud_pixels_label.setText(f"Cloud pixels: {self.cloud_pixels}")
        self.analyzed_pixels_label.setText(
            f"Analyzed pixels: {self.analyzed_pixels}"
        )

    def set_tool_mode(self, mode: str):
        self.mask_editor.set_mode(mode)

        self.brush_btn.setChecked(mode == "brush")
        self.fill_btn.setChecked(mode == "fill")
        self.magic_btn.setChecked(mode == "magic")

        self.brush_action.setChecked(mode == "brush")
        self.fill_action.setChecked(mode == "fill")
        self.magic_action.setChecked(mode == "magic")

    def on_tool_value_changed(self, value: int):
        self.tool_value_label.setText(f"Threshold / size: {value}")
        self.mask_editor.set_tool_value(value)

    def mask_editor_undo(self):
        self.mask_editor.undo_last()
        self.recompute_stats_from_editor()

    def toggle_overlay(self):
        self.set_overlay_visible(not self.overlay_btn.isChecked())

    def on_overlay_toggled(self, checked: bool):
        self.set_overlay_visible(checked)

    def set_overlay_visible(self, checked: bool):
        self.overlay_btn.setChecked(checked)
        self.overlay_action.setChecked(checked)
        self.mask_editor.set_overlay_enabled(checked)

    def on_enable_cuda_changed(self, checked: bool):
        self.enable_cuda = checked
        self.identifier = None

        
        self.device_label.setText("Device: not loaded")

    def error(self, message: str):
        QMessageBox.critical(self, "Error", message)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cloud mask prediction and editing tool.",
    )

    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Optional cloudy input image.",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Path to the trained model checkpoint. Default: {DEFAULT_MODEL}",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional model threshold override.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.55,
        help="Cloud overlay alpha. Default: 0.55",
    )

    parser.add_argument(
        "--valid-mask",
        default=None,
        help="Optional valid sky mask. White/non-zero pixels are analyzed.",
    )

    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Disable CUDA.",
    )

    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run cloud detection in console mode and print statistics.",
    )

    return parser


def run_no_gui(args: argparse.Namespace) -> int:
    if not args.input:
        raise SystemExit("error: --no-gui requires an input image")

    image_path = Path(args.input)
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise SystemExit(f"error: could not read image: {image_path}")

    valid_zone_mask = None
    if args.valid_mask:
        valid_zone_mask = load_valid_zone_mask_for_image(
            Path(args.valid_mask),
            image_bgr.shape[:2],
        )

    cloud_mask, identifier = predict_cloud_mask(
        image_bgr=image_bgr,
        model_path=args.model,
        threshold=args.threshold,
        alpha=args.alpha,
        enable_cuda=not args.no_cuda,
        valid_zone_mask=valid_zone_mask,
    )

    cloud_percent, cloud_pixels, valid_pixels = compute_cloud_stats(
        cloud_mask,
        valid_zone_mask,
    )

    print(f"Image: {image_path}")
    print(f"Model: {args.model}")
    print(f"Device: {identifier.device}")

    if identifier.device_warning:
        print(f"CUDA warning: {identifier.device_warning}")

    print(f"Model threshold: {identifier.threshold:.3f}")
    print(f"Cloud coverage: {cloud_percent:.2f}%")
    print(f"Cloud pixels: {cloud_pixels}")
    print(f"Valid sky pixels: {valid_pixels}")

    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.no_gui:
        return run_no_gui(args)

    app = QApplication([])

    window = MainWindow(
        model_path=args.model,
        input_path=args.input,
        threshold=args.threshold,
        alpha=args.alpha,
        enable_cuda=not args.no_cuda,
    )

    if args.valid_mask:
        window.load_and_apply_valid_zone_mask(Path(args.valid_mask))

    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
