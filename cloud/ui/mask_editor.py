from __future__ import annotations

import cv2
import numpy as np

from PyQt6.QtCore import Qt, QPoint, QRect, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QWheelEvent, QPainter
from PyQt6.QtWidgets import QWidget


def bgr_to_qpixmap(image_bgr: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimage = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimage)


def mask_to_qpixmap(mask: np.ndarray) -> QPixmap:
    gray = mask.astype(np.uint8) * 255
    h, w = gray.shape
    qimage = QImage(gray.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    return QPixmap.fromImage(qimage)


class MaskEditor(QWidget):
    mask_changed = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.mask: np.ndarray | None = None
        self.base_image_bgr: np.ndarray | None = None

        self.mode = "brush"
        self.brush_size = 25
        self.magic_tolerance = 2
        self.paint_value = 255

        self.overlay_enabled = True
        self.overlay_alpha = 0.55

        self._base_pixmap: QPixmap | None = None
        self._zoom = 1.0
        self._display_rect: QRect | None = None

        self._history: list[np.ndarray] = []
        self._max_history = 20
        self._fill_points: list[tuple[int, int]] = []

        self.setMinimumSize(800, 500)
        self.setMouseTracking(True)

    def clear_mask(self):
        self.mask = None
        self.base_image_bgr = None
        self._base_pixmap = None
        self._display_rect = None
        self._history.clear()
        self._fill_points.clear()
        self._zoom = 1.0
        self.update()

    def set_mode(self, mode: str):
        if mode not in {"brush", "magic", "fill"}:
            raise ValueError(f"Unknown edit mode: {mode}")
        self.mode = mode

    def set_tool_value(self, value: int):
        self.brush_size = max(1, int(value))

        # Magic wand tolerance must be much finer than brush size.
        self.magic_tolerance = max(0, min(25, int(value / 12)))

    def set_overlay_enabled(self, enabled: bool):
        self.overlay_enabled = enabled
        self._rebuild_pixmap()

    def set_overlay_alpha(self, alpha: float):
        self.overlay_alpha = 0.55 if alpha is None else float(alpha)
        self._rebuild_pixmap()

    def set_mask(
        self,
        mask: np.ndarray,
        base_image_bgr: np.ndarray,
        reset_zoom: bool = True,
    ):
        self.mask = mask.astype(np.uint8) * 255
        self.base_image_bgr = base_image_bgr
        self._history.clear()
        self._fill_points.clear()

        if reset_zoom:
            self._zoom = 1.0

        self._rebuild_pixmap()
        self.mask_changed.emit()

    def get_mask_bool(self) -> np.ndarray:
        if self.mask is None:
            raise ValueError("No mask loaded")
        return self.mask > 0

    def undo_last(self):
        if not self._history:
            return

        self.mask = self._history.pop()
        self._fill_points.clear()
        self._rebuild_pixmap()
        self.mask_changed.emit()

    def _push_history(self):
        if self.mask is None:
            return

        self._history.append(self.mask.copy())

        if len(self._history) > self._max_history:
            self._history.pop(0)

    def wheelEvent(self, event: QWheelEvent):
        if self._base_pixmap is None:
            return

        self._zoom *= 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self._zoom = max(0.1, min(self._zoom, 20.0))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)

        if self._base_pixmap is None:
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "No image loaded",
            )
            self._display_rect = None
            return

        target = self._compute_display_rect()
        self._display_rect = target

        scaled = self._base_pixmap.scaled(
            target.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        painter.drawPixmap(target, scaled)

    def mousePressEvent(self, event):
        if self.mask is None:
            return

        if event.button() == Qt.MouseButton.LeftButton:
            self.paint_value = 255
        elif event.button() == Qt.MouseButton.RightButton:
            self.paint_value = 0
        else:
            return

        if self.mode == "brush":
            self._push_history()
            self._paint_at(event.pos(), emit=False)
            self.mask_changed.emit()

        elif self.mode == "magic":
            self._push_history()
            self._magic_select_at(event.pos())
            self.mask_changed.emit()

        elif self.mode == "fill":
            self._push_history()
            self._fill_points.clear()
            self._add_fill_point(event.pos())

    def mouseMoveEvent(self, event):
        if self.mask is None:
            return

        if not (
            event.buttons() & Qt.MouseButton.LeftButton
            or event.buttons() & Qt.MouseButton.RightButton
        ):
            return

        if self.mode == "brush":
            self._paint_at(event.pos(), emit=True)

        elif self.mode == "fill":
            self._add_fill_point(event.pos())
            self._rebuild_pixmap(show_fill_preview=True)

    def mouseReleaseEvent(self, event):
        if self.mask is None or self.mode != "fill":
            return

        if len(self._fill_points) >= 3:
            pts = np.array(self._fill_points, dtype=np.int32)
            cv2.fillPoly(self.mask, [pts], self.paint_value)

        self._fill_points.clear()
        self._rebuild_pixmap()
        self.mask_changed.emit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update()

    def _compute_display_rect(self) -> QRect:
        if self._base_pixmap is None:
            return QRect()

        pix_w = self._base_pixmap.width()
        pix_h = self._base_pixmap.height()

        if pix_w <= 0 or pix_h <= 0:
            return QRect()

        widget_w = self.width()
        widget_h = self.height()

        scale = min(
            widget_w / pix_w,
            widget_h / pix_h,
        )

        scale *= self._zoom

        display_w = max(1, int(pix_w * scale))
        display_h = max(1, int(pix_h * scale))

        x = (widget_w - display_w) // 2
        y = (widget_h - display_h) // 2

        return QRect(x, y, display_w, display_h)

    def _paint_at(self, pos: QPoint, emit: bool):
        x, y = self._widget_to_image_coords(pos)

        if x is None or y is None:
            return

        cv2.circle(
            self.mask,
            (x, y),
            self.brush_size,
            self.paint_value,
            thickness=-1,
        )

        self._rebuild_pixmap()

        if emit:
            self.mask_changed.emit()

    def _magic_select_at(self, pos: QPoint):
        if self.mask is None or self.base_image_bgr is None:
            return

        x, y = self._widget_to_image_coords(pos)

        if x is None or y is None:
            return

        if self.paint_value == 0:
            selected = self._flood_from_point(x, y)
            self.mask[selected] = 0

        else:
            clicked_region = self._flood_from_point(x, y)
            current_cloud = self.mask > 0

            if np.any(current_cloud):
                grown_region = self._grow_from_existing_mask(
                    clicked_region=clicked_region,
                    current_cloud=current_cloud,
                )
                selected = clicked_region | grown_region
            else:
                selected = clicked_region

            self.mask[selected] = 255

        self._rebuild_pixmap()

    def _flood_from_point(self, x: int, y: int) -> np.ndarray:
        h, w = self.base_image_bgr.shape[:2]

        flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        work = self.base_image_bgr.copy()

        tol = self.magic_tolerance

        flags = (
            4
            | cv2.FLOODFILL_MASK_ONLY
            | (255 << 8)
        )

        cv2.floodFill(
            work,
            flood_mask,
            (x, y),
            (0, 0, 0),
            (tol, tol, tol),
            (tol, tol, tol),
            flags,
        )

        return flood_mask[1:-1, 1:-1] > 0

    def _grow_from_existing_mask(
        self,
        clicked_region: np.ndarray,
        current_cloud: np.ndarray,
    ) -> np.ndarray:
        image = self.base_image_bgr

        if image is None:
            return np.zeros(current_cloud.shape, dtype=bool)

        if not np.any(clicked_region):
            return np.zeros(current_cloud.shape, dtype=bool)

        ref_color = image[clicked_region].mean(axis=0)

        kernel_size = max(3, self.brush_size // 2)

        kernel = np.ones(
            (kernel_size, kernel_size),
            dtype=np.uint8,
        )

        dilated = cv2.dilate(
            current_cloud.astype(np.uint8),
            kernel,
            iterations=1,
        ) > 0

        border = dilated & ~current_cloud

        diff = np.linalg.norm(
            image.astype(np.float32) - ref_color.astype(np.float32),
            axis=2,
        )

        color_limit = max(4.0, self.magic_tolerance * 2.0)
        color_close = diff <= color_limit

        return border & color_close

    def _add_fill_point(self, pos: QPoint):
        x, y = self._widget_to_image_coords(pos)

        if x is None or y is None:
            return

        self._fill_points.append((x, y))

    def _widget_to_image_coords(self, pos: QPoint):
        if self.mask is None or self._base_pixmap is None:
            return None, None

        rect = self._display_rect or self._compute_display_rect()

        if rect.isNull():
            return None, None

        if not rect.contains(pos):
            return None, None

        img_h, img_w = self.mask.shape

        px = pos.x() - rect.x()
        py = pos.y() - rect.y()

        x = int(px * img_w / rect.width())
        y = int(py * img_h / rect.height())

        x = max(0, min(img_w - 1, x))
        y = max(0, min(img_h - 1, y))

        return x, y

    def _rebuild_pixmap(self, show_fill_preview: bool = False):
        if self.mask is None:
            return

        if self.base_image_bgr is not None:
            image = self.base_image_bgr.copy()

            if self.overlay_enabled:
                cloud = self.mask > 0

                if np.any(cloud):
                    red = np.zeros_like(image)
                    red[:, :] = (0, 0, 255)

                    alpha = 0.55 if self.overlay_alpha is None else float(self.overlay_alpha)
                    alpha = max(0.0, min(1.0, alpha))

                    image[cloud] = cv2.addWeighted(
                        image[cloud],
                        1.0 - alpha,
                        red[cloud],
                        alpha,
                        0,
                    )

            if show_fill_preview and len(self._fill_points) >= 2:
                pts = np.array(self._fill_points, dtype=np.int32)
                cv2.polylines(
                    image,
                    [pts],
                    isClosed=False,
                    color=(0, 255, 255),
                    thickness=2,
                )

            self._base_pixmap = bgr_to_qpixmap(image)

        else:
            preview_mask = self.mask.copy()

            if show_fill_preview and len(self._fill_points) >= 2:
                pts = np.array(self._fill_points, dtype=np.int32)
                cv2.polylines(
                    preview_mask,
                    [pts],
                    isClosed=False,
                    color=128,
                    thickness=2,
                )

            self._base_pixmap = mask_to_qpixmap(preview_mask > 0)

        self.update()