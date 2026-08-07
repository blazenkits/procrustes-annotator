"""Zoomable RGB image widget with numbered correspondence markers."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QImage,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QTransform,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QLabel,
)


class ImageView(QGraphicsView):
    """Display an RGB array and emit coordinates for clicks on the image."""

    point_clicked = Signal(float, float)
    delete_last_requested = Signal()
    clear_all_requested = Signal()
    view_changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._image_item = QGraphicsPixmapItem()
        self._depth_validity_item = QGraphicsPixmapItem()
        self._overlay_item = QGraphicsPixmapItem()
        self._scene.addItem(self._image_item)
        self._scene.addItem(self._depth_validity_item)
        self._scene.addItem(self._overlay_item)
        self._depth_validity_item.setZValue(0.5)
        self._overlay_item.setZValue(1)
        self._color_pixmap = QPixmap()
        self._greyscale_pixmap = QPixmap()
        self._marker_items: list[QGraphicsEllipseItem | QGraphicsSimpleTextItem] = []
        self._has_image = False
        self._user_zoomed = False
        self._greyscale = False
        self._navigation_mode = False
        self._magnifier_enabled = False
        self._magnification = 4.0
        self._last_mouse_position: QPoint | None = None
        self._magnifier = QLabel(self.viewport())
        self._magnifier.setFixedSize(190, 190)
        self._magnifier.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._magnifier.setStyleSheet(
            "background: #111; border: 2px solid white; border-radius: 4px;"
        )
        self._magnifier.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._magnifier.hide()
        self.setBackgroundBrush(QColor("#171a21"))
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def set_image(self, rgb: np.ndarray) -> None:
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("ImageView expects an HxWx3 uint8 RGB array.")
        height, width, _ = rgb.shape
        image = QImage(
            rgb.data,
            width,
            height,
            int(rgb.strides[0]),
            QImage.Format.Format_RGB888,
        ).copy()
        first_image = not self._has_image
        self._color_pixmap = QPixmap.fromImage(image)
        self._greyscale_pixmap = QPixmap.fromImage(image.convertToFormat(QImage.Format.Format_Grayscale8))
        self._image_item.setPixmap(
            self._greyscale_pixmap if self._greyscale else self._color_pixmap
        )
        self._depth_validity_item.setPixmap(QPixmap())
        self._overlay_item.setPixmap(QPixmap())
        self._scene.setSceneRect(0, 0, width, height)
        self._has_image = True
        self.clear_markers()
        if first_image:
            self.reset_view()
        self._update_magnifier()

    def set_greyscale(self, enabled: bool) -> None:
        self._greyscale = enabled
        if self._has_image:
            self._image_item.setPixmap(
                self._greyscale_pixmap if enabled else self._color_pixmap
            )
        self._update_magnifier()

    def set_magnifier_enabled(self, enabled: bool) -> None:
        self._magnifier_enabled = enabled
        if enabled and self._last_mouse_position is not None:
            self._update_magnifier()
        else:
            self._magnifier.hide()

    def set_navigation_mode(self, enabled: bool) -> None:
        self._navigation_mode = enabled
        self.setCursor(
            Qt.CursorShape.OpenHandCursor if enabled else Qt.CursorShape.CrossCursor
        )

    def set_overlay(self, rgba: np.ndarray | None) -> None:
        """Set or clear an HxWx4 uint8 overlay aligned to the RGB image."""
        if rgba is None:
            self._overlay_item.setPixmap(QPixmap())
            self._update_magnifier()
            return
        if rgba.ndim != 3 or rgba.shape[2] != 4 or rgba.dtype != np.uint8:
            raise ValueError("ImageView expects an HxWx4 uint8 RGBA overlay.")
        height, width, _ = rgba.shape
        image = QImage(
            rgba.data,
            width,
            height,
            int(rgba.strides[0]),
            QImage.Format.Format_RGBA8888,
        ).copy()
        self._overlay_item.setPixmap(QPixmap.fromImage(image))
        self._update_magnifier()

    def set_invalid_depth_mask(self, invalid: np.ndarray | None) -> None:
        """Tint invalid depth pixels light red without changing valid pixels."""
        if invalid is None:
            self._depth_validity_item.setPixmap(QPixmap())
            self._update_magnifier()
            return
        if invalid.ndim != 2 or invalid.dtype != np.bool_:
            raise ValueError("ImageView expects an HxW boolean invalid-depth mask.")
        height, width = invalid.shape
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[invalid] = (255, 72, 72, 76)
        image = QImage(
            rgba.data,
            width,
            height,
            int(rgba.strides[0]),
            QImage.Format.Format_RGBA8888,
        ).copy()
        self._depth_validity_item.setPixmap(QPixmap.fromImage(image))
        self._update_magnifier()

    def set_markers(self, markers: list[tuple[int, float, float, QColor]]) -> None:
        self.clear_markers()
        radius = 8.0
        for index, x, y, color in markers:
            circle = self._scene.addEllipse(
                x - radius,
                y - radius,
                radius * 2,
                radius * 2,
                QPen(color, 3),
            )
            # Marker geometry is expressed in image pixels. It therefore
            # grows/shrinks with image zoom instead of staying viewport-fixed.
            circle.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations,
                False,
            )
            circle.setZValue(10)
            label = self._scene.addSimpleText(str(index))
            label.setBrush(color)
            label.setPos(x + radius, y - radius * 2)
            label.setScale(1.5)
            label.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations,
                False,
            )
            label.setZValue(11)
            self._marker_items.extend((circle, label))
        self._update_magnifier()

    def clear_markers(self) -> None:
        for item in self._marker_items:
            self._scene.removeItem(item)
        self._marker_items.clear()
        self._update_magnifier()

    def reset_view(self) -> None:
        if self._has_image:
            self._user_zoomed = False
            self.fitInView(self._image_item, Qt.AspectRatioMode.KeepAspectRatio)

    def view_state(self) -> dict[str, object]:
        """Return the zoom, pan, and magnifier settings for project storage."""
        transform = self.transform()
        return {
            "transform": [
                transform.m11(),
                transform.m12(),
                transform.m13(),
                transform.m21(),
                transform.m22(),
                transform.m23(),
                transform.m31(),
                transform.m32(),
                transform.m33(),
            ],
            "horizontal_scroll": self.horizontalScrollBar().value(),
            "vertical_scroll": self.verticalScrollBar().value(),
            "user_zoomed": self._user_zoomed,
            "magnification": self._magnification,
        }

    def restore_view_state(self, state: dict[str, object]) -> None:
        """Restore a state returned by :meth:`view_state`."""
        saved_user_zoomed = bool(state.get("user_zoomed", False))
        # Changing the transform can show/hide scrollbars, which queues a
        # viewport resize. Freeze auto-fit until that layout pass completes,
        # otherwise the queued resize immediately overwrites the saved zoom.
        self._user_zoomed = True
        values = state.get("transform")
        if isinstance(values, list) and len(values) == 9:
            self.setTransform(QTransform(*(float(value) for value in values)))
        self._magnification = min(
            12.0, max(2.0, float(state.get("magnification", 4.0)))
        )
        self.horizontalScrollBar().setValue(int(state.get("horizontal_scroll", 0)))
        self.verticalScrollBar().setValue(int(state.get("vertical_scroll", 0)))
        QTimer.singleShot(
            0, lambda: setattr(self, "_user_zoomed", saved_user_zoomed)
        )

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        super().resizeEvent(event)
        if self._has_image and not self._user_zoomed:
            self.reset_view()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API name
        if self._navigation_mode:
            super().mousePressEvent(event)
            return
        if event.button() == Qt.MouseButton.RightButton:
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self.clear_all_requested.emit()
            else:
                self.delete_last_requested.emit()
            event.accept()
            return
        if self._has_image and event.button() == Qt.MouseButton.LeftButton:
            point: QPointF = self._image_item.mapFromScene(self.mapToScene(event.position().toPoint()))
            bounds = self._image_item.boundingRect()
            if bounds.contains(point):
                self.point_clicked.emit(point.x(), point.y())
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API name
        super().mouseReleaseEvent(event)
        if self._navigation_mode:
            self.view_changed.emit()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API name
        self._last_mouse_position = event.position().toPoint()
        self._update_magnifier()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt API name
        self._magnifier.hide()
        super().leaveEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt API name
        if not self._has_image:
            return
        if self._magnifier_enabled:
            step = 0.5 if event.angleDelta().y() > 0 else -0.5
            self._magnification = min(12.0, max(2.0, self._magnification + step))
            self._update_magnifier()
            self.view_changed.emit()
            event.accept()
            return
        self._user_zoomed = True
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        self.scale(factor, factor)
        self.view_changed.emit()

    def _update_magnifier(self) -> None:
        if not self._magnifier_enabled or not self._has_image or self._last_mouse_position is None:
            self._magnifier.hide()
            return
        scene_point = self.mapToScene(self._last_mouse_position)
        image_point = self._image_item.mapFromScene(scene_point)
        if not self._image_item.boundingRect().contains(image_point):
            self._magnifier.hide()
            return

        output_size = 180
        source_size = max(12, int(round(output_size / self._magnification)))
        x = image_point.x() - source_size / 2
        y = image_point.y() - source_size / 2
        crop = QPixmap(output_size, output_size)
        crop.fill(QColor("#111"))
        painter = QPainter(crop)
        # Render the composed scene—not just the RGB pixmap—so the magnifier
        # includes invalid-depth tint, solved overlay, circles, and numbers.
        self._scene.render(
            painter,
            QRectF(0, 0, output_size, output_size),
            QRectF(x, y, source_size, source_size),
            Qt.AspectRatioMode.IgnoreAspectRatio,
        )
        painter.setPen(QPen(QColor("#ff4050"), 1))
        center = output_size // 2
        painter.drawLine(center - 12, center, center + 12, center)
        painter.drawLine(center, center - 12, center, center + 12)
        painter.end()
        self._magnifier.setPixmap(crop)

        margin = 12
        position = self._last_mouse_position - QPoint(
            self._magnifier.width() // 2, self._magnifier.height() // 2
        )
        max_x = max(margin, self.viewport().width() - self._magnifier.width() - margin)
        max_y = max(margin, self.viewport().height() - self._magnifier.height() - margin)
        self._magnifier.move(
            min(max_x, max(margin, position.x())),
            min(max_y, max(margin, position.y())),
        )
        self._magnifier.show()
