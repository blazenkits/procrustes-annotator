"""Small, explicit human-review dialogs for the annotation workflow."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QEvent, Qt, QThread, Signal
from PySide6.QtGui import QColor, QMouseEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from .image_view import ImageView


class BackgroundTask(QThread):
    result_ready = Signal(object)
    failed = Signal(str)
    progress = Signal(int, int, int, int)

    def __init__(self, function: Callable[[], Any], parent=None):
        super().__init__(parent)
        self.function = function

    def run(self) -> None:
        try:
            self.result_ready.emit(self.function())
        except Exception as error:  # noqa: BLE001 - report optional GPU failures in the GUI
            self.failed.emit(str(error))


def rgba_mask_overlay(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    rgba = np.zeros((*binary.shape, 4), np.uint8)
    boundary = binary & (cv2.dilate((~binary).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    rgba[boundary] = (0, 70, 255, 240)
    return rgba


class PromptImageView(ImageView):
    negative_clicked = Signal(float, float)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.RightButton and not self._navigation_mode:
            point = self._image_item.mapFromScene(self.mapToScene(event.position().toPoint()))
            if self._image_item.boundingRect().contains(point):
                self.negative_clicked.emit(point.x(), point.y())
                event.accept()
                return
        super().mousePressEvent(event)


class ICPOptionsDialog(QDialog):
    """Choose one refinement configuration before exporting the GPU job."""

    def __init__(self, method: str = "silhouette", weight: float = 0.10, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("ICP batch settings")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("SAM masks are generated for all solved poses. Choose the refinement that uses them."))
        form = QFormLayout()
        self.method = QComboBox()
        self.method.addItem("Bounded silhouette ICP", "silhouette")
        self.method.addItem("SAM-masked color ICP", "colored")
        self.method.setCurrentIndex(max(0, self.method.findData(method)))
        form.addRow("Method", self.method)
        self.weight = QDoubleSpinBox()
        self.weight.setRange(0.0, 2.0)
        self.weight.setDecimals(3)
        self.weight.setSingleStep(0.025)
        self.weight.setValue(weight)
        form.addRow("Silhouette weight", self.weight)
        self.method.currentIndexChanged.connect(
            lambda: self.weight.setEnabled(self.method.currentData() == "silhouette")
        )
        self.weight.setEnabled(self.method.currentData() == "silhouette")
        layout.addLayout(form)
        layout.addWidget(QLabel("Color ICP uses the 4-pixel-eroded SAM region. The weight applies only to silhouette ICP."))
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class BatchReviewDialog(QDialog):
    """One navigable review window; decisions commit only on Confirm."""

    def __init__(
        self, keys: list[tuple[int, int]],
        load_page: Callable[[tuple[int, int]], dict[str, Any]],
        *,
        rejected: dict[tuple[int, int], bool],
        current_key: tuple[int, int] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        if not keys:
            raise ValueError("Review requires at least one frame/object pair.")
        self.setWindowTitle("Review SAM + ICP")
        self.resize(1550, 880)
        self.keys = sorted(keys)
        self.load_page = load_page
        self.rejected = dict(rejected)
        self.prompts: dict[tuple[int, int], dict[str, Any]] = {}
        self._pages: dict[tuple[int, int], dict[str, Any]] = {}
        self.retry_key: tuple[int, int] | None = None
        self.retry_prompts: dict[str, Any] | None = None
        self.confirmed = False
        self._raw_held = False
        self._loading = False

        layout = QVBoxLayout(self)
        navigation = QHBoxLayout()
        navigation.addWidget(QLabel("Frame"))
        self.frame_combo = QComboBox()
        for frame in sorted({frame for frame, _ in self.keys}):
            self.frame_combo.addItem(f"{frame:06d}", frame)
        navigation.addWidget(self.frame_combo)
        navigation.addWidget(QLabel("Object"))
        self.object_combo = QComboBox()
        navigation.addWidget(self.object_combo)
        previous = QPushButton("◀ Previous")
        previous.clicked.connect(lambda: self._step(-1))
        navigation.addWidget(previous)
        next_button = QPushButton("Next ▶")
        next_button.clicked.connect(lambda: self._step(1))
        navigation.addWidget(next_button)
        navigation.addStretch()
        self.position_label = QLabel()
        navigation.addWidget(self.position_label)
        layout.addLayout(navigation)
        self.frame_combo.currentIndexChanged.connect(self._frame_changed)
        self.object_combo.currentIndexChanged.connect(self._load_current)

        self.metric_label = QLabel()
        layout.addWidget(self.metric_label)
        self.error_label = QLabel()
        self.error_label.setStyleSheet("color: #b23535;")
        layout.addWidget(self.error_label)
        images = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(QLabel("SAM boundary (blue) · left click include · right click exclude"))
        self.mask_view = PromptImageView()
        self.mask_view.point_clicked.connect(self._add_positive)
        self.mask_view.negative_clicked.connect(self._add_negative)
        left.addWidget(self.mask_view, stretch=1)
        mask_controls = QHBoxLayout()
        pan = QCheckBox("Drag/pan mask image")
        pan.toggled.connect(self.mask_view.set_navigation_mode)
        mask_controls.addWidget(pan)
        undo = QPushButton("Undo added prompt")
        undo.clicked.connect(self._undo_point)
        mask_controls.addWidget(undo)
        mask_controls.addWidget(QLabel("CAD box padding"))
        self.padding = QDoubleSpinBox()
        self.padding.setRange(0.0, 0.75)
        self.padding.setSingleStep(0.05)
        self.padding.valueChanged.connect(self._padding_changed)
        mask_controls.addWidget(self.padding)
        left.addLayout(mask_controls)
        images.addLayout(left, stretch=1)

        right = QVBoxLayout()
        right.addWidget(QLabel("ICP overlay · hold Tab for raw Procrustes · drag to pan · wheel to zoom"))
        self.pose_view = ImageView()
        self.pose_view.set_navigation_mode(True)
        right.addWidget(self.pose_view, stretch=1)
        images.addLayout(right, stretch=1)
        layout.addLayout(images, stretch=1)

        settings = QHBoxLayout()
        self.reject_box = QCheckBox("Reject ICP — use raw pose")
        self.reject_box.toggled.connect(self._reject_changed)
        settings.addWidget(self.reject_box)
        settings.addStretch()
        settings.addWidget(QLabel("Regenerate with"))
        self.method = QComboBox()
        self.method.addItem("Silhouette ICP", "silhouette")
        self.method.addItem("Color ICP", "colored")
        self.method.currentIndexChanged.connect(self._method_changed)
        settings.addWidget(self.method)
        settings.addWidget(QLabel("weight"))
        self.weight = QDoubleSpinBox()
        self.weight.setRange(0.0, 2.0)
        self.weight.setDecimals(3)
        self.weight.setSingleStep(0.025)
        self.weight.valueChanged.connect(self._weight_changed)
        settings.addWidget(self.weight)
        layout.addLayout(settings)
        self.prompt_note = QLabel("Keep is the default. Changes to prompts or settings require regeneration.")
        layout.addWidget(self.prompt_note)
        buttons = QHBoxLayout()
        regenerate = QPushButton("Regenerate SAM + ICP…")
        regenerate.clicked.connect(self._retry)
        buttons.addWidget(regenerate)
        buttons.addStretch()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        buttons.addWidget(cancel)
        confirm = QPushButton("Confirm all selections")
        confirm.clicked.connect(self._confirm)
        buttons.addWidget(confirm)
        layout.addLayout(buttons)

        target = current_key if current_key in self.keys else self.keys[0]
        self.frame_combo.blockSignals(True)
        self.frame_combo.setCurrentIndex(self.frame_combo.findData(target[0]))
        self.frame_combo.blockSignals(False)
        self._frame_changed()
        self.object_combo.setCurrentIndex(self.object_combo.findData(target[1]))
        QApplication.instance().installEventFilter(self)

    def _current_key(self) -> tuple[int, int]:
        return int(self.frame_combo.currentData()), int(self.object_combo.currentData())

    def _frame_changed(self) -> None:
        frame = self.frame_combo.currentData()
        objects = sorted(obj for candidate, obj in self.keys if candidate == frame)
        self.object_combo.blockSignals(True)
        self.object_combo.clear()
        for obj in objects:
            self.object_combo.addItem(f"obj_{obj:06d}", obj)
        self.object_combo.blockSignals(False)
        self._load_current()

    def _step(self, offset: int) -> None:
        index = self.keys.index(self._current_key())
        target = self.keys[min(max(index + offset, 0), len(self.keys) - 1)]
        if self.frame_combo.currentData() != target[0]:
            self.frame_combo.setCurrentIndex(self.frame_combo.findData(target[0]))
        self.object_combo.setCurrentIndex(self.object_combo.findData(target[1]))

    def _prompt_state(self, key: tuple[int, int], page: dict[str, Any]) -> dict[str, Any]:
        if key not in self.prompts:
            self.prompts[key] = {
                "positive": list(page["positive"]),
                "negative": list(page["negative"]),
                "padding": float(page["box_padding"]),
                "method": page["method"],
                "weight": float(page["weight"]),
                "dirty": False,
                "history": [],
            }
        return self.prompts[key]

    def _load_current(self) -> None:
        if self.object_combo.currentData() is None:
            return
        key = self._current_key()
        if key not in self._pages:
            self._pages[key] = self.load_page(key)
        page = self._pages[key]
        prompt = self._prompt_state(key, page)
        self._loading = True
        self.mask_view.set_image(page["rgb"])
        self.mask_view.set_overlay(rgba_mask_overlay(page["mask"]) if page["mask"] is not None else None)
        self.mask_view.reset_view()
        self.pose_view.set_image(page["rgb"])
        self.pose_view.reset_view()
        self.padding.setValue(prompt["padding"])
        self.method.setCurrentIndex(max(0, self.method.findData(prompt["method"])))
        self.weight.setValue(prompt["weight"])
        self.weight.setEnabled(self.method.currentData() == "silhouette")
        self.reject_box.setChecked(self.rejected.get(key, False) or page["icp_overlay"] is None)
        self.reject_box.setEnabled(page["icp_overlay"] is not None)
        self._loading = False
        self._refresh_points()
        self._refresh_pose()
        self.position_label.setText(f"{self.keys.index(key) + 1} / {len(self.keys)}")
        self.metric_label.setText(
            f"Rendered-depth RMSE: raw {_metric(page['raw_rmse_mm'])} → ICP {_metric(page['icp_rmse_mm'])}  "
            f"· SAM score {_metric_score(page['sam_score'])}  "
            "· sensor agreement, not pose ground truth"
        )
        self.error_label.setText(page.get("error") or "")
        self.prompt_note.setText(
            "Prompts/settings changed; regenerate before using them."
            if prompt["dirty"] else "Keep is the default. Check Reject ICP only for proposals you do not want."
        )

    def _refresh_pose(self) -> None:
        page = self._pages[self._current_key()]
        overlay = page["raw_overlay"] if self._raw_held or page["icp_overlay"] is None else page["icp_overlay"]
        self.pose_view.set_overlay(overlay)

    def _refresh_points(self) -> None:
        page = self._pages[self._current_key()]
        prompt = self.prompts[self._current_key()]
        markers = []
        for points, color in (
            (page["manual_clicks"], QColor("#ffcf33")),
            (prompt["positive"], QColor("#35d65a")),
            (prompt["negative"], QColor("#ff5252")),
        ):
            offset = len(markers)
            markers.extend((offset + i + 1, x, y, color) for i, (x, y) in enumerate(points))
        self.mask_view.set_markers(markers)

    def _mark_dirty(self) -> None:
        if self._loading:
            return
        self.prompts[self._current_key()]["dirty"] = True
        self.prompt_note.setText("Prompts/settings changed. Regenerate SAM + ICP to use them.")

    def _add_positive(self, x: float, y: float) -> None:
        prompt = self.prompts[self._current_key()]
        prompt["positive"].append((x, y))
        prompt["history"].append("positive")
        self._refresh_points()
        self._mark_dirty()

    def _add_negative(self, x: float, y: float) -> None:
        prompt = self.prompts[self._current_key()]
        prompt["negative"].append((x, y))
        prompt["history"].append("negative")
        self._refresh_points()
        self._mark_dirty()

    def _undo_point(self) -> None:
        prompt = self.prompts[self._current_key()]
        if not prompt["history"]:
            return
        (prompt["negative"] if prompt["history"].pop() == "negative" else prompt["positive"]).pop()
        self._refresh_points()
        self._mark_dirty()

    def _padding_changed(self, value: float) -> None:
        if not self._loading:
            self.prompts[self._current_key()]["padding"] = value
            self._mark_dirty()

    def _method_changed(self) -> None:
        self.weight.setEnabled(self.method.currentData() == "silhouette")
        if not self._loading:
            self.prompts[self._current_key()]["method"] = self.method.currentData()
            self._mark_dirty()

    def _weight_changed(self, value: float) -> None:
        if not self._loading:
            self.prompts[self._current_key()]["weight"] = value
            self._mark_dirty()

    def _reject_changed(self, checked: bool) -> None:
        if not self._loading:
            self.rejected[self._current_key()] = checked

    def _retry(self) -> None:
        self.retry_key = self._current_key()
        self.retry_prompts = dict(self.prompts[self.retry_key])
        self.accept()

    def _confirm(self) -> None:
        changed = sum(bool(prompt["dirty"]) for prompt in self.prompts.values())
        if changed:
            answer = QMessageBox.question(
                self, "Unprocessed prompt changes",
                f"{changed} pairs have prompt/settings changes not yet regenerated. "
                "Discard those changes and confirm the current ICP selections?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.confirmed = True
        self.accept()

    def eventFilter(self, watched, event) -> bool:
        if (QApplication.activeModalWidget() is self
                and event.type() in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease)
                and event.key() == Qt.Key.Key_Tab):
            self._raw_held = event.type() == QEvent.Type.KeyPress
            self._refresh_pose()
            return True
        return super().eventFilter(watched, event)

    def done(self, result: int) -> None:
        QApplication.instance().removeEventFilter(self)
        super().done(result)


def _metric_score(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.3f}"


def _metric(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.2f} mm"


class SetupOverviewDialog(QDialog):
    def __init__(self, rows: list[tuple[int, int, str]], setup_id: str, report: dict | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Setup overview — {setup_id}")
        self.resize(640, 700)
        self.open_key: tuple[int, int] | None = None
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Every solved Procrustes pose is a setup target. Double-click a row to open it."))
        if report:
            layout.addWidget(QLabel(
                f"Last setup fit: {report['accepted_target_count']} accepted targets, "
                f"{report['inferred_count']} inferred poses; maximum discrepancy "
                f"{report['max_translation_residual_mm']:.1f} mm / "
                f"{report['max_rotation_residual_deg']:.1f}°."
            ))
        self.table = QTableWidget(len(rows), 3)
        self.table.setHorizontalHeaderLabels(["Frame", "Object", "Progress"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.keys = []
        for index, (frame, obj, status) in enumerate(rows):
            self.keys.append((frame, obj))
            self.table.setItem(index, 0, QTableWidgetItem(f"{frame:06d}"))
            self.table.setItem(index, 1, QTableWidgetItem(f"obj_{obj:06d}"))
            self.table.setItem(index, 2, QTableWidgetItem(status))
        self.table.cellDoubleClicked.connect(self._open_row)
        layout.addWidget(self.table)
        buttons = QHBoxLayout()
        open_button = QPushButton("Open selected pair")
        open_button.clicked.connect(self._open_selected)
        buttons.addWidget(open_button)
        done_button = QPushButton("Close")
        done_button.clicked.connect(self.accept)
        buttons.addWidget(done_button)
        layout.addLayout(buttons)

    def _open_row(self, row: int, _column: int) -> None:
        self.open_key = self.keys[row]
        self.accept()

    def _open_selected(self) -> None:
        row = self.table.currentRow()
        if row >= 0:
            self._open_row(row, 0)
