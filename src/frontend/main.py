"""PySide6/PyVista desktop application for RGBD mesh annotation."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pyvista as pv
from PySide6.QtCore import (
    QByteArray,
    QElapsedTimer,
    QEvent,
    QKeyCombination,
    QLibraryInfo,
    QSettings,
    Qt,
    QTimer,
)
from PySide6.QtGui import QAction, QCloseEvent, QColor, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QKeySequenceEdit,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from src.backend.annotate import (
    Vector2,
    Vector3,
    get_pixel_point,
    solve_procrustes,
    unproject,
)
from src.backend.batch_pipeline import (
    RAW_FORMAT,
    RESULT_FORMAT,
    decode_mask,
    make_job_archive,
    record_hash,
    run_batch,
    write_json,
)
from src.backend.loader import DataSet, Pose
from src.backend.review_pipeline import SAM_MODEL_ID
from src.backend.setup_solver import RigidPose, solve_setup
from src.frontend.image_view import ImageView
from src.frontend.review_dialogs import (
    BackgroundTask,
    BatchReviewDialog,
    ICPOptionsDialog,
    SetupOverviewDialog,
)


class AnnotationInteractor(QtInteractor):
    """Qt/VTK interactor that reserves Shift for the annotation UI.

    Qt still delivers Shift key events to the application-level event filter,
    but VTK never sees Shift as a mouse/key modifier.  This prevents its native
    Shift+drag pan behavior from conflicting with temporary Navigation mode.
    """

    def _GetCtrlShift(self, event):  # noqa: N802 - inherited VTK/Qt API name
        modifiers = (
            event.modifiers()
            if hasattr(event, "modifiers")
            else QApplication.keyboardModifiers()
        )
        control = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
        return control, False


COLORS = (
    "#ff5f56",
    "#27c93f",
    "#1e90ff",
    "#ffbd2e",
    "#c678dd",
    "#00d4c7",
    "#ff7f50",
    "#e6e6e6",
)

DEFAULT_SHORTCUTS = {
    "rotate_up": "W",
    "rotate_left": "A",
    "rotate_down": "S",
    "rotate_right": "D",
    "spin_counterclockwise": "Q",
    "spin_clockwise": "E",
    "toggle_overlay": "Tab",
    "previous_frame": "X",
    "next_frame": "C",
    "solve": "Return",
    "defer_pose": "R",
    "hold_navigation": "Shift",
    "hold_magnifier": "Z",
}

RAW_HOLD_ACTIONS = {
    "rotate_up",
    "rotate_left",
    "rotate_down",
    "rotate_right",
    "spin_counterclockwise",
    "spin_clockwise",
    "toggle_overlay",
    "hold_navigation",
    "hold_magnifier",
}

PROJECT_FORMAT = "procrustes-annotator-project"
PROJECT_VERSION = 6


@dataclass(frozen=True)
class ImageSelection:
    pixel: Vector2
    camera: Vector3


@dataclass(frozen=True)
class PoseSolution:
    rotation: np.ndarray
    translation: Vector3
    rmse: float
    residuals_mm: np.ndarray


@dataclass
class MaskReview:
    mask: np.ndarray
    score: float
    box_xyxy: list[float]
    positive_xy: list[tuple[float, float]] = field(default_factory=list)
    negative_xy: list[tuple[float, float]] = field(default_factory=list)
    box_padding: float = 0.25
    model_id: str = SAM_MODEL_ID


@dataclass
class ObjectAnnotationState:
    """Independent pose data plus the object's most recently used landmarks.

    Within a frame, indices in ``frame_model_points`` and ``frame_points``
    define correspondence identity. ``last_model_points`` is only a suggestion
    for untouched frames and is copied before that frame is edited/annotated.
    """

    last_model_points: list[Vector3] = field(default_factory=list)
    frame_model_points: dict[int, list[Vector3]] = field(default_factory=dict)
    frame_points: dict[int, list[ImageSelection]] = field(default_factory=dict)
    solutions: dict[int, PoseSolution] = field(default_factory=dict)
    inferred: dict[int, PoseSolution] = field(default_factory=dict)
    masks: dict[int, MaskReview] = field(default_factory=dict)
    icp_candidates: dict[int, PoseSolution] = field(default_factory=dict)
    icp_info: dict[int, dict[str, object]] = field(default_factory=dict)
    icp_decisions: dict[int, str] = field(default_factory=dict)
    batch_metrics: dict[int, dict[str, float | None]] = field(default_factory=dict)
    batch_errors: dict[int, str] = field(default_factory=dict)
    review_prompts: dict[int, dict[str, object]] = field(default_factory=dict)
    deferred_frames: set[int] = field(default_factory=set)


class ShortcutDialog(QDialog):
    """Edit application shortcuts and return portable key-sequence strings."""

    def __init__(self, actions: dict[str, QAction], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Configure shortcuts")
        self.setMinimumWidth(440)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.editors: dict[str, QKeySequenceEdit] = {}
        for name, action in actions.items():
            editor = QKeySequenceEdit(_action_shortcut(action))
            editor.setMaximumSequenceLength(1)
            self.editors[name] = editor
            label = str(action.property("base_label") or action.text()).replace("&", "")
            form.addRow(label, editor)
        layout.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def shortcuts(self) -> dict[str, str]:
        return {
            name: editor.keySequence().toString(QKeySequence.SequenceFormat.PortableText)
            for name, editor in self.editors.items()
        }


class ShortcutHelpDialog(QDialog):
    """Bilingual mouse and keyboard reference."""

    def __init__(self, actions: dict[str, QAction], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Annotation help / 주석 도움말")
        self.resize(680, 560)
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        shortcuts = {
            name: _action_shortcut(action).toString(QKeySequence.SequenceFormat.NativeText)
            for name, action in actions.items()
        }
        english = QTextBrowser()
        english.setHtml(_help_html(shortcuts, korean=False))
        korean = QTextBrowser()
        korean.setHtml(_help_html(shortcuts, korean=True))
        tabs.addTab(english, "English")
        tabs.addTab(korean, "한국어")
        layout.addWidget(tabs)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class AnnotationWindow(QMainWindow):
    """Collect reusable model landmarks and per-frame RGBD observations."""

    def __init__(
        self,
        dataset_path: str | Path,
        *,
        project_path: str | Path | None = None,
        project_data: dict[str, object] | None = None,
    ) -> None:
        super().__init__()
        self.resize(1600, 950)

        with warnings.catch_warnings(record=True) as caught:
            self.dataset = DataSet.load(dataset_path, depth_preset=2)
        self._startup_warnings = [str(item.message) for item in caught]
        if not self.dataset.models:
            raise ValueError(f"No obj_*.ply models found below {self.dataset.root / 'models'}")
        if not len(self.dataset):
            raise ValueError(f"No complete RGBD frames found below {self.dataset.root}")

        self.current_pose: Pose | None = None
        self.current_object_id: int | None = None
        self.annotation_states = {
            object_id: ObjectAnnotationState() for object_id in self.dataset.models
        }
        self.frame_setups = {pose.frame_id: "setup_0" for pose in self.dataset.poses["0"]}
        self.setup_reports: dict[str, dict[str, object]] = {}
        self._icp_method = "silhouette"
        self._silhouette_weight = 0.10
        self._review_worker: BackgroundTask | None = None
        self._mesh: pv.PolyData | None = None
        self._mesh_texture: pv.Texture | None = None
        self._overlay_key_held = False
        self._overlay_held_key: int | None = None
        self._active_rotation_keys: dict[int, str] = {}
        self._active_view = "model"
        self._temporary_navigation_held = False
        self._navigation_held_key: int | None = None
        self._magnifier_held_key: int | None = None
        self._persistent_navigation = False
        self._show_invalid_depth = True
        self._project_path = Path(project_path).resolve() if project_path else None
        self._dirty = False
        self._restoring_project = False
        self._pending_image_view_state: dict[str, object] | None = None
        self._pending_splitter_sizes: list[int] | None = None
        self._settings = QSettings("procrustes-annotator", "annotator")
        self._migrate_shortcut_defaults()
        self.actions: dict[str, QAction] = {}

        self._rotation_clock = QElapsedTimer()
        self._rotation_clock.start()
        self._rotation_timer = QTimer(self)
        self._rotation_timer.setInterval(16)
        self._rotation_timer.timeout.connect(self._rotation_tick)

        self._build_ui()
        self._create_actions_and_menus()
        self._populate_controls()
        self._load_current_selection()
        if project_data is not None:
            self._restore_project(project_data)
        self._set_dirty(False)
        QApplication.instance().installEventFilter(self)
        if self._startup_warnings:
            self.statusBar().showMessage("  ".join(self._startup_warnings), 15000)

    def _update_window_title(self) -> None:
        project_name = self._project_path.name if self._project_path else "Untitled"
        marker = "*" if self._dirty else ""
        self.setWindowTitle(f"{project_name}{marker} — Procrustes Annotator")

    def _set_dirty(self, dirty: bool = True) -> None:
        if self._dirty == dirty and self.windowTitle():
            return
        self._dirty = dirty
        self._update_window_title()

    @property
    def state(self) -> ObjectAnnotationState:
        if self.current_object_id is None:
            raise RuntimeError("No object is selected.")
        return self.annotation_states[self.current_object_id]

    @property
    def frame_observations(self) -> list[ImageSelection]:
        if self.current_pose is None:
            raise RuntimeError("No frame is selected.")
        return self.state.frame_points.get(self.current_pose.frame_id, [])

    @property
    def model_points(self) -> list[Vector3]:
        if self.current_pose is None:
            raise RuntimeError("No frame is selected.")
        return self.state.frame_model_points.get(
            self.current_pose.frame_id, self.state.last_model_points
        )

    @property
    def model_points_are_inherited(self) -> bool:
        return bool(
            self.current_pose is not None
            and self.current_pose.frame_id not in self.state.frame_model_points
            and self.state.last_model_points
        )

    def _materialize_current_model_points(self) -> list[Vector3]:
        if self.current_pose is None:
            raise RuntimeError("No frame is selected.")
        frame_id = self.current_pose.frame_id
        if frame_id not in self.state.frame_model_points:
            self.state.frame_model_points[frame_id] = list(
                self.state.last_model_points
            )
        return self.state.frame_model_points[frame_id]

    def _remember_current_model_points(self) -> None:
        self.state.last_model_points = list(self.model_points)

    def _migrate_shortcut_defaults(self) -> None:
        migrations = {
            "hold_navigation": ("Space", "Shift"),
            "previous_frame": ("Q", "X"),
            "next_frame": ("E", "C"),
        }
        for name, (old, new) in migrations.items():
            key = f"shortcuts/{name}"
            stored = self._settings.value(key)
            if stored is None or str(stored) == old:
                self._settings.setValue(key, new)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        controls = QHBoxLayout()
        self.frame_combo = QComboBox()
        self.object_combo = QComboBox()
        self.frame_combo.currentIndexChanged.connect(self._frame_selection_changed)
        self.object_combo.currentIndexChanged.connect(self._object_selection_changed)
        controls.addWidget(QLabel("Frame"))
        controls.addWidget(self.frame_combo)
        controls.addSpacing(16)
        controls.addWidget(QLabel("Object"))
        controls.addWidget(self.object_combo)
        controls.addSpacing(16)
        self.setup_edit = QLineEdit("setup_0")
        self.setup_edit.setMaximumWidth(120)
        self.setup_edit.setToolTip("Frames with the same setup ID share one rigid object arrangement.")
        controls.addWidget(QLabel("Setup"))
        controls.addWidget(self.setup_edit)
        assign_setup_button = QPushButton("Assign")
        assign_setup_button.setToolTip("Assign this frame to the typed setup ID.")
        assign_setup_button.clicked.connect(self._assign_current_setup)
        controls.addWidget(assign_setup_button)
        overview_button = QPushButton("Setup overview…")
        overview_button.clicked.connect(self._show_setup_overview)
        controls.addWidget(overview_button)
        controls.addSpacing(16)
        self.mode_button = QToolButton()
        self.mode_button.setCheckable(True)
        self.mode_button.setMinimumWidth(130)
        self.mode_button.toggled.connect(self._persistent_mode_changed)
        controls.addWidget(self.mode_button)
        controls.addStretch()

        self.instruction = QLabel()
        self.instruction.setStyleSheet("font-weight: 600; padding: 6px;")
        controls.addWidget(self.instruction)
        root.addLayout(controls)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.mesh_view = AnnotationInteractor(self.splitter)
        self.mesh_view.set_background("#171a21")
        self.image_view = ImageView()
        self.image_view.point_clicked.connect(self._image_point_picked)
        self.image_view.delete_last_requested.connect(self._delete_last_image_point)
        self.image_view.clear_all_requested.connect(self._clear_frame_points)
        self.image_view.view_changed.connect(self._view_state_changed)
        self.splitter.addWidget(self.mesh_view.interactor)
        self.splitter.addWidget(self.image_view)
        self.splitter.setSizes([800, 800])
        self.splitter.splitterMoved.connect(lambda *_args: self._view_state_changed())
        root.addWidget(self.splitter, stretch=1)

        bottom = QHBoxLayout()
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            ["#", "model x", "model y", "model z", "u", "v", "depth m", "residual mm"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        bottom.addWidget(self.table, stretch=1)

        actions = QVBoxLayout()
        self.solve_button = QPushButton("1. Solve Procrustes")
        self.solve_button.clicked.connect(self._solve)
        self.solve_setup_button = QPushButton("Solve setup (all objects)")
        self.solve_setup_button.clicked.connect(self._solve_setup)
        self.defer_button = QPushButton("Defer pose")
        self.defer_button.setCheckable(True)
        self.defer_button.clicked.connect(self._toggle_current_deferred)
        clear_frame_button = QPushButton("Clear frame points")
        clear_frame_button.clicked.connect(self._clear_frame_points)
        reset_camera_button = QPushButton("Reset 3D camera")
        reset_camera_button.clicked.connect(self._reset_model_camera)
        reset_image_button = QPushButton("Fit RGB image")
        reset_image_button.clicked.connect(self._reset_image_view)
        actions.addWidget(self.solve_button)
        actions.addWidget(self.defer_button)
        actions.addWidget(clear_frame_button)
        actions.addSpacing(12)
        actions.addWidget(reset_camera_button)
        actions.addWidget(reset_image_button)
        actions.addStretch()

        result_form = QFormLayout()
        self.progress_label = QLabel("0 / 0")
        self.rmse_label = QLabel("—")
        self.translation_label = QLabel("—")
        result_form.addRow("Frame points", self.progress_label)
        result_form.addRow("Landmark RMSD", self.rmse_label)
        result_form.addRow("Translation", self.translation_label)
        actions.addLayout(result_form)
        bottom.addLayout(actions)
        root.addLayout(bottom)

        workflow_footer = QHBoxLayout()
        workflow_footer.addWidget(QLabel("Manual depth: High Density (2) · ICP depth: High Accuracy (1)"))
        workflow_footer.addStretch()
        self.proceed_button = QPushButton("Proceed to ICP…")
        self.proceed_button.clicked.connect(self._proceed_to_icp)
        workflow_footer.addWidget(self.proceed_button)
        self.import_batch_button = QPushButton("Import GPU results…")
        self.import_batch_button.clicked.connect(self._import_gpu_results)
        workflow_footer.addWidget(self.import_batch_button)
        self.batch_review_button = QPushButton("Review SAM + ICP…")
        self.batch_review_button.clicked.connect(self._review_batch_results)
        workflow_footer.addWidget(self.batch_review_button)
        workflow_footer.addWidget(self.solve_setup_button)
        root.addLayout(workflow_footer)

        self.mesh_view.iren.add_observer("RightButtonPressEvent", self._mesh_right_clicked)

    def _view_state_changed(self) -> None:
        if not self._restoring_project:
            self._set_dirty()

    def _reset_model_camera(self) -> None:
        self.mesh_view.reset_camera()
        self._view_state_changed()

    def _reset_image_view(self) -> None:
        self.image_view.reset_view()
        self._view_state_changed()

    def _create_actions_and_menus(self) -> None:
        definitions = {
            "solve": ("&Solve Procrustes", self._solve),
            "defer_pose": ("&Defer/unannotatable pose", self._toggle_current_deferred),
            "toggle_overlay": ("Preview solved &overlay", self._preview_overlay_from_menu),
            "previous_frame": ("&Previous object / frame", self._previous_frame),
            "next_frame": ("&Next object / frame", self._next_frame),
            "rotate_up": ("Rotate &up", lambda: self._rotate_mesh(elevation=5)),
            "rotate_left": ("Rotate &left", lambda: self._rotate_mesh(azimuth=-5)),
            "rotate_down": ("Rotate &down", lambda: self._rotate_mesh(elevation=-5)),
            "rotate_right": ("Rotate &right", lambda: self._rotate_mesh(azimuth=5)),
            "spin_counterclockwise": (
                "Spin &counterclockwise",
                lambda: self._rotate_mesh(roll=-5),
            ),
            "spin_clockwise": ("Spin cl&ockwise", lambda: self._rotate_mesh(roll=5)),
            "hold_navigation": ("Hold &navigation mode", self._toggle_persistent_navigation),
            "hold_magnifier": ("Hold RGB magni&fier", lambda: None),
        }
        for name, (label, callback) in definitions.items():
            action = QAction(label, self)
            shortcut = self._settings.value(f"shortcuts/{name}", DEFAULT_SHORTCUTS[name], str)
            action.setProperty("base_label", label)
            if name in RAW_HOLD_ACTIONS:
                action.setProperty("raw_shortcut", shortcut)
                _update_raw_action_label(action)
            else:
                action.setShortcut(QKeySequence(shortcut))
                action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
            action.triggered.connect(callback)
            self.actions[name] = action

        file_menu = self.menuBar().addMenu("&File")
        new_project = file_menu.addAction("&New project…")
        new_project.setShortcut(QKeySequence.StandardKey.New)
        new_project.triggered.connect(self._new_project)
        load_project = file_menu.addAction("&Load project…")
        load_project.setShortcut(QKeySequence.StandardKey.Open)
        load_project.triggered.connect(self._load_project_dialog)
        file_menu.addSeparator()
        save_project = file_menu.addAction("&Save project")
        save_project.setShortcut(QKeySequence.StandardKey.Save)
        save_project.triggered.connect(self._save_project)
        save_project_as = file_menu.addAction("Save project &as…")
        save_project_as.setShortcut(QKeySequence.StandardKey.SaveAs)
        save_project_as.triggered.connect(self._save_project_as)
        file_menu.addSeparator()
        export_poses = file_menu.addAction("&Export solved poses as JSON…")
        export_poses.triggered.connect(self._export_solved_poses)

        workflow = self.menuBar().addMenu("&Workflow")
        workflow.addAction(self.actions["solve"])
        workflow.addAction(self.actions["defer_pose"])
        workflow.addAction(self.actions["toggle_overlay"])
        workflow.addSeparator()
        clear_action = workflow.addAction("Clear current &frame points")
        clear_action.triggered.connect(self._clear_frame_points)

        navigate = self.menuBar().addMenu("&Navigate")
        navigate.addAction(self.actions["previous_frame"])
        navigate.addAction(self.actions["next_frame"])

        view = self.menuBar().addMenu("&View")
        for name in (
            "rotate_up",
            "rotate_left",
            "rotate_down",
            "rotate_right",
            "spin_counterclockwise",
            "spin_clockwise",
        ):
            view.addAction(self.actions[name])
        view.addSeparator()
        view.addAction(self.actions["hold_navigation"])
        view.addAction(self.actions["hold_magnifier"])
        view.addSeparator()
        self.invalid_depth_action = view.addAction("Show &invalid depth regions")
        self.invalid_depth_action.setCheckable(True)
        self.invalid_depth_action.setChecked(self._show_invalid_depth)
        self.invalid_depth_action.toggled.connect(self._toggle_invalid_depth_overlay)

        settings = self.menuBar().addMenu("&Settings")
        configure = settings.addAction("Configure &shortcuts…")
        configure.triggered.connect(self._configure_shortcuts)

        help_menu = self.menuBar().addMenu("&Help")
        help_action = help_menu.addAction("사용법")
        help_action.triggered.connect(self._show_help)

    def _populate_controls(self) -> None:
        self.frame_combo.blockSignals(True)
        self.frame_combo.clear()
        for pose in self.dataset.poses["0"]:
            self.frame_combo.addItem(f"{pose.frame_id:06}", pose.frame_id)
        self.frame_combo.blockSignals(False)
        self._populate_object_combo()

    def _frame_object_ids(self, frame_id: int) -> list[int]:
        """Return known object IDs for one frame, or all models without a manifest.

        BOP ``scene_gt.json`` doubles as a scene-object manifest here. Only its
        object IDs are used to populate the annotation workflow; its supplied
        poses remain read-only and are never used to solve an annotation.
        """
        pose = next(
            (pose for pose in self.dataset.poses["0"] if pose.frame_id == frame_id),
            None,
        )
        if pose is None:
            raise ValueError(f"Frame {frame_id} is outside the annotation workflow.")
        return self._pose_object_ids(pose)

    def _pose_object_ids(self, pose: Pose) -> list[int]:
        referenced_ids = sorted(
            {
                instance.object_id
                for instance in pose.reference_instances
                if instance.object_id in self.dataset.models
            }
        )
        return referenced_ids or sorted(self.dataset.models)

    def _populate_object_combo(
        self,
        *,
        preferred_object_id: int | None = None,
        selected_index: int | None = None,
    ) -> None:
        frame_id = self.frame_combo.currentData()
        object_ids = self._frame_object_ids(int(frame_id)) if frame_id is not None else []
        self.object_combo.blockSignals(True)
        self.object_combo.clear()
        for object_id in object_ids:
            self.object_combo.addItem(f"obj_{object_id:06}", object_id)
        if selected_index is not None and object_ids:
            self.object_combo.setCurrentIndex(
                min(max(0, selected_index), len(object_ids) - 1)
            )
        elif preferred_object_id is not None:
            preferred_index = self.object_combo.findData(preferred_object_id)
            self.object_combo.setCurrentIndex(max(0, preferred_index))
        self.object_combo.blockSignals(False)

    def _frame_selection_changed(self) -> None:
        self._populate_object_combo(preferred_object_id=self.current_object_id)
        self._selection_changed()

    def _object_selection_changed(self) -> None:
        self._selection_changed()

    def _selection_changed(self) -> None:
        if self.frame_combo.currentData() is None or self.object_combo.currentData() is None:
            return
        next_key = (int(self.frame_combo.currentData()), int(self.object_combo.currentData()))
        if self.current_pose is not None and self.current_object_id is not None and not self._restoring_project:
            old_key = (self.current_pose.frame_id, self.current_object_id)
            old_state = self.annotation_states[self.current_object_id]
            if (next_key != old_key and old_state.frame_points.get(old_key[0])
                    and old_key[0] not in old_state.solutions):
                answer = QMessageBox.warning(
                    self, "Points not solved",
                    f"Frame {old_key[0]:06d}, object {old_key[1]:06d} has selected RGB-D points "
                    "but no Procrustes pose. Leave it unfinished?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                    QMessageBox.StandardButton.Cancel,
                )
                if answer != QMessageBox.StandardButton.Yes:
                    self.frame_combo.blockSignals(True)
                    self.frame_combo.setCurrentIndex(self.frame_combo.findData(old_key[0]))
                    self.frame_combo.blockSignals(False)
                    self._populate_object_combo(preferred_object_id=old_key[1])
                    return
        self._load_current_selection()
        if not self._restoring_project:
            self._set_dirty()

    def _load_current_selection(self) -> None:
        frame_id = int(self.frame_combo.currentData())
        object_id = int(self.object_combo.currentData())
        object_changed = object_id != self.current_object_id
        self.current_object_id = object_id
        self.current_pose = self.dataset[frame_id]
        self.setup_edit.blockSignals(True)
        self.setup_edit.setText(self.frame_setups.get(frame_id, "setup_0"))
        self.setup_edit.blockSignals(False)
        self.image_view.set_image(self.current_pose.rgb)
        self._refresh_invalid_depth_overlay()

        if object_changed or self._mesh is None:
            model = self.dataset.models[object_id]
            self._mesh = pv.read(model.mesh_path)
            self._mesh_texture = None
            self.mesh_view.disable_picking()
            self.mesh_view.clear()
            texture_path = _find_mesh_texture_path(model.mesh_path)
            texture_coordinates = next(
                (
                    name
                    for name in ("TCoords", "Texture Coordinates", "texture_uv")
                    if name in self._mesh.point_data
                ),
                None,
            )
            color_scalars = next(
                (name for name in ("RGBA", "RGB", "rgba", "rgb") if name in self._mesh.point_data),
                None,
            )
            if texture_path is not None and texture_coordinates is not None:
                self._mesh.active_texture_coordinates = self._mesh.point_data[
                    texture_coordinates
                ]
                self._mesh_texture = pv.read_texture(texture_path)
                self.mesh_view.add_mesh(
                    self._mesh,
                    texture=self._mesh_texture,
                    opacity=1.0,
                )
            elif color_scalars:
                color_values = np.asarray(self._mesh.point_data[color_scalars])
                if color_values.ndim == 2 and color_values.shape[1] == 4:
                    # VTK honors the fourth scalar component as per-vertex
                    # alpha. Keep the PLY colors but deliberately discard that
                    # alpha for a fully opaque annotation/picking view.
                    color_scalars = "_annotation_opaque_rgb"
                    self._mesh.point_data[color_scalars] = np.ascontiguousarray(
                        color_values[:, :3]
                    )
                self.mesh_view.add_mesh(
                    self._mesh,
                    scalars=color_scalars,
                    rgb=True,
                    opacity=1.0,
                )
            else:
                self.mesh_view.add_mesh(self._mesh, color="wheat", opacity=1.0)
            self.mesh_view.reset_camera()
            self._apply_interaction_mode()
        self._refresh()

    def _toggle_invalid_depth_overlay(self, enabled: bool) -> None:
        self._show_invalid_depth = enabled
        self._refresh_invalid_depth_overlay()
        if not self._restoring_project:
            self._set_dirty()

    def _refresh_invalid_depth_overlay(self) -> None:
        if not self._show_invalid_depth or self.current_pose is None:
            self.image_view.set_invalid_depth_mask(None)
            return
        depth = self.current_pose.depth
        if depth.shape != self.current_pose.rgb.shape[:2]:
            self.image_view.set_invalid_depth_mask(None)
            self.statusBar().showMessage(
                f"Depth/RGB size mismatch for frame {self.current_pose.frame_id}.",
                6000,
            )
            return
        invalid = ~np.isfinite(depth) | (depth <= 0)
        self.image_view.set_invalid_depth_mask(invalid)

    @property
    def navigation_active(self) -> bool:
        return self._persistent_navigation or self._temporary_navigation_held

    def _persistent_mode_changed(self, navigation: bool) -> None:
        self._persistent_navigation = navigation
        self._apply_interaction_mode()
        if not self._restoring_project:
            self._set_dirty()

    def _toggle_persistent_navigation(self) -> None:
        self.mode_button.toggle()

    def _apply_interaction_mode(self) -> None:
        navigation = self.navigation_active
        self.mesh_view.disable_picking()
        if navigation:
            self.mesh_view.enable_trackball_style()
            self.mesh_view.interactor.setCursor(Qt.CursorShape.OpenHandCursor)
            self.mode_button.setText("Navigate")
            self.mode_button.setStyleSheet(
                "QToolButton { background: #3273dc; color: white; font-weight: 600; padding: 6px; }"
            )
        else:
            # Keep PyVista's managed style because its picker relies on the
            # style's internal ``_parent`` link. Camera drag/wheel events are
            # suppressed in ``eventFilter`` while Annotate mode is active.
            self.mesh_view.enable_trackball_style()
            self.mesh_view.enable_surface_point_picking(
                callback=self._model_point_picked,
                picker="cell",
                show_point=False,
                show_message=False,
                pickable_window=False,
                left_clicking=True,
            )
            self.mesh_view.interactor.setCursor(Qt.CursorShape.CrossCursor)
            self.mode_button.setText("Annotate")
            self.mode_button.setStyleSheet(
                "QToolButton { background: #d97706; color: white; font-weight: 600; padding: 6px; }"
            )
        self.image_view.set_navigation_mode(navigation)

    def _model_point_picked(self, point: np.ndarray) -> None:
        if self.navigation_active:
            return
        self._active_view = "model"
        coordinates = np.asarray(point, dtype=np.float64).reshape(3) / 1000.0
        if not np.isfinite(coordinates).all():
            return
        selected = Vector3.from_array(coordinates)
        if any(np.linalg.norm(selected.as_array() - point.as_array()) < 0.0005 for point in self.model_points):
            self.statusBar().showMessage("That model landmark is already selected.", 4000)
            return
        self._materialize_current_model_points().append(selected)
        self._remember_current_model_points()
        self._invalidate_current_solution()
        self._set_dirty()
        self._refresh()

    def _mesh_right_clicked(self, *_args) -> None:
        if self.navigation_active:
            return
        self._active_view = "model"
        if QApplication.keyboardModifiers() & Qt.KeyboardModifier.ControlModifier:
            self._clear_model_points()
        else:
            self._delete_last_model_point()

    def _delete_last_model_point(self) -> None:
        if not self.model_points:
            return
        points = self._materialize_current_model_points()
        removed_index = len(points) - 1
        points.pop()
        self._remember_current_model_points()
        observations = self.frame_observations
        if len(observations) > removed_index:
            observations.pop(removed_index)
        if not observations and self.current_pose is not None:
            self.state.frame_points.pop(self.current_pose.frame_id, None)
        self._invalidate_current_solution()
        self._set_dirty()
        self._refresh()

    def _clear_model_points(self) -> None:
        if not self.model_points:
            return
        points = self._materialize_current_model_points()
        points.clear()
        self._remember_current_model_points()
        if self.current_pose is not None:
            frame_id = self.current_pose.frame_id
            self.state.frame_points.pop(frame_id, None)
            self._invalidate_current_solution()
        self._set_dirty()
        self._refresh()

    def _image_point_picked(self, x: float, y: float) -> None:
        self._active_view = "image"
        if self.current_pose is None:
            return
        if len(self.frame_observations) >= len(self.model_points):
            self.statusBar().showMessage(
                "Select another model landmark before adding another RGB point.", 4000
            )
            return
        pixel = Vector2(x, y)
        try:
            image_point = get_pixel_point(self.current_pose, pixel)
            camera_point = unproject(image_point, self.current_pose.camera_intrinsics)
        except ValueError as error:
            self.statusBar().showMessage(str(error), 6000)
            return

        observations = self.state.frame_points.setdefault(self.current_pose.frame_id, [])
        self._materialize_current_model_points()
        observations.append(ImageSelection(pixel=pixel, camera=camera_point))
        self._invalidate_current_solution()
        self._set_dirty()
        self._refresh()

    def _delete_last_image_point(self) -> None:
        self._active_view = "image"
        observations = self.frame_observations
        if observations:
            observations.pop()
            if not observations and self.current_pose is not None:
                self.state.frame_points.pop(self.current_pose.frame_id, None)
            self._invalidate_current_solution()
            self._set_dirty()
            self._refresh()

    def _clear_frame_points(self) -> None:
        self._active_view = "image"
        if self.current_pose is None:
            return
        had_points = self.current_pose.frame_id in self.state.frame_points
        self.state.frame_points.pop(self.current_pose.frame_id, None)
        self._invalidate_current_solution()
        if had_points:
            self._set_dirty()
        self._refresh()

    def _invalidate_current_solution(self) -> None:
        if self.current_pose is not None:
            frame_id = self.current_pose.frame_id
            if frame_id in self.state.solutions or frame_id in self.state.inferred:
                self._clear_setup_inferences(self.frame_setups[frame_id])
            self.state.solutions.pop(frame_id, None)
            self._invalidate_local_review(self.state, frame_id)

    @staticmethod
    def _invalidate_local_review(state: ObjectAnnotationState, frame_id: int) -> None:
        state.masks.pop(frame_id, None)
        state.icp_candidates.pop(frame_id, None)
        state.icp_info.pop(frame_id, None)
        state.icp_decisions.pop(frame_id, None)
        state.batch_metrics.pop(frame_id, None)
        state.batch_errors.pop(frame_id, None)
        state.review_prompts.pop(frame_id, None)

    def _clear_setup_inferences(self, setup_id: str) -> None:
        self.setup_reports.pop(setup_id, None)
        frames = {frame for frame, group in self.frame_setups.items() if group == setup_id}
        for state in self.annotation_states.values():
            for frame in frames:
                state.inferred.pop(frame, None)

    def _assign_current_setup(self) -> None:
        if self.current_pose is None:
            return
        setup_id = self.setup_edit.text().strip()
        frame_id = self.current_pose.frame_id
        if not setup_id:
            self.setup_edit.setText(self.frame_setups[frame_id])
            return
        old_id = self.frame_setups[frame_id]
        if setup_id == old_id:
            return
        self._clear_setup_inferences(old_id)
        self._clear_setup_inferences(setup_id)
        self.frame_setups[frame_id] = setup_id
        self._set_dirty()
        self._refresh()

    def _stage_text(self, state: ObjectAnnotationState, frame_id: int) -> str:
        if frame_id in state.inferred:
            return "Inferred from setup"
        if frame_id not in state.solutions:
            return "Not manually solved"
        if frame_id in state.batch_errors:
            return "GPU batch failed; reprompt or use raw"
        if frame_id not in state.masks:
            return "Manual Procrustes target"
        if frame_id not in state.icp_candidates:
            return "SAM mask ready; ICP unavailable"
        decision = state.icp_decisions.get(frame_id, "pending")
        if decision == "keep":
            return "ICP accepted"
        if decision == "revert":
            return "Raw pose selected (ICP rejected)"
        return "Review SAM + ICP proposal"

    def _show_setup_overview(self) -> None:
        if self.current_pose is None:
            return
        setup_id = self.frame_setups[self.current_pose.frame_id]
        rows = [
            (frame, object_id, self._stage_text(self.annotation_states[object_id], frame))
            for frame in sorted(self.frame_setups)
            if self.frame_setups[frame] == setup_id
            for object_id in self._frame_object_ids(frame)
        ]
        dialog = SetupOverviewDialog(rows, setup_id, self.setup_reports.get(setup_id), self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        if dialog.open_key is not None:
            frame, object_id = dialog.open_key
            frame_index = self.frame_combo.findData(frame)
            object_index = self._frame_object_ids(frame).index(object_id)
            self._set_annotation_selection(frame_index, object_index)
        self._refresh()

    def _toggle_current_deferred(self, _checked: bool = False) -> None:
        if self.current_pose is None:
            return
        frame_id = self.current_pose.frame_id
        if frame_id in self.state.deferred_frames:
            self.state.deferred_frames.remove(frame_id)
            message = f"Frame {frame_id} restored for annotation."
        else:
            self.state.deferred_frames.add(frame_id)
            self._clear_setup_inferences(self.frame_setups[frame_id])
            self.state.solutions.pop(frame_id, None)
            self._invalidate_local_review(self.state, frame_id)
            self._hide_overlay()
            message = f"Frame {frame_id} marked deferred/unannotatable."
        self._set_dirty()
        self._refresh()
        self.statusBar().showMessage(message, 5000)

    def _solve(self, _checked: bool = False, *, show_errors: bool = True) -> bool:
        if self.current_pose is None:
            return False
        if self.current_pose.frame_id in self.state.deferred_frames:
            if show_errors:
                QMessageBox.information(
                    self,
                    "Pose deferred",
                    "This pose is marked unannotatable. Press R to restore it before solving.",
                )
            return False
        if len(self.frame_observations) != len(self.model_points) or len(self.model_points) < 3:
            if show_errors:
                QMessageBox.warning(
                    self,
                    "Incomplete correspondences",
                    "Add at least three model points and one RGB point for each model point.",
                )
            return False
        model_points = self.model_points
        camera_points = [observation.camera for observation in self.frame_observations]
        try:
            rotation, translation, rmse = solve_procrustes(model_points, camera_points)
        except ValueError as error:
            if show_errors:
                QMessageBox.warning(self, "Cannot solve pose", str(error))
            else:
                self.statusBar().showMessage(str(error), 5000)
            return False

        transformed = np.stack(
            [rotation @ point.as_array() + translation.as_array() for point in model_points]
        )
        targets = np.stack([point.as_array() for point in camera_points])
        residuals_mm = np.linalg.norm(targets - transformed, axis=1) * 1000.0
        solution = PoseSolution(rotation, translation, rmse, residuals_mm)
        self._clear_setup_inferences(self.frame_setups[self.current_pose.frame_id])
        self._invalidate_local_review(self.state, self.current_pose.frame_id)
        self.state.deferred_frames.discard(self.current_pose.frame_id)
        self.state.solutions[self.current_pose.frame_id] = solution
        self._set_dirty()
        result = {
            "frame_id": self.current_pose.frame_id,
            "obj_id": self.current_object_id,
            "cam_R_m2c": rotation.reshape(-1).tolist(),
            "cam_t_m2c": (translation.as_array() * 1000.0).tolist(),
            "rmse_metres": rmse,
        }
        QApplication.clipboard().setText(json.dumps(result, indent=2))
        self._refresh()
        self.statusBar().showMessage(
            "Pose solved and copied to the clipboard. Hold Tab to preview the overlay.", 8000
        )
        return True

    def _raw_record(self, frame_id: int, object_id: int) -> dict[str, object]:
        state = self.annotation_states[object_id]
        solution = state.solutions[frame_id]
        clicks = [list(point.pixel.as_array()) for point in state.frame_points.get(frame_id, [])]
        if len(clicks) < 3:
            raise ValueError(f"Frame {frame_id}/object {object_id} needs at least three saved RGB clicks.")
        prompts = state.review_prompts.get(frame_id, {})
        return {
            "obj_id": object_id,
            "cam_R_m2c": solution.rotation.reshape(-1).tolist(),
            "cam_t_m2c": (solution.translation.as_array() * 1000.0).tolist(),
            "annotation_rmsd_mm": solution.rmse * 1000.0,
            "manual_clicks_xy": clicks,
            "extra_positive_xy": prompts.get("extra_positive_xy", []),
            "extra_negative_xy": prompts.get("extra_negative_xy", []),
            "box_padding": prompts.get("box_padding", 0.25),
        }

    def _raw_batch_document(self, keys: list[tuple[int, int]] | None = None) -> dict[str, object]:
        if keys is None:
            keys = sorted(
                (frame, object_id)
                for object_id, state in self.annotation_states.items()
                for frame in state.solutions
            )
        if not keys:
            raise ValueError("Solve at least one pose with Procrustes before proceeding.")
        poses: dict[str, list[dict[str, object]]] = {}
        for frame, object_id in keys:
            poses.setdefault(str(frame), []).append(self._raw_record(frame, object_id))
        return {
            "format": RAW_FORMAT,
            "method": self._icp_method,
            "silhouette_weight": self._silhouette_weight,
            "poses": poses,
        }

    @staticmethod
    def _standard_raw_export(raw: dict[str, object]) -> dict[str, list[dict[str, object]]]:
        """Normal pose JSON, separate from the prompt-rich GPU job manifest."""
        return {
            frame: [
                {
                    "obj_id": record["obj_id"],
                    "annotation_status": "solved",
                    "annotation_source": "manual_procrustes",
                    "cam_R_m2c": record["cam_R_m2c"],
                    "cam_t_m2c": record["cam_t_m2c"],
                    "annotation_rmsd_mm": record["annotation_rmsd_mm"],
                }
                for record in records
            ]
            for frame, records in raw["poses"].items()
        }

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch
            return bool(torch.cuda.is_available())
        except ImportError:
            return False

    def _proceed_to_icp(self) -> None:
        settings = ICPOptionsDialog(self._icp_method, self._silhouette_weight, self)
        if settings.exec() != QDialog.DialogCode.Accepted:
            return
        self._icp_method = settings.method.currentData()
        self._silhouette_weight = settings.weight.value()
        unfinished = [
            (frame, object_id)
            for object_id, state in self.annotation_states.items()
            for frame, points in state.frame_points.items()
            if points and frame not in state.solutions
        ]
        if unfinished:
            examples = ", ".join(f"{frame:06d}/obj_{obj:06d}" for frame, obj in sorted(unfinished)[:6])
            answer = QMessageBox.warning(
                self, "Selected points without a pose",
                f"{len(unfinished)} frame/object pairs have RGB-D points but no Procrustes solution "
                f"({examples}). Continue with solved poses only?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        try:
            raw = self._raw_batch_document()
        except ValueError as error:
            QMessageBox.warning(self, "Manual poses needed", str(error))
            return
        suggested = self._project_path.with_name(self._project_path.stem + "_raw_poses.json") if self._project_path else Path.cwd() / "raw_poses.json"
        filename, _ = QFileDialog.getSaveFileName(self, "Save raw Procrustes poses", str(suggested), "JSON files (*.json)")
        if not filename:
            return
        path = Path(filename).with_suffix(".json")
        batch_input = path.with_name(path.stem + "_batch_input.json")
        archive_path = path.with_name(path.stem + "_gpu_job.zip")
        try:
            write_json(path, self._standard_raw_export(raw))
            write_json(batch_input, raw)
            make_job_archive(self.dataset.root, batch_input, archive_path)
        except (OSError, ValueError, TypeError) as error:
            QMessageBox.critical(self, "Could not prepare ICP batch", str(error))
            return
        if self._has_cuda():
            self._launch_local_batch(raw, path.with_name(path.stem + "_gpu_results.json"))
        else:
            QMessageBox.information(
                self, "GPU job ready",
                f"Raw poses were saved to {path}.\n\n"
                f"Upload {archive_path} to the GPU machine and run:\n\n"
                "uv run --extra segmentation python -m src.backend.batch_pipeline "
                "--job <uploaded.zip> --output gpu_results.json\n\n"
                "Download gpu_results.json, then click Import GPU results… here.",
            )

    def _launch_local_batch(self, raw: dict[str, object], output_path: Path) -> None:
        total = sum(len(records) for records in raw["poses"].values())
        progress = QProgressDialog("Generating SAM masks and ICP proposals…", "", 0, total, self)
        progress.setCancelButton(None)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()

        def work():
            return run_batch(
                DataSet.load(self.dataset.root, depth_preset=1), raw,
                progress=lambda i, n, frame, obj: worker.progress.emit(i, n, frame, obj),
            )

        worker = BackgroundTask(work, self)
        self._review_worker = worker

        def update_progress(index: int, count: int, frame: int, object_id: int) -> None:
            progress.setLabelText(f"Processing {index}/{count}: frame {frame:06d}, object {object_id:06d}")
            progress.setValue(index - 1)

        worker.progress.connect(update_progress)

        def success(result: dict[str, object]) -> None:
            progress.close()
            try:
                write_json(output_path, result)
                count = self._apply_batch_results(result)
            except (OSError, ValueError, TypeError, KeyError) as error:
                QMessageBox.critical(self, "Could not use GPU results", str(error))
                return
            self.statusBar().showMessage(f"Saved {output_path}; imported {count} ICP proposals.", 10000)
            self._review_batch_results()

        worker.result_ready.connect(success)
        worker.failed.connect(lambda message: (progress.close(), QMessageBox.warning(self, "GPU batch failed", message)))
        worker.finished.connect(lambda current=worker: setattr(self, "_review_worker", None) if self._review_worker is current else None)
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _import_gpu_results(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "Import GPU batch results", str(self._project_path.parent if self._project_path else Path.cwd()), "JSON files (*.json)")
        if not filename:
            return
        try:
            with Path(filename).open(encoding="utf-8") as file:
                count = self._apply_batch_results(json.load(file))
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            QMessageBox.critical(self, "Could not import GPU results", str(error))
            return
        QMessageBox.information(self, "GPU results imported", f"Imported {count} ICP proposals. Review each mask and pose before accepting it.")
        self._review_batch_results()

    def _apply_batch_results(self, document: dict[str, object]) -> int:
        if document.get("format") != RESULT_FORMAT or not isinstance(document.get("results"), dict):
            raise ValueError("This is not a compatible SAM + ICP batch result.")
        prepared = []
        for frame_text, objects in document["results"].items():
            if not isinstance(objects, dict):
                raise TypeError("Batch result contains an invalid frame entry.")
            frame = int(frame_text)
            for object_text, entry in objects.items():
                object_id = int(object_text)
                if object_id not in self.annotation_states or frame not in self.annotation_states[object_id].solutions:
                    raise ValueError(f"Result frame {frame}/object {object_id} is not manually solved here.")
                raw = self._raw_record(frame, object_id)
                if entry.get("raw_sha256") != record_hash(raw):
                    raise ValueError(f"Result frame {frame}/object {object_id} does not match current pose or prompts.")
                if entry.get("status") != "ready":
                    prepared.append((frame, object_id, None, str(entry.get("error", "GPU batch failed")), None, None, {}))
                    continue
                mask = decode_mask(entry["mask_png_base64"], self.dataset[frame].rgb.shape[:2])
                rotation = np.asarray(entry["candidate_rotation_m2c"], np.float64).reshape(3, 3)
                translation = np.asarray(entry["candidate_translation_m2c_mm"], np.float64).reshape(3) / 1000.0
                RigidPose(rotation, translation)
                score = float(entry["sam_score"])
                if not np.isfinite(score):
                    raise ValueError("SAM score is not finite.")
                box = _float_vector(entry["sam_box_xyxy"], 4, "SAM box")
                metrics = {
                    "raw_depth_rmse_mm": None if entry.get("raw_depth_rmse_mm") is None else float(entry["raw_depth_rmse_mm"]),
                    "icp_depth_rmse_mm": None if entry.get("icp_depth_rmse_mm") is None else float(entry["icp_depth_rmse_mm"]),
                }
                for value in metrics.values():
                    if value is not None and (not np.isfinite(value) or value < 0):
                        raise ValueError("Depth RMSE must be finite and non-negative.")
                info = dict(_mapping(entry.get("icp_info", {}), "ICP info"))
                info.setdefault("review_method", entry.get("method", document.get("method", "silhouette")))
                info.setdefault("review_silhouette_weight", entry.get("silhouette_weight", document.get("silhouette_weight", 0.10)))
                prepared.append((frame, object_id, (mask, score, box, rotation, translation), None, metrics, raw, info))
        count = 0
        for frame, object_id, ready, error, metrics, raw, info in prepared:
            state = self.annotation_states[object_id]
            state.icp_candidates.pop(frame, None)
            state.icp_decisions.pop(frame, None)
            state.icp_info.pop(frame, None)
            state.batch_metrics.pop(frame, None)
            if ready is None:
                state.masks.pop(frame, None)
                state.batch_errors[frame] = error
                continue
            mask, score, box, rotation, translation = ready
            state.batch_errors.pop(frame, None)
            state.masks[frame] = MaskReview(mask, score, list(box),
                [tuple(point) for point in raw["extra_positive_xy"]],
                [tuple(point) for point in raw["extra_negative_xy"]],
                float(raw["box_padding"]))
            state.icp_candidates[frame] = PoseSolution(rotation, Vector3.from_array(translation), state.solutions[frame].rmse, np.empty(0))
            state.icp_info[frame] = info
            state.batch_metrics[frame] = metrics
            state.review_prompts[frame] = {
                "extra_positive_xy": raw["extra_positive_xy"],
                "extra_negative_xy": raw["extra_negative_xy"],
                "box_padding": raw["box_padding"],
            }
            state.icp_decisions[frame] = "pending"
            count += 1
        if prepared:
            method = document.get("method", "silhouette")
            if method in ("silhouette", "colored"):
                self._icp_method = method
            weight = document.get("silhouette_weight")
            if weight is not None and np.isfinite(float(weight)) and float(weight) >= 0:
                self._silhouette_weight = float(weight)
            for frame, *_ in prepared:
                self._clear_setup_inferences(self.frame_setups[frame])
            self._set_dirty()
            self._refresh()
        return count

    def _review_batch_results(self) -> None:
        keys = sorted(
            (frame, object_id)
            for object_id, state in self.annotation_states.items()
            for frame in state.icp_candidates.keys() | state.batch_errors.keys()
        )
        if not keys:
            QMessageBox.information(self, "No batch results", "Proceed to ICP or import GPU results first.")
            return
        if self._project_path is None and not self._save_project_as():
            return
        meshes: dict[int, pv.PolyData] = {}

        def load_page(key: tuple[int, int]) -> dict[str, object]:
            frame, object_id = key
            state = self.annotation_states[object_id]
            pose = self.dataset[frame]
            if object_id not in meshes:
                meshes[object_id] = pv.read(self.dataset.models[object_id].mesh_path)
            mesh = meshes[object_id]
            manual = state.solutions[frame]
            candidate = state.icp_candidates.get(frame)
            mask = state.masks.get(frame)
            prompts = state.review_prompts.get(frame, {})
            metrics = state.batch_metrics.get(frame, {})
            info = state.icp_info.get(frame, {})
            return {
                "rgb": pose.rgb,
                "mask": mask.mask if mask else None,
                "raw_overlay": _project_mesh_overlay(mesh, pose, manual),
                "icp_overlay": _project_mesh_overlay(mesh, pose, candidate) if candidate else None,
                "raw_rmse_mm": metrics.get("raw_depth_rmse_mm"),
                "icp_rmse_mm": metrics.get("icp_depth_rmse_mm"),
                "sam_score": mask.score if mask else None,
                "manual_clicks": [tuple(point.pixel.as_array()) for point in state.frame_points.get(frame, [])],
                "positive": [tuple(point) for point in prompts.get("extra_positive_xy", [])],
                "negative": [tuple(point) for point in prompts.get("extra_negative_xy", [])],
                "box_padding": float(prompts.get("box_padding", 0.25)),
                "method": info.get("review_method", self._icp_method),
                "weight": info.get("review_silhouette_weight") or self._silhouette_weight,
                "error": state.batch_errors.get(frame),
            }

        rejected = {
            key: self.annotation_states[key[1]].icp_decisions.get(key[0]) == "revert"
            or key[0] not in self.annotation_states[key[1]].icp_candidates
            for key in keys
        }
        current = (self.current_pose.frame_id, self.current_object_id) if self.current_pose else None
        dialog = BatchReviewDialog(keys, load_page, rejected=rejected, current_key=current, parent=self)
        dialog.exec()
        if dialog.retry_key is not None:
            frame, object_id = dialog.retry_key
            prompt = dialog.retry_prompts
            state = self.annotation_states[object_id]
            self._icp_method = prompt["method"]
            self._silhouette_weight = prompt["weight"]
            self._invalidate_local_review(state, frame)
            # Keep the new prompts after clearing the stale mask and proposal.
            state.review_prompts[frame] = {
                "extra_positive_xy": [list(point) for point in prompt["positive"]],
                "extra_negative_xy": [list(point) for point in prompt["negative"]],
                "box_padding": prompt["padding"],
            }
            self._clear_setup_inferences(self.frame_setups[frame])
            self._set_dirty()
            self._refresh()
            self._queue_retry_job(frame, object_id)
            return
        if not dialog.confirmed:
            return
        affected_setups = set()
        for frame, object_id in keys:
            state = self.annotation_states[object_id]
            decision = "revert" if dialog.rejected.get((frame, object_id), False) else "keep"
            if frame not in state.icp_candidates:
                decision = "revert"
            state.icp_decisions[frame] = decision
            if decision == "revert":
                state.batch_errors.pop(frame, None)
            affected_setups.add(self.frame_setups[frame])
        for setup_id in affected_setups:
            self._clear_setup_inferences(setup_id)
        self._set_dirty()
        self._refresh()
        self._save_project()

    def _queue_retry_job(self, frame: int, object_id: int) -> None:
        raw = self._raw_batch_document([(frame, object_id)])
        base = self._project_path.with_name(self._project_path.stem) if self._project_path else Path.cwd() / "procrustes"
        attempt = 1
        while True:
            path = base.with_name(f"{base.name}_retry_f{frame:06d}_o{object_id:06d}_v{attempt}_raw.json")
            archive = path.with_name(path.stem + "_gpu_job.zip")
            if not path.exists() and not archive.exists():
                break
            attempt += 1
        try:
            write_json(path, raw)
            make_job_archive(self.dataset.root, path, archive)
        except (OSError, ValueError, TypeError) as error:
            QMessageBox.critical(self, "Could not prepare retry", str(error))
            return
        if self._has_cuda():
            self._launch_local_batch(raw, path.with_name(path.stem + "_gpu_results.json"))
        else:
            QMessageBox.information(self, "Retry job ready", f"Upload {archive} to the GPU machine, run the batch command from the README, and import its returned JSON.")

    def _solve_setup(self, _checked: bool = False) -> None:
        if self.current_pose is None:
            return
        setup_id = self.frame_setups[self.current_pose.frame_id]
        frames = sorted(frame for frame, group in self.frame_setups.items() if group == setup_id)
        object_ids = sorted(self.dataset.models)
        observations, incomplete = _accepted_setup_observations(self.annotation_states, frames)
        if incomplete:
            QMessageBox.warning(
                self, "ICP proposals need review",
                "Choose Keep ICP or Use raw pose for these pairs before solving the setup:\n"
                + "\n".join(incomplete[:12]),
            )
            return
        try:
            fit = solve_setup(observations, frames, object_ids)
        except ValueError as error:
            QMessageBox.warning(self, "Cannot solve setup", str(error))
            return
        self._clear_setup_inferences(setup_id)
        count = 0
        for (frame, object_id), pose in fit.predicted.items():
            state = self.annotation_states[object_id]
            if frame in state.solutions:
                continue
            state.deferred_frames.discard(frame)
            state.inferred[frame] = PoseSolution(
                pose.rotation, Vector3.from_array(pose.translation), 0.0, np.empty(0)
            )
            count += 1
        maximum_mm = max(fit.translation_residual_mm.values(), default=0.0)
        maximum_deg = max(fit.rotation_residual_deg.values(), default=0.0)
        self.setup_reports[setup_id] = {
            "accepted_target_count": len(observations),
            "inferred_count": count,
            "max_translation_residual_mm": maximum_mm,
            "max_rotation_residual_deg": maximum_deg,
            "observations": [
                {
                    "frame_id": frame,
                    "object_id": object_id,
                    "translation_residual_mm": fit.translation_residual_mm[(frame, object_id)],
                    "rotation_residual_deg": fit.rotation_residual_deg[(frame, object_id)],
                }
                for frame, object_id in sorted(observations)
            ],
        }
        self._set_dirty()
        self._refresh()
        message = (f"Setup {setup_id}: inferred {count} hidden poses from {len(observations)} "
                   f"accepted target poses. Maximum fit discrepancy: {maximum_mm:.1f} mm / {maximum_deg:.1f}°.")
        self.statusBar().showMessage(message, 20000)
        if maximum_mm > 15 or maximum_deg > 10:
            QMessageBox.warning(self, "Review setup fit", message + "\n\nCheck object identity and manual poses before export.")

    def _show_overlay_held(self) -> None:
        if self.current_pose is None:
            return
        if (self.current_pose.frame_id not in self.state.solutions
                and self.current_pose.frame_id not in self.state.inferred
                and not self._solve(show_errors=False)):
            return
        self._overlay_key_held = True
        self._refresh_overlay()

    def _hide_overlay(self) -> None:
        self._overlay_key_held = False
        self._overlay_held_key = None
        self._refresh_overlay()

    def _preview_overlay_from_menu(self) -> None:
        self._show_overlay_held()
        QTimer.singleShot(1000, self._hide_overlay)

    def _refresh_overlay(self) -> None:
        if (
            not self._overlay_key_held
            or self.current_pose is None
            or self._mesh is None
            or (self.current_pose.frame_id not in self.state.solutions
                and self.current_pose.frame_id not in self.state.inferred)
        ):
            self.image_view.set_overlay(None)
            self.image_view.set_greyscale(False)
            return
        frame_id = self.current_pose.frame_id
        solution = (
            self.state.icp_candidates.get(frame_id)
            if self.state.icp_decisions.get(frame_id) == "keep"
            else None
        ) or self.state.solutions.get(frame_id) or self.state.inferred.get(frame_id)
        self.image_view.set_greyscale(True)
        self.image_view.set_overlay(
            _project_mesh_overlay(self._mesh, self.current_pose, solution)
        )

    def _previous_frame(self) -> None:
        frame_index, object_index = _step_annotation_selection(
            self.frame_combo.currentIndex(),
            self.object_combo.currentIndex(),
            self._object_counts_by_frame(),
            direction=-1,
        )
        self._set_annotation_selection(frame_index, object_index)

    def _next_frame(self) -> None:
        frame_index, object_index = _step_annotation_selection(
            self.frame_combo.currentIndex(),
            self.object_combo.currentIndex(),
            self._object_counts_by_frame(),
            direction=1,
        )
        self._set_annotation_selection(frame_index, object_index)

    def _object_counts_by_frame(self) -> list[int]:
        return [
            len(self._pose_object_ids(pose)) for pose in self.dataset.poses["0"]
        ]

    def _set_annotation_selection(self, frame_index: int, object_index: int) -> None:
        changed = (
            frame_index != self.frame_combo.currentIndex()
            or object_index != self.object_combo.currentIndex()
        )
        if not changed:
            return
        self.frame_combo.blockSignals(True)
        self.frame_combo.setCurrentIndex(frame_index)
        self.frame_combo.blockSignals(False)
        self._populate_object_combo(selected_index=object_index)
        self._selection_changed()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API name
        event_type = event.type()
        if (
            watched is self.setup_edit
            and event_type == QEvent.Type.ShortcutOverride
            and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
        ):
            event.accept()
            return True
        # Child/native VTK windows may emit WindowDeactivate while the main
        # application is still active. Only a true application deactivation
        # should cancel held controls.
        if event_type == QEvent.Type.ApplicationDeactivate:
            self._active_rotation_keys.clear()
            self._rotation_timer.stop()
            self._hide_overlay()
            self.image_view.set_magnifier_enabled(False)
            self._magnifier_held_key = None
            if self._temporary_navigation_held:
                self._temporary_navigation_held = False
                self._navigation_held_key = None
                self._apply_interaction_mode()
            return super().eventFilter(watched, event)
        if (
            watched is self.mesh_view.interactor
            and self.navigation_active
            and event_type == QEvent.Type.MouseButtonRelease
        ):
            self._view_state_changed()
        if watched is self.mesh_view.interactor and not self.navigation_active:
            if event_type == QEvent.Type.Wheel:
                return True
            if event_type == QEvent.Type.MouseMove and event.buttons() != Qt.MouseButton.NoButton:
                return True
        if event_type not in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            return super().eventFilter(watched, event)
        if QApplication.activeModalWidget() is not None or not self.isActiveWindow():
            return super().eventFilter(watched, event)
        if isinstance(QApplication.focusWidget(), QLineEdit):
            if (watched is self.setup_edit and event_type == QEvent.Type.KeyPress
                    and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)):
                self._assign_current_setup()
                self.setup_edit.clearFocus()
                return True
            return super().eventFilter(watched, event)

        pressed = event_type == QEvent.Type.KeyPress
        if pressed and self._event_matches_action(event, "hold_navigation"):
            if not event.isAutoRepeat():
                self._temporary_navigation_held = True
                self._navigation_held_key = event.key()
                self._apply_interaction_mode()
            return True
        if not pressed and event.key() == self._navigation_held_key:
            if event.isAutoRepeat():
                return True
            self._temporary_navigation_held = False
            self._navigation_held_key = None
            self._apply_interaction_mode()
            return True

        if pressed and self._event_matches_action(event, "hold_magnifier"):
            if not event.isAutoRepeat():
                self._magnifier_held_key = event.key()
                self.image_view.set_magnifier_enabled(True)
            return True
        if not pressed and event.key() == self._magnifier_held_key:
            if event.isAutoRepeat():
                return True
            self._magnifier_held_key = None
            self.image_view.set_magnifier_enabled(False)
            return True

        if pressed and self._event_matches_action(event, "toggle_overlay"):
            if not event.isAutoRepeat():
                self._overlay_held_key = event.key()
                self._show_overlay_held()
            return True
        if not pressed and event.key() == self._overlay_held_key:
            if event.isAutoRepeat():
                return True
            self._hide_overlay()
            return True

        rotation_names = (
            "rotate_up",
            "rotate_left",
            "rotate_down",
            "rotate_right",
            "spin_counterclockwise",
            "spin_clockwise",
        )
        if pressed:
            for name in rotation_names:
                if self._event_matches_action(
                    event, name, ignore_temporary_navigation_shift=True
                ):
                    if not event.isAutoRepeat():
                        self._active_rotation_keys[event.key()] = name
                        if not self._rotation_timer.isActive():
                            self._rotation_clock.restart()
                            self._rotation_timer.start()
                    return True
        elif event.key() in self._active_rotation_keys:
            if event.isAutoRepeat():
                return True
            del self._active_rotation_keys[event.key()]
            if not self._active_rotation_keys:
                self._rotation_timer.stop()
            return True
        return super().eventFilter(watched, event)

    def _event_matches_action(
        self,
        event,
        name: str,
        *,
        ignore_temporary_navigation_shift: bool = False,
    ) -> bool:
        shortcut = _action_shortcut(self.actions[name])
        portable = shortcut.toString(QKeySequence.SequenceFormat.PortableText)
        if portable == "Shift":
            return event.key() == Qt.Key.Key_Shift
        if shortcut.isEmpty():
            return False
        if QKeySequence(event.keyCombination()) == shortcut:
            return True
        if ignore_temporary_navigation_shift and self._temporary_navigation_held:
            modifiers = event.modifiers() & ~Qt.KeyboardModifier.ShiftModifier
            key_without_navigation_shift = QKeyCombination(
                modifiers, Qt.Key(event.key())
            )
            return QKeySequence(key_without_navigation_shift) == shortcut
        return False

    def _rotation_tick(self) -> None:
        elapsed_seconds = min(self._rotation_clock.restart(), 50) / 1000.0
        angle = 65.0 * elapsed_seconds
        active = set(self._active_rotation_keys.values())
        azimuth = angle * (("rotate_right" in active) - ("rotate_left" in active))
        elevation = angle * (("rotate_up" in active) - ("rotate_down" in active))
        roll = angle * (
            ("spin_clockwise" in active) - ("spin_counterclockwise" in active)
        )
        if azimuth:
            self.mesh_view.camera.Azimuth(azimuth)
        if elevation:
            self.mesh_view.camera.Elevation(elevation)
            self.mesh_view.camera.OrthogonalizeViewUp()
        if roll:
            self.mesh_view.camera.Roll(roll)
        if azimuth or elevation or roll:
            self.mesh_view.render()
            self._view_state_changed()

    def _rotate_mesh(
        self, *, azimuth: float = 0, elevation: float = 0, roll: float = 0
    ) -> None:
        if azimuth:
            self.mesh_view.camera.Azimuth(azimuth)
        if elevation:
            self.mesh_view.camera.Elevation(elevation)
            self.mesh_view.camera.OrthogonalizeViewUp()
        if roll:
            self.mesh_view.camera.Roll(roll)
        self.mesh_view.render()
        self._view_state_changed()

    def _new_project(self, _checked: bool = False) -> None:
        if not self._maybe_save_changes():
            return
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select BOP-style dataset directory",
            str(self.dataset.root),
        )
        if not directory:
            return
        try:
            dataset, messages = _load_dataset(directory)
        except (FileNotFoundError, ValueError) as error:
            QMessageBox.critical(self, "Cannot create project", str(error))
            return
        self._install_dataset(dataset, messages)
        self._project_path = None
        self._set_dirty(False)
        self.statusBar().showMessage(f"New project created for {dataset.root}", 6000)

    def _load_project_dialog(self, _checked: bool = False) -> None:
        if not self._maybe_save_changes():
            return
        start = self._project_path.parent if self._project_path else Path.cwd()
        filename, _filter = QFileDialog.getOpenFileName(
            self,
            "Load annotation project",
            str(start),
            "Procrustes Annotator projects (*.project);;JSON files (*.json);;All files (*)",
        )
        if filename:
            self._load_project_file(Path(filename))

    def _load_project_file(self, path: Path) -> bool:
        try:
            document = _read_project(path)
            dataset_path = _resolve_project_dataset_path(document, path)
            dataset, messages = _load_dataset(dataset_path)
            states = _deserialize_annotation_states(document, dataset)
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
            QMessageBox.critical(self, "Cannot load project", str(error))
            return False

        self._install_dataset(dataset, messages)
        self.annotation_states = states
        self._project_path = path.resolve()
        self._restore_project(document)
        self._set_dirty(False)
        self.statusBar().showMessage(f"Loaded project {self._project_path}", 6000)
        return True

    def _install_dataset(self, dataset: DataSet, messages: list[str]) -> None:
        """Replace the active dataset after it has been fully validated."""
        self._restoring_project = True
        try:
            self._hide_overlay()
            self._active_rotation_keys.clear()
            self._rotation_timer.stop()
            self.dataset = dataset
            self._startup_warnings = messages
            self.annotation_states = {
                object_id: ObjectAnnotationState() for object_id in dataset.models
            }
            self.frame_setups = {pose.frame_id: "setup_0" for pose in dataset.poses["0"]}
            self.setup_reports = {}
            self._icp_method = "silhouette"
            self._silhouette_weight = 0.10
            self.current_pose = None
            self.current_object_id = None
            self._mesh = None
            self._mesh_texture = None
            self._active_view = "model"
            self._persistent_navigation = False
            self._temporary_navigation_held = False
            self.mode_button.setChecked(False)
            self._populate_controls()
            self._load_current_selection()
            self.image_view.reset_view()
        finally:
            self._restoring_project = False

    def _save_project(self, _checked: bool = False) -> bool:
        if self._project_path is None:
            return self._save_project_as()
        try:
            document = self._project_document()
            _deserialize_annotation_states(document, self.dataset)
            _deserialize_frame_setups(document, self.dataset)
            _write_json_atomic(self._project_path, document)
            if _read_project(self._project_path) != document:
                raise ValueError("Post-write project verification failed.")
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
            QMessageBox.critical(self, "Cannot save project", str(error))
            return False
        self._set_dirty(False)
        counts = _annotation_counts(self.annotation_states)
        self.statusBar().showMessage(
            f"Saved {self._project_path} — {counts['model_points']} landmarks, "
            f"{counts['observations']} RGB points, {counts['solved']} solved, "
            f"{counts['inferred']} inferred, {counts['deferred']} deferred (verified).",
            8000,
        )
        return True

    def _save_project_as(self, _checked: bool = False) -> bool:
        start = self._project_path or (Path.cwd() / "annotation.project")
        filename, _filter = QFileDialog.getSaveFileName(
            self,
            "Save annotation project",
            str(start),
            "Procrustes Annotator project (*.project)",
        )
        if not filename:
            return False
        path = Path(filename)
        if path.suffix.lower() != ".project":
            path = path.with_suffix(".project")
        old_path = self._project_path
        self._project_path = path.resolve()
        if not self._save_project():
            self._project_path = old_path
            self._update_window_title()
            return False
        return True

    def _project_document(self) -> dict[str, object]:
        camera = self.mesh_view.camera
        relative_dataset_path = None
        if self._project_path is not None:
            relative_dataset_path = os.path.relpath(
                self.dataset.root, self._project_path.parent
            )
        current_frame = self.current_pose.frame_id if self.current_pose else None
        return {
            "format": PROJECT_FORMAT,
            "version": PROJECT_VERSION,
            "dataset_path": str(self.dataset.root),
            "dataset_path_relative": relative_dataset_path,
            "current": {
                "frame_id": current_frame,
                "object_id": self.current_object_id,
                "active_view": self._active_view,
                "persistent_navigation": self._persistent_navigation,
            },
            "annotations": _serialize_annotation_states(self.annotation_states),
            "frame_setups": {str(frame): setup for frame, setup in sorted(self.frame_setups.items())},
            "setup_reports": self.setup_reports,
            "icp_settings": {"method": self._icp_method, "silhouette_weight": self._silhouette_weight},
            "view": {
                "show_invalid_depth": self._show_invalid_depth,
                "model_camera": {
                    "position": list(camera.GetPosition()),
                    "focal_point": list(camera.GetFocalPoint()),
                    "view_up": list(camera.GetViewUp()),
                    "clipping_range": list(camera.GetClippingRange()),
                    "parallel_scale": camera.GetParallelScale(),
                    "view_angle": camera.GetViewAngle(),
                    "parallel_projection": bool(camera.GetParallelProjection()),
                },
                "image": self.image_view.view_state(),
                "splitter_sizes": self.splitter.sizes(),
                "window_geometry": bytes(self.saveGeometry().toBase64()).decode("ascii"),
            },
        }

    def _restore_project(self, document: dict[str, object]) -> None:
        self._restoring_project = True
        try:
            self.annotation_states = _deserialize_annotation_states(document, self.dataset)
            self.frame_setups = _deserialize_frame_setups(document, self.dataset)
            self.setup_reports = _mapping(document.get("setup_reports", {}), "setup reports")
            settings = _mapping(document.get("icp_settings", {}), "ICP settings")
            method = str(settings.get("method", "silhouette"))
            weight = float(settings.get("silhouette_weight", 0.10))
            self._icp_method = method if method in ("silhouette", "colored") else "silhouette"
            self._silhouette_weight = weight if np.isfinite(weight) and weight >= 0 else 0.10
            current = _mapping(document.get("current"), "current")
            frame_id = current.get("frame_id")
            object_id = current.get("object_id")
            self.frame_combo.blockSignals(True)
            frame_index = self.frame_combo.findData(frame_id)
            self.frame_combo.setCurrentIndex(max(0, frame_index))
            self.frame_combo.blockSignals(False)
            self._populate_object_combo(preferred_object_id=object_id)
            self.current_pose = None
            self.current_object_id = None
            self._mesh = None
            self._mesh_texture = None
            self._active_view = str(current.get("active_view", "model"))
            self._persistent_navigation = bool(
                current.get("persistent_navigation", False)
            )
            self.mode_button.setChecked(self._persistent_navigation)
            self._load_current_selection()

            view = _mapping(document.get("view", {}), "view")
            self._show_invalid_depth = bool(view.get("show_invalid_depth", True))
            self.invalid_depth_action.blockSignals(True)
            self.invalid_depth_action.setChecked(self._show_invalid_depth)
            self.invalid_depth_action.blockSignals(False)
            self._refresh_invalid_depth_overlay()
            camera_state = _mapping(view.get("model_camera", {}), "model camera")
            camera = self.mesh_view.camera
            if camera_state:
                camera.SetPosition(*_float_vector(camera_state["position"], 3, "camera position"))
                camera.SetFocalPoint(
                    *_float_vector(camera_state["focal_point"], 3, "camera focal point")
                )
                camera.SetViewUp(*_float_vector(camera_state["view_up"], 3, "camera view up"))
                camera.SetClippingRange(
                    *_float_vector(camera_state["clipping_range"], 2, "camera clipping range")
                )
                camera.SetParallelScale(float(camera_state["parallel_scale"]))
                camera.SetViewAngle(float(camera_state["view_angle"]))
                camera.SetParallelProjection(bool(camera_state["parallel_projection"]))
                self.mesh_view.render()

            image_state = _mapping(view.get("image", {}), "image view")
            self._pending_image_view_state = dict(image_state) if image_state else None
            sizes = view.get("splitter_sizes")
            if isinstance(sizes, list) and len(sizes) == 2:
                self._pending_splitter_sizes = [int(value) for value in sizes]
            geometry = view.get("window_geometry")
            if isinstance(geometry, str) and geometry:
                self.restoreGeometry(QByteArray.fromBase64(geometry.encode("ascii")))
            if self.isVisible():
                QTimer.singleShot(0, self._apply_pending_view_state)
            self._refresh()
        finally:
            self._restoring_project = False

    def _apply_pending_view_state(self) -> None:
        was_restoring = self._restoring_project
        self._restoring_project = True
        try:
            if self._pending_splitter_sizes is not None:
                self.splitter.setSizes(self._pending_splitter_sizes)
                self._pending_splitter_sizes = None
            if self._pending_image_view_state is not None:
                self.image_view.restore_view_state(self._pending_image_view_state)
                self._pending_image_view_state = None
        finally:
            self._restoring_project = was_restoring

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API name
        super().showEvent(event)
        if (
            self._pending_image_view_state is not None
            or self._pending_splitter_sizes is not None
        ):
            QTimer.singleShot(0, self._apply_pending_view_state)

    def _export_solved_poses(self, _checked: bool = False) -> bool:
        try:
            export = _serialize_solved_poses(self.annotation_states)
        except ValueError as error:
            QMessageBox.warning(self, "Review poses before export", str(error))
            return False
        if not export:
            QMessageBox.information(
                self,
                "Nothing to export",
                "Solve or defer at least one pose before exporting.",
            )
            return False
        if self._project_path:
            default = self._project_path.with_name(
                f"{self._project_path.stem}_poses.json"
            )
        else:
            default = Path.cwd() / "solved_poses.json"
        filename, _filter = QFileDialog.getSaveFileName(
            self,
            "Export solved poses",
            str(default),
            "JSON files (*.json)",
        )
        if not filename:
            return False
        path = Path(filename)
        if not path.suffix:
            path = path.with_suffix(".json")
        try:
            _write_json_atomic(path, export)
            with path.expanduser().resolve().open(encoding="utf-8") as file:
                if json.load(file) != export:
                    raise ValueError("Post-write export verification failed.")
            mask_dir = path.with_name(path.stem + "_masks")
            approved_masks = [
                (frame, object_id, review.mask)
                for object_id, state in self.annotation_states.items()
                for frame, review in state.masks.items()
            ]
            if approved_masks:
                mask_dir.mkdir(parents=True, exist_ok=True)
                for frame, object_id, mask in approved_masks:
                    mask_path = mask_dir / f"{frame:06d}_obj_{object_id:06d}.png"
                    if not cv2.imwrite(str(mask_path), np.asarray(mask, np.uint8) * 255):
                        raise OSError(f"Could not save approved mask {mask_path}")
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            QMessageBox.critical(self, "Cannot export poses", str(error))
            return False
        counts = _annotation_counts(self.annotation_states)
        self.statusBar().showMessage(
            f"Exported {counts['solved']} manual, {counts['inferred']} inferred, and {counts['deferred']} deferred "
            f"pose records to {path} (verified).",
            8000,
        )
        return True

    def _maybe_save_changes(self) -> bool:
        if not self._dirty:
            return True
        choice = QMessageBox.warning(
            self,
            "Unsaved project changes",
            "Save changes to the current project?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if choice == QMessageBox.StandardButton.Cancel:
            return False
        if choice == QMessageBox.StandardButton.Save:
            return self._save_project()
        return True

    def _configure_shortcuts(self) -> None:
        dialog = ShortcutDialog(self.actions, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        shortcuts = dialog.shortcuts()
        nonempty = [shortcut for shortcut in shortcuts.values() if shortcut]
        if len(nonempty) != len(set(nonempty)):
            QMessageBox.warning(self, "Duplicate shortcuts", "Each action must use a unique shortcut.")
            return
        for name, shortcut in shortcuts.items():
            action = self.actions[name]
            if name in RAW_HOLD_ACTIONS:
                action.setProperty("raw_shortcut", shortcut)
                _update_raw_action_label(action)
            else:
                action.setShortcut(QKeySequence(shortcut))
            self._settings.setValue(f"shortcuts/{name}", shortcut)

    def _show_help(self) -> None:
        ShortcutHelpDialog(self.actions, self).exec()

    def _refresh(self) -> None:
        self._refresh_table()
        self._refresh_image_markers()
        self._refresh_mesh_markers()
        self._refresh_overlay()

        total = len(self.model_points)
        selected = len(self.frame_observations)
        deferred = bool(
            self.current_pose
            and self.current_pose.frame_id in self.state.deferred_frames
        )
        frame_id = self.current_pose.frame_id if self.current_pose else -1
        self.batch_review_button.setEnabled(any(
            state.icp_candidates or state.batch_errors for state in self.annotation_states.values()
        ))
        self.progress_label.setText(f"{selected} / {total}")
        self.solve_button.setEnabled(not deferred and total >= 3 and selected == total)
        self.defer_button.blockSignals(True)
        self.defer_button.setChecked(deferred)
        self.defer_button.setText(
            "Deferred — restore pose" if deferred else "Defer pose"
        )
        self.defer_button.blockSignals(False)
        inferred = bool(self.current_pose and self.current_pose.frame_id in self.state.inferred)
        if inferred:
            self.instruction.setText("Inferred from setup. Hold Tab to inspect; add landmarks to replace it with a manual pose.")
        elif deferred:
            self.instruction.setText(
                "Pose deferred/unannotatable. Press the defer shortcut again to restore it."
            )
        elif self._active_view == "image":
            if self.model_points_are_inherited and selected == 0:
                self.instruction.setText(
                    f"Using {total} last-used model landmarks. First RGB click creates this scene's copy."
                )
            elif selected < total:
                self.instruction.setText(
                    f"RGB list: {selected} / {total}. Left-click to add, right-click to undo."
                )
            elif total:
                self.instruction.setText("RGB list complete. Hold the overlay key to solve and inspect.")
            else:
                self.instruction.setText("Add model points before adding RGB observations.")
        else:
            source = "Last-used suggestion" if self.model_points_are_inherited else "Scene model list"
            self.instruction.setText(
                f"{source}: {total} landmarks. Editing creates/updates this scene's copy."
            )

        solution = (
            (self.state.icp_candidates.get(frame_id)
             if self.state.icp_decisions.get(frame_id) == "keep" else None)
            or self.state.solutions.get(frame_id)
            or self.state.inferred.get(frame_id)
            if self.current_pose else None
        )
        if solution is None:
            self.rmse_label.setText("—")
            self.translation_label.setText("—")
        else:
            self.rmse_label.setText(
                "Inferred (setup)" if inferred else
                f"Manual {solution.rmse * 1000.0:.2f} mm; ICP kept"
                if self.state.icp_decisions.get(frame_id) == "keep" else
                f"{solution.rmse * 1000.0:.2f} mm"
            )
            t = solution.translation
            self.translation_label.setText(f"[{t.x:.4f}, {t.y:.4f}, {t.z:.4f}] m")

    def _refresh_table(self) -> None:
        solution = (
            self.state.solutions.get(self.current_pose.frame_id) if self.current_pose else None
        )
        self.table.setRowCount(len(self.model_points))
        next_index = len(self.frame_observations)
        for row, model_point in enumerate(self.model_points):
            observation = self.frame_observations[row] if row < len(self.frame_observations) else None
            marker_color = QColor(COLORS[row % len(COLORS)])
            values = [
                f"▶ {row + 1}" if row == next_index else str(row + 1),
                f"{model_point.x:.4f}",
                f"{model_point.y:.4f}",
                f"{model_point.z:.4f}",
                f"{observation.pixel.x:.1f}" if observation else "—",
                f"{observation.pixel.y:.1f}" if observation else "—",
                f"{observation.camera.z:.4f}" if observation else "—",
                (
                    f"{solution.residuals_mm[row]:.2f}"
                    if solution is not None and row < len(solution.residuals_mm)
                    else "—"
                ),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setBackground(marker_color)
                    brightness = (
                        0.299 * marker_color.red()
                        + 0.587 * marker_color.green()
                        + 0.114 * marker_color.blue()
                    )
                    item.setForeground(
                        QColor("black") if brightness > 150 else QColor("white")
                    )
                if row == next_index:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                self.table.setItem(row, column, item)

    def _refresh_image_markers(self) -> None:
        markers = []
        for index, observation in enumerate(self.frame_observations, start=1):
            markers.append(
                (
                    index,
                    observation.pixel.x,
                    observation.pixel.y,
                    QColor(COLORS[(index - 1) % len(COLORS)]),
                )
            )
        self.image_view.set_markers(markers)

    def _refresh_mesh_markers(self) -> None:
        for actor_name in (
            "model-landmarks",
            "model-landmark-labels",
            "model-next-landmark",
        ):
            self.mesh_view.remove_actor(actor_name, reset_camera=False)
        if self.model_points:
            points_mm = np.stack([point.as_array() * 1000.0 for point in self.model_points])
            next_index = len(self.frame_observations)
            if next_index < len(points_mm):
                next_point = pv.PolyData(points_mm[next_index : next_index + 1])
                self.mesh_view.add_mesh(
                    next_point,
                    name="model-next-landmark",
                    color="white",
                    point_size=28,
                    opacity=0.8,
                    render_points_as_spheres=True,
                    pickable=False,
                )
            colors = np.array(
                [QColor(COLORS[index % len(COLORS)]).getRgb()[:3] for index in range(len(points_mm))],
                dtype=np.uint8,
            )
            cloud = pv.PolyData(points_mm)
            cloud["marker_colors"] = colors
            self.mesh_view.add_mesh(
                cloud,
                name="model-landmarks",
                scalars="marker_colors",
                rgb=True,
                point_size=16,
                render_points_as_spheres=True,
                pickable=False,
            )
            labels = [
                f"▶ {index + 1}" if index == next_index else str(index + 1)
                for index in range(len(points_mm))
            ]
            self.mesh_view.add_point_labels(
                points_mm,
                labels,
                # PyVista appends ``-labels`` to this base actor name.
                name="model-landmark",
                show_points=False,
                font_size=18,
                text_color="white",
                shape="rounded_rect",
                shape_color="#171a21",
                shape_opacity=0.72,
                always_visible=True,
                pickable=False,
                reset_camera=False,
            )
        self.mesh_view.render()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API name
        if self._review_worker is not None and self._review_worker.isRunning():
            self.statusBar().showMessage("Wait for SAM/ICP to finish before closing.", 5000)
            event.ignore()
            return
        if not self._maybe_save_changes():
            event.ignore()
            return
        QApplication.instance().removeEventFilter(self)
        self.mesh_view.close()
        super().closeEvent(event)


def _encode_mask(mask: np.ndarray) -> str:
    success, encoded = cv2.imencode(".png", np.asarray(mask, np.uint8) * 255)
    if not success:
        raise ValueError("Could not encode reviewed SAM mask.")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _decode_mask(encoded: str) -> np.ndarray:
    raw = base64.b64decode(encoded, validate=True)
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None or image.ndim != 2 or not np.any(image):
        raise ValueError("Reviewed SAM mask is empty or invalid.")
    return image > 0


def _serialize_annotation_states(
    states: dict[int, ObjectAnnotationState],
) -> dict[str, object]:
    annotations: dict[str, object] = {}
    for object_id, state in sorted(states.items()):
        annotations[str(object_id)] = {
            "last_model_points_m": [
                point.as_array().tolist() for point in state.last_model_points
            ],
            "frame_model_points_m": {
                str(frame_id): [point.as_array().tolist() for point in points]
                for frame_id, points in sorted(state.frame_model_points.items())
                if points
            },
            "deferred_frames": sorted(state.deferred_frames),
            "frame_points": {
                str(frame_id): [
                    {
                        "pixel": observation.pixel.as_array().tolist(),
                        "camera_m": observation.camera.as_array().tolist(),
                    }
                    for observation in observations
                ]
                for frame_id, observations in sorted(state.frame_points.items())
            },
            "solutions": {
                str(frame_id): {
                    "rotation_m2c": solution.rotation.reshape(-1).tolist(),
                    "translation_m2c_m": solution.translation.as_array().tolist(),
                    "rmsd_m": solution.rmse,
                    "residuals_mm": solution.residuals_mm.tolist(),
                }
                for frame_id, solution in sorted(state.solutions.items())
            },
            "inferred": {
                str(frame_id): {
                    "rotation_m2c": solution.rotation.reshape(-1).tolist(),
                    "translation_m2c_m": solution.translation.as_array().tolist(),
                }
                for frame_id, solution in sorted(state.inferred.items())
            },
            "reviewed_masks": {
                str(frame_id): {
                    "png_base64": _encode_mask(review.mask),
                    "score": review.score,
                    "box_xyxy": review.box_xyxy,
                    "box_padding": review.box_padding,
                    "positive_xy": [list(point) for point in review.positive_xy],
                    "negative_xy": [list(point) for point in review.negative_xy],
                    "model_id": review.model_id,
                }
                for frame_id, review in sorted(state.masks.items())
            },
            "icp_candidates": {
                str(frame_id): {
                    "rotation_m2c": solution.rotation.reshape(-1).tolist(),
                    "translation_m2c_m": solution.translation.as_array().tolist(),
                    "details": state.icp_info.get(frame_id, {}),
                    "decision": state.icp_decisions.get(frame_id, "pending"),
                }
                for frame_id, solution in sorted(state.icp_candidates.items())
            },
            "batch_metrics": {str(frame): metrics for frame, metrics in sorted(state.batch_metrics.items())},
            "batch_errors": {str(frame): message for frame, message in sorted(state.batch_errors.items())},
            "review_prompts": {str(frame): prompts for frame, prompts in sorted(state.review_prompts.items())},
        }
    return annotations


def _deserialize_annotation_states(
    document: dict[str, object], dataset: DataSet
) -> dict[int, ObjectAnnotationState]:
    states = {
        object_id: ObjectAnnotationState() for object_id in dataset.models
    }
    annotations = _mapping(document.get("annotations", {}), "annotations")
    valid_frames = {pose.frame_id for pose in dataset.poses["0"]}
    for raw_object_id, raw_state in annotations.items():
        object_id = int(raw_object_id)
        if object_id not in states:
            raise ValueError(f"Project references missing model obj_{object_id:06}.")
        state_data = _mapping(raw_state, f"object {object_id}")
        state = states[object_id]
        last_model_values = state_data.get("last_model_points_m", [])
        if not isinstance(last_model_values, list):
            raise ValueError(
                f"Object {object_id} last_model_points_m must be a list."
            )
        state.last_model_points = [
            Vector3.from_array(_float_vector(value, 3, "last model point"))
            for value in last_model_values
        ]
        frame_model_points = _mapping(
            state_data.get("frame_model_points_m", {}),
            f"object {object_id} frame model points",
        )
        for raw_frame_id, model_values in frame_model_points.items():
            frame_id = int(raw_frame_id)
            if frame_id not in valid_frames:
                raise ValueError(f"Project references missing frame {frame_id}.")
            if not isinstance(model_values, list):
                raise ValueError(f"Frame {frame_id} model points must be a list.")
            points = [
                Vector3.from_array(_float_vector(value, 3, "model point"))
                for value in model_values
            ]
            if points:
                state.frame_model_points[frame_id] = points
        deferred_values = state_data.get("deferred_frames", [])
        if not isinstance(deferred_values, list):
            raise ValueError(f"Object {object_id} deferred_frames must be a list.")
        state.deferred_frames = {int(value) for value in deferred_values}
        missing_deferred = state.deferred_frames - valid_frames
        if missing_deferred:
            raise ValueError(
                f"Project references missing deferred frame {min(missing_deferred)}."
            )

        frame_points = _mapping(
            state_data.get("frame_points", {}), f"object {object_id} frame points"
        )
        for raw_frame_id, raw_observations in frame_points.items():
            frame_id = int(raw_frame_id)
            if frame_id not in valid_frames:
                raise ValueError(f"Project references missing frame {frame_id}.")
            if not isinstance(raw_observations, list):
                raise ValueError(f"Frame {frame_id} observations must be a list.")
            observations: list[ImageSelection] = []
            for raw_observation in raw_observations:
                observation = _mapping(raw_observation, "image observation")
                pixel = _float_vector(observation["pixel"], 2, "pixel")
                camera = _float_vector(observation["camera_m"], 3, "camera point")
                observations.append(
                    ImageSelection(
                        pixel=Vector2(*pixel),
                        camera=Vector3.from_array(camera),
                    )
                )
            if len(observations) > len(state.frame_model_points.get(frame_id, [])):
                raise ValueError(
                    f"Frame {frame_id} has more observations than model landmarks."
                )
            if observations:
                state.frame_points[frame_id] = observations

        solutions = _mapping(
            state_data.get("solutions", {}), f"object {object_id} solutions"
        )
        for raw_frame_id, raw_solution in solutions.items():
            frame_id = int(raw_frame_id)
            if frame_id not in valid_frames:
                raise ValueError(f"Project references missing frame {frame_id}.")
            solution = _mapping(raw_solution, "pose solution")
            rotation = np.asarray(
                _float_vector(solution["rotation_m2c"], 9, "rotation"),
                dtype=np.float64,
            ).reshape(3, 3)
            translation = Vector3.from_array(
                _float_vector(solution["translation_m2c_m"], 3, "translation")
            )
            rmsd = float(solution["rmsd_m"])
            residuals = np.asarray(solution["residuals_mm"], dtype=np.float64).reshape(-1)
            if not np.isfinite(rmsd) or rmsd < 0 or not np.isfinite(residuals).all():
                raise ValueError("Pose RMSD and residuals must be finite and non-negative.")
            if (residuals < 0).any() or len(residuals) != len(
                state.frame_model_points.get(frame_id, [])
            ):
                raise ValueError(
                    f"Frame {frame_id} residual count must match its model landmarks."
                )
            state.solutions[frame_id] = PoseSolution(
                rotation=rotation,
                translation=translation,
                rmse=rmsd,
                residuals_mm=residuals,
            )
        inferred = _mapping(state_data.get("inferred", {}), f"object {object_id} inferred poses")
        for raw_frame_id, raw_pose in inferred.items():
            frame_id = int(raw_frame_id)
            if frame_id not in valid_frames or frame_id in state.solutions:
                raise ValueError(f"Invalid inferred pose for object {object_id}, frame {frame_id}.")
            pose = _mapping(raw_pose, "inferred pose")
            rigid = RigidPose(
                np.asarray(_float_vector(pose["rotation_m2c"], 9, "inferred rotation")).reshape(3, 3),
                np.asarray(_float_vector(pose["translation_m2c_m"], 3, "inferred translation")),
            )
            state.inferred[frame_id] = PoseSolution(
                rigid.rotation, Vector3.from_array(rigid.translation), 0.0, np.empty(0)
            )
        masks = _mapping(state_data.get("reviewed_masks", {}), f"object {object_id} reviewed masks")
        for raw_frame, raw_review in masks.items():
            frame = int(raw_frame)
            if frame not in state.solutions:
                raise ValueError(f"Object {object_id} frame {frame} has a mask without a manual pose.")
            review = _mapping(raw_review, "reviewed mask")
            mask = _decode_mask(str(review["png_base64"]))
            if mask.shape != dataset[frame].rgb.shape[:2]:
                raise ValueError(f"Object {object_id} frame {frame} mask dimensions differ from RGB.")
            score = float(review["score"])
            padding = float(review.get("box_padding", 0.25))
            if not np.isfinite(score) or not np.isfinite(padding) or not 0.0 <= padding <= 0.75:
                raise ValueError(f"Object {object_id} frame {frame} has invalid SAM review values.")
            state.masks[frame] = MaskReview(
                mask=mask,
                score=score,
                box_xyxy=_float_vector(review["box_xyxy"], 4, "mask box"),
                positive_xy=[tuple(_float_vector(point, 2, "positive point")) for point in review.get("positive_xy", [])],
                negative_xy=[tuple(_float_vector(point, 2, "negative point")) for point in review.get("negative_xy", [])],
                box_padding=padding,
                model_id=str(review.get("model_id", SAM_MODEL_ID)),
            )
        candidates = _mapping(state_data.get("icp_candidates", {}), f"object {object_id} ICP candidates")
        for raw_frame, raw_candidate in candidates.items():
            frame = int(raw_frame)
            if frame not in state.masks:
                raise ValueError(f"Object {object_id} frame {frame} has ICP without an approved mask.")
            candidate = _mapping(raw_candidate, "ICP candidate")
            rigid = RigidPose(
                np.asarray(_float_vector(candidate["rotation_m2c"], 9, "ICP rotation")).reshape(3, 3),
                np.asarray(_float_vector(candidate["translation_m2c_m"], 3, "ICP translation")),
            )
            state.icp_candidates[frame] = PoseSolution(
                rigid.rotation, Vector3.from_array(rigid.translation), state.solutions[frame].rmse, np.empty(0)
            )
            state.icp_info[frame] = _mapping(candidate.get("details", {}), "ICP details")
            decision = str(candidate.get("decision", "pending"))
            if decision not in ("pending", "keep", "revert"):
                raise ValueError(f"Invalid ICP decision {decision!r}.")
            state.icp_decisions[frame] = decision
        for raw_frame, raw_metrics in _mapping(state_data.get("batch_metrics", {}), "batch metrics").items():
            frame = int(raw_frame)
            if frame not in state.solutions:
                raise ValueError(f"Object {object_id} frame {frame} has metrics without a manual pose.")
            metrics = _mapping(raw_metrics, "batch metrics")
            state.batch_metrics[frame] = {
                key: None if value is None else float(value) for key, value in metrics.items()
            }
        for raw_frame, message in _mapping(state_data.get("batch_errors", {}), "batch errors").items():
            frame = int(raw_frame)
            if frame not in state.solutions:
                raise ValueError(f"Object {object_id} frame {frame} has a batch error without a manual pose.")
            state.batch_errors[frame] = str(message)
        for raw_frame, raw_prompts in _mapping(state_data.get("review_prompts", {}), "review prompts").items():
            frame = int(raw_frame)
            if frame not in state.solutions:
                raise ValueError(f"Object {object_id} frame {frame} has prompts without a manual pose.")
            prompts = _mapping(raw_prompts, "review prompts")
            state.review_prompts[frame] = {
                "extra_positive_xy": [list(_float_vector(point, 2, "positive prompt")) for point in prompts.get("extra_positive_xy", [])],
                "extra_negative_xy": [list(_float_vector(point, 2, "negative prompt")) for point in prompts.get("extra_negative_xy", [])],
                "box_padding": float(prompts.get("box_padding", 0.25)),
            }
        overlap = state.deferred_frames & state.solutions.keys()
        if overlap:
            raise ValueError(
                f"Frame {min(overlap)} cannot be both solved and deferred."
            )
    return states


def _deserialize_frame_setups(document: dict[str, object], dataset: DataSet) -> dict[int, str]:
    frame_ids = {pose.frame_id for pose in dataset.poses["0"]}
    result = {frame: "setup_0" for frame in frame_ids}
    for raw_frame, raw_setup in _mapping(document.get("frame_setups", {}), "frame setups").items():
        frame = int(raw_frame)
        if frame not in frame_ids or not isinstance(raw_setup, str) or not raw_setup.strip():
            raise ValueError(f"Invalid setup assignment for frame {raw_frame}.")
        result[frame] = raw_setup.strip()
    return result


def _accepted_setup_observations(
    states: dict[int, ObjectAnnotationState], frame_ids: list[int]
) -> tuple[dict[tuple[int, int], RigidPose], list[str]]:
    """Every manual Procrustes solution is a target; ICP is used only if kept."""
    frames = set(frame_ids)
    observations: dict[tuple[int, int], RigidPose] = {}
    incomplete: list[str] = []
    for object_id, state in sorted(states.items()):
        for frame in sorted(state.solutions.keys() & frames):
            manual = state.solutions[frame]
            candidate = state.icp_candidates.get(frame)
            decision = state.icp_decisions.get(frame)
            if candidate is not None and decision not in ("keep", "revert"):
                incomplete.append(f"frame {frame}/obj {object_id}: review ICP proposal")
                continue
            accepted = candidate if decision == "keep" and candidate is not None else manual
            observations[(frame, object_id)] = RigidPose(
                accepted.rotation, accepted.translation.as_array(), manual.rmse
            )
    return observations, incomplete


def _serialize_solved_poses(
    states: dict[int, ObjectAnnotationState],
) -> dict[str, list[dict[str, object]]]:
    """Export human-accepted poses plus setup-inferred and deferred records."""
    frames: dict[int, list[dict[str, object]]] = {}
    solved_frames: set[int] = set()
    for object_id, state in sorted(states.items()):
        for frame_id, solution in sorted(state.solutions.items()):
            decision = state.icp_decisions.get(frame_id)
            candidate = state.icp_candidates.get(frame_id)
            if candidate is not None and decision not in ("keep", "revert"):
                raise ValueError(f"Frame {frame_id}/object {object_id} has an unreviewed ICP proposal.")
            accepted = candidate if decision == "keep" else solution
            solved_frames.add(frame_id)
            record = {
                "obj_id": object_id,
                "annotation_status": "solved",
                "annotation_source": "reviewed_silhouette_icp" if decision == "keep" else "manual_procrustes",
                "cam_R_m2c": accepted.rotation.reshape(-1).tolist(),
                "cam_t_m2c": (accepted.translation.as_array() * 1000.0).tolist(),
                "annotation_rmsd_mm": solution.rmse * 1000.0,
                "annotation_residuals_mm": solution.residuals_mm.tolist(),
            }
            if candidate is not None:
                record["icp_review_decision"] = decision
                record["manual_cam_R_m2c"] = solution.rotation.reshape(-1).tolist()
                record["manual_cam_t_m2c"] = (solution.translation.as_array() * 1000.0).tolist()
                record["icp_proposal"] = state.icp_info.get(frame_id, {})
            if frame_id in state.batch_metrics:
                record["depth_rmse"] = dict(state.batch_metrics[frame_id])
            frames.setdefault(frame_id, []).append(record)
        for frame_id, solution in sorted(state.inferred.items()):
            solved_frames.add(frame_id)
            frames.setdefault(frame_id, []).append(
                {
                    "obj_id": object_id,
                    "annotation_status": "solved",
                    "annotation_source": "inferred_setup",
                    "cam_R_m2c": solution.rotation.reshape(-1).tolist(),
                    "cam_t_m2c": (solution.translation.as_array() * 1000.0).tolist(),
                }
            )
        for frame_id in sorted(state.deferred_frames):
            if frame_id in state.solutions or frame_id in state.inferred:
                continue
            frames.setdefault(frame_id, []).append(
                {
                    "obj_id": object_id,
                    "annotation_status": "deferred",
                    "deferred_reason": "unannotatable",
                }
            )
    # Put solved records first so a large deferred block cannot make a valid
    # export look as though it contains only deferrals when inspected by eye.
    ordered_frames = sorted(
        frames,
        key=lambda frame_id: (frame_id not in solved_frames, frame_id),
    )
    return {str(frame_id): frames[frame_id] for frame_id in ordered_frames}


def _annotation_counts(
    states: dict[int, ObjectAnnotationState],
) -> dict[str, int]:
    return {
        "model_points": sum(
            len(points)
            for state in states.values()
            for points in state.frame_model_points.values()
        ),
        "observations": sum(
            len(observations)
            for state in states.values()
            for observations in state.frame_points.values()
        ),
        "solved": sum(len(state.solutions) for state in states.values()),
        "inferred": sum(len(state.inferred) for state in states.values()),
        "deferred": sum(len(state.deferred_frames - state.inferred.keys()) for state in states.values()),
    }


def _step_annotation_selection(
    frame_index: int,
    object_index: int,
    object_counts: list[int],
    *,
    direction: int,
) -> tuple[int, int]:
    """Step through objects within frames, clamping at workflow endpoints."""
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1.")
    if not object_counts or any(count < 1 for count in object_counts):
        raise ValueError("Every annotation frame must contain at least one object.")
    if not 0 <= frame_index < len(object_counts):
        raise ValueError("frame_index is outside the annotation workflow.")
    if not 0 <= object_index < object_counts[frame_index]:
        raise ValueError("object_index is outside the current frame.")

    if direction > 0:
        if object_index + 1 < object_counts[frame_index]:
            return frame_index, object_index + 1
        if frame_index + 1 < len(object_counts):
            return frame_index + 1, 0
        return frame_index, object_index

    if object_index > 0:
        return frame_index, object_index - 1
    if frame_index > 0:
        previous_frame = frame_index - 1
        return previous_frame, object_counts[previous_frame] - 1
    return frame_index, object_index


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"Project {label} must be a JSON object.")
    return value


def _float_vector(value: object, length: int, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"Project {label} must contain {length} numbers.")
    numbers = [float(item) for item in value]
    if not np.isfinite(numbers).all():
        raise ValueError(f"Project {label} must contain finite numbers.")
    return numbers


def _read_project(path: Path) -> dict[str, object]:
    with path.expanduser().resolve().open(encoding="utf-8") as file:
        document = json.load(file)
    document = _mapping(document, "root")
    if document.get("format") != PROJECT_FORMAT:
        raise ValueError(f"{path} is not a Procrustes Annotator project.")
    version = document.get("version")
    if version not in (3, 4, 5, PROJECT_VERSION):
        raise ValueError(
            f"Unsupported project version {version!r}; expected 3, 4, 5 or {PROJECT_VERSION}."
        )
    return document


def _resolve_project_dataset_path(document: dict[str, object], project_path: Path) -> Path:
    absolute = Path(str(document["dataset_path"])).expanduser()
    if absolute.is_dir():
        return absolute.resolve()
    relative = document.get("dataset_path_relative")
    if isinstance(relative, str) and relative:
        candidate = (project_path.resolve().parent / relative).resolve()
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Project dataset is unavailable: {absolute}. "
        "Move the dataset back or create a new project with its new location."
    )


def _load_dataset(path: str | Path) -> tuple[DataSet, list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        dataset = DataSet.load(path, depth_preset=2)
    if not dataset.models:
        raise ValueError(f"No obj_*.ply models found below {dataset.root / 'models'}")
    if not len(dataset):
        raise ValueError(f"No complete RGBD frames found below {dataset.root}")
    return dataset, [str(item.message) for item in caught]


def _write_json_atomic(path: Path, value: object) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(value, temporary, indent=2, ensure_ascii=False, allow_nan=False)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _find_mesh_texture_path(mesh_path: str | Path) -> Path | None:
    """Resolve a PLY TextureFile comment or a same-stem image sidecar."""
    path = Path(mesh_path)
    if path.suffix.lower() == ".ply":
        try:
            with path.open("rb") as file:
                for _ in range(256):
                    raw_line = file.readline(4096)
                    if not raw_line:
                        break
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if line == "end_header":
                        break
                    prefix = "comment TextureFile "
                    if line.startswith(prefix):
                        texture_name = line[len(prefix) :].strip()
                        candidate = path.parent / texture_name
                        if texture_name and candidate.is_file():
                            return candidate
        except OSError:
            return None

    for extension in (".png", ".jpg", ".jpeg"):
        candidate = path.with_suffix(extension)
        if candidate.is_file():
            return candidate
    return None


def _project_mesh_overlay(mesh: pv.PolyData, pose: Pose, solution: PoseSolution) -> np.ndarray:
    """Rasterize visible transformed mesh vertices into an RGBA image."""
    height, width, _ = pose.rgb.shape
    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    points = np.asarray(mesh.points, dtype=np.float64) / 1000.0
    camera = points @ solution.rotation.T + solution.translation.as_array()
    valid = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 0)
    camera = camera[valid]
    if not len(camera):
        return overlay

    K = pose.camera_intrinsics
    u = np.rint(K[0, 0] * camera[:, 0] / camera[:, 2] + K[0, 2]).astype(int)
    v = np.rint(K[1, 1] * camera[:, 1] / camera[:, 2] + K[1, 2]).astype(int)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, z = u[inside], v[inside], camera[inside, 2]

    color_name = next(
        (name for name in ("RGBA", "RGB", "rgba", "rgb") if name in mesh.point_data), None
    )
    if color_name:
        colors = np.asarray(mesh.point_data[color_name])[valid][inside, :3].astype(np.uint8)
    else:
        colors = np.tile(np.array([[0, 255, 80]], dtype=np.uint8), (len(u), 1))

    flat = v * width + u
    nearest_depth = np.full(height * width, np.inf, dtype=np.float64)
    np.minimum.at(nearest_depth, flat, z)
    visible = z <= nearest_depth[flat] + 1e-6
    u, v, colors = u[visible], v[visible], colors[visible]
    for dx, dy in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
        px, py = u + dx, v + dy
        keep = (px >= 0) & (px < width) & (py >= 0) & (py < height)
        overlay[py[keep], px[keep], :3] = colors[keep]
        overlay[py[keep], px[keep], 3] = 180
    return overlay


def _action_shortcut(action: QAction) -> QKeySequence:
    raw = action.property("raw_shortcut")
    return QKeySequence(str(raw)) if raw is not None else action.shortcut()


def _update_raw_action_label(action: QAction) -> None:
    base_label = str(action.property("base_label") or action.text().split("\t", 1)[0])
    shortcut = _action_shortcut(action).toString(QKeySequence.SequenceFormat.NativeText)
    action.setText(f"{base_label}\t{shortcut}" if shortcut else base_label)


def _help_html(shortcuts: dict[str, str], *, korean: bool) -> str:
    if korean:
        return f"""
        <h2>RGBD Procrustes 주석 도구</h2>
        <h3>작업 순서</h3>
        <ol>
          <li>왼쪽 창에서 클릭하여 점을 선택합니다.</li>
          <li>Shift를 누르고 드래그/ WASDQE로 모델을 움직일 수 있습니다.</li> 
          <li>오른쪽 창에서 클릭하여 매칭되는 점을 순서대로 선택합니다.</li>
          <li>매칭이 완료된 후 TAB키를 누르면 자동 계산 후 결과를 표시합니다.</li>
          <li>R 키를 누르면 (Annotation 불가) 상태로 표시할 수 있습니다.</li>
          <li>X/C키로 이전/다음 객체로 이동하고, 현재 프레임의 객체가 끝나면 프레임을 바꿉니다.</li>
        </ol>
        <h3>마우스</h3>
        <p><b>왼쪽 클릭:</b> 클릭한 뷰의 점 추가<br>
        <b>오른쪽 클릭:</b> 클릭한 뷰의 마지막 점 삭제<br>
        <b>Ctrl + 오른쪽 클릭:</b> 클릭한 뷰의 모든 점 삭제<br>
        <b>Z 누르기:</b> RGB 확대창 표시 (휠로 배율 변경)<br>
        <b>Shift 누르기:</b> 누르는 동안 탐색 모드 활성화<br>
        상단 버튼으로 주석/탐색 모드를 계속 유지할 수도 있습니다.<br>
        보기 메뉴의 <b>유효하지 않은 깊이 영역 표시</b>로 빨간색 깊이 마스크를 켜거나 끕니다.<br>
        새 프레임은 가장 최근에 사용한 모델 점을 제안으로 표시하며, 첫 편집 또는 RGB 클릭 시 해당 프레임에 복사합니다.</p>
        <h3>단축키</h3>
        {_shortcut_table(shortcuts, korean=True)}
        """
    return """
    <h2>Procrustes Annotator</h2>
    
    """


def _shortcut_table(shortcuts: dict[str, str], *, korean: bool) -> str:
    labels = {
        "rotate_up": ("Rotate mesh up", "메시 위로 회전"),
        "rotate_left": ("Rotate mesh left", "메시 왼쪽 회전"),
        "rotate_down": ("Rotate mesh down", "메시 아래로 회전"),
        "rotate_right": ("Rotate mesh right", "메시 오른쪽 회전"),
        "spin_counterclockwise": ("Spin model counterclockwise", "모델 반시계 방향 회전"),
        "spin_clockwise": ("Spin model clockwise", "모델 시계 방향 회전"),
        "toggle_overlay": ("Hold solved overlay", "정합 오버레이 누르고 보기"),
        "previous_frame": ("Previous object / frame", "이전 객체 / 프레임"),
        "next_frame": ("Next object / frame", "다음 객체 / 프레임"),
        "solve": ("Solve Procrustes", "Procrustes 계산"),
        "defer_pose": ("Defer/unannotatable pose", "주석 불가 포즈 연기"),
        "hold_navigation": ("Hold navigation mode", "탐색 모드 누르고 사용"),
        "hold_magnifier": ("Hold RGB magnifier", "RGB 확대창 누르고 사용"),
    }
    rows = "".join(
        f"<tr><td>{labels[name][1 if korean else 0]}</td><td><b>{shortcut or '—'}</b></td></tr>"
        for name, shortcut in shortcuts.items()
    )
    return f"<table cellspacing='8'>{rows}</table>"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Procrustes Annotator.")
    parser.add_argument(
        "source",
        nargs="?",
        default="dataset",
        help="BOP-style dataset directory or .project file (default: dataset)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    # OpenCV's Qt-enabled wheel can redirect Qt to its own incompatible xcb
    # plugin. This is a PySide6 application, so use PySide6's plugin bundle.
    if "/cv2/qt/plugins" in os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH", ""):
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.path(
            QLibraryInfo.LibraryPath.PluginsPath
        )
    app = QApplication(sys.argv)
    try:
        source = Path(args.source).expanduser()
        if source.is_file():
            project_path = source.resolve()
            project = _read_project(project_path)
            dataset_path = _resolve_project_dataset_path(project, project_path)
            window = AnnotationWindow(
                dataset_path,
                project_path=project_path,
                project_data=project,
            )
        else:
            window = AnnotationWindow(source)
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
        QMessageBox.critical(None, "Cannot start annotator", str(error))
        return 1
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
