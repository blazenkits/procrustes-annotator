"""PySide6 viewer for rough and ICP-refined pose overlays."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyvista as pv
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

try:
    from ..backend.loader import DataSet, Pose
    from ..frontend.image_view import ImageView
    from .icp_scene_view import ICPSceneView
except ImportError:  # pragma: no cover - direct ``python src/tools/...`` use.
    repository_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repository_root))
    from src.backend.loader import DataSet, Pose
    from src.frontend.image_view import ImageView
    from src.tools.icp_scene_view import ICPSceneView


ROUGH_COLOR = (235, 45, 55)
ICP_COLOR = (35, 220, 90)
OVERLAP_COLOR = (255, 210, 35)


class ViewerMode(IntEnum):
    Normal = 1
    Rough = 2
    ICP = 3
    RoughAndICP = 4


@dataclass(frozen=True)
class ReviewPose:
    """Rough and refined model-to-camera transforms for one object/frame."""

    frame_id: int
    object_id: int
    rough_rotation_m2c: np.ndarray
    rough_translation_m2c_m: np.ndarray
    icp_rotation_m2c: np.ndarray
    icp_translation_m2c_m: np.ndarray
    fitness: float
    inlier_rmse_mm: float
    correspondence_count: int


def load_review_poses(path: str | Path) -> list[ReviewPose]:
    """Load refined records from the structure written by the ICP backend."""
    input_path = Path(path).expanduser().resolve()
    with input_path.open(encoding="utf-8") as file:
        document = json.load(file)
    if not isinstance(document, dict):
        raise ValueError(f"Expected a JSON object in {input_path}.")
    return parse_review_poses(document)


def parse_review_poses(document: dict[str, Any]) -> list[ReviewPose]:
    """Parse every solved record containing successful ICP metadata."""
    poses: list[ReviewPose] = []
    for raw_frame_id, records in document.items():
        try:
            frame_id = int(raw_frame_id)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Unexpected non-frame JSON key: {raw_frame_id!r}") from error
        if not isinstance(records, list):
            raise ValueError(f"Frame {frame_id} must contain a list of annotations.")
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"Frame {frame_id} contains a non-object annotation.")
            if record.get("annotation_status", "solved") != "solved":
                continue
            metadata = record.get("icp")
            if not isinstance(metadata, dict) or metadata.get("status") != "refined":
                continue
            object_id = int(record["obj_id"])
            rough_rotation = _finite_array(
                metadata["rough_cam_R_m2c"],
                (3, 3),
                f"frame {frame_id} rough rotation",
            )
            rough_translation = _finite_array(
                metadata["rough_cam_t_m2c"],
                (3,),
                f"frame {frame_id} rough translation",
            ) / 1000.0
            icp_rotation = _finite_array(
                record["cam_R_m2c"], (3, 3), f"frame {frame_id} ICP rotation"
            )
            icp_translation = _finite_array(
                record["cam_t_m2c"], (3,), f"frame {frame_id} ICP translation"
            ) / 1000.0
            fitness = float(metadata["fitness"])
            inlier_rmse_mm = float(metadata["inlier_rmse_mm"])
            correspondence_count = int(metadata["correspondence_count"])
            if not np.isfinite([fitness, inlier_rmse_mm]).all():
                raise ValueError(f"Frame {frame_id} has non-finite ICP metrics.")
            poses.append(
                ReviewPose(
                    frame_id=frame_id,
                    object_id=object_id,
                    rough_rotation_m2c=rough_rotation,
                    rough_translation_m2c_m=rough_translation,
                    icp_rotation_m2c=icp_rotation,
                    icp_translation_m2c_m=icp_translation,
                    fitness=fitness,
                    inlier_rmse_mm=inlier_rmse_mm,
                    correspondence_count=correspondence_count,
                )
            )
    if not poses:
        raise ValueError("The pose file contains no successfully refined ICP records.")
    return sorted(poses, key=lambda pose: (pose.frame_id, pose.object_id))


def _finite_array(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    try:
        array = array.reshape(shape)
    except ValueError as error:
        raise ValueError(f"{label} must have shape {shape}.") from error
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite.")
    return array


def project_mesh_overlay(
    model_points_m: np.ndarray,
    pose: Pose,
    rotation_m2c: np.ndarray,
    translation_m2c_m: np.ndarray,
    color: tuple[int, int, int],
    *,
    alpha: int = 190,
) -> np.ndarray:
    """Project frontmost transformed mesh vertices into a fixed-color overlay."""
    height, width = pose.rgb.shape[:2]
    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    points = np.asarray(model_points_m, dtype=np.float64).reshape(-1, 3)
    rotation = np.asarray(rotation_m2c, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(translation_m2c_m, dtype=np.float64).reshape(3)
    camera = points @ rotation.T + translation
    valid = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 0)
    camera = camera[valid]
    if not len(camera):
        return overlay

    intrinsics = np.asarray(pose.camera_intrinsics, dtype=np.float64).reshape(3, 3)
    u = np.rint(
        intrinsics[0, 0] * camera[:, 0] / camera[:, 2] + intrinsics[0, 2]
    ).astype(int)
    v = np.rint(
        intrinsics[1, 1] * camera[:, 1] / camera[:, 2] + intrinsics[1, 2]
    ).astype(int)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, z = u[inside], v[inside], camera[inside, 2]
    if not len(u):
        return overlay

    flat = v * width + u
    nearest_depth = np.full(height * width, np.inf, dtype=np.float64)
    np.minimum.at(nearest_depth, flat, z)
    visible = z <= nearest_depth[flat] + 1e-6
    u, v = u[visible], v[visible]
    rgba = np.array((*color, alpha), dtype=np.uint8)
    for dx, dy in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
        px, py = u + dx, v + dy
        keep = (px >= 0) & (px < width) & (py >= 0) & (py < height)
        overlay[py[keep], px[keep]] = rgba
    return overlay


def combine_pose_overlays(rough: np.ndarray, refined: np.ndarray) -> np.ndarray:
    """Composite rough/red and ICP/green overlays, using yellow for agreement."""
    if rough.shape != refined.shape or rough.dtype != np.uint8 or refined.dtype != np.uint8:
        raise ValueError("Pose overlays must be matching uint8 RGBA arrays.")
    output = np.zeros_like(rough)
    rough_mask = rough[..., 3] > 0
    refined_mask = refined[..., 3] > 0
    output[rough_mask] = rough[rough_mask]
    output[refined_mask] = refined[refined_mask]
    overlap = rough_mask & refined_mask
    output[overlap, :3] = OVERLAP_COLOR
    output[overlap, 3] = np.maximum(rough[overlap, 3], refined[overlap, 3])
    return output


class ICPViewerWindow(QMainWindow):
    """Inspect a refined pose export over its source RGB frames."""

    def __init__(
        self,
        dataset: DataSet,
        review_poses: list[ReviewPose],
        *,
        model_scale: float = 0.001,
    ) -> None:
        super().__init__()
        if model_scale <= 0:
            raise ValueError("model_scale must be positive.")
        self.dataset = dataset
        self.model_scale = model_scale
        self.review_poses = [
            review_pose
            for review_pose in review_poses
            if review_pose.object_id in dataset.models
            and _dataset_has_frame(dataset, review_pose.frame_id)
        ]
        if not self.review_poses:
            raise ValueError("No refined records have both a dataset frame and object model.")
        self._mesh_points: dict[int, np.ndarray] = {}
        self._mode = ViewerMode.Normal
        self._shortcuts: list[QShortcut] = []
        self._mode_buttons: dict[ViewerMode, QPushButton] = {}

        self.setWindowTitle("ICP Overlay Viewer")
        self.resize(1450, 900)
        self._build_ui()
        self._populate_pose_selector()
        self._install_shortcuts()
        self._load_selection()

    @property
    def current_review_pose(self) -> ReviewPose:
        index = self.pose_selector.currentIndex()
        if index < 0:
            raise RuntimeError("No review pose is selected.")
        return self.review_poses[index]

    def _build_ui(self) -> None:
        central = QWidget(self)
        layout = QVBoxLayout(central)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Frame/object:"))
        self.previous_button = QPushButton("Previous")
        self.previous_button.clicked.connect(self._previous_pose)
        controls.addWidget(self.previous_button)
        self.pose_selector = QComboBox()
        self.pose_selector.setMinimumWidth(250)
        self.pose_selector.currentIndexChanged.connect(self._load_selection)
        controls.addWidget(self.pose_selector)
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self._next_pose)
        controls.addWidget(self.next_button)
        controls.addSpacing(24)

        button_group = QButtonGroup(self)
        button_group.setExclusive(True)
        labels = {
            ViewerMode.Normal: "1  Normal",
            ViewerMode.Rough: "2  Rough (red)",
            ViewerMode.ICP: "3  ICP (green)",
            ViewerMode.RoughAndICP: "4  Rough + ICP",
        }
        for mode, label in labels.items():
            button = QPushButton(label)
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, value=mode: self.set_mode(value))
            button_group.addButton(button)
            controls.addWidget(button)
            self._mode_buttons[mode] = button
        self._mode_buttons[self._mode].setChecked(True)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.image_view = ImageView()
        self.image_view.set_navigation_mode(True)
        self.scene_view = ICPSceneView()
        self.tabs = QTabWidget()
        self.tabs.addTab(self.image_view, "2D RGB overlay")
        self.tabs.addTab(self.scene_view, "3D RGB-D scene")
        layout.addWidget(self.tabs, stretch=1)
        self.details_label = QLabel()
        self.details_label.setStyleSheet("padding: 6px; color: #d7dce5;")
        layout.addWidget(self.details_label)
        self.setCentralWidget(central)

    def _populate_pose_selector(self) -> None:
        self.pose_selector.blockSignals(True)
        for review_pose in self.review_poses:
            self.pose_selector.addItem(
                f"Frame {review_pose.frame_id:06d} · obj_{review_pose.object_id:06d}"
            )
        self.pose_selector.blockSignals(False)

    def _install_shortcuts(self) -> None:
        for mode in ViewerMode:
            shortcut = QShortcut(QKeySequence(str(mode.value)), self)
            shortcut.activated.connect(lambda value=mode: self.set_mode(value))
            self._shortcuts.append(shortcut)

    def _load_selection(self, _index: int = -1) -> None:
        if self.pose_selector.currentIndex() < 0:
            return
        review_pose = self.current_review_pose
        pose = self.dataset[review_pose.frame_id]
        self.image_view.set_image(pose.rgb)
        self.scene_view.set_scene(
            pose,
            self.dataset.models[review_pose.object_id].mesh_path,
            review_pose.rough_rotation_m2c,
            review_pose.rough_translation_m2c_m,
            review_pose.icp_rotation_m2c,
            review_pose.icp_translation_m2c_m,
            model_scale=self.model_scale,
        )
        self.scene_view.set_mode(self._mode.value)
        self._refresh_overlay()
        self._update_details()
        index = self.pose_selector.currentIndex()
        self.previous_button.setEnabled(index > 0)
        self.next_button.setEnabled(index + 1 < len(self.review_poses))

    def set_mode(self, mode: ViewerMode) -> None:
        self._mode = ViewerMode(mode)
        self._mode_buttons[self._mode].setChecked(True)
        self.scene_view.set_mode(self._mode.value)
        self._refresh_overlay()
        self._update_details()

    def _refresh_overlay(self) -> None:
        if self.pose_selector.currentIndex() < 0 or self._mode is ViewerMode.Normal:
            self.image_view.set_overlay(None)
            return
        review_pose = self.current_review_pose
        pose = self.dataset[review_pose.frame_id]
        points = self._model_points(review_pose.object_id)
        rough = None
        refined = None
        if self._mode in (ViewerMode.Rough, ViewerMode.RoughAndICP):
            rough = project_mesh_overlay(
                points,
                pose,
                review_pose.rough_rotation_m2c,
                review_pose.rough_translation_m2c_m,
                ROUGH_COLOR,
            )
        if self._mode in (ViewerMode.ICP, ViewerMode.RoughAndICP):
            refined = project_mesh_overlay(
                points,
                pose,
                review_pose.icp_rotation_m2c,
                review_pose.icp_translation_m2c_m,
                ICP_COLOR,
            )
        if rough is not None and refined is not None:
            overlay = combine_pose_overlays(rough, refined)
        else:
            overlay = rough if rough is not None else refined
        self.image_view.set_overlay(overlay)

    def _model_points(self, object_id: int) -> np.ndarray:
        if object_id not in self._mesh_points:
            mesh = pv.read(self.dataset.models[object_id].mesh_path)
            points = np.asarray(mesh.points, dtype=np.float64) * self.model_scale
            if not len(points) or not np.isfinite(points).all():
                raise ValueError(f"obj_{object_id:06d}.ply has no finite vertices.")
            self._mesh_points[object_id] = points
        return self._mesh_points[object_id]

    def _update_details(self) -> None:
        review_pose = self.current_review_pose
        mode_text = {
            ViewerMode.Normal: "Normal RGB",
            ViewerMode.Rough: "Rough pose — red",
            ViewerMode.ICP: "ICP pose — green",
            ViewerMode.RoughAndICP: "Rough red · ICP green · overlap yellow",
        }[self._mode]
        self.details_label.setText(
            f"{mode_text}    |    fitness {review_pose.fitness:.4f}    |    "
            f"ICP inlier RMSE {review_pose.inlier_rmse_mm:.3f} mm    |    "
            f"correspondences {review_pose.correspondence_count:,}"
        )

    def _previous_pose(self) -> None:
        self.pose_selector.setCurrentIndex(max(0, self.pose_selector.currentIndex() - 1))

    def _next_pose(self) -> None:
        self.pose_selector.setCurrentIndex(
            min(len(self.review_poses) - 1, self.pose_selector.currentIndex() + 1)
        )

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        self.scene_view.close()
        super().closeEvent(event)


def _dataset_has_frame(dataset: DataSet, frame_id: int) -> bool:
    try:
        dataset[frame_id]
    except KeyError:
        return False
    return True


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review rough and ICP pose overlays.")
    parser.add_argument("--dataset", type=Path, required=True, help="BOP dataset root")
    parser.add_argument(
        "--poses", "--input", dest="poses", type=Path, required=True, help="ICP pose JSON"
    )
    parser.add_argument(
        "--model-scale",
        type=float,
        default=0.001,
        help="PLY coordinate-to-metre scale (default: 0.001)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    application = QApplication.instance() or QApplication([sys.argv[0]])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataset = DataSet.load(args.dataset)
        review_poses = load_review_poses(args.poses)
        window = ICPViewerWindow(dataset, review_poses, model_scale=args.model_scale)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        QMessageBox.critical(None, "Cannot start ICP viewer", str(error))
        return 1
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
