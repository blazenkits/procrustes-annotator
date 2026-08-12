"""Reusable PySide6/PyVista view for RGB-D clouds and ICP pose meshes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
from PySide6.QtWidgets import QVBoxLayout, QWidget
from pyvistaqt import QtInteractor

from src.backend.icp import image_to_point_cloud
from src.backend.loader import Pose


ROUGH_COLOR = "#eb2d37"
ICP_COLOR = "#23dc5a"


def transform_points_to_camera(
    points_m: np.ndarray,
    rotation_m2c: np.ndarray,
    translation_m2c_m: np.ndarray,
) -> np.ndarray:
    """Apply a model-to-camera pose to Nx3 model points in metres."""
    points = np.asarray(points_m, dtype=np.float64).reshape(-1, 3)
    rotation = np.asarray(rotation_m2c, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(translation_m2c_m, dtype=np.float64).reshape(3)
    if not (
        np.isfinite(points).all()
        and np.isfinite(rotation).all()
        and np.isfinite(translation).all()
    ):
        raise ValueError("Scene pose inputs must be finite.")
    return points @ rotation.T + translation


class ICPSceneView(QWidget):
    """Show an RGB-colored depth cloud with rough/refined mesh outlines.

    Modes follow the 2D viewer: 1 shows only the measured point cloud, 2 adds
    the rough red mesh, 3 adds the refined green mesh, and 4 adds both.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        point_cloud_voxel_size: float = 0.003,
    ) -> None:
        super().__init__(parent)
        if point_cloud_voxel_size < 0:
            raise ValueError("point_cloud_voxel_size cannot be negative.")
        self.point_cloud_voxel_size = point_cloud_voxel_size
        self._mode = 1
        self._mesh_cache: dict[tuple[Path, float], pv.PolyData] = {}
        self._point_actor = None
        self._rough_actor = None
        self._icp_actor = None
        self._camera_initialized = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(self)
        self.plotter.set_background("#171a21")
        self.plotter.show_axes()
        layout.addWidget(self.plotter.interactor)

    def set_scene(
        self,
        pose: Pose,
        model_path: str | Path,
        rough_rotation_m2c: np.ndarray,
        rough_translation_m2c_m: np.ndarray,
        icp_rotation_m2c: np.ndarray,
        icp_translation_m2c_m: np.ndarray,
        *,
        model_scale: float = 0.001,
    ) -> None:
        """Replace the frame cloud and both posed mesh actors."""
        if model_scale <= 0:
            raise ValueError("model_scale must be positive.")
        self._remove_scene_actors()

        rgbd_cloud = image_to_point_cloud(
            pose,
            voxel_size=self.point_cloud_voxel_size,
            include_colors=True,
        )
        measured_points = np.asarray(rgbd_cloud.points, dtype=np.float64)
        measured_colors = np.rint(
            np.asarray(rgbd_cloud.colors, dtype=np.float64) * 255.0
        ).astype(np.uint8)
        cloud = pv.PolyData(measured_points)
        cloud["rgb"] = measured_colors
        self._point_actor = self.plotter.add_mesh(
            cloud,
            name="rgbd-point-cloud",
            scalars="rgb",
            rgb=True,
            point_size=2,
            render_points_as_spheres=False,
            lighting=False,
            pickable=False,
            reset_camera=False,
        )

        model = self._load_model(model_path, model_scale)
        rough_mesh = model.copy(deep=True)
        rough_mesh.points = transform_points_to_camera(
            model.points, rough_rotation_m2c, rough_translation_m2c_m
        )
        icp_mesh = model.copy(deep=True)
        icp_mesh.points = transform_points_to_camera(
            model.points, icp_rotation_m2c, icp_translation_m2c_m
        )
        self._rough_actor = self._add_outline_mesh(
            rough_mesh, name="rough-model", color=ROUGH_COLOR
        )
        self._icp_actor = self._add_outline_mesh(
            icp_mesh, name="icp-model", color=ICP_COLOR
        )
        self._update_actor_visibility()
        self._focus_camera(rough_mesh, icp_mesh)
        self.plotter.render()

    def set_mode(self, mode: int) -> None:
        """Set display mode 1, 2, 3, or 4."""
        if mode not in (1, 2, 3, 4):
            raise ValueError("3D scene mode must be 1, 2, 3, or 4.")
        self._mode = mode
        self._update_actor_visibility()
        self.plotter.render()

    def _load_model(self, model_path: str | Path, model_scale: float) -> pv.PolyData:
        path = Path(model_path).expanduser().resolve()
        key = (path, model_scale)
        if key not in self._mesh_cache:
            if not path.is_file():
                raise FileNotFoundError(f"Missing object model: {path}")
            mesh = pv.read(path)
            if not isinstance(mesh, pv.PolyData):
                mesh = mesh.extract_surface()
            mesh = mesh.copy(deep=True)
            mesh.points = np.asarray(mesh.points, dtype=np.float64) * model_scale
            if not len(mesh.points) or not np.isfinite(mesh.points).all():
                raise ValueError(f"Object model has no finite vertices: {path}")
            self._mesh_cache[key] = mesh
        return self._mesh_cache[key]

    def _add_outline_mesh(self, mesh: pv.PolyData, *, name: str, color: str):
        return self.plotter.add_mesh(
            mesh,
            name=name,
            color=color,
            opacity=0.18,
            show_edges=True,
            edge_color=color,
            line_width=2,
            lighting=False,
            pickable=False,
            reset_camera=False,
        )

    def _remove_scene_actors(self) -> None:
        for name in ("rgbd-point-cloud", "rough-model", "icp-model"):
            self.plotter.remove_actor(name, reset_camera=False, render=False)
        self._point_actor = None
        self._rough_actor = None
        self._icp_actor = None

    def _update_actor_visibility(self) -> None:
        if self._point_actor is not None:
            self._point_actor.SetVisibility(True)
        if self._rough_actor is not None:
            self._rough_actor.SetVisibility(self._mode in (2, 4))
        if self._icp_actor is not None:
            self._icp_actor.SetVisibility(self._mode in (3, 4))

    def _focus_camera(self, rough_mesh: pv.PolyData, icp_mesh: pv.PolyData) -> None:
        """Track the object without allowing distant sensor glitches to set zoom."""
        points = np.vstack((rough_mesh.points, icp_mesh.points))
        center = (points.min(axis=0) + points.max(axis=0)) / 2.0
        diameter = max(
            float(np.linalg.norm(points.max(axis=0) - points.min(axis=0))), 0.05
        )
        camera = self.plotter.camera
        if self._camera_initialized:
            old_focus = np.asarray(camera.focal_point, dtype=np.float64)
            offset = np.asarray(camera.position, dtype=np.float64) - old_focus
            if not np.isfinite(offset).all() or np.linalg.norm(offset) < 1e-6:
                offset = np.array(
                    [diameter * 1.4, -diameter * 1.0, -diameter * 1.8]
                )
        else:
            offset = np.array([diameter * 1.4, -diameter * 1.0, -diameter * 1.8])
            camera.view_up = (0.0, -1.0, 0.0)
            self._camera_initialized = True
        camera.focal_point = center
        camera.position = center + offset
        self.plotter.reset_camera_clipping_range()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        self.plotter.close()
        super().closeEvent(event)
