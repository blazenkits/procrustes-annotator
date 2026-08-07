"""Geometry primitives for RGBD correspondence annotation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

try:  # Supports both ``python -m src.backend...`` and direct execution.
    from .loader import Pose
except ImportError:  # pragma: no cover - exercised only when run as a script.
    from loader import Pose


@dataclass(frozen=True)
class Vector2:
    """A pixel coordinate, using the image convention ``(x, y)``."""

    x: float
    y: float

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float64)


@dataclass(frozen=True)
class Vector3:
    """A three-dimensional point, using metres for geometric coordinates."""

    x: float
    y: float
    z: float

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @classmethod
    def from_array(cls, values: Iterable[float]) -> "Vector3":
        x, y, z = np.asarray(tuple(values), dtype=np.float64).reshape(3)
        return cls(float(x), float(y), float(z))


# Get xyz coord in image space of image point
def get_pixel_point(pose: Pose, point: Vector2, *, neighborhood_radius: int = 2) -> Vector3:
    """Return ``(u, v, depth)`` for a clicked RGB pixel.

    The loader supplies RGB-aligned depth in metres.  Clicks can be
    fractional; their image coordinate is rounded to the nearest pixel. Depth
    is averaged over valid samples in a small circular neighborhood so an
    isolated missing/speckle pixel does not prevent annotation.
    """
    u, v = int(round(point.x)), int(round(point.y))
    height, width = pose.depth.shape
    if not (0 <= u < width and 0 <= v < height):
        raise ValueError(f"Pixel ({point.x}, {point.y}) is outside a {width}x{height} frame.")

    if neighborhood_radius < 0:
        raise ValueError("neighborhood_radius must be non-negative.")
    x0, x1 = max(0, u - neighborhood_radius), min(width, u + neighborhood_radius + 1)
    y0, y1 = max(0, v - neighborhood_radius), min(height, v + neighborhood_radius + 1)
    neighborhood = pose.depth[y0:y1, x0:x1]
    yy, xx = np.ogrid[y0:y1, x0:x1]
    inside_radius = (xx - u) ** 2 + (yy - v) ** 2 <= neighborhood_radius**2
    valid = inside_radius & np.isfinite(neighborhood) & (neighborhood > 0)
    if not valid.any():
        raise ValueError(
            f"Pixel ({u}, {v}) has no valid depth data within "
            f"{neighborhood_radius} pixels."
        )
    depth = float(np.mean(neighborhood[valid], dtype=np.float64))
    return Vector3(float(u), float(v), depth)


# Convert camera coords (xy + depth) to model xyz coords
def unproject(camera_coords: Vector3, camera_intrinsics: np.ndarray) -> Vector3:
    """Unproject ``(u, v, depth)`` into a 3D point in camera coordinates.

    ``camera_intrinsics`` is the 3x3 ``cam_K`` matrix from the corresponding
    :class:`Pose`.  The returned point is *not* in model coordinates: it is
    the camera-side 3D counterpart used by Procrustes to solve model-to-camera
    alignment.
    """
    K = np.asarray(camera_intrinsics, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError("camera_intrinsics must have shape (3, 3).")
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    if fx == 0 or fy == 0:
        raise ValueError("camera_intrinsics must have non-zero focal lengths.")
    if not np.isfinite(camera_coords.as_array()).all() or camera_coords.z <= 0:
        raise ValueError("camera_coords must contain a finite, positive depth.")

    x = (camera_coords.x - cx) * camera_coords.z / fx
    y = (camera_coords.y - cy) * camera_coords.z / fy
    return Vector3(x, y, camera_coords.z)


def solve_procrustes(
    cloud_1: list[Vector3], cloud_2: list[Vector3]
) -> tuple[np.ndarray, Vector3, float]:
    """Solve the rigid transform mapping ``cloud_1`` onto ``cloud_2``.

    This is the Kabsch form of 3D Procrustes.  It returns ``(R, t, rmse)``
    such that ``target = R @ source + t``.  ``rmse`` is the root-mean-square
    3D correspondence residual in metres.  At least three non-collinear,
    paired points are required and reflections are rejected to preserve a
    proper rotation.
    """
    if len(cloud_1) != len(cloud_2):
        raise ValueError("Point clouds must contain the same number of correspondences.")
    if len(cloud_1) < 3:
        raise ValueError("At least three point correspondences are required.")

    source = np.stack([point.as_array() for point in cloud_1])
    target = np.stack([point.as_array() for point in cloud_2])
    if not (np.isfinite(source).all() and np.isfinite(target).all()):
        raise ValueError("Point correspondences must be finite.")

    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    centered_source = source - source_center
    centered_target = target - target_center
    if np.linalg.matrix_rank(centered_source) < 2 or np.linalg.matrix_rank(centered_target) < 2:
        raise ValueError("Point correspondences must include three non-collinear points.")

    U, _, Vt = np.linalg.svd(centered_source.T @ centered_target)
    rotation = Vt.T @ U.T
    if np.linalg.det(rotation) < 0:
        Vt[-1, :] *= -1
        rotation = Vt.T @ U.T
    translation = target_center - rotation @ source_center
    residuals = target - (source @ rotation.T + translation)
    rmse = float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))
    return rotation, Vector3.from_array(translation), rmse
