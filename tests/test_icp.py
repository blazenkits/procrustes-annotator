"""Focused geometry tests for the ICP backend."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import open3d as o3d

from src.backend.icp import ICPMode, image_to_point_cloud, run_icp


def _cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))


class ICPTests(unittest.TestCase):
    def test_image_unprojection_uses_camera_coordinates(self) -> None:
        pose = SimpleNamespace(
            frame_id=4,
            depth=np.array([[1.0, np.nan], [2.0, 1.5]], dtype=np.float32),
            rgb=np.zeros((2, 2, 3), dtype=np.uint8),
            camera_intrinsics=np.array(
                [[2.0, 0.0, 0.5], [0.0, 4.0, 0.5], [0.0, 0.0, 1.0]]
            ),
        )
        result = image_to_point_cloud(pose, voxel_size=0, include_colors=False)
        actual = np.asarray(result.points)
        expected = np.array(
            [
                [-0.25, -0.125, 1.0],
                [-0.50, 0.25, 2.0],
                [0.375, 0.1875, 1.5],
            ]
        )
        np.testing.assert_allclose(actual, expected)

    def test_point_to_point_recovers_model_to_camera_transform(self) -> None:
        random = np.random.default_rng(7)
        source_points = random.normal(size=(800, 3)) * np.array([0.04, 0.03, 0.06])
        angle = np.deg2rad(3.0)
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        translation = np.array([0.006, -0.004, 0.55])
        target_points = source_points @ rotation.T + translation
        initial = np.eye(4)
        initial[:3, 3] = [0.0, 0.0, 0.545]

        result = run_icp(
            _cloud(source_points),
            _cloud(target_points),
            initial,
            mode=ICPMode.PointToPoint,
            max_correspondence_distance=0.03,
            max_iteration=100,
        )

        np.testing.assert_allclose(result.R, rotation, atol=1e-6)
        np.testing.assert_allclose(result.t, translation, atol=1e-6)
        self.assertAlmostEqual(result.fitness, 1.0)
        self.assertLess(result.inlier_rmse, 1e-7)
        self.assertEqual(len(result.correspondence_set), len(source_points))


if __name__ == "__main__":
    unittest.main()
