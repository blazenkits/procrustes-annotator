"""Tests for ICP review-pose parsing and overlay composition."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from src.tools.icp_viewer import (
    ICP_COLOR,
    OVERLAP_COLOR,
    ROUGH_COLOR,
    combine_pose_overlays,
    parse_review_poses,
    project_mesh_overlay,
)
from src.tools.icp_scene_view import transform_points_to_camera


class ICPViewerTests(unittest.TestCase):
    def test_model_to_camera_transform_for_3d_scene(self) -> None:
        points = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        translation = np.array([0.1, 0.2, 0.5])
        actual = transform_points_to_camera(points, rotation, translation)
        expected = np.array([[0.1, 1.2, 0.5], [-1.9, 0.2, 0.5]])
        np.testing.assert_allclose(actual, expected)

    def test_parse_preserves_rough_and_refined_pose_conventions(self) -> None:
        identity = np.eye(3).reshape(-1).tolist()
        document = {
            "7": [
                {
                    "obj_id": 3,
                    "annotation_status": "solved",
                    "cam_R_m2c": identity,
                    "cam_t_m2c": [10.0, 20.0, 600.0],
                    "icp": {
                        "status": "refined",
                        "rough_cam_R_m2c": identity,
                        "rough_cam_t_m2c": [5.0, 15.0, 590.0],
                        "fitness": 0.9,
                        "inlier_rmse_mm": 2.5,
                        "correspondence_count": 120,
                    },
                }
            ],
            "8": [{"obj_id": 3, "annotation_status": "deferred"}],
        }
        review_pose = parse_review_poses(document)[0]
        self.assertEqual(review_pose.frame_id, 7)
        self.assertEqual(review_pose.object_id, 3)
        np.testing.assert_allclose(review_pose.rough_translation_m2c_m, [0.005, 0.015, 0.59])
        np.testing.assert_allclose(review_pose.icp_translation_m2c_m, [0.01, 0.02, 0.6])

    def test_projection_and_combined_overlap_colors(self) -> None:
        pose = SimpleNamespace(
            rgb=np.zeros((21, 21, 3), dtype=np.uint8),
            camera_intrinsics=np.array(
                [[10.0, 0.0, 10.0], [0.0, 10.0, 10.0], [0.0, 0.0, 1.0]]
            ),
        )
        points = np.array([[0.0, 0.0, 1.0]])
        rough = project_mesh_overlay(points, pose, np.eye(3), np.zeros(3), ROUGH_COLOR)
        refined = project_mesh_overlay(points, pose, np.eye(3), np.zeros(3), ICP_COLOR)
        self.assertTupleEqual(tuple(rough[10, 10, :3]), ROUGH_COLOR)
        self.assertTupleEqual(tuple(refined[10, 10, :3]), ICP_COLOR)

        combined = combine_pose_overlays(rough, refined)
        self.assertTupleEqual(tuple(combined[10, 10, :3]), OVERLAP_COLOR)
        self.assertGreater(int(combined[10, 10, 3]), 0)


if __name__ == "__main__":
    unittest.main()
