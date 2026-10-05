"""Focused geometry tests for the ICP backend."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import cv2
import numpy as np
import open3d as o3d

from src.backend.icp import (
    ICPConfig,
    ICPMode,
    _silhouette_distance_field,
    _silhouette_terms,
    _sample_texture_at_uv,
    erode_binary_mask,
    green_mask_overlay,
    image_to_point_cloud,
    projected_model_box,
    run_icp,
    run_projective_icp,
    run_silhouette_constrained_icp,
    select_target_neighborhood,
    segment_object_sam2,
)


def _cloud(
    points: np.ndarray,
    colors: np.ndarray | None = None,
) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    if colors is not None:
        cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


class ICPTests(unittest.TestCase):
    def test_texture_sampling_uses_bottom_left_uv_origin(self) -> None:
        texture = np.array(
            [
                [[255, 0, 0], [0, 255, 0]],
                [[0, 0, 255], [255, 255, 255]],
            ],
            dtype=np.uint8,
        )
        uv = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0]])

        colors = _sample_texture_at_uv(texture, uv)

        np.testing.assert_allclose(
            colors,
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
        )

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

    def test_image_unprojection_respects_pixel_mask(self) -> None:
        pose = SimpleNamespace(
            frame_id=4,
            depth=np.ones((3, 3), dtype=np.float32),
            rgb=np.zeros((3, 3, 3), dtype=np.uint8),
            camera_intrinsics=np.eye(3),
        )
        mask = np.zeros((3, 3), dtype=bool)
        mask[0, 0] = True
        mask[1, 1] = True
        mask[2, 2] = True

        result = image_to_point_cloud(
            pose,
            voxel_size=0,
            include_colors=False,
            pixel_mask=mask,
        )

        np.testing.assert_allclose(
            np.asarray(result.points),
            [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [2.0, 2.0, 1.0]],
        )

    def test_mask_erosion_uses_pixel_radius(self) -> None:
        mask = np.zeros((11, 11), dtype=bool)
        mask[1:10, 1:10] = True

        eroded = erode_binary_mask(mask, radius_px=2)

        self.assertTrue(eroded[5, 5])
        self.assertFalse(eroded[1, 5])
        self.assertLess(np.count_nonzero(eroded), np.count_nonzero(mask))

    def test_green_mask_overlay_changes_only_selected_pixels(self) -> None:
        rgb = np.full((2, 2, 3), [120, 80, 40], dtype=np.uint8)
        mask = np.array([[False, True], [False, False]])

        overlay = green_mask_overlay(rgb, mask, alpha=1.0)

        np.testing.assert_array_equal(overlay[0, 1], [0, 255, 0])
        np.testing.assert_array_equal(overlay[1, 1], rgb[1, 1])

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

    def test_colored_icp_requires_source_and_target_colors(self) -> None:
        points = np.array(
            [[0.0, 0.0, 1.0], [0.01, 0.0, 1.0], [0.0, 0.01, 1.0]]
        )
        target = _cloud(points, colors=np.ones((3, 3)))

        with self.assertRaisesRegex(ValueError, "requires colors"):
            run_icp(
                _cloud(points),
                target,
                np.eye(4),
                mode=ICPMode.Colored,
                max_correspondence_distance=0.02,
            )

    def test_projective_icp_recovers_depth_translation(self) -> None:
        intrinsics = np.array(
            [[200.0, 0.0, 49.5], [0.0, 200.0, 49.5], [0.0, 0.0, 1.0]]
        )
        depth = np.ones((100, 100), dtype=np.float64)
        x, y = np.meshgrid(np.linspace(-0.15, 0.15, 61), np.linspace(-0.15, 0.15, 61))
        source = _cloud(np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))))
        initial = np.eye(4)
        initial[2, 3] = 0.995

        result = run_projective_icp(
            source,
            depth,
            intrinsics,
            initial,
            max_correspondence_distance=0.02,
            max_iteration=20,
        )

        self.assertGreater(len(result.correspondence_set), 100)
        self.assertAlmostEqual(result.t[2], 1.0, places=5)
        self.assertLess(result.inlier_rmse, 1e-5)

    def test_silhouette_configuration_requires_sam(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires projected-box SAM"):
            ICPConfig(silhouette_weight=0.25).validate()

    def test_silhouette_depth_gate_requires_enabled_silhouette(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires silhouette_weight"):
            ICPConfig(silhouette_depth_gate=0.01).validate()

    def test_silhouette_depth_gate_rejects_depth_inconsistent_contour(self) -> None:
        x, y = np.meshgrid(
            np.linspace(-0.1, 0.1, 31), np.linspace(-0.1, 0.1, 31)
        )
        source = _cloud(np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))))
        intrinsics = np.array(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]
        )
        transformation = np.eye(4)
        transformation[2, 3] = 1.0
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(mask, (38, 38), (62, 62), 1, thickness=cv2.FILLED)
        distance, gradient_x, gradient_y = _silhouette_distance_field(
            mask, image_scale=1.0
        )

        _, _, accepted, rejected = _silhouette_terms(
            source,
            transformation,
            intrinsics,
            mask.shape,
            distance,
            gradient_x,
            gradient_y,
            image_scale=1.0,
            splat_radius_px=1,
            observed_depth=np.ones(mask.shape, dtype=np.float64),
            depth_gate=0.01,
        )
        self.assertGreater(accepted, 6)
        self.assertEqual(rejected, 0)

        _, _, accepted, rejected = _silhouette_terms(
            source,
            transformation,
            intrinsics,
            mask.shape,
            distance,
            gradient_x,
            gradient_y,
            image_scale=1.0,
            splat_radius_px=1,
            observed_depth=np.full(mask.shape, 1.02),
            depth_gate=0.01,
        )
        self.assertEqual(accepted, 0)
        self.assertGreater(rejected, 6)

    def test_one_sided_silhouette_gate_rejects_only_foreground_depth(self) -> None:
        x, y = np.meshgrid(
            np.linspace(-0.1, 0.1, 31), np.linspace(-0.1, 0.1, 31)
        )
        source = _cloud(np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))))
        intrinsics = np.array(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]
        )
        transformation = np.eye(4)
        transformation[2, 3] = 1.0
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(mask, (38, 38), (62, 62), 1, thickness=cv2.FILLED)
        distance, gradient_x, gradient_y = _silhouette_distance_field(
            mask, image_scale=1.0
        )

        common = dict(
            image_scale=1.0,
            splat_radius_px=1,
            depth_gate=0.01,
            depth_gate_occlusion_only=True,
        )
        _, _, accepted, rejected = _silhouette_terms(
            source, transformation, intrinsics, mask.shape, distance, gradient_x,
            gradient_y, observed_depth=np.full(mask.shape, 1.02), **common
        )
        self.assertGreater(accepted, 6)  # Background depth is retained.
        self.assertEqual(rejected, 0)
        _, _, accepted, rejected = _silhouette_terms(
            source, transformation, intrinsics, mask.shape, distance, gradient_x,
            gradient_y, observed_depth=np.full(mask.shape, 0.98), **common
        )
        self.assertEqual(accepted, 0)
        self.assertGreater(rejected, 6)
        _, _, accepted, rejected = _silhouette_terms(
            source, transformation, intrinsics, mask.shape, distance, gradient_x,
            gradient_y, observed_depth=np.full(mask.shape, np.nan), **common
        )
        self.assertGreater(accepted, 6)  # Invalid depth is not affirmative occlusion.
        self.assertEqual(rejected, 0)

    def test_target_neighborhood_configuration_requires_positive_distance(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            ICPConfig(target_neighborhood_distance=0.0).validate()

    def test_target_neighborhood_uses_rough_transformed_model(self) -> None:
        source = _cloud(np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]]))
        colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        target = _cloud(
            np.array([[0.0, 0.0, 1.0], [0.012, 0.0, 1.0], [0.2, 0.0, 1.0]]),
            colors=colors,
        )
        rough = np.eye(4)
        rough[2, 3] = 1.0

        selected = select_target_neighborhood(
            target, source, rough, max_distance=0.005
        )

        np.testing.assert_allclose(
            np.asarray(selected.points),
            [[0.0, 0.0, 1.0], [0.012, 0.0, 1.0]],
        )
        np.testing.assert_allclose(np.asarray(selected.colors), colors[:2])

    def test_silhouette_constraint_recovers_tangential_plane_motion(self) -> None:
        x, y = np.meshgrid(
            np.linspace(-0.1, 0.1, 61), np.linspace(-0.2, 0.2, 121)
        )
        model_points = np.column_stack(
            (x.reshape(-1), y.reshape(-1), np.zeros(x.size))
        )
        true_translation = np.array([0.02, 0.0, 1.0])
        target_points = model_points + true_translation
        target = _cloud(target_points)
        target.normals = o3d.utility.Vector3dVector(
            np.tile([0.0, 0.0, -1.0], (len(target_points), 1))
        )
        rough = np.eye(4)
        rough[:3, 3] = [0.0, 0.0, 1.0]
        intrinsics = np.array(
            [[200.0, 0.0, 100.0], [0.0, 200.0, 100.0], [0.0, 0.0, 1.0]]
        )
        projected = (
            target_points[:, :2] / target_points[:, 2, np.newaxis] * 200.0
            + 100.0
        )
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(
            mask,
            tuple(np.floor(projected.min(axis=0)).astype(int)),
            tuple(np.ceil(projected.max(axis=0)).astype(int)),
            1,
            thickness=cv2.FILLED,
        )

        result, diagnostics = run_silhouette_constrained_icp(
            _cloud(model_points),
            target,
            rough,
            rough,
            intrinsics,
            mask.shape,
            mask,
            max_correspondence_distance=0.05,
            silhouette_weight=1.0,
            max_iteration=15,
            image_scale=0.5,
            distance_scale_px=3.0,
            splat_radius_px=1,
            pose_prior_weight=0.001,
            max_rotation_deg=5.0,
            max_translation_m=0.05,
            max_step_translation_m=0.005,
        )

        self.assertGreater(result.t[0], 0.01)
        self.assertLess(result.t[0], 0.025)
        self.assertLess(diagnostics.final_objective, diagnostics.initial_objective)
        self.assertLessEqual(diagnostics.translation_delta_m, 0.05 + 1e-9)

    def test_projected_model_box_applies_padding_and_clips_to_image(self) -> None:
        points = np.array(
            [
                [-1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0],
                [-1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )
        intrinsics = np.array([[10.0, 0.0, 10.0], [0.0, 10.0, 10.0], [0.0, 0.0, 1.0]])

        box = projected_model_box(
            points,
            np.eye(3),
            np.array([0.0, 0.0, 1.0]),
            intrinsics,
            (20, 20),
            padding_fraction=0.5,
        )

        np.testing.assert_allclose(box, [0.0, 0.0, 19.0, 19.0])

    def test_projected_model_box_rejects_points_behind_camera(self) -> None:
        with self.assertRaisesRegex(ValueError, "in front of the camera"):
            projected_model_box(
                np.array([[0.0, 0.0, -2.0], [1.0, 0.0, -2.0], [0.0, 1.0, -2.0]]),
                np.eye(3),
                np.zeros(3),
                np.eye(3),
                (10, 10),
            )

    def test_segment_object_selects_highest_scoring_candidate(self) -> None:
        class FakeSegmenter:
            def __init__(self) -> None:
                self.image_shape = None
                self.box = None
                self.selected_mask = None

            def predict_masks(self, image, box_xyxy, *, multimask_output):
                self.image_shape = image.shape
                self.box = box_xyxy.copy()
                masks = np.zeros((3, 4, 5), dtype=bool)
                masks[0, 1:3, 1:3] = True
                masks[1, 1:4, 1:4] = True
                masks[2, 0, 0] = True
                return masks, np.array([0.4, 0.9, 0.2])

            def predict_target_mask(self, image, box_xyxy, selected_mask):
                self.selected_mask = selected_mask.copy()
                return ~selected_mask

        pose = SimpleNamespace(
            frame_id=8,
            rgb=np.zeros((4, 5, 3), dtype=np.uint8),
            depth=np.ones((4, 5), dtype=np.float32),
            camera_intrinsics=np.array(
                [[2.0, 0.0, 2.0], [0.0, 2.0, 1.5], [0.0, 0.0, 1.0]]
            ),
        )
        points = np.array([[-0.5, -0.5, 1.0], [0.5, -0.5, 1.0], [0.0, 0.5, 1.0]])
        segmenter = FakeSegmenter()

        result = segment_object_sam2(
            pose,
            points,
            np.eye(3),
            np.array([0.0, 0.0, 1.0]),
            segmenter,
            padding_fraction=0.25,
        )

        self.assertEqual(segmenter.image_shape, (4, 5, 3))
        self.assertEqual(result.selected_index, 1)
        self.assertAlmostEqual(result.selected_score, 0.9)
        self.assertEqual(result.mask.dtype, np.bool_)
        self.assertEqual(result.mask.shape, (4, 5))
        np.testing.assert_array_equal(result.mask, result.candidate_masks[1])
        np.testing.assert_allclose(result.prompt_box_xyxy, segmenter.box)
        np.testing.assert_array_equal(result.target_mask, ~result.mask)
        np.testing.assert_array_equal(segmenter.selected_mask, result.mask)


if __name__ == "__main__":
    unittest.main()
