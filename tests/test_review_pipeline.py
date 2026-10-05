"""Verify reviewed SAM prompts and the fixed V1 ICP configuration."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import open3d as o3d

from src.backend.icp import ICPMode
from src.backend.review_pipeline import generate_sam_mask, propose_silhouette_icp


class PredictorStub:
    def __init__(self, mask: np.ndarray):
        self.mask = mask
        self.kwargs = None

    def set_image(self, image):
        self.image = image

    def predict(self, **kwargs):
        self.kwargs = kwargs
        return self.mask[None], np.array([0.94]), None


class ReviewPipelineTests(TestCase):
    def test_sam_receives_box_manual_clicks_extra_prompts_and_prior(self) -> None:
        mask = np.zeros((64, 64), bool)
        mask[10:55, 10:55] = True
        predictor = PredictorStub(mask)
        segmenter = SimpleNamespace(
            predictor=predictor, device="cpu",
            _torch=SimpleNamespace(inference_mode=nullcontext),
        )
        pose = SimpleNamespace(
            rgb=np.zeros((64, 64, 3), np.uint8),
            camera_intrinsics=np.eye(3),
        )
        class DatasetStub:
            models = {1: object()}

            def __getitem__(self, frame):
                assert frame == 0
                return pose

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1]]))
        with (
            patch("src.backend.review_pipeline.load_sampled_model", return_value=cloud),
            patch("src.backend.review_pipeline.projected_model_box", return_value=np.array([2, 2, 60, 60])),
            patch("src.backend.review_pipeline.rough_silhouette", return_value=mask),
        ):
            output, score, box = generate_sam_mask(
                DatasetStub(), 0, 1, np.eye(3), np.array([0, 0, 1]),
                np.array([[20, 20], [25, 25], [30, 30]]),
                np.array([[35, 35]]), np.array([[50, 50]]), segmenter,
            )
        self.assertEqual(output.shape, (64, 64))
        self.assertAlmostEqual(score, 0.94)
        self.assertEqual(len(box), 4)
        np.testing.assert_array_equal(predictor.kwargs["point_labels"], [1, 1, 1, 1, 0])
        self.assertEqual(predictor.kwargs["mask_input"].shape, (1, 256, 256))
        self.assertFalse(predictor.kwargs["multimask_output"])

    def test_icp_uses_only_approved_mask_and_v1_parameters(self) -> None:
        mask = np.ones((40, 40), bool)

        def fake_refine(dataset, document, *, config, segmenter):
            self.assertEqual(config.silhouette_weight, 0.10)
            self.assertEqual(config.sam_erosion_radius_px, 4)
            np.testing.assert_array_equal(segmenter.mask, mask)
            record = document["0"][0].copy()
            record["icp"] = {"status": "refined", "fitness": 0.8}
            return {"0": [record]}

        with patch("src.backend.review_pipeline.refine_annotations", side_effect=fake_refine):
            rotation, translation, info = propose_silhouette_icp(
                object(), 0, 1, np.eye(3), np.array([0, 0, 0.6]), 0.003, mask,
            )
        np.testing.assert_allclose(rotation, np.eye(3))
        np.testing.assert_allclose(translation, [0, 0, 0.6])
        self.assertEqual(info["status"], "refined")

    def test_colored_mode_and_adjustable_silhouette_weight(self) -> None:
        seen = []

        def fake_refine(_dataset, document, *, config, segmenter):
            seen.append((config.mode, config.silhouette_weight, config.sam_erosion_radius_px))
            record = document["0"][0].copy()
            record["icp"] = {"status": "refined"}
            return {"0": [record]}

        with patch("src.backend.review_pipeline.refine_annotations", side_effect=fake_refine):
            _, _, color = propose_silhouette_icp(
                object(), 0, 1, np.eye(3), np.array([0, 0, 0.6]), 0.003,
                np.ones((40, 40), bool), method="colored",
            )
            _, _, silhouette = propose_silhouette_icp(
                object(), 0, 1, np.eye(3), np.array([0, 0, 0.6]), 0.003,
                np.ones((40, 40), bool), method="silhouette", silhouette_weight=0.2,
            )
        self.assertEqual(seen, [(ICPMode.Colored, 0.0, 4), (ICPMode.PointToPlane, 0.2, 4)])
        self.assertEqual(color["review_method"], "colored")
        self.assertEqual(silhouette["review_silhouette_weight"], 0.2)
