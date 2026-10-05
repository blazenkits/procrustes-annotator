"""Tests for silhouette sweep metadata and multi-file validation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.tools.silhouette_sweep_viewer import (
    load_silhouette_sweeps,
    parse_silhouette_sweep,
)


def _document(weight: float, *, frame_id: int = 0) -> dict:
    return {
        str(frame_id): [
            {
                "obj_id": 1,
                "annotation_status": "solved",
                "cam_R_m2c": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                "cam_t_m2c": [0.0, 0.0, 600.0],
                "icp": {
                    "status": "refined",
                    "fitness": 0.9,
                    "inlier_rmse_mm": 2.0,
                    "correspondence_count": 100,
                    "rough_cam_R_m2c": [
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                    "rough_cam_t_m2c": [0.0, 0.0, 600.0],
                    "silhouette": {"parameters": {"weight": weight}},
                },
            }
        ]
    }


class SilhouetteSweepViewerTests(unittest.TestCase):
    def test_parse_reads_weight_from_pose_metadata(self) -> None:
        sweep = parse_silhouette_sweep(_document(0.1))

        self.assertEqual(sweep.weight, 0.1)
        self.assertEqual(len(sweep.review_poses), 1)
        self.assertEqual(sweep.review_poses[0].frame_id, 0)

    def test_loader_sorts_weights_and_requires_matching_pose_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            high = root / "high.json"
            low = root / "low.json"
            high.write_text(json.dumps(_document(0.5)), encoding="utf-8")
            low.write_text(json.dumps(_document(0.05)), encoding="utf-8")

            sweeps = load_silhouette_sweeps([high, low])

            self.assertEqual([sweep.weight for sweep in sweeps], [0.05, 0.5])

            high.write_text(json.dumps(_document(0.5, frame_id=2)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "same frame/object records"):
                load_silhouette_sweeps([low, high])

    def test_loader_rejects_duplicate_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.json"
            second = root / "second.json"
            first.write_text(json.dumps(_document(0.1)), encoding="utf-8")
            second.write_text(json.dumps(_document(0.1)), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Duplicate silhouette weight"):
                load_silhouette_sweeps([first, second])


if __name__ == "__main__":
    unittest.main()
