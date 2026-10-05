"""Portable GPU handoff and rendered-depth RMSE regression tests."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import open3d as o3d
from PIL import Image

from src.backend.batch_pipeline import (
    RAW_FORMAT,
    RESULT_FORMAT,
    decode_mask,
    depth_rmse_mm,
    encode_mask,
    make_job_archive,
    record_hash,
    run_batch,
    write_json,
)
from src.backend.loader import DataSet


class BatchPipelineTests(unittest.TestCase):
    def test_depth_preset_two_for_manual_and_one_for_icp(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rgb").mkdir()
            (root / "depth" / "1").mkdir(parents=True)
            (root / "depth" / "2").mkdir(parents=True)
            (root / "scene_camera.json").write_text(json.dumps({
                "0": {"cam_K": [100, 0, 1, 0, 100, 1, 0, 0, 1], "depth_scale": 1}
            }), encoding="utf-8")
            Image.fromarray(np.zeros((2, 2, 3), np.uint8)).save(root / "rgb" / "000000.png")
            Image.fromarray(np.full((2, 2), 500, np.uint16)).save(root / "depth" / "1" / "000000.png")
            Image.fromarray(np.full((2, 2), 600, np.uint16)).save(root / "depth" / "2" / "000000.png")
            self.assertAlmostEqual(float(DataSet.load(root, depth_preset=2)[0].depth[0, 0]), 0.6)
            self.assertAlmostEqual(float(DataSet.load(root, depth_preset=1)[0].depth[0, 0]), 0.5)

    def test_rendered_depth_rmse_uses_sensor_depth_and_50mm_cutoff(self) -> None:
        with TemporaryDirectory() as directory:
            mesh_path = Path(directory) / "obj_000001.ply"
            mesh = o3d.geometry.TriangleMesh.create_box(width=100, height=100, depth=100)
            self.assertTrue(o3d.io.write_triangle_mesh(str(mesh_path), mesh))
            pose = SimpleNamespace(
                depth=np.full((10, 10), 0.5, np.float32),
                camera_intrinsics=np.array([[100., 0., 5.], [0., 100., 5.], [0., 0., 1.]]),
            )
            class DatasetStub:
                def __init__(self):
                    self.models = {1: SimpleNamespace(mesh_path=mesh_path)}

                def __getitem__(self, _frame):
                    return pose

            dataset = DatasetStub()
            result = depth_rmse_mm(dataset, 0, 1, np.eye(3), np.array([-0.05, -0.05, 0.5]))
            self.assertIsNotNone(result)
            self.assertLess(result, 0.01)
            pose.depth[0, 0] = 0.6  # 100 mm residual is excluded.
            result = depth_rmse_mm(dataset, 0, 1, np.eye(3), np.array([-0.05, -0.05, 0.5]))
            self.assertLess(result, 0.01)

    def test_batch_result_is_bound_to_raw_pose_and_prompt(self) -> None:
        record = {
            "obj_id": 1, "cam_R_m2c": np.eye(3).reshape(-1).tolist(),
            "cam_t_m2c": [0., 0., 500.], "annotation_rmsd_mm": 1.,
            "manual_clicks_xy": [[1., 2.], [3., 4.], [5., 6.]],
        }
        raw = {"format": RAW_FORMAT, "method": "colored", "silhouette_weight": 0.2,
               "poses": {"0": [record]}}
        pose = SimpleNamespace(rgb=np.zeros((8, 8, 3), np.uint8))
        class DatasetStub:
            def __getitem__(self, _frame):
                return pose
        with (
            patch("src.backend.batch_pipeline.generate_sam_mask", return_value=(np.ones((8, 8), bool), 0.9, np.array([0, 0, 7, 7]))),
            patch("src.backend.batch_pipeline.propose_silhouette_icp", return_value=(np.eye(3), np.array([0., 0., 0.501]), {"status": "refined"})) as refine,
            patch("src.backend.batch_pipeline.depth_rmse_mm", side_effect=[3.0, 2.8]),
        ):
            result = run_batch(DatasetStub(), raw, segmenter=object())
        entry = result["results"]["0"]["1"]
        self.assertEqual(result["format"], RESULT_FORMAT)
        self.assertEqual(entry["raw_sha256"], record_hash(record))
        self.assertEqual(entry["status"], "ready")
        self.assertEqual(entry["method"], "colored")
        self.assertEqual(refine.call_args.kwargs["method"], "colored")
        self.assertEqual(refine.call_args.kwargs["silhouette_weight"], 0.2)
        self.assertEqual(entry["raw_depth_rmse_mm"], 3.0)
        self.assertEqual(entry["icp_depth_rmse_mm"], 2.8)
        self.assertEqual(int(decode_mask(entry["mask_png_base64"], (8, 8)).sum()), 64)
        changed = dict(record, extra_negative_xy=[[0., 0.]])
        self.assertNotEqual(record_hash(changed), entry["raw_sha256"])

    def test_job_archive_contains_raw_and_dataset(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            dataset.mkdir()
            (dataset / "scene_camera.json").write_text("{}", encoding="utf-8")
            (dataset / "rgb").mkdir()
            (dataset / "rgb" / "000000.png").touch()
            (dataset / "depth" / "1").mkdir(parents=True)
            (dataset / "depth" / "2").mkdir(parents=True)
            (dataset / "depth" / "1" / "000000.png").touch()
            (dataset / "depth" / "2" / "000000.png").touch()
            (dataset / "models").mkdir()
            (dataset / "models" / "obj_000001.ply").touch()
            raw = root / "raw.json"
            write_json(raw, {"format": RAW_FORMAT, "poses": {"0": [{"obj_id": 1}]}})
            archive = root / "job.zip"
            make_job_archive(dataset, raw, archive)
            import zipfile
            with zipfile.ZipFile(archive) as file:
                self.assertEqual(set(file.namelist()), {
                    "raw_poses.json", "dataset/scene_camera.json", "dataset/rgb/000000.png",
                    "dataset/depth/1/000000.png", "dataset/models/obj_000001.ply",
                })
                self.assertEqual(json.loads(file.read("raw_poses.json"))["format"], RAW_FORMAT)

    def test_mask_codec_checks_shape(self) -> None:
        encoded = encode_mask(np.ones((5, 7), bool))
        with self.assertRaisesRegex(ValueError, "does not match"):
            decode_mask(encoded, (5, 5))


if __name__ == "__main__":
    unittest.main()
