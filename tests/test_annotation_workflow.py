"""Tests for the multi-object raw annotation workflow."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from src.backend.annotate import Vector3
from src.frontend.main import (
    AnnotationWindow,
    MaskReview,
    ObjectAnnotationState,
    PoseSolution,
    _accepted_setup_observations,
    _deserialize_annotation_states,
    _deserialize_frame_setups,
    _find_mesh_texture_path,
    _serialize_annotation_states,
    _serialize_solved_poses,
    _step_annotation_selection,
)


class AnnotationWorkflowTests(unittest.TestCase):
    def test_mesh_texture_uses_ply_texture_file_comment(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            mesh_path = root / "object.ply"
            texture_path = root / "object texture.png"
            mesh_path.write_text(
                "ply\nformat ascii 1.0\n"
                "comment TextureFile object texture.png\n"
                "element vertex 0\nend_header\n",
                encoding="utf-8",
            )
            texture_path.touch()

            self.assertEqual(_find_mesh_texture_path(mesh_path), texture_path)

    def test_mesh_texture_falls_back_to_same_stem_png(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            mesh_path = root / "object.ply"
            texture_path = root / "object.png"
            mesh_path.write_text(
                "ply\nformat ascii 1.0\nelement vertex 0\nend_header\n",
                encoding="utf-8",
            )
            texture_path.touch()

            self.assertEqual(_find_mesh_texture_path(mesh_path), texture_path)

    def test_next_visits_every_object_before_next_frame(self) -> None:
        counts = [2, 3]

        self.assertEqual(
            _step_annotation_selection(0, 0, counts, direction=1), (0, 1)
        )
        self.assertEqual(
            _step_annotation_selection(0, 1, counts, direction=1), (1, 0)
        )
        self.assertEqual(
            _step_annotation_selection(1, 2, counts, direction=1), (1, 2)
        )

    def test_previous_reverses_object_then_frame_order(self) -> None:
        counts = [2, 3]

        self.assertEqual(
            _step_annotation_selection(1, 0, counts, direction=-1), (0, 1)
        )
        self.assertEqual(
            _step_annotation_selection(1, 2, counts, direction=-1), (1, 1)
        )
        self.assertEqual(
            _step_annotation_selection(0, 0, counts, direction=-1), (0, 0)
        )

    def test_frame_selector_uses_scene_object_manifest(self) -> None:
        window = AnnotationWindow.__new__(AnnotationWindow)
        pose = SimpleNamespace(
            reference_instances=(
                SimpleNamespace(object_id=7),
                SimpleNamespace(object_id=2),
                SimpleNamespace(object_id=7),
                SimpleNamespace(object_id=99),
            )
        )

        class DatasetStub:
            models = {2: object(), 7: object(), 8: object()}

        window.dataset = DatasetStub()

        self.assertEqual(window._pose_object_ids(pose), [2, 7])

    def test_frame_selector_falls_back_to_all_models_without_manifest(self) -> None:
        window = AnnotationWindow.__new__(AnnotationWindow)

        class DatasetStub:
            models = {5: object(), 1: object()}

        window.dataset = DatasetStub()

        pose = SimpleNamespace(reference_instances=())
        self.assertEqual(window._pose_object_ids(pose), [1, 5])

    def test_export_keeps_multiple_objects_in_the_same_frame(self) -> None:
        solution = PoseSolution(
            rotation=np.eye(3),
            translation=Vector3(0.1, 0.2, 0.3),
            rmse=0.001,
            residuals_mm=np.array([1.0, 2.0, 3.0]),
        )
        states = {
            2: ObjectAnnotationState(solutions={4: solution}),
            7: ObjectAnnotationState(solutions={4: solution}),
        }

        exported = _serialize_solved_poses(states)

        self.assertEqual([record["obj_id"] for record in exported["4"]], [2, 7])

    def test_export_distinguishes_manual_and_inferred_poses(self) -> None:
        solution = PoseSolution(np.eye(3), Vector3(0.1, 0.2, 0.3), 0.001, np.array([]))
        states = {
            2: ObjectAnnotationState(solutions={0: solution}, inferred={1: solution}),
        }
        exported = _serialize_solved_poses(states)
        self.assertEqual(exported["0"][0]["annotation_source"], "manual_procrustes")
        self.assertEqual(exported["1"][0]["annotation_source"], "inferred_setup")

    def test_project_roundtrip_keeps_inferred_poses_and_setup_groups(self) -> None:
        solution = PoseSolution(np.eye(3), Vector3(0.1, 0.2, 0.3), 0.0, np.array([]))
        dataset = SimpleNamespace(models={2: object()}, poses={"0": [SimpleNamespace(frame_id=0), SimpleNamespace(frame_id=1)]})
        document = {
            "annotations": _serialize_annotation_states({2: ObjectAnnotationState(inferred={1: solution})}),
            "frame_setups": {"0": "turntable_A", "1": "turntable_A"},
        }
        restored = _deserialize_annotation_states(document, dataset)
        self.assertIn(1, restored[2].inferred)
        np.testing.assert_allclose(restored[2].inferred[1].translation.as_array(), [0.1, 0.2, 0.3])
        self.assertEqual(_deserialize_frame_setups(document, dataset), {0: "turntable_A", 1: "turntable_A"})

    def test_approved_mask_and_icp_choice_survive_project_roundtrip(self) -> None:
        manual = PoseSolution(np.eye(3), Vector3(0.1, 0.2, 0.3), 0.001, np.array([]))
        proposal = PoseSolution(np.eye(3), Vector3(0.11, 0.2, 0.3), 0.001, np.array([]))
        mask = np.zeros((10, 12), dtype=bool)
        mask[2:7, 3:9] = True
        review = MaskReview(mask, 0.93, [1.0, 1.0, 10.0, 8.0], [(4.0, 5.0)], [(9.0, 4.0)])
        state = ObjectAnnotationState(
            solutions={0: manual},
            masks={0: review}, icp_candidates={0: proposal},
            icp_info={0: {"fitness": 0.8}}, icp_decisions={0: "keep"},
            batch_metrics={0: {"raw_depth_rmse_mm": 3.2, "icp_depth_rmse_mm": 2.9}},
            review_prompts={0: {"extra_positive_xy": [[4.0, 5.0]], "extra_negative_xy": [[9.0, 4.0]], "box_padding": 0.25}},
        )

        class DatasetStub:
            models = {2: object()}
            poses = {"0": [SimpleNamespace(frame_id=0)]}

            def __getitem__(self, _frame):
                return SimpleNamespace(rgb=np.zeros((10, 12, 3), np.uint8))

        restored = _deserialize_annotation_states(
            {"annotations": _serialize_annotation_states({2: state})}, DatasetStub()
        )[2]
        np.testing.assert_array_equal(restored.masks[0].mask, mask)
        self.assertEqual(restored.icp_decisions[0], "keep")
        self.assertAlmostEqual(restored.batch_metrics[0]["raw_depth_rmse_mm"], 3.2)
        self.assertEqual(restored.review_prompts[0]["extra_positive_xy"], [[4.0, 5.0]])
        exported = _serialize_solved_poses({2: restored})["0"][0]
        self.assertEqual(exported["annotation_source"], "reviewed_silhouette_icp")
        self.assertAlmostEqual(exported["cam_t_m2c"][0], 110.0)

    def test_unreviewed_icp_cannot_be_exported(self) -> None:
        solution = PoseSolution(np.eye(3), Vector3(0.1, 0.2, 0.3), 0.001, np.array([]))
        state = ObjectAnnotationState(
            solutions={0: solution}, icp_candidates={0: solution},
            icp_decisions={0: "pending"},
        )
        with self.assertRaisesRegex(ValueError, "unreviewed ICP"):
            _serialize_solved_poses({2: state})

    def test_manual_pose_is_immediately_exportable(self) -> None:
        solution = PoseSolution(np.eye(3), Vector3(0.1, 0.2, 0.3), 0.001, np.array([]))
        exported = _serialize_solved_poses({2: ObjectAnnotationState(solutions={0: solution})})
        self.assertEqual(exported["0"][0]["annotation_source"], "manual_procrustes")

    def test_setup_uses_all_manually_solved_poses_and_reviewed_icp(self) -> None:
        manual = PoseSolution(np.eye(3), Vector3(0.1, 0.2, 0.3), 0.003, np.array([]))
        candidate = PoseSolution(np.eye(3), Vector3(0.11, 0.2, 0.3), 0.003, np.array([]))
        state = ObjectAnnotationState(
            solutions={0: manual, 1: manual},
            icp_candidates={0: candidate}, icp_decisions={0: "keep"},
        )
        observations, incomplete = _accepted_setup_observations({2: state}, [0, 1])
        self.assertFalse(incomplete)
        self.assertEqual(list(observations), [(0, 2), (1, 2)])
        self.assertAlmostEqual(observations[(0, 2)].translation[0], 0.11)
        state.icp_decisions[0] = "revert"
        observations, _ = _accepted_setup_observations({2: state}, [0, 1])
        self.assertAlmostEqual(observations[(0, 2)].translation[0], 0.1)
        state.icp_decisions[0] = "pending"
        _, incomplete = _accepted_setup_observations({2: state}, [0, 1])
        self.assertIn("review ICP proposal", incomplete[0])


if __name__ == "__main__":
    unittest.main()
