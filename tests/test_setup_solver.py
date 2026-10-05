"""Synthetic, ground-truth checks for the joint turntable setup fit."""

from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from src.backend.setup_solver import RigidPose, solve_setup


def compose(a: RigidPose, b: RigidPose) -> RigidPose:
    return RigidPose(a.rotation @ b.rotation, a.rotation @ b.translation + a.translation)


class SetupSolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frames = {
            0: RigidPose(np.eye(3), np.array([0.0, 0.0, 0.6])),
            1: RigidPose(Rotation.from_euler("z", 35, degrees=True).as_matrix(), np.array([0.01, -0.02, 0.6])),
            2: RigidPose(Rotation.from_euler("z", 75, degrees=True).as_matrix(), np.array([0.02, 0.01, 0.6])),
        }
        self.objects = {
            1: RigidPose(Rotation.from_euler("x", 15, degrees=True).as_matrix(), np.array([0.1, 0.0, 0.03])),
            2: RigidPose(Rotation.from_euler("y", -20, degrees=True).as_matrix(), np.array([-0.08, 0.02, 0.04])),
            3: RigidPose(Rotation.from_euler("xyz", [5, 9, -6], degrees=True).as_matrix(), np.array([0.0, 0.11, 0.06])),
        }

    def test_fills_hidden_pairs_from_connected_manual_observations(self) -> None:
        visible = ((0, 1), (0, 2), (1, 2), (1, 3), (2, 1), (2, 3))
        observations = {(f, o): compose(self.frames[f], self.objects[o]) for f, o in visible}
        fit = solve_setup(observations, [0, 1, 2], [1, 2, 3])
        self.assertEqual(len(fit.predicted), 9)
        for frame in self.frames:
            for obj in self.objects:
                expected = compose(self.frames[frame], self.objects[obj])
                actual = fit.predicted[(frame, obj)]
                np.testing.assert_allclose(actual.translation, expected.translation, atol=1e-6)
                np.testing.assert_allclose(actual.rotation, expected.rotation, atol=1e-6)

    def test_rejects_unobserved_object_and_frame(self) -> None:
        observations = {(0, 1): compose(self.frames[0], self.objects[1])}
        with self.assertRaisesRegex(ValueError, "unconnected frames"):
            solve_setup(observations, [0, 1], [1, 2])

    def test_rejects_disconnected_observation_groups(self) -> None:
        observations = {
            (0, 1): compose(self.frames[0], self.objects[1]),
            (1, 2): compose(self.frames[1], self.objects[2]),
        }
        with self.assertRaisesRegex(ValueError, "unconnected frames"):
            solve_setup(observations, [0, 1], [1, 2])


if __name__ == "__main__":
    unittest.main()
