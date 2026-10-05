"""Joint rigid setup fit from manually solved model-to-camera poses.

For each observed (frame, object), M[f, o] ≈ F[f] @ O[o], where F is the
setup-to-camera pose in that frame and O is the fixed object-in-setup pose.
The first observed frame fixes the otherwise arbitrary setup-coordinate gauge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class RigidPose:
    rotation: np.ndarray
    translation: np.ndarray  # metres
    rmse: float = 0.003

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=np.float64)
        translation = np.asarray(self.translation, dtype=np.float64)
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError("Rigid pose must contain a 3x3 rotation and 3-vector translation.")
        if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            raise ValueError("Rigid pose contains non-finite values.")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3) or np.linalg.det(rotation) < 0.99:
            raise ValueError("Rigid pose rotation must be a proper orthonormal matrix.")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)


@dataclass(frozen=True)
class SetupFit:
    predicted: dict[tuple[int, int], RigidPose]
    frame_poses: dict[int, RigidPose]
    object_poses: dict[int, RigidPose]
    translation_residual_mm: dict[tuple[int, int], float]
    rotation_residual_deg: dict[tuple[int, int], float]


def _compose(left: RigidPose, right: RigidPose) -> RigidPose:
    return RigidPose(left.rotation @ right.rotation, left.rotation @ right.translation + left.translation)


def _inverse(pose: RigidPose) -> RigidPose:
    rotation = pose.rotation.T
    return RigidPose(rotation, -(rotation @ pose.translation))


def _pack(pose: RigidPose) -> np.ndarray:
    return np.r_[Rotation.from_matrix(pose.rotation).as_rotvec(), pose.translation]


def _unpack(values: np.ndarray) -> RigidPose:
    return RigidPose(Rotation.from_rotvec(values[:3]).as_matrix(), values[3:6])


def solve_setup(
    observations: dict[tuple[int, int], RigidPose],
    frame_ids: list[int],
    object_ids: list[int],
) -> SetupFit:
    """Fit all frames and objects in one connected rigid setup.

    The graph must connect every requested frame and object. Missing objects
    cannot be inferred from this data; missing frames require a motion cue.
    """
    frames, objects = sorted(set(frame_ids)), sorted(set(object_ids))
    if not frames or not objects:
        raise ValueError("A setup needs at least one frame and object.")
    if not observations:
        raise ValueError("Solve at least one visible object pose before solving the setup.")
    invalid = set(observations) - {(f, o) for f in frames for o in objects}
    if invalid:
        raise ValueError(f"Observation {min(invalid)} is outside this setup.")

    # The first observed frame establishes the setup coordinate gauge.
    anchor = min(frame for frame, _ in observations)
    identity = RigidPose(np.eye(3), np.zeros(3))
    frame_guess: dict[int, RigidPose] = {anchor: identity}
    object_guess: dict[int, RigidPose] = {}
    for _ in range(len(frames) + len(objects)):
        changed = False
        for (frame, obj), measured in sorted(observations.items()):
            if frame in frame_guess and obj not in object_guess:
                object_guess[obj] = _compose(_inverse(frame_guess[frame]), measured)
                changed = True
            elif obj in object_guess and frame not in frame_guess:
                frame_guess[frame] = _compose(measured, _inverse(object_guess[obj]))
                changed = True
        if not changed:
            break
    missing_frames = set(frames) - frame_guess.keys()
    missing_objects = set(objects) - object_guess.keys()
    if missing_frames or missing_objects:
        details = []
        if missing_frames:
            details.append(f"unconnected frames: {sorted(missing_frames)}")
        if missing_objects:
            details.append(f"unobserved or unconnected objects: {sorted(missing_objects)}")
        raise ValueError("Cannot solve setup; " + "; ".join(details) + ". Annotate bridging objects/frames first.")

    variable_frames = [frame for frame in frames if frame != anchor]
    x0 = np.concatenate([_pack(frame_guess[f]) for f in variable_frames] + [_pack(object_guess[o]) for o in objects])
    measured_items = sorted(observations.items())

    def decode(values: np.ndarray) -> tuple[dict[int, RigidPose], dict[int, RigidPose]]:
        estimates_f = {anchor: identity}
        estimates_o: dict[int, RigidPose] = {}
        offset = 0
        for frame in variable_frames:
            estimates_f[frame] = _unpack(values[offset : offset + 6]); offset += 6
        for obj in objects:
            estimates_o[obj] = _unpack(values[offset : offset + 6]); offset += 6
        return estimates_f, estimates_o

    def residual(values: np.ndarray) -> np.ndarray:
        estimates_f, estimates_o = decode(values)
        terms = []
        for (frame, obj), measured in measured_items:
            fitted = _compose(estimates_f[frame], estimates_o[obj])
            scale = float(np.clip(measured.rmse, 0.003, 0.015))
            rotation_error = Rotation.from_matrix(measured.rotation.T @ fitted.rotation).as_rotvec()
            terms.extend((0.05 * rotation_error / scale).tolist())
            terms.extend(((fitted.translation - measured.translation) / scale).tolist())
        return np.asarray(terms)

    optimized = least_squares(residual, x0, loss="soft_l1", f_scale=1.0, max_nfev=300)
    if not optimized.success and np.linalg.norm(optimized.fun) > np.linalg.norm(residual(x0)):
        raise ValueError(f"Setup optimization failed: {optimized.message}")
    fitted_frames, fitted_objects = decode(optimized.x)
    predicted = {(frame, obj): _compose(fitted_frames[frame], fitted_objects[obj]) for frame in frames for obj in objects}
    translation_errors = {}
    rotation_errors = {}
    for key, measured in measured_items:
        fitted = predicted[key]
        translation_errors[key] = float(np.linalg.norm(fitted.translation - measured.translation) * 1000.0)
        rotation_errors[key] = float(np.linalg.norm(Rotation.from_matrix(measured.rotation.T @ fitted.rotation).as_rotvec()) * 180.0 / np.pi)
    return SetupFit(predicted, fitted_frames, fitted_objects, translation_errors, rotation_errors)
