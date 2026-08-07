"""Loader for the small BOP-style RGBD dataset shipped with this project."""

from __future__ import annotations

import json
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Model:
    """Object-model metadata; mesh decoding is deliberately deferred."""

    object_id: int
    mesh_path: Path


@dataclass(frozen=True)
class ReferenceInstance:
    """Read-only supplied ground truth, expressed in metres."""

    object_id: int
    rotation_m2c: np.ndarray
    translation_m2c: np.ndarray


@dataclass
class Pose:
    """One complete RGBD frame with lazily decoded image arrays."""

    frame_id: int
    pose_class: str
    camera_intrinsics: np.ndarray
    reference_instances: tuple[ReferenceInstance, ...]
    rgb_path: Path
    depth_path: Path
    depth_scale: float
    _rgb_cache: np.ndarray | None = None
    _depth_cache: np.ndarray | None = None

    @property
    def rgb(self) -> np.ndarray:
        if self._rgb_cache is None:
            self._rgb_cache = _load_rgb(self.rgb_path)
        return self._rgb_cache

    @property
    def depth(self) -> np.ndarray:
        if self._depth_cache is None:
            self._depth_cache = _load_depth_metres(self.depth_path, self.depth_scale)
        return self._depth_cache

    def release_images(self) -> None:
        """Release decoded arrays; they are transparently reloaded on demand."""
        self._rgb_cache = None
        self._depth_cache = None


class DataSet:
    """Lazy RGBD frames, object metadata, and optional reference labels.

    Accessing ``Pose.rgb`` or ``Pose.depth`` returns decoded arrays while a
    bounded cache prevents large datasets from remaining fully resident in
    memory. All public geometry uses metres: ``depth`` is ``float32`` in
    metres and reference translations are ``float64`` metres.
    """

    def __init__(
        self,
        root: Path,
        frames: dict[int, Pose],
        models: dict[int, Model],
        load_warnings: tuple[str, ...],
    ) -> None:
        self.root = root
        self._frames = frames
        self.models = models
        self.load_warnings = load_warnings
        self._image_cache: OrderedDict[int, Pose] = OrderedDict()
        self._image_cache_size = 3

        # Each scene corresponds to 1 taken rgbd image.
        # The dataset will contain multiple scenes for the same pose, (e.g. with different lighting etc.)
        # However we only need annotate 1 of those scenes for the pose.
        # Because we will use different pose classes, use a dict of
        # {pose_class: [pose1, pose2, ...]} as the datatype.
        # Currently we have only 1 pose_class of '0'.
        self.poses: dict[str, list[Pose]] = {"0": list(frames.values())}

    @classmethod
    def load(cls, path: str | Path) -> "DataSet":
        """Load complete frames from a BOP-style dataset directory.

        Frames that lack RGB, depth, or camera intrinsics are skipped and
        recorded in ``load_warnings``.  ``scene_gt.json`` is optional and is
        only exposed as read-only reference data for overlays/evaluation.
        """
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {root}")

        camera_path = root / "scene_camera.json"
        if not camera_path.is_file():
            raise FileNotFoundError(f"Missing camera metadata: {camera_path}")
        camera_data = _read_json(camera_path)
        ground_truth_path = root / "scene_gt.json"
        ground_truth = _read_json(ground_truth_path) if ground_truth_path.is_file() else {}

        models = _load_models(root / "models")
        messages: list[str] = []
        frames: dict[int, Pose] = {}
        frame_keys = sorted(set(camera_data) | set(ground_truth), key=int)
        for key in frame_keys:
            frame_id = int(key)
            camera = camera_data.get(key)
            rgb_path = root / "rgb" / f"{frame_id:06}.png"
            depth_path = root / "depth" / f"{frame_id:06}.png"
            missing = [
                name
                for name, present in (
                    ("camera intrinsics", camera is not None),
                    ("RGB image", rgb_path.is_file()),
                    ("depth image", depth_path.is_file()),
                )
                if not present
            ]
            if missing:
                message = f"Skipping frame {frame_id}: missing {', '.join(missing)}."
                messages.append(message)
                warnings.warn(message, stacklevel=2)
                continue

            intrinsics = np.asarray(camera["cam_K"], dtype=np.float64).reshape(3, 3)
            frames[frame_id] = Pose(
                frame_id=frame_id,
                pose_class="0",
                camera_intrinsics=intrinsics,
                reference_instances=_reference_instances(ground_truth.get(key, [])),
                rgb_path=rgb_path,
                depth_path=depth_path,
                depth_scale=float(camera.get("depth_scale", 1.0)),
            )

        return cls(root=root, frames=frames, models=models, load_warnings=tuple(messages))

    @property
    def camera_intrinsics(self) -> dict[int, np.ndarray]:
        """Per-frame intrinsic matrices, retained for the skeleton API."""
        return {frame_id: pose.camera_intrinsics for frame_id, pose in self._frames.items()}

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, key: int | str) -> Pose:
        """Return a complete frame by its numeric BOP frame ID."""
        frame_id = int(key)
        pose = self._frames[frame_id]
        if frame_id in self._image_cache:
            self._image_cache.move_to_end(frame_id)
        else:
            self._image_cache[frame_id] = pose
            while len(self._image_cache) > self._image_cache_size:
                _, evicted = self._image_cache.popitem(last=False)
                evicted.release_images()
        return pose


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def _load_models(models_dir: Path) -> dict[int, Model]:
    if not models_dir.is_dir():
        return {}
    models: dict[int, Model] = {}
    for mesh_path in sorted(models_dir.glob("obj_*.ply")):
        try:
            object_id = int(mesh_path.stem.removeprefix("obj_"))
        except ValueError:
            continue
        models[object_id] = Model(object_id=object_id, mesh_path=mesh_path)
    return models


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB")).copy()


def _load_depth_metres(path: Path, depth_scale: float) -> np.ndarray:
    with Image.open(path) as image:
        raw_depth = np.asarray(image).copy()
    if raw_depth.ndim != 2:
        raise ValueError(f"Expected a single-channel depth image: {path}")

    # BOP's ``depth_scale`` converts stored depth values to millimetres.
    depth = raw_depth.astype(np.float32) * depth_scale / 1000.0
    invalid = (raw_depth == 0) | (raw_depth == np.iinfo(raw_depth.dtype).max)
    depth[invalid] = np.nan
    return depth


def _reference_instances(instances: list[dict[str, Any]]) -> tuple[ReferenceInstance, ...]:
    return tuple(
        ReferenceInstance(
            object_id=int(instance["obj_id"]),
            rotation_m2c=np.asarray(instance["cam_R_m2c"], dtype=np.float64).reshape(3, 3),
            translation_m2c=np.asarray(instance["cam_t_m2c"], dtype=np.float64).reshape(3) / 1000.0,
        )
        for instance in instances
    )
