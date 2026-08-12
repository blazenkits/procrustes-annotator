"""Open3D ICP refinement for Procrustes Annotator pose exports.

All geometry is expressed in metres internally. Input and output annotation
JSON follows the app's BOP-style convention: ``cam_R_m2c`` maps model points
to camera coordinates and ``cam_t_m2c`` is stored in millimetres.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import open3d as o3d

try:  # Supports package imports and ``python src/backend/icp.py``.
    from .loader import DataSet, Model, Pose
except ImportError:  # pragma: no cover - used only for direct script execution.
    from loader import DataSet, Model, Pose


class ICPMode(str, Enum):
    """Correspondence error minimized by Open3D ICP."""

    PointToPlane = "point-to-plane"
    PointToPoint = "point-to-point"


@dataclass(frozen=True)
class ICPResults:
    """A complete model-to-camera ICP result.

    ``inlier_rmse`` is in metres. ``correspondence_set`` contains source and
    target point indices and is intentionally not serialized to annotation
    JSON; only its length is saved there.
    """

    R: np.ndarray
    t: np.ndarray
    fitness: float
    inlier_rmse: float
    transformation: np.ndarray
    correspondence_set: np.ndarray


@dataclass(frozen=True)
class ICPConfig:
    """Preprocessing and registration settings, all distances in metres."""

    mode: ICPMode = ICPMode.PointToPlane
    model_sample_count: int = 50_000
    model_scale: float = 0.001
    target_voxel_size: float = 0.003
    max_correspondence_distance: float = 0.01
    max_iteration: int = 50
    relative_fitness: float = 1e-6
    relative_rmse: float = 1e-6
    normal_radius: float = 0.01
    normal_max_nn: int = 30
    visibility_radius_factor: float = 100.0
    random_seed: int = 0

    def validate(self) -> None:
        if not isinstance(self.mode, ICPMode):
            raise ValueError("mode must be an ICPMode value.")
        if self.model_sample_count < 3:
            raise ValueError("model_sample_count must be at least 3.")
        if self.model_scale <= 0:
            raise ValueError("model_scale must be positive.")
        if self.target_voxel_size < 0:
            raise ValueError("target_voxel_size cannot be negative.")
        if self.max_correspondence_distance <= 0:
            raise ValueError("max_correspondence_distance must be positive.")
        if self.max_iteration < 1:
            raise ValueError("max_iteration must be at least 1.")
        if self.relative_fitness < 0 or self.relative_rmse < 0:
            raise ValueError("relative convergence tolerances cannot be negative.")
        if self.normal_radius <= 0 or self.normal_max_nn < 3:
            raise ValueError("Normal estimation requires a positive radius and max_nn >= 3.")
        if self.visibility_radius_factor <= 1:
            raise ValueError("visibility_radius_factor must be greater than 1.")


def load_sampled_model(
    model: Model | str | Path,
    *,
    sample_count: int = 50_000,
    model_scale: float = 0.001,
) -> o3d.geometry.PointCloud:
    """Load a triangle mesh and uniformly sample its surface.

    BOP PLY models are conventionally stored in millimetres; ``model_scale``
    converts their coordinates to the metre convention used by this module.
    """
    if sample_count < 3:
        raise ValueError("sample_count must be at least 3.")
    if model_scale <= 0:
        raise ValueError("model_scale must be positive.")
    path = model.mesh_path if isinstance(model, Model) else Path(model)
    if not path.is_file():
        raise FileNotFoundError(f"Missing object model: {path}")

    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty() or not mesh.has_triangles():
        raise ValueError(f"Object model is not a non-empty triangle mesh: {path}")
    mesh.scale(model_scale, center=(0.0, 0.0, 0.0))
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    sampled = mesh.sample_points_uniformly(number_of_points=sample_count)
    if sampled.is_empty():
        raise ValueError(f"Uniform sampling produced no points for {path}")
    return sampled


def image_to_point_cloud(
    pose: Pose,
    *,
    voxel_size: float = 0.003,
    include_colors: bool = True,
) -> o3d.geometry.PointCloud:
    """Vectorize the RGB-D image into a camera-coordinate point cloud.

    This is the array form of :func:`annotate.unproject`: ``x=(u-cx)z/fx``
    and ``y=(v-cy)z/fy``. The loader has already converted raw depth values
    to metres and replaced raw 0 and 65535 with NaN.

    TODO: Restrict the target to an object mask or rough-pose region instead
    of including every valid image-depth pixel.
    """
    if voxel_size < 0:
        raise ValueError("voxel_size cannot be negative.")
    depth = np.asarray(pose.depth, dtype=np.float64)
    valid = np.isfinite(depth) & (depth > 0)
    rows, columns = np.nonzero(valid)
    if not len(rows):
        raise ValueError(f"Frame {pose.frame_id} contains no valid depth pixels.")

    z = depth[rows, columns]
    intrinsics = np.asarray(pose.camera_intrinsics, dtype=np.float64).reshape(3, 3)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    if not np.isfinite(intrinsics).all() or fx == 0 or fy == 0:
        raise ValueError(f"Frame {pose.frame_id} has invalid camera intrinsics.")
    points = np.column_stack(
        (
            (columns - cx) * z / fx,
            (rows - cy) * z / fy,
            z,
        )
    )

    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    if include_colors:
        rgb = np.asarray(pose.rgb)
        if rgb.shape[:2] != depth.shape:
            raise ValueError(
                f"Frame {pose.frame_id} RGB and depth dimensions do not match."
            )
        cloud.colors = o3d.utility.Vector3dVector(
            rgb[rows, columns, :3].astype(np.float64) / 255.0
        )
    if voxel_size > 0:
        cloud = cloud.voxel_down_sample(voxel_size)
    if len(cloud.points) < 3:
        raise ValueError(f"Frame {pose.frame_id} target cloud has fewer than 3 points.")
    return cloud


def get_visible_points(
    model_point_cloud: o3d.geometry.PointCloud,
    rough_model_rotation: np.ndarray,
    rough_model_translation: np.ndarray,
    camera_intrinsics: np.ndarray,
    image_shape: tuple[int, int] | tuple[int, int, int],
    *,
    visibility_radius_factor: float = 100.0,
) -> o3d.geometry.PointCloud:
    """Return sampled model points visible from the camera under a rough pose.

    Hidden-point removal handles model self-occlusion. Positive depth and
    projection bounds then restrict candidates to points visible in the RGB-D
    image. Returned points remain in model coordinates so the rough
    model-to-camera transform can be supplied to Open3D as ICP's initializer.
    """
    if visibility_radius_factor <= 1:
        raise ValueError("visibility_radius_factor must be greater than 1.")
    model_points = np.asarray(model_point_cloud.points, dtype=np.float64)
    if len(model_points) < 3:
        raise ValueError("Model point cloud must contain at least 3 points.")
    rotation = np.asarray(rough_model_rotation, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(rough_model_translation, dtype=np.float64).reshape(3)
    intrinsics = np.asarray(camera_intrinsics, dtype=np.float64).reshape(3, 3)
    if not (
        np.isfinite(model_points).all()
        and np.isfinite(rotation).all()
        and np.isfinite(translation).all()
        and np.isfinite(intrinsics).all()
    ):
        raise ValueError("Visibility inputs must be finite.")
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("image_shape must have positive height and width.")

    camera_points = model_points @ rotation.T + translation
    positive_indices = np.flatnonzero(camera_points[:, 2] > 0)
    if len(positive_indices) < 3:
        raise ValueError("Fewer than 3 model samples are in front of the camera.")
    camera_cloud = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(camera_points[positive_indices])
    )
    farthest = float(np.linalg.norm(camera_points[positive_indices], axis=1).max())
    radius = farthest * visibility_radius_factor
    _, visible_local_indices = camera_cloud.hidden_point_removal(
        camera_location=np.zeros(3), radius=radius
    )
    visible_indices = positive_indices[np.asarray(visible_local_indices, dtype=int)]
    visible_camera = camera_points[visible_indices]

    u = intrinsics[0, 0] * visible_camera[:, 0] / visible_camera[:, 2] + intrinsics[0, 2]
    v = intrinsics[1, 1] * visible_camera[:, 1] / visible_camera[:, 2] + intrinsics[1, 2]
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    visible_indices = visible_indices[inside]
    if len(visible_indices) < 3:
        raise ValueError("Fewer than 3 visible model samples project inside the image.")
    return model_point_cloud.select_by_index(visible_indices.tolist())


def get_icp_candidates(
    model_point_cloud: o3d.geometry.PointCloud,
    rough_model_rotation: np.ndarray,
    rough_model_translation: np.ndarray,
    camera_intrinsics: np.ndarray,
    image_shape: tuple[int, int] | tuple[int, int, int],
    *,
    visibility_radius_factor: float = 100.0,
) -> o3d.geometry.PointCloud:
    """Choose model samples used by ICP; currently all camera-visible points."""
    return get_visible_points(
        model_point_cloud,
        rough_model_rotation,
        rough_model_translation,
        camera_intrinsics,
        image_shape,
        visibility_radius_factor=visibility_radius_factor,
    )


def estimate_target_normals(
    target: o3d.geometry.PointCloud,
    *,
    radius: float = 0.01,
    max_nn: int = 30,
) -> None:
    """Estimate target normals and orient them toward the camera origin."""
    if radius <= 0 or max_nn < 3:
        raise ValueError("Normal estimation requires radius > 0 and max_nn >= 3.")
    target.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    target.orient_normals_towards_camera_location(np.zeros(3))


def run_icp(
    icp_candidate_points: o3d.geometry.PointCloud,
    image_point_cloud: o3d.geometry.PointCloud,
    initial_transformation: np.ndarray,
    *,
    mode: ICPMode = ICPMode.PointToPlane,
    max_correspondence_distance: float = 0.01,
    max_iteration: int = 50,
    relative_fitness: float = 1e-6,
    relative_rmse: float = 1e-6,
    normal_radius: float = 0.01,
    normal_max_nn: int = 30,
) -> ICPResults:
    """Refine a full model-to-camera transform using single-scale ICP."""
    if len(icp_candidate_points.points) < 3 or len(image_point_cloud.points) < 3:
        raise ValueError("ICP source and target must each contain at least 3 points.")
    if max_correspondence_distance <= 0:
        raise ValueError("max_correspondence_distance must be positive.")
    if max_iteration < 1:
        raise ValueError("max_iteration must be at least 1.")
    if relative_fitness < 0 or relative_rmse < 0:
        raise ValueError("relative convergence tolerances cannot be negative.")
    transformation = np.asarray(initial_transformation, dtype=np.float64)
    if transformation.shape != (4, 4) or not np.isfinite(transformation).all():
        raise ValueError("initial_transformation must be a finite 4x4 matrix.")
    if not isinstance(mode, ICPMode):
        mode = ICPMode(mode)

    if mode is ICPMode.PointToPlane:
        if not image_point_cloud.has_normals():
            estimate_target_normals(
                image_point_cloud, radius=normal_radius, max_nn=normal_max_nn
            )
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=relative_fitness,
        relative_rmse=relative_rmse,
        max_iteration=max_iteration,
    )
    registration = o3d.pipelines.registration.registration_icp(
        icp_candidate_points,
        image_point_cloud,
        max_correspondence_distance,
        transformation,
        estimation,
        criteria,
    )
    refined = np.asarray(registration.transformation, dtype=np.float64).copy()
    correspondences = np.asarray(registration.correspondence_set, dtype=np.int64).copy()
    return ICPResults(
        R=refined[:3, :3].copy(),
        t=refined[:3, 3].copy(),
        fitness=float(registration.fitness),
        inlier_rmse=float(registration.inlier_rmse),
        transformation=refined,
        correspondence_set=correspondences,
    )


def refine_annotations(
    dataset: DataSet,
    annotation_document: dict[str, Any],
    *,
    config: ICPConfig = ICPConfig(),
) -> dict[str, Any]:
    """Refine every solved pose while preserving the input JSON structure."""
    config.validate()
    o3d.utility.random.seed(config.random_seed)
    output = copy.deepcopy(annotation_document)
    sampled_models: dict[int, o3d.geometry.PointCloud] = {}
    target_clouds: dict[int, o3d.geometry.PointCloud] = {}
    solved_count = 0

    for raw_frame_id, records in output.items():
        try:
            frame_id = int(raw_frame_id)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Unexpected non-frame JSON key: {raw_frame_id!r}") from error
        if not isinstance(records, list):
            raise ValueError(f"Frame {frame_id} must contain a list of annotations.")
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"Frame {frame_id} contains a non-object annotation.")
            status = record.get("annotation_status", "solved")
            if status == "deferred":
                continue
            if status != "solved":
                raise ValueError(f"Frame {frame_id} has unknown status {status!r}.")
            object_id = int(record["obj_id"])
            if object_id not in dataset.models:
                raise FileNotFoundError(
                    f"Frame {frame_id} references missing model obj_{object_id:06d}.ply"
                )
            try:
                pose = dataset[frame_id]
            except KeyError as error:
                raise ValueError(
                    f"Solved frame {frame_id} is unavailable in the loaded dataset."
                ) from error

            rough_rotation = np.asarray(record["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
            rough_translation_m = (
                np.asarray(record["cam_t_m2c"], dtype=np.float64).reshape(3) / 1000.0
            )
            if not (
                np.isfinite(rough_rotation).all()
                and np.isfinite(rough_translation_m).all()
            ):
                raise ValueError(f"Frame {frame_id}, object {object_id} has a non-finite pose.")
            rough_transform = np.eye(4, dtype=np.float64)
            rough_transform[:3, :3] = rough_rotation
            rough_transform[:3, 3] = rough_translation_m

            if object_id not in sampled_models:
                sampled_models[object_id] = load_sampled_model(
                    dataset.models[object_id],
                    sample_count=config.model_sample_count,
                    model_scale=config.model_scale,
                )
            sampled_model = sampled_models[object_id]
            candidates = get_icp_candidates(
                sampled_model,
                rough_rotation,
                rough_translation_m,
                pose.camera_intrinsics,
                pose.depth.shape,
                visibility_radius_factor=config.visibility_radius_factor,
            )
            if frame_id not in target_clouds:
                target = image_to_point_cloud(
                    pose, voxel_size=config.target_voxel_size, include_colors=False
                )
                if config.mode is ICPMode.PointToPlane:
                    estimate_target_normals(
                        target,
                        radius=config.normal_radius,
                        max_nn=config.normal_max_nn,
                    )
                target_clouds[frame_id] = target
            target = target_clouds[frame_id]
            result = run_icp(
                candidates,
                target,
                rough_transform,
                mode=config.mode,
                max_correspondence_distance=config.max_correspondence_distance,
                max_iteration=config.max_iteration,
                relative_fitness=config.relative_fitness,
                relative_rmse=config.relative_rmse,
                normal_radius=config.normal_radius,
                normal_max_nn=config.normal_max_nn,
            )

            record["cam_R_m2c"] = result.R.reshape(-1).tolist()
            record["cam_t_m2c"] = (result.t * 1000.0).tolist()
            record["icp"] = {
                "status": "refined",
                "mode": config.mode.value,
                "fitness": result.fitness,
                "inlier_rmse_mm": result.inlier_rmse * 1000.0,
                "correspondence_count": int(len(result.correspondence_set)),
                "candidate_point_count": int(len(candidates.points)),
                "target_point_count": int(len(target.points)),
                "rough_cam_R_m2c": rough_rotation.reshape(-1).tolist(),
                "rough_cam_t_m2c": (rough_translation_m * 1000.0).tolist(),
                "transformation_m2c": result.transformation.reshape(-1).tolist(),
                "transformation_translation_unit": "metres",
                "annotation_rmsd_describes": "rough_pose",
                "parameters": {
                    "model_sample_count": config.model_sample_count,
                    "model_scale": config.model_scale,
                    "target_voxel_size_mm": config.target_voxel_size * 1000.0,
                    "max_correspondence_distance_mm": (
                        config.max_correspondence_distance * 1000.0
                    ),
                    "max_iteration": config.max_iteration,
                    "relative_fitness": config.relative_fitness,
                    "relative_rmse": config.relative_rmse,
                    "normal_radius_mm": config.normal_radius * 1000.0,
                    "normal_max_nn": config.normal_max_nn,
                    "visibility_radius_factor": config.visibility_radius_factor,
                    "random_seed": config.random_seed,
                },
            }
            solved_count += 1
            print(
                f"frame={frame_id} obj={object_id} candidates={len(candidates.points)} "
                f"fitness={result.fitness:.4f} "
                f"inlier_rmse={result.inlier_rmse * 1000.0:.3f} mm"
            )

    if not solved_count:
        raise ValueError("Input annotation JSON contains no solved poses.")
    return output


def load_annotation_document(path: str | Path) -> dict[str, Any]:
    """Load a BOP-style pose export."""
    input_path = Path(path).expanduser().resolve()
    with input_path.open(encoding="utf-8") as file:
        document = json.load(file)
    if not isinstance(document, dict):
        raise ValueError(f"Expected a JSON object in {input_path}.")
    return document


def write_json_atomic(path: str | Path, value: object) -> None:
    """Write JSON completely before atomically replacing the destination."""
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(value, temporary, indent=2, ensure_ascii=False, allow_nan=False)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, output_path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def refine_annotation_file(
    dataset_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    *,
    config: ICPConfig = ICPConfig(),
) -> dict[str, Any]:
    """Load, refine, and atomically save an annotation export."""
    dataset = DataSet.load(dataset_path)
    document = load_annotation_document(input_path)
    refined = refine_annotations(dataset, document, config=config)
    write_json_atomic(output_path, refined)
    return refined


def _positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refine Procrustes Annotator poses using single-scale Open3D ICP."
    )
    parser.add_argument("--dataset", type=Path, required=True, help="BOP dataset root")
    parser.add_argument("--input", type=Path, required=True, help="Rough pose JSON")
    parser.add_argument("--output", type=Path, required=True, help="Refined pose JSON")
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in ICPMode],
        default=ICPMode.PointToPlane.value,
    )
    parser.add_argument("--model-sample-count", type=int, default=50_000)
    parser.add_argument("--model-scale", type=_positive_float, default=0.001)
    parser.add_argument("--target-voxel-size-mm", type=float, default=3.0)
    parser.add_argument(
        "--max-correspondence-distance-mm", type=_positive_float, default=10.0
    )
    parser.add_argument("--max-iteration", type=int, default=50)
    parser.add_argument("--relative-fitness", type=float, default=1e-6)
    parser.add_argument("--relative-rmse", type=float, default=1e-6)
    parser.add_argument("--normal-radius-mm", type=_positive_float, default=10.0)
    parser.add_argument("--normal-max-nn", type=int, default=30)
    parser.add_argument("--visibility-radius-factor", type=float, default=100.0)
    parser.add_argument("--random-seed", type=int, default=0)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    config = ICPConfig(
        mode=ICPMode(args.mode),
        model_sample_count=args.model_sample_count,
        model_scale=args.model_scale,
        target_voxel_size=args.target_voxel_size_mm / 1000.0,
        max_correspondence_distance=args.max_correspondence_distance_mm / 1000.0,
        max_iteration=args.max_iteration,
        relative_fitness=args.relative_fitness,
        relative_rmse=args.relative_rmse,
        normal_radius=args.normal_radius_mm / 1000.0,
        normal_max_nn=args.normal_max_nn,
        visibility_radius_factor=args.visibility_radius_factor,
        random_seed=args.random_seed,
    )
    try:
        refined = refine_annotation_file(
            args.dataset, args.input, args.output, config=config
        )
    except (
        OSError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    refined_count = sum(
        record.get("icp", {}).get("status") == "refined"
        for records in refined.values()
        for record in records
        if isinstance(record, dict)
    )
    print(f"Wrote {refined_count} refined poses to {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
