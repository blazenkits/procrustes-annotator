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
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

import cv2
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
    Colored = "colored"
    ProjectivePointToPlane = "projective-point-to-plane"


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
    target_neighborhood_distance: float | None = None
    max_iteration: int = 50
    relative_fitness: float = 1e-6
    relative_rmse: float = 1e-6
    normal_radius: float = 0.01
    normal_max_nn: int = 30
    colored_icp_lambda_geometric: float = 0.999
    visibility_radius_factor: float = 100.0
    projective_normal_depth_delta: float = 0.02
    projective_huber_delta: float = 0.003
    projective_max_step_rotation_deg: float = 1.0
    projective_max_step_translation: float = 0.002
    random_seed: int = 0
    sam_model_id: str | None = None
    sam_device: str = "auto"
    sam_box_padding_fraction: float = 0.25
    sam_erosion_radius_px: int = 0
    sam_overlay_output_dir: Path | None = None
    sam_mask_output_dir: Path | None = None
    sam_overlay_alpha: float = 0.45
    silhouette_weight: float = 0.0
    silhouette_max_iteration: int = 12
    silhouette_image_scale: float = 0.5
    silhouette_distance_scale_px: float = 5.0
    silhouette_splat_radius_px: int = 2
    silhouette_huber_delta_px: float = 4.0
    silhouette_pose_prior_weight: float = 0.05
    silhouette_damping: float = 1e-3
    silhouette_max_rotation_deg: float = 8.0
    silhouette_max_translation: float = 0.012
    silhouette_max_step_rotation_deg: float = 1.0
    silhouette_max_step_translation: float = 0.002
    # When set, contour samples whose rendered and observed depths disagree by
    # more than this many metres do not contribute a silhouette residual.
    # ``None`` retains the original purely 2D contour objective.
    silhouette_depth_gate: float | None = None
    # If true, reject only measured foreground that is in front of the rendered
    # contour by more than ``silhouette_depth_gate``; retain background depth
    # and invalid depth because neither establishes external occlusion.
    silhouette_depth_gate_occlusion_only: bool = False

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
        if (
            self.target_neighborhood_distance is not None
            and self.target_neighborhood_distance <= 0
        ):
            raise ValueError("target_neighborhood_distance must be positive.")
        if self.max_iteration < 1:
            raise ValueError("max_iteration must be at least 1.")
        if self.relative_fitness < 0 or self.relative_rmse < 0:
            raise ValueError("relative convergence tolerances cannot be negative.")
        if self.normal_radius <= 0 or self.normal_max_nn < 3:
            raise ValueError(
                "Normal estimation requires a positive radius and max_nn >= 3."
            )
        if not np.isfinite(self.colored_icp_lambda_geometric) or not (
            0 <= self.colored_icp_lambda_geometric <= 1
        ):
            raise ValueError("colored_icp_lambda_geometric must be in [0, 1].")
        if self.visibility_radius_factor <= 1:
            raise ValueError("visibility_radius_factor must be greater than 1.")
        if self.projective_normal_depth_delta <= 0:
            raise ValueError("projective_normal_depth_delta must be positive.")
        if self.projective_huber_delta <= 0:
            raise ValueError("projective_huber_delta must be positive.")
        if self.projective_max_step_rotation_deg <= 0:
            raise ValueError("projective_max_step_rotation_deg must be positive.")
        if self.projective_max_step_translation <= 0:
            raise ValueError("projective_max_step_translation must be positive.")
        if self.sam_model_id is not None and not self.sam_model_id.strip():
            raise ValueError("sam_model_id cannot be empty.")
        if (
            not np.isfinite(self.sam_box_padding_fraction)
            or self.sam_box_padding_fraction < 0
        ):
            raise ValueError(
                "sam_box_padding_fraction must be finite and non-negative."
            )
        if self.sam_erosion_radius_px < 0:
            raise ValueError("sam_erosion_radius_px cannot be negative.")
        if self.sam_overlay_output_dir is not None and self.sam_model_id is None:
            raise ValueError("sam_overlay_output_dir requires sam_model_id.")
        if self.sam_mask_output_dir is not None and self.sam_model_id is None:
            raise ValueError("sam_mask_output_dir requires sam_model_id.")
        if not np.isfinite(self.sam_overlay_alpha) or not (
            0 < self.sam_overlay_alpha <= 1
        ):
            raise ValueError("sam_overlay_alpha must be in the interval (0, 1].")
        if not np.isfinite(self.silhouette_weight) or self.silhouette_weight < 0:
            raise ValueError("silhouette_weight must be finite and non-negative.")
        if self.silhouette_weight > 0 and self.sam_model_id is None:
            raise ValueError("Silhouette refinement requires projected-box SAM masking.")
        if self.silhouette_weight > 0 and self.mode is not ICPMode.PointToPlane:
            raise ValueError(
                "Silhouette refinement currently supports point-to-plane ICP only."
            )
        if self.silhouette_max_iteration < 1:
            raise ValueError("silhouette_max_iteration must be at least 1.")
        if not np.isfinite(self.silhouette_image_scale) or not (
            0 < self.silhouette_image_scale <= 1
        ):
            raise ValueError("silhouette_image_scale must be in (0, 1].")
        if (
            not np.isfinite(self.silhouette_distance_scale_px)
            or self.silhouette_distance_scale_px <= 0
        ):
            raise ValueError("silhouette_distance_scale_px must be positive.")
        if self.silhouette_splat_radius_px < 1:
            raise ValueError("silhouette_splat_radius_px must be at least 1.")
        if (
            not np.isfinite(self.silhouette_huber_delta_px)
            or self.silhouette_huber_delta_px <= 0
        ):
            raise ValueError("silhouette_huber_delta_px must be positive.")
        if (
            not np.isfinite(self.silhouette_pose_prior_weight)
            or self.silhouette_pose_prior_weight < 0
        ):
            raise ValueError(
                "silhouette_pose_prior_weight must be finite and non-negative."
            )
        if not np.isfinite(self.silhouette_damping) or self.silhouette_damping <= 0:
            raise ValueError("silhouette_damping must be positive.")
        if (
            not np.isfinite(self.silhouette_max_rotation_deg)
            or self.silhouette_max_rotation_deg <= 0
        ):
            raise ValueError("silhouette_max_rotation_deg must be positive.")
        if (
            not np.isfinite(self.silhouette_max_translation)
            or self.silhouette_max_translation <= 0
        ):
            raise ValueError("silhouette_max_translation must be positive.")
        if (
            not np.isfinite(self.silhouette_max_step_rotation_deg)
            or self.silhouette_max_step_rotation_deg <= 0
        ):
            raise ValueError("silhouette_max_step_rotation_deg must be positive.")
        if (
            not np.isfinite(self.silhouette_max_step_translation)
            or self.silhouette_max_step_translation <= 0
        ):
            raise ValueError("silhouette_max_step_translation must be positive.")
        if (
            self.silhouette_depth_gate is not None
            and (
                not np.isfinite(self.silhouette_depth_gate)
                or self.silhouette_depth_gate <= 0
            )
        ):
            raise ValueError("silhouette_depth_gate must be finite and positive.")
        if self.silhouette_depth_gate is not None and self.silhouette_weight <= 0:
            raise ValueError("silhouette_depth_gate requires silhouette_weight > 0.")


@dataclass(frozen=True)
class SAM2SegmentationResult:
    """Box-prompted SAM 2.1 masks for one object in one RGB frame."""

    mask: np.ndarray
    candidate_masks: np.ndarray
    scores: np.ndarray
    selected_index: int
    prompt_box_xyxy: np.ndarray
    target_mask: np.ndarray | None = None

    @property
    def selected_score(self) -> float:
        return float(self.scores[self.selected_index])


@dataclass(frozen=True)
class SilhouetteRefinementDiagnostics:
    """Audit information for the bounded joint geometry/silhouette stage."""

    initial_candidate: str
    initial_objective: float
    final_objective: float
    accepted_iterations: int
    attempted_iterations: int
    geometric_correspondence_count: int
    silhouette_point_count: int
    silhouette_depth_rejected_point_count: int
    rotation_delta_deg: float
    translation_delta_m: float


class SAM2ImageSegmenter:
    """Reusable SAM 2.1 image predictor loaded only when requested."""

    def __init__(self, predictor: Any, torch_module: Any, device: str, model_id: str):
        self.predictor = predictor
        self._torch = torch_module
        self.device = device
        self.model_id = model_id

    def predict_masks(
        self,
        image: np.ndarray,
        box_xyxy: np.ndarray,
        *,
        multimask_output: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return SAM masks and quality scores for one RGB image and box prompt."""
        autocast = (
            self._torch.autocast(device_type="cuda", dtype=self._torch.bfloat16)
            if self.device.startswith("cuda")
            else nullcontext()
        )
        with self._torch.inference_mode(), autocast:
            self.predictor.set_image(image)
            masks, scores, _ = self.predictor.predict(
                box=np.asarray(box_xyxy, dtype=np.float32),
                multimask_output=multimask_output,
            )
        return np.asarray(masks), np.asarray(scores)


def load_sam2_predictor(
    model_id: str = "facebook/sam2.1-hiera-small",
    *,
    device: str = "auto",
) -> SAM2ImageSegmenter:
    """Load and configure a reusable SAM 2.1 predictor from Hugging Face.

    SAM and PyTorch remain optional so normal annotator installations do not
    inherit their sizeable runtime. The checkpoint is downloaded once and
    subsequently reused from the Hugging Face cache.
    """
    try:
        import torch
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as error:
        raise RuntimeError(
            "SAM 2.1 segmentation dependencies are unavailable. Install them "
            "with `uv sync --extra segmentation`."
        ) from error

    resolved_device = device
    if resolved_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        torch_device = torch.device(resolved_device)
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"Invalid SAM device: {device!r}") from error
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SAM was configured for CUDA, but CUDA is unavailable.")

    predictor = SAM2ImagePredictor.from_pretrained(model_id)
    predictor.model.to(torch_device)
    predictor.model.eval()
    return SAM2ImageSegmenter(
        predictor=predictor,
        torch_module=torch,
        device=str(torch_device),
        model_id=model_id,
    )


def projected_model_box(
    model_point_cloud: o3d.geometry.PointCloud | np.ndarray,
    rotation_m2c: np.ndarray,
    translation_m2c_m: np.ndarray,
    camera_intrinsics: np.ndarray,
    image_shape: tuple[int, int] | tuple[int, int, int],
    *,
    padding_fraction: float = 0.25,
) -> np.ndarray:
    """Project model points and return a clipped, padded ``xyxy`` prompt box."""
    if padding_fraction < 0 or not np.isfinite(padding_fraction):
        raise ValueError("padding_fraction must be finite and non-negative.")
    points = (
        np.asarray(model_point_cloud.points, dtype=np.float64)
        if isinstance(model_point_cloud, o3d.geometry.PointCloud)
        else np.asarray(model_point_cloud, dtype=np.float64)
    )
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 3:
        raise ValueError(
            "model_point_cloud must contain at least 3 three-dimensional points."
        )
    rotation = np.asarray(rotation_m2c, dtype=np.float64)
    translation = np.asarray(translation_m2c_m, dtype=np.float64)
    intrinsics = np.asarray(camera_intrinsics, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("rotation_m2c must be a 3x3 matrix.")
    if translation.size != 3:
        raise ValueError("translation_m2c_m must contain 3 values.")
    if intrinsics.shape != (3, 3):
        raise ValueError("camera_intrinsics must be a 3x3 matrix.")
    translation = translation.reshape(3)
    if not (
        np.isfinite(points).all()
        and np.isfinite(rotation).all()
        and np.isfinite(translation).all()
        and np.isfinite(intrinsics).all()
    ):
        raise ValueError("Projection inputs must be finite.")
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("image_shape must have positive height and width.")
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    if fx <= 0 or fy <= 0:
        raise ValueError("Camera focal lengths must be positive.")

    camera_points = points @ rotation.T + translation
    visible = camera_points[:, 2] > 0
    if np.count_nonzero(visible) < 3:
        raise ValueError("Fewer than 3 model points are in front of the camera.")
    camera_points = camera_points[visible]
    u = fx * camera_points[:, 0] / camera_points[:, 2] + cx
    v = fy * camera_points[:, 1] / camera_points[:, 2] + cy
    finite = np.isfinite(u) & np.isfinite(v)
    if np.count_nonzero(finite) < 3:
        raise ValueError("Fewer than 3 model points have finite image projections.")
    u, v = u[finite], v[finite]

    x0, x1 = float(np.min(u)), float(np.max(u))
    y0, y1 = float(np.min(v)), float(np.max(v))
    box_width, box_height = x1 - x0, y1 - y0
    if box_width <= 0 or box_height <= 0:
        raise ValueError("Projected model points do not form a non-empty box.")
    x_padding = box_width * padding_fraction
    y_padding = box_height * padding_fraction
    box = np.array(
        [
            np.clip(x0 - x_padding, 0.0, width - 1.0),
            np.clip(y0 - y_padding, 0.0, height - 1.0),
            np.clip(x1 + x_padding, 0.0, width - 1.0),
            np.clip(y1 + y_padding, 0.0, height - 1.0),
        ],
        dtype=np.float32,
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("Projected model box lies outside the image.")
    return box


def segment_object_sam2(
    pose: Pose,
    model_point_cloud: o3d.geometry.PointCloud | np.ndarray,
    rotation_m2c: np.ndarray,
    translation_m2c_m: np.ndarray,
    segmenter: Any,
    *,
    padding_fraction: float = 0.25,
    multimask_output: bool = True,
) -> SAM2SegmentationResult:
    """Segment the projected object using a padded CAD-derived box prompt."""
    rgb = np.asarray(pose.rgb)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(
            f"Frame {pose.frame_id} RGB image must have at least 3 channels."
        )
    if rgb.shape[:2] != np.asarray(pose.depth).shape:
        raise ValueError(
            f"Frame {pose.frame_id} RGB and depth dimensions do not match."
        )
    if rgb.dtype != np.uint8:
        raise ValueError(f"Frame {pose.frame_id} RGB image must use uint8 values.")
    rgb = np.ascontiguousarray(rgb[:, :, :3])
    prompt_box = projected_model_box(
        model_point_cloud,
        rotation_m2c,
        translation_m2c_m,
        pose.camera_intrinsics,
        rgb.shape,
        padding_fraction=padding_fraction,
    )
    try:
        raw_masks, raw_scores = segmenter.predict_masks(
            rgb,
            prompt_box,
            multimask_output=multimask_output,
        )
    except AttributeError as error:
        raise TypeError("segmenter must provide a predict_masks() method.") from error
    masks = np.asarray(raw_masks)
    scores = np.asarray(raw_scores, dtype=np.float64).reshape(-1)
    if masks.ndim == 2:
        masks = masks[np.newaxis, ...]
    if masks.ndim != 3 or masks.shape[1:] != rgb.shape[:2]:
        raise ValueError("SAM masks must have shape (N, height, width).")
    if len(masks) == 0 or len(scores) != len(masks):
        raise ValueError("SAM must return matching non-empty mask and score arrays.")
    finite_scores = np.where(np.isfinite(scores), scores, -np.inf)
    if not np.isfinite(finite_scores).any():
        raise ValueError("SAM returned no finite mask scores.")
    candidate_masks = np.asarray(masks > 0, dtype=bool)
    selected_index = int(np.argmax(finite_scores))
    selected_mask = candidate_masks[selected_index].copy()
    if not np.any(selected_mask):
        raise ValueError("SAM selected an empty object mask.")
    target_mask: np.ndarray | None = None
    target_mask_provider = getattr(segmenter, "predict_target_mask", None)
    if callable(target_mask_provider):
        target_mask = np.asarray(
            target_mask_provider(rgb, prompt_box, selected_mask), dtype=bool
        )
        if target_mask.shape != selected_mask.shape:
            raise ValueError("Segmenter target mask must match the selected mask shape.")
    return SAM2SegmentationResult(
        mask=selected_mask,
        candidate_masks=candidate_masks,
        scores=scores.copy(),
        selected_index=selected_index,
        prompt_box_xyxy=prompt_box.copy(),
        target_mask=target_mask,
    )


def erode_binary_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    """Erode a 2D object mask with an elliptical pixel-radius kernel."""
    binary = np.asarray(mask)
    if binary.ndim != 2:
        raise ValueError("mask must be a two-dimensional array.")
    if radius_px < 0:
        raise ValueError("radius_px cannot be negative.")
    binary = np.asarray(binary > 0, dtype=bool)
    if radius_px == 0:
        return binary.copy()
    diameter = 2 * int(radius_px) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    return cv2.erode(binary.astype(np.uint8), kernel).astype(bool)


def green_mask_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend a binary mask over an RGB image in green."""
    image = np.asarray(rgb)
    binary = np.asarray(mask)
    if image.ndim != 3 or image.shape[2] < 3 or image.dtype != np.uint8:
        raise ValueError("rgb must be a uint8 image with at least three channels.")
    if binary.ndim != 2 or binary.shape != image.shape[:2]:
        raise ValueError("mask dimensions must match the RGB image.")
    if not np.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("alpha must be in the interval (0, 1].")
    overlay = image[:, :, :3].astype(np.float32).copy()
    selected = binary > 0
    green = np.array([0.0, 255.0, 0.0], dtype=np.float32)
    overlay[selected] = (1.0 - alpha) * overlay[selected] + alpha * green
    return np.clip(np.rint(overlay), 0, 255).astype(np.uint8)


def write_rgb_image(path: str | Path, rgb: np.ndarray) -> None:
    """Write an RGB uint8 image, creating its parent directory if needed."""
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("rgb must be a three-channel uint8 image.")
    if not cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not write image: {output_path}")


def write_binary_mask(path: str | Path, mask: np.ndarray) -> None:
    """Write a two-dimensional binary mask as a lossless 0/255 PNG."""
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    binary = np.asarray(mask)
    if binary.ndim != 2:
        raise ValueError("mask must be a two-dimensional array.")
    image = np.asarray(binary > 0, dtype=np.uint8) * 255
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Could not write mask: {output_path}")


def load_sampled_model(
    model: Model | str | Path,
    *,
    sample_count: int = 50_000,
    model_scale: float = 0.001,
    include_colors: bool = False,
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
    if include_colors and not mesh.has_vertex_colors():
        _apply_uv_texture_as_vertex_colors(mesh, path)
    mesh.scale(model_scale, center=(0.0, 0.0, 0.0))
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    sampled = mesh.sample_points_uniformly(number_of_points=sample_count)
    if sampled.is_empty():
        raise ValueError(f"Uniform sampling produced no points for {path}")
    return sampled


def _find_ply_texture_path(path: Path) -> Path | None:
    """Resolve a PLY TextureFile comment or a same-stem image sidecar."""
    try:
        with path.open("rb") as file:
            for _ in range(256):
                raw_line = file.readline(4096)
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", errors="replace").strip()
                if line == "end_header":
                    break
                prefix = "comment TextureFile "
                if line.startswith(prefix):
                    texture_name = line[len(prefix) :].strip()
                    candidate = path.parent / texture_name
                    if texture_name and candidate.is_file():
                        return candidate
    except OSError:
        return None
    for extension in (".png", ".jpg", ".jpeg"):
        candidate = path.with_suffix(extension)
        if candidate.is_file():
            return candidate
    return None


def _sample_texture_at_uv(texture_rgb: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinearly sample an RGB image at bottom-left-origin UV coordinates."""
    image = np.asarray(texture_rgb)
    coordinates = np.asarray(uv, dtype=np.float64)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Texture image must contain at least three channels.")
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("Texture coordinates must have shape (N, 2).")
    if not np.isfinite(coordinates).all():
        raise ValueError("Texture coordinates must be finite.")
    height, width = image.shape[:2]
    map_x = np.clip(coordinates[:, 0], 0.0, 1.0) * (width - 1)
    map_y = (1.0 - np.clip(coordinates[:, 1], 0.0, 1.0)) * (height - 1)
    sampled = cv2.remap(
        np.asarray(image[:, :, :3], dtype=np.float32),
        map_x.astype(np.float32).reshape(-1, 1),
        map_y.astype(np.float32).reshape(-1, 1),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return sampled.reshape(-1, 3).astype(np.float64) / 255.0


def _apply_uv_texture_as_vertex_colors(
    mesh: o3d.geometry.TriangleMesh,
    path: Path,
) -> None:
    """Populate Open3D vertex colors from a PyVista-readable PLY UV map."""
    texture_path = _find_ply_texture_path(path)
    if texture_path is None:
        raise ValueError(f"Colored ICP requires colors or a texture map in {path}.")
    try:
        import pyvista as pv
    except ImportError as error:  # pragma: no cover - base app requires PyVista.
        raise RuntimeError("UV-textured colored ICP requires PyVista.") from error
    poly_data = pv.read(path)
    texture_coordinates = poly_data.active_texture_coordinates
    if texture_coordinates is None:
        raise ValueError(f"Colored ICP found no UV coordinates in {path}.")
    open3d_vertices = np.asarray(mesh.vertices)
    pyvista_vertices = np.asarray(poly_data.points)
    if open3d_vertices.shape != pyvista_vertices.shape or not np.allclose(
        open3d_vertices,
        pyvista_vertices,
        rtol=0.0,
        atol=5e-6,
    ):
        raise ValueError(f"PLY readers disagree on vertex ordering for {path}.")
    texture_bgr = cv2.imread(str(texture_path), cv2.IMREAD_COLOR)
    if texture_bgr is None:
        raise ValueError(f"Could not read model texture: {texture_path}")
    texture_rgb = cv2.cvtColor(texture_bgr, cv2.COLOR_BGR2RGB)
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        _sample_texture_at_uv(texture_rgb, texture_coordinates)
    )


def image_to_point_cloud(
    pose: Pose,
    *,
    voxel_size: float = 0.003,
    include_colors: bool = True,
    pixel_mask: np.ndarray | None = None,
) -> o3d.geometry.PointCloud:
    """Vectorize the RGB-D image into a camera-coordinate point cloud.

    This is the array form of :func:`annotate.unproject`: ``x=(u-cx)z/fx``
    and ``y=(v-cy)z/fy``. The loader has already converted raw depth values
    to metres and replaced raw 0 and 65535 with NaN.

    When ``pixel_mask`` is supplied, only valid-depth pixels inside that mask
    become ICP target points.
    """
    if voxel_size < 0:
        raise ValueError("voxel_size cannot be negative.")
    depth = np.asarray(pose.depth, dtype=np.float64)
    valid = np.isfinite(depth) & (depth > 0)
    if pixel_mask is not None:
        mask = np.asarray(pixel_mask)
        if mask.ndim != 2 or mask.shape != depth.shape:
            raise ValueError(
                f"Frame {pose.frame_id} pixel mask must match the depth dimensions."
            )
        valid &= mask > 0
    rows, columns = np.nonzero(valid)
    if not len(rows):
        qualifier = " inside the pixel mask" if pixel_mask is not None else ""
        raise ValueError(
            f"Frame {pose.frame_id} contains no valid depth pixels{qualifier}."
        )

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

    u = (
        intrinsics[0, 0] * visible_camera[:, 0] / visible_camera[:, 2]
        + intrinsics[0, 2]
    )
    v = (
        intrinsics[1, 1] * visible_camera[:, 1] / visible_camera[:, 2]
        + intrinsics[1, 2]
    )
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


def select_target_neighborhood(
    target: o3d.geometry.PointCloud,
    source_model: o3d.geometry.PointCloud,
    rough_transformation: np.ndarray,
    *,
    max_distance: float,
) -> o3d.geometry.PointCloud:
    """Keep target points within a 3D epsilon-neighborhood of the rough model."""
    if max_distance <= 0:
        raise ValueError("max_distance must be positive.")
    transformed_source = copy.deepcopy(source_model)
    transformed_source.transform(
        np.asarray(rough_transformation, dtype=np.float64).reshape(4, 4)
    )
    distances = np.asarray(
        target.compute_point_cloud_distance(transformed_source), dtype=np.float64
    )
    selected = np.flatnonzero(np.isfinite(distances) & (distances <= max_distance))
    return target.select_by_index(selected.tolist())


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
    colored_icp_lambda_geometric: float = 0.999,
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
    if not np.isfinite(colored_icp_lambda_geometric) or not (
        0 <= colored_icp_lambda_geometric <= 1
    ):
        raise ValueError("colored_icp_lambda_geometric must be in [0, 1].")
    transformation = np.asarray(initial_transformation, dtype=np.float64)
    if transformation.shape != (4, 4) or not np.isfinite(transformation).all():
        raise ValueError("initial_transformation must be a finite 4x4 matrix.")
    if not isinstance(mode, ICPMode):
        mode = ICPMode(mode)

    if mode in (ICPMode.PointToPlane, ICPMode.Colored):
        if not image_point_cloud.has_normals():
            estimate_target_normals(
                image_point_cloud, radius=normal_radius, max_nn=normal_max_nn
            )
    if mode is ICPMode.Colored:
        if not icp_candidate_points.has_colors() or not image_point_cloud.has_colors():
            raise ValueError("Colored ICP requires colors on both source and target.")
        estimation = (
            o3d.pipelines.registration.TransformationEstimationForColoredICP(
                lambda_geometric=colored_icp_lambda_geometric
            )
        )
    elif mode is ICPMode.PointToPlane:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=relative_fitness,
        relative_rmse=relative_rmse,
        max_iteration=max_iteration,
    )
    if mode is ICPMode.Colored:
        registration = o3d.pipelines.registration.registration_colored_icp(
            icp_candidate_points,
            image_point_cloud,
            max_correspondence_distance,
            transformation,
            estimation,
            criteria,
        )
    else:
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


def _depth_vertex_and_normal_maps(
    depth: np.ndarray,
    camera_intrinsics: np.ndarray,
    *,
    maximum_neighbor_depth_delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Unproject an RGB-D frame and estimate organized central-difference normals."""
    if maximum_neighbor_depth_delta <= 0:
        raise ValueError("maximum_neighbor_depth_delta must be positive.")
    depth_image = np.asarray(depth, dtype=np.float64)
    if depth_image.ndim != 2:
        raise ValueError("depth must be a two-dimensional array.")
    intrinsics = np.asarray(camera_intrinsics, dtype=np.float64).reshape(3, 3)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    if not np.isfinite(intrinsics).all() or fx <= 0 or fy <= 0:
        raise ValueError("camera_intrinsics must contain finite positive focal lengths.")

    height, width = depth_image.shape
    rows, columns = np.indices((height, width), dtype=np.float64)
    valid = np.isfinite(depth_image) & (depth_image > 0)
    vertices = np.full((height, width, 3), np.nan, dtype=np.float64)
    vertices[..., 0] = (columns - cx) * depth_image / fx
    vertices[..., 1] = (rows - cy) * depth_image / fy
    vertices[..., 2] = depth_image
    vertices[~valid] = np.nan

    normals = np.full_like(vertices, np.nan)
    center = depth_image[1:-1, 1:-1]
    neighbor_depths = np.stack(
        (
            depth_image[1:-1, :-2],
            depth_image[1:-1, 2:],
            depth_image[:-2, 1:-1],
            depth_image[2:, 1:-1],
        ),
        axis=-1,
    )
    locally_valid = np.isfinite(center) & (center > 0)
    locally_valid &= np.all(np.isfinite(neighbor_depths) & (neighbor_depths > 0), axis=-1)
    locally_valid &= np.max(np.abs(neighbor_depths - center[..., None]), axis=-1) <= (
        maximum_neighbor_depth_delta
    )
    horizontal = vertices[1:-1, 2:] - vertices[1:-1, :-2]
    vertical = vertices[2:, 1:-1] - vertices[:-2, 1:-1]
    central_normals = np.cross(horizontal, vertical)
    lengths = np.linalg.norm(central_normals, axis=2)
    locally_valid &= np.isfinite(lengths) & (lengths > 1e-12)
    central_normals[locally_valid] /= lengths[locally_valid, None]
    central_normals[~locally_valid] = np.nan
    normals[1:-1, 1:-1] = central_normals
    return vertices, normals


def _projective_correspondences(
    source_points: np.ndarray,
    transformation: np.ndarray,
    camera_intrinsics: np.ndarray,
    target_vertices: np.ndarray,
    target_normals: np.ndarray,
    *,
    max_correspondence_distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Match the frontmost transformed model sample to depth at the same pixel."""
    source = np.asarray(source_points, dtype=np.float64)
    pose = np.asarray(transformation, dtype=np.float64).reshape(4, 4)
    intrinsics = np.asarray(camera_intrinsics, dtype=np.float64).reshape(3, 3)
    height, width = target_vertices.shape[:2]
    camera_points = source @ pose[:3, :3].T + pose[:3, 3]
    positive = np.flatnonzero(camera_points[:, 2] > 1e-8)
    if len(positive) < 3:
        raise ValueError("Fewer than 3 projective ICP samples are in front of the camera.")
    points = camera_points[positive]
    columns = np.rint(
        intrinsics[0, 0] * points[:, 0] / points[:, 2] + intrinsics[0, 2]
    ).astype(np.int64)
    rows = np.rint(
        intrinsics[1, 1] * points[:, 1] / points[:, 2] + intrinsics[1, 2]
    ).astype(np.int64)
    inside = (columns >= 0) & (columns < width) & (rows >= 0) & (rows < height)
    source_indices = positive[inside]
    points, rows, columns = points[inside], rows[inside], columns[inside]
    if len(points) < 3:
        raise ValueError("Fewer than 3 projective ICP samples lie inside the image.")

    # A sampled-model z-buffer: only the nearest model sample owns each pixel.
    linear_pixels = rows * width + columns
    order = np.lexsort((points[:, 2], linear_pixels))
    ordered_pixels = linear_pixels[order]
    front = order[np.r_[True, ordered_pixels[1:] != ordered_pixels[:-1]]]
    source_indices = source_indices[front]
    points, rows, columns = points[front], rows[front], columns[front]
    targets = target_vertices[rows, columns]
    normals = target_normals[rows, columns]
    finite = np.isfinite(targets).all(axis=1) & np.isfinite(normals).all(axis=1)
    delta = points - targets
    finite &= np.abs(delta[:, 2]) <= max_correspondence_distance
    finite &= np.linalg.norm(delta, axis=1) <= max_correspondence_distance
    return source_indices[finite], points[finite], targets[finite], normals[finite]


def run_projective_icp(
    source: o3d.geometry.PointCloud,
    depth: np.ndarray,
    camera_intrinsics: np.ndarray,
    initial_transformation: np.ndarray,
    *,
    max_correspondence_distance: float = 0.01,
    max_iteration: int = 50,
    normal_depth_delta: float = 0.02,
    huber_delta: float = 0.003,
    max_step_rotation_deg: float = 1.0,
    max_step_translation: float = 0.002,
) -> ICPResults:
    """Visibility-aware projective point-to-plane ICP for organized RGB-D depth."""
    if len(source.points) < 3:
        raise ValueError("Projective ICP source must contain at least 3 points.")
    if max_correspondence_distance <= 0 or normal_depth_delta <= 0:
        raise ValueError("Projective ICP distance thresholds must be positive.")
    if max_iteration < 1 or huber_delta <= 0:
        raise ValueError("Projective ICP iteration count and Huber delta must be positive.")
    transformation = np.asarray(initial_transformation, dtype=np.float64).copy()
    if transformation.shape != (4, 4) or not np.isfinite(transformation).all():
        raise ValueError("initial_transformation must be a finite 4x4 matrix.")
    target_vertices, target_normals = _depth_vertex_and_normal_maps(
        depth,
        camera_intrinsics,
        maximum_neighbor_depth_delta=normal_depth_delta,
    )
    source_points = np.asarray(source.points, dtype=np.float64)
    accepted_source = np.empty(0, dtype=np.int64)
    residual = np.empty(0, dtype=np.float64)
    for _ in range(max_iteration):
        accepted_source, camera_points, targets, normals = _projective_correspondences(
            source_points,
            transformation,
            camera_intrinsics,
            target_vertices,
            target_normals,
            max_correspondence_distance=max_correspondence_distance,
        )
        if len(accepted_source) < 6:
            break
        residual = np.einsum("ni,ni->n", normals, camera_points - targets)
        jacobian = np.einsum("ni,nij->nj", normals, _point_jacobian(camera_points))
        _, robust_weight = _huber_cost_and_weight(residual, huber_delta)
        weighted_jacobian = jacobian * np.sqrt(robust_weight)[:, None]
        weighted_residual = residual * np.sqrt(robust_weight)
        system = weighted_jacobian.T @ weighted_jacobian
        gradient = weighted_jacobian.T @ weighted_residual
        system += 1e-8 * np.diag(np.maximum(np.diag(system), 1.0))
        try:
            step = -np.linalg.solve(system, gradient)
        except np.linalg.LinAlgError:
            step = -np.linalg.lstsq(system, gradient, rcond=None)[0]
        step[:3] = _clamp_vector_norm(step[:3], np.deg2rad(max_step_rotation_deg))
        step[3:] = _clamp_vector_norm(step[3:], max_step_translation)
        if np.linalg.norm(step[:3]) < 1e-7 and np.linalg.norm(step[3:]) < 1e-7:
            break
        transformation = _se3_exp(step) @ transformation

    accepted_source, camera_points, targets, normals = _projective_correspondences(
        source_points,
        transformation,
        camera_intrinsics,
        target_vertices,
        target_normals,
        max_correspondence_distance=max_correspondence_distance,
    )
    residual = np.einsum("ni,ni->n", normals, camera_points - targets)
    target_pixels = np.rint(
        np.asarray(
            [
                camera_intrinsics[1, 1] * camera_points[:, 1] / camera_points[:, 2]
                + camera_intrinsics[1, 2],
                camera_intrinsics[0, 0] * camera_points[:, 0] / camera_points[:, 2]
                + camera_intrinsics[0, 2],
            ]
        ).T
    ).astype(np.int64)
    correspondences = np.column_stack(
        (accepted_source, target_pixels[:, 0] * depth.shape[1] + target_pixels[:, 1])
    )
    return ICPResults(
        R=transformation[:3, :3].copy(),
        t=transformation[:3, 3].copy(),
        fitness=float(len(correspondences) / len(source_points)),
        inlier_rmse=float(np.sqrt(np.mean(residual**2))) if len(residual) else float("inf"),
        transformation=transformation,
        correspondence_set=correspondences,
    )


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _rotation_vector_to_matrix(rotation_vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(rotation_vector, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3) + _skew(vector)
    axis_skew = _skew(vector / angle)
    return (
        np.eye(3)
        + np.sin(angle) * axis_skew
        + (1.0 - np.cos(angle)) * (axis_skew @ axis_skew)
    )


def _matrix_to_rotation_vector(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    cosine = float(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-10:
        return np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / 2.0,
                (matrix[0, 2] - matrix[2, 0]) / 2.0,
                (matrix[1, 0] - matrix[0, 1]) / 2.0,
            ]
        )
    if np.pi - angle < 1e-5:
        diagonal = np.maximum((np.diag(matrix) + 1.0) / 2.0, 0.0)
        axis = np.sqrt(diagonal)
        largest = int(np.argmax(axis))
        if axis[largest] < 1e-8:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            if largest == 0:
                axis[1] = (matrix[0, 1] + matrix[1, 0]) / (4.0 * axis[0])
                axis[2] = (matrix[0, 2] + matrix[2, 0]) / (4.0 * axis[0])
            elif largest == 1:
                axis[0] = (matrix[0, 1] + matrix[1, 0]) / (4.0 * axis[1])
                axis[2] = (matrix[1, 2] + matrix[2, 1]) / (4.0 * axis[1])
            else:
                axis[0] = (matrix[0, 2] + matrix[2, 0]) / (4.0 * axis[2])
                axis[1] = (matrix[1, 2] + matrix[2, 1]) / (4.0 * axis[2])
            axis /= np.linalg.norm(axis)
        return axis * angle
    factor = angle / (2.0 * np.sin(angle))
    return factor * np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ]
    )


def _se3_exp(delta: np.ndarray) -> np.ndarray:
    """Map a small ``[rotation, translation]`` tangent vector to SE(3)."""
    tangent = np.asarray(delta, dtype=np.float64).reshape(6)
    omega, velocity = tangent[:3], tangent[3:]
    angle = float(np.linalg.norm(omega))
    omega_skew = _skew(omega)
    rotation = _rotation_vector_to_matrix(omega)
    if angle < 1e-8:
        left_jacobian = np.eye(3) + 0.5 * omega_skew
    else:
        left_jacobian = (
            np.eye(3)
            + (1.0 - np.cos(angle)) / angle**2 * omega_skew
            + (angle - np.sin(angle)) / angle**3 * (omega_skew @ omega_skew)
        )
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = left_jacobian @ velocity
    return result


def _clamp_vector_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    return value if norm <= maximum else value * (maximum / norm)


def _clamp_pose_to_rough(
    transformation: np.ndarray,
    rough_transformation: np.ndarray,
    *,
    max_rotation_rad: float,
    max_translation_m: float,
) -> np.ndarray:
    """Hard-limit rotation and translation differences from the rough pose."""
    pose = np.asarray(transformation, dtype=np.float64).reshape(4, 4)
    rough = np.asarray(rough_transformation, dtype=np.float64).reshape(4, 4)
    relative_rotation = pose[:3, :3] @ rough[:3, :3].T
    rotation_vector = _clamp_vector_norm(
        _matrix_to_rotation_vector(relative_rotation), max_rotation_rad
    )
    translation_delta = _clamp_vector_norm(
        pose[:3, 3] - rough[:3, 3], max_translation_m
    )
    clamped = np.eye(4)
    clamped[:3, :3] = _rotation_vector_to_matrix(rotation_vector) @ rough[:3, :3]
    clamped[:3, 3] = rough[:3, 3] + translation_delta
    return clamped


def _silhouette_distance_field(
    mask: np.ndarray,
    *,
    image_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a signed contour distance and its image gradients in pixels."""
    binary = np.asarray(mask)
    if binary.ndim != 2 or not np.any(binary):
        raise ValueError("silhouette mask must be a non-empty 2D array.")
    if not 0 < image_scale <= 1:
        raise ValueError("image_scale must be in (0, 1].")
    height = max(2, int(round(binary.shape[0] * image_scale)))
    width = max(2, int(round(binary.shape[1] * image_scale)))
    resized = cv2.resize(
        (binary > 0).astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    inside = cv2.distanceTransform(resized, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(1 - resized, cv2.DIST_L2, 5)
    signed_distance = outside - inside
    gradient_x = cv2.Sobel(signed_distance, cv2.CV_64F, 1, 0, ksize=3) / 8.0
    gradient_y = cv2.Sobel(signed_distance, cv2.CV_64F, 0, 1, ksize=3) / 8.0
    return signed_distance.astype(np.float64), gradient_x, gradient_y


def _bilinear_sample(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    height, width = image.shape
    u = np.clip(np.asarray(u, dtype=np.float64), 0.0, width - 1.000001)
    v = np.clip(np.asarray(v, dtype=np.float64), 0.0, height - 1.000001)
    x0, y0 = np.floor(u).astype(int), np.floor(v).astype(int)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    dx, dy = u - x0, v - y0
    return (
        (1.0 - dx) * (1.0 - dy) * image[y0, x0]
        + dx * (1.0 - dy) * image[y0, x1]
        + (1.0 - dx) * dy * image[y1, x0]
        + dx * dy * image[y1, x1]
    )


def _projected_contour_samples(
    model_points: np.ndarray,
    transformation: np.ndarray,
    camera_intrinsics: np.ndarray,
    image_shape: tuple[int, int],
    *,
    image_scale: float,
    splat_radius_px: int,
    maximum_samples: int = 2500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Approximate visible model-contour samples using a dense point raster."""
    points = np.asarray(model_points, dtype=np.float64)
    pose = np.asarray(transformation, dtype=np.float64).reshape(4, 4)
    intrinsics = np.asarray(camera_intrinsics, dtype=np.float64).reshape(3, 3).copy()
    intrinsics[:2] *= image_scale
    height = max(2, int(round(image_shape[0] * image_scale)))
    width = max(2, int(round(image_shape[1] * image_scale)))
    camera_points = points @ pose[:3, :3].T + pose[:3, 3]
    positive = camera_points[:, 2] > 1e-8
    indices = np.flatnonzero(positive)
    camera_points = camera_points[positive]
    u = intrinsics[0, 0] * camera_points[:, 0] / camera_points[:, 2] + intrinsics[0, 2]
    v = intrinsics[1, 1] * camera_points[:, 1] / camera_points[:, 2] + intrinsics[1, 2]
    inside = (u >= 1) & (u < width - 1) & (v >= 1) & (v < height - 1)
    if np.count_nonzero(inside) < 3:
        raise ValueError("Too few model samples project inside the silhouette image.")
    indices, camera_points, u, v = (
        indices[inside],
        camera_points[inside],
        u[inside],
        v[inside],
    )
    columns = np.rint(u).astype(int)
    rows = np.rint(v).astype(int)
    occupancy = np.zeros((height, width), dtype=np.uint8)
    occupancy[rows, columns] = 255
    diameter = 2 * splat_radius_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    occupancy = cv2.dilate(occupancy, kernel)
    occupancy = cv2.morphologyEx(occupancy, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        occupancy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    rendered = np.zeros_like(occupancy)
    usable_contours = [contour for contour in contours if cv2.contourArea(contour) >= 4]
    if not usable_contours:
        raise ValueError("Projected model samples did not form a rendered silhouette.")
    cv2.drawContours(rendered, usable_contours, -1, 255, thickness=cv2.FILLED)
    boundary = cv2.morphologyEx(
        rendered, cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8)
    )
    distance_to_boundary = cv2.distanceTransform(
        (boundary == 0).astype(np.uint8), cv2.DIST_L2, 3
    )
    near_boundary = distance_to_boundary[rows, columns] <= splat_radius_px + 1.0
    selected = np.flatnonzero(near_boundary)
    if len(selected) < 6:
        raise ValueError("Rendered silhouette contains too few model contour samples.")
    if len(selected) > maximum_samples:
        selected = selected[np.linspace(0, len(selected) - 1, maximum_samples).astype(int)]
    return indices[selected], camera_points[selected], np.column_stack((u[selected], v[selected]))


def _point_jacobian(camera_points: np.ndarray) -> np.ndarray:
    points = np.asarray(camera_points, dtype=np.float64)
    jacobian = np.zeros((len(points), 3, 6), dtype=np.float64)
    jacobian[:, :, 3:] = np.eye(3)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    jacobian[:, :, :3] = np.stack(
        (
            np.column_stack((np.zeros(len(points)), z, -y)),
            np.column_stack((-z, np.zeros(len(points)), x)),
            np.column_stack((y, -x, np.zeros(len(points)))),
        ),
        axis=1,
    )
    return jacobian


def _geometry_terms(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transformation: np.ndarray,
    max_correspondence_distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source, target, max_correspondence_distance, transformation
    )
    correspondences = np.asarray(evaluation.correspondence_set, dtype=np.int64)
    if len(correspondences) < 6:
        raise ValueError("Silhouette ICP found fewer than 6 geometric correspondences.")
    source_points = np.asarray(source.points)[correspondences[:, 0]]
    target_points = np.asarray(target.points)[correspondences[:, 1]]
    target_normals = np.asarray(target.normals)[correspondences[:, 1]]
    pose = np.asarray(transformation, dtype=np.float64).reshape(4, 4)
    camera_points = source_points @ pose[:3, :3].T + pose[:3, 3]
    residual = np.einsum("ni,ni->n", target_normals, camera_points - target_points)
    jacobian = np.einsum(
        "ni,nij->nj", target_normals, _point_jacobian(camera_points)
    )
    return residual, jacobian, correspondences


def _silhouette_terms(
    source: o3d.geometry.PointCloud,
    transformation: np.ndarray,
    camera_intrinsics: np.ndarray,
    image_shape: tuple[int, int],
    distance_field: np.ndarray,
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    *,
    image_scale: float,
    splat_radius_px: int,
    observed_depth: np.ndarray | None = None,
    depth_gate: float | None = None,
    depth_gate_occlusion_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    _, camera_points, image_points = _projected_contour_samples(
        np.asarray(source.points),
        transformation,
        camera_intrinsics,
        image_shape,
        image_scale=image_scale,
        splat_radius_px=splat_radius_px,
    )
    u, v = image_points[:, 0], image_points[:, 1]
    depth_rejected_count = 0
    if depth_gate is not None:
        if observed_depth is None:
            raise ValueError("observed_depth is required when depth_gate is enabled.")
        depth = np.asarray(observed_depth, dtype=np.float64)
        if depth.ndim != 2 or depth.shape != tuple(image_shape):
            raise ValueError("observed_depth must match the full-resolution image shape.")
        if not np.isfinite(depth_gate) or depth_gate <= 0:
            raise ValueError("depth_gate must be finite and positive.")
        # ``image_points`` live in the downscaled SDF coordinate system.  Depth
        # is deliberately sampled at its native RGB-D pixel, so the gate does
        # not blur an occlusion boundary a second time.
        full_u = u / image_scale
        full_v = v / image_scale
        columns = np.rint(full_u).astype(np.int64)
        rows = np.rint(full_v).astype(np.int64)
        inside = (
            (columns >= 0) & (columns < depth.shape[1])
            & (rows >= 0) & (rows < depth.shape[0])
        )
        measured_depth = np.full(len(camera_points), np.nan, dtype=np.float64)
        measured_depth[inside] = depth[rows[inside], columns[inside]]
        valid_depth = np.isfinite(measured_depth) & (measured_depth > 0)
        if depth_gate_occlusion_only:
            # Only a nearer measured surface establishes external occlusion.
            # Background bleed and invalid depth provide no reason to discard a
            # silhouette constraint, so they remain active.
            keep = ~valid_depth | (measured_depth >= camera_points[:, 2] - depth_gate)
        else:
            keep = valid_depth & (
                np.abs(camera_points[:, 2] - measured_depth) <= depth_gate
            )
        depth_rejected_count = int(np.count_nonzero(~keep))
        camera_points, u, v = camera_points[keep], u[keep], v[keep]
    if not len(camera_points):
        return np.empty(0), np.empty((0, 6)), 0, depth_rejected_count
    residual = _bilinear_sample(distance_field, u, v)
    image_gradient = np.column_stack(
        (_bilinear_sample(gradient_x, u, v), _bilinear_sample(gradient_y, u, v))
    )
    intrinsics = np.asarray(camera_intrinsics, dtype=np.float64).reshape(3, 3)
    fx, fy = intrinsics[0, 0] * image_scale, intrinsics[1, 1] * image_scale
    x, y, z = camera_points[:, 0], camera_points[:, 1], camera_points[:, 2]
    projection_jacobian = np.zeros((len(camera_points), 2, 3), dtype=np.float64)
    projection_jacobian[:, 0, 0] = fx / z
    projection_jacobian[:, 0, 2] = -fx * x / z**2
    projection_jacobian[:, 1, 1] = fy / z
    projection_jacobian[:, 1, 2] = -fy * y / z**2
    image_pose_jacobian = projection_jacobian @ _point_jacobian(camera_points)
    jacobian = np.einsum("ni,nij->nj", image_gradient, image_pose_jacobian)
    return residual, jacobian, len(camera_points), depth_rejected_count


def _huber_cost_and_weight(
    residual: np.ndarray, delta: float
) -> tuple[np.ndarray, np.ndarray]:
    absolute = np.abs(residual)
    quadratic = absolute <= delta
    cost = np.where(quadratic, 0.5 * residual**2, delta * (absolute - 0.5 * delta))
    weight = np.ones_like(residual)
    weight[~quadratic] = delta / np.maximum(absolute[~quadratic], 1e-12)
    return cost, weight


def _joint_silhouette_system(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transformation: np.ndarray,
    rough_transformation: np.ndarray,
    camera_intrinsics: np.ndarray,
    image_shape: tuple[int, int],
    distance_field: np.ndarray,
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    *,
    max_correspondence_distance: float,
    silhouette_weight: float,
    silhouette_distance_scale_px: float,
    silhouette_splat_radius_px: int,
    silhouette_huber_delta_px: float,
    silhouette_pose_prior_weight: float,
    silhouette_image_scale: float,
    observed_depth: np.ndarray | None,
    silhouette_depth_gate: float | None,
    silhouette_depth_gate_occlusion_only: bool,
    max_rotation_rad: float,
    max_translation_m: float,
    with_jacobian: bool,
) -> tuple[float, np.ndarray | None, np.ndarray | None, int, int, int]:
    geometric_residual, geometric_jacobian, correspondences = _geometry_terms(
        source, target, transformation, max_correspondence_distance
    )
    (
        silhouette_residual,
        silhouette_jacobian,
        silhouette_count,
        silhouette_depth_rejected_count,
    ) = _silhouette_terms(
        source,
        transformation,
        camera_intrinsics,
        image_shape,
        distance_field,
        gradient_x,
        gradient_y,
        image_scale=silhouette_image_scale,
        splat_radius_px=silhouette_splat_radius_px,
        observed_depth=observed_depth,
        depth_gate=silhouette_depth_gate,
        depth_gate_occlusion_only=silhouette_depth_gate_occlusion_only,
    )
    geometric_residual = geometric_residual / max_correspondence_distance
    geometric_jacobian = geometric_jacobian / max_correspondence_distance
    silhouette_residual = silhouette_residual / silhouette_distance_scale_px
    silhouette_jacobian = silhouette_jacobian / silhouette_distance_scale_px
    geometric_cost, geometric_weight = _huber_cost_and_weight(
        geometric_residual, 0.5
    )
    silhouette_cost, silhouette_robust_weight = _huber_cost_and_weight(
        silhouette_residual,
        silhouette_huber_delta_px / silhouette_distance_scale_px,
    )
    pose = np.asarray(transformation, dtype=np.float64).reshape(4, 4)
    rough = np.asarray(rough_transformation, dtype=np.float64).reshape(4, 4)
    prior_residual = np.concatenate(
        (
            _matrix_to_rotation_vector(pose[:3, :3] @ rough[:3, :3].T)
            / max_rotation_rad,
            (pose[:3, 3] - rough[:3, 3]) / max_translation_m,
        )
    )
    objective = float(
        np.mean(geometric_cost)
        + (silhouette_weight * np.mean(silhouette_cost) if silhouette_count else 0.0)
        + 0.5 * silhouette_pose_prior_weight * np.dot(prior_residual, prior_residual)
    )
    if not with_jacobian:
        return (
            objective,
            None,
            None,
            len(correspondences),
            silhouette_count,
            silhouette_depth_rejected_count,
        )

    row_blocks: list[np.ndarray] = []
    residual_blocks: list[np.ndarray] = []
    geometric_scale = np.sqrt(geometric_weight / len(geometric_residual))
    row_blocks.append(geometric_jacobian * geometric_scale[:, None])
    residual_blocks.append(geometric_residual * geometric_scale)
    if silhouette_weight > 0 and silhouette_count > 0:
        silhouette_scale = np.sqrt(
            silhouette_weight * silhouette_robust_weight / len(silhouette_residual)
        )
        row_blocks.append(silhouette_jacobian * silhouette_scale[:, None])
        residual_blocks.append(silhouette_residual * silhouette_scale)
    if silhouette_pose_prior_weight > 0:
        prior_scale = np.sqrt(silhouette_pose_prior_weight)
        prior_jacobian = np.diag(
            [1.0 / max_rotation_rad] * 3 + [1.0 / max_translation_m] * 3
        )
        row_blocks.append(prior_jacobian * prior_scale)
        residual_blocks.append(prior_residual * prior_scale)
    return (
        objective,
        np.vstack(row_blocks),
        np.concatenate(residual_blocks),
        len(correspondences),
        silhouette_count,
        silhouette_depth_rejected_count,
    )


def run_silhouette_constrained_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    icp_transformation: np.ndarray,
    rough_transformation: np.ndarray,
    camera_intrinsics: np.ndarray,
    image_shape: tuple[int, int],
    silhouette_mask: np.ndarray,
    *,
    max_correspondence_distance: float = 0.01,
    silhouette_weight: float = 0.25,
    max_iteration: int = 12,
    image_scale: float = 0.5,
    distance_scale_px: float = 5.0,
    splat_radius_px: int = 2,
    huber_delta_px: float = 4.0,
    pose_prior_weight: float = 0.05,
    damping: float = 1e-3,
    max_rotation_deg: float = 8.0,
    max_translation_m: float = 0.012,
    max_step_rotation_deg: float = 1.0,
    max_step_translation_m: float = 0.002,
    observed_depth: np.ndarray | None = None,
    depth_gate: float | None = None,
    depth_gate_occlusion_only: bool = False,
) -> tuple[ICPResults, SilhouetteRefinementDiagnostics]:
    """Jointly minimize point-to-plane and depth-eligible 2D contour residuals.

    When ``depth_gate`` is set, the gate is recomputed for each evaluated pose.
    With ``depth_gate_occlusion_only``, only a measured foreground surface more
    than the threshold in front of the rendered sample rejects that sample.
    """
    if silhouette_weight <= 0:
        raise ValueError("silhouette_weight must be positive.")
    if not target.has_normals():
        raise ValueError("Silhouette-constrained ICP requires target normals.")
    max_rotation_rad = np.deg2rad(max_rotation_deg)
    max_step_rotation_rad = np.deg2rad(max_step_rotation_deg)
    distance_field, gradient_x, gradient_y = _silhouette_distance_field(
        silhouette_mask, image_scale=image_scale
    )
    system_arguments = dict(
        source=source,
        target=target,
        rough_transformation=rough_transformation,
        camera_intrinsics=camera_intrinsics,
        image_shape=image_shape,
        distance_field=distance_field,
        gradient_x=gradient_x,
        gradient_y=gradient_y,
        max_correspondence_distance=max_correspondence_distance,
        silhouette_weight=silhouette_weight,
        silhouette_distance_scale_px=distance_scale_px,
        silhouette_splat_radius_px=splat_radius_px,
        silhouette_huber_delta_px=huber_delta_px,
        silhouette_pose_prior_weight=pose_prior_weight,
        silhouette_image_scale=image_scale,
        observed_depth=observed_depth,
        silhouette_depth_gate=depth_gate,
        silhouette_depth_gate_occlusion_only=depth_gate_occlusion_only,
        max_rotation_rad=max_rotation_rad,
        max_translation_m=max_translation_m,
    )
    rough = np.asarray(rough_transformation, dtype=np.float64).reshape(4, 4)
    icp = _clamp_pose_to_rough(
        icp_transformation,
        rough,
        max_rotation_rad=max_rotation_rad,
        max_translation_m=max_translation_m,
    )
    rough_objective = _joint_silhouette_system(
        transformation=rough, with_jacobian=False, **system_arguments
    )[0]
    icp_objective = _joint_silhouette_system(
        transformation=icp, with_jacobian=False, **system_arguments
    )[0]
    if rough_objective <= icp_objective:
        transformation, initial_candidate, initial_objective = (
            rough.copy(),
            "rough",
            rough_objective,
        )
    else:
        transformation, initial_candidate, initial_objective = (
            icp.copy(),
            "point-to-plane-icp",
            icp_objective,
        )

    accepted_iterations = 0
    attempted_iterations = 0
    geometric_count = silhouette_count = depth_rejected_count = 0
    current_objective = initial_objective
    for attempted_iterations in range(1, max_iteration + 1):
        (
            current_objective,
            jacobian,
            residual,
            geometric_count,
            silhouette_count,
            depth_rejected_count,
        ) = _joint_silhouette_system(
            transformation=transformation, with_jacobian=True, **system_arguments
        )
        assert jacobian is not None and residual is not None
        diagonal_scale = np.maximum(np.diag(jacobian.T @ jacobian), 1.0)
        system = jacobian.T @ jacobian + damping * np.diag(diagonal_scale)
        gradient = jacobian.T @ residual
        try:
            step = -np.linalg.solve(system, gradient)
        except np.linalg.LinAlgError:
            step = -np.linalg.lstsq(system, gradient, rcond=None)[0]
        step[:3] = _clamp_vector_norm(step[:3], max_step_rotation_rad)
        step[3:] = _clamp_vector_norm(step[3:], max_step_translation_m)
        if np.linalg.norm(step[:3]) < 1e-7 and np.linalg.norm(step[3:]) < 1e-7:
            break

        accepted = False
        for line_scale in (1.0, 0.5, 0.25, 0.125):
            trial = _se3_exp(step * line_scale) @ transformation
            trial = _clamp_pose_to_rough(
                trial,
                rough,
                max_rotation_rad=max_rotation_rad,
                max_translation_m=max_translation_m,
            )
            try:
                trial_objective = _joint_silhouette_system(
                    transformation=trial, with_jacobian=False, **system_arguments
                )[0]
            except ValueError:
                continue
            if trial_objective < current_objective - 1e-10:
                transformation = trial
                current_objective = trial_objective
                accepted_iterations += 1
                accepted = True
                break
        if not accepted:
            break

    final_evaluation = o3d.pipelines.registration.evaluate_registration(
        source, target, max_correspondence_distance, transformation
    )
    correspondences = np.asarray(
        final_evaluation.correspondence_set, dtype=np.int64
    ).copy()
    rotation_delta = _matrix_to_rotation_vector(
        transformation[:3, :3] @ rough[:3, :3].T
    )
    diagnostics = SilhouetteRefinementDiagnostics(
        initial_candidate=initial_candidate,
        initial_objective=float(initial_objective),
        final_objective=float(current_objective),
        accepted_iterations=accepted_iterations,
        attempted_iterations=attempted_iterations,
        geometric_correspondence_count=int(len(correspondences)),
        silhouette_point_count=int(silhouette_count),
        silhouette_depth_rejected_point_count=int(depth_rejected_count),
        rotation_delta_deg=float(np.rad2deg(np.linalg.norm(rotation_delta))),
        translation_delta_m=float(
            np.linalg.norm(transformation[:3, 3] - rough[:3, 3])
        ),
    )
    return (
        ICPResults(
            R=transformation[:3, :3].copy(),
            t=transformation[:3, 3].copy(),
            fitness=float(final_evaluation.fitness),
            inlier_rmse=float(final_evaluation.inlier_rmse),
            transformation=transformation.copy(),
            correspondence_set=correspondences,
        ),
        diagnostics,
    )


def refine_annotations(
    dataset: DataSet,
    annotation_document: dict[str, Any],
    *,
    config: ICPConfig = ICPConfig(),
    segmenter: Any | None = None,
) -> dict[str, Any]:
    """Refine every solved pose while preserving the input JSON structure."""
    config.validate()
    if config.sam_model_id is None and segmenter is not None:
        raise ValueError("A segmenter was supplied without configuring sam_model_id.")
    if config.sam_model_id is not None and segmenter is None:
        print(
            f"Loading SAM model {config.sam_model_id} on {config.sam_device}...",
            flush=True,
        )
        segmenter = load_sam2_predictor(
            config.sam_model_id,
            device=config.sam_device,
        )
        print(f"Loaded SAM model on {segmenter.device}.", flush=True)
    o3d.utility.random.seed(config.random_seed)
    output = copy.deepcopy(annotation_document)
    sampled_models: dict[int, o3d.geometry.PointCloud] = {}
    target_clouds: dict[int, o3d.geometry.PointCloud] = {}
    solved_count = 0
    use_colors = config.mode is ICPMode.Colored

    for raw_frame_id, records in output.items():
        try:
            frame_id = int(raw_frame_id)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Unexpected non-frame JSON key: {raw_frame_id!r}"
            ) from error
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

            rough_rotation = np.asarray(record["cam_R_m2c"], dtype=np.float64).reshape(
                3, 3
            )
            rough_translation_m = (
                np.asarray(record["cam_t_m2c"], dtype=np.float64).reshape(3) / 1000.0
            )
            if not (
                np.isfinite(rough_rotation).all()
                and np.isfinite(rough_translation_m).all()
            ):
                raise ValueError(
                    f"Frame {frame_id}, object {object_id} has a non-finite pose."
                )
            rough_transform = np.eye(4, dtype=np.float64)
            rough_transform[:3, :3] = rough_rotation
            rough_transform[:3, 3] = rough_translation_m

            if object_id not in sampled_models:
                sampled_models[object_id] = load_sampled_model(
                    dataset.models[object_id],
                    sample_count=config.model_sample_count,
                    model_scale=config.model_scale,
                    include_colors=use_colors,
                )
            sampled_model = sampled_models[object_id]
            if use_colors and not sampled_model.has_colors():
                raise ValueError(
                    f"Colored ICP requires vertex colors in obj_{object_id:06d}.ply."
                )
            candidates = get_icp_candidates(
                sampled_model,
                rough_rotation,
                rough_translation_m,
                pose.camera_intrinsics,
                pose.depth.shape,
                visibility_radius_factor=config.visibility_radius_factor,
            )
            segmentation: SAM2SegmentationResult | None = None
            eroded_mask: np.ndarray | None = None
            overlay_path: Path | None = None
            selected_mask_path: Path | None = None
            eroded_mask_path: Path | None = None
            if segmenter is not None:
                segmentation = segment_object_sam2(
                    pose,
                    sampled_model,
                    rough_rotation,
                    rough_translation_m,
                    segmenter,
                    padding_fraction=config.sam_box_padding_fraction,
                    multimask_output=True,
                )
                eroded_mask = erode_binary_mask(
                    (
                        segmentation.mask
                        if segmentation.target_mask is None
                        else segmentation.target_mask
                    ),
                    config.sam_erosion_radius_px,
                )
                if not np.any(eroded_mask):
                    raise ValueError(
                        f"Frame {frame_id}, object {object_id} has an empty SAM mask "
                        f"after {config.sam_erosion_radius_px}px erosion."
                    )
                if config.sam_mask_output_dir is not None:
                    mask_name = f"{frame_id:06d}_obj_{object_id:06d}.png"
                    selected_mask_path = (
                        Path(config.sam_mask_output_dir) / "selected" / mask_name
                    ).resolve()
                    eroded_mask_path = (
                        Path(config.sam_mask_output_dir)
                        / f"eroded_{config.sam_erosion_radius_px}px"
                        / mask_name
                    ).resolve()
                    write_binary_mask(selected_mask_path, segmentation.mask)
                    write_binary_mask(eroded_mask_path, eroded_mask)
                valid_masked_depth_count = int(
                    np.count_nonzero(
                        eroded_mask
                        & np.isfinite(np.asarray(pose.depth))
                        & (np.asarray(pose.depth) > 0)
                    )
                )
                if valid_masked_depth_count < 3:
                    _record_icp_failure(
                        record,
                        config=config,
                        rough_rotation=rough_rotation,
                        rough_translation_m=rough_translation_m,
                        reason=(
                            "SAM mask contains fewer than 3 valid depth pixels "
                            f"after erosion ({valid_masked_depth_count})."
                        ),
                        candidate_point_count=len(candidates.points),
                        target_point_count=valid_masked_depth_count,
                        segmentation=segmentation,
                        eroded_mask=eroded_mask,
                        segmenter=segmenter,
                    )
                    solved_count += 1
                    print(
                        f"frame={frame_id} obj={object_id} refinement_failed="
                        f"empty_masked_depth"
                    )
                    continue
                target = image_to_point_cloud(
                    pose,
                    voxel_size=config.target_voxel_size,
                    include_colors=use_colors,
                    pixel_mask=eroded_mask,
                )
            elif frame_id not in target_clouds:
                target = image_to_point_cloud(
                    pose,
                    voxel_size=config.target_voxel_size,
                    include_colors=use_colors,
                )
                target_clouds[frame_id] = target
            if segmenter is None:
                target = target_clouds[frame_id]
            if config.target_neighborhood_distance is not None:
                target = select_target_neighborhood(
                    target,
                    candidates,
                    rough_transform,
                    max_distance=config.target_neighborhood_distance,
                )
                if len(target.points) < 3:
                    _record_icp_failure(
                        record,
                        config=config,
                        rough_rotation=rough_rotation,
                        rough_translation_m=rough_translation_m,
                        reason=(
                            "3D rough-model neighborhood contains fewer than 3 "
                            f"target points ({len(target.points)})."
                        ),
                        candidate_point_count=len(candidates.points),
                        target_point_count=len(target.points),
                        segmentation=segmentation,
                        eroded_mask=eroded_mask,
                        segmenter=segmenter,
                    )
                    solved_count += 1
                    print(
                        f"frame={frame_id} obj={object_id} refinement_failed="
                        f"empty_target_neighborhood"
                    )
                    continue
            if config.mode in (ICPMode.PointToPlane, ICPMode.Colored):
                estimate_target_normals(
                    target,
                    radius=config.normal_radius,
                    max_nn=config.normal_max_nn,
                )
            try:
                if config.mode is ICPMode.ProjectivePointToPlane:
                    result = run_projective_icp(
                        candidates,
                        np.asarray(pose.depth),
                        pose.camera_intrinsics,
                        rough_transform,
                        max_correspondence_distance=(
                            config.max_correspondence_distance
                        ),
                        max_iteration=config.max_iteration,
                        normal_depth_delta=config.projective_normal_depth_delta,
                        huber_delta=config.projective_huber_delta,
                        max_step_rotation_deg=(
                            config.projective_max_step_rotation_deg
                        ),
                        max_step_translation=(
                            config.projective_max_step_translation
                        ),
                    )
                else:
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
                        colored_icp_lambda_geometric=(
                            config.colored_icp_lambda_geometric
                        ),
                    )
            except RuntimeError as error:
                _record_icp_failure(
                    record,
                    config=config,
                    rough_rotation=rough_rotation,
                    rough_translation_m=rough_translation_m,
                    reason=f"ICP backend error: {error}",
                    candidate_point_count=len(candidates.points),
                    target_point_count=len(target.points),
                    segmentation=segmentation,
                    eroded_mask=eroded_mask,
                    segmenter=segmenter,
                )
                solved_count += 1
                print(
                    f"frame={frame_id} obj={object_id} refinement_failed="
                    f"backend_error"
                )
                continue
            if len(result.correspondence_set) < 3:
                _record_icp_failure(
                    record,
                    config=config,
                    rough_rotation=rough_rotation,
                    rough_translation_m=rough_translation_m,
                    reason=(
                        "ICP found fewer than 3 inlier correspondences "
                        f"({len(result.correspondence_set)})."
                    ),
                    candidate_point_count=len(candidates.points),
                    target_point_count=len(target.points),
                    segmentation=segmentation,
                    eroded_mask=eroded_mask,
                    segmenter=segmenter,
                )
                solved_count += 1
                print(
                    f"frame={frame_id} obj={object_id} refinement_failed="
                    f"insufficient_correspondences"
                )
                continue
            silhouette_diagnostics: SilhouetteRefinementDiagnostics | None = None
            if config.silhouette_weight > 0:
                if segmentation is None:
                    raise RuntimeError(
                        "Silhouette refinement requires a segmentation result."
                    )
                if config.mode is not ICPMode.PointToPlane:
                    raise ValueError(
                        "Silhouette refinement currently supports point-to-plane "
                        "ICP only."
                    )
                result, silhouette_diagnostics = run_silhouette_constrained_icp(
                    candidates,
                    target,
                    result.transformation,
                    rough_transform,
                    pose.camera_intrinsics,
                    pose.depth.shape,
                    segmentation.mask,
                    max_correspondence_distance=(
                        config.max_correspondence_distance
                    ),
                    silhouette_weight=config.silhouette_weight,
                    max_iteration=config.silhouette_max_iteration,
                    image_scale=config.silhouette_image_scale,
                    distance_scale_px=config.silhouette_distance_scale_px,
                    splat_radius_px=config.silhouette_splat_radius_px,
                    huber_delta_px=config.silhouette_huber_delta_px,
                    pose_prior_weight=config.silhouette_pose_prior_weight,
                    damping=config.silhouette_damping,
                    max_rotation_deg=config.silhouette_max_rotation_deg,
                    max_translation_m=config.silhouette_max_translation,
                    max_step_rotation_deg=(
                        config.silhouette_max_step_rotation_deg
                    ),
                    max_step_translation_m=(
                        config.silhouette_max_step_translation
                    ),
                    observed_depth=np.asarray(pose.depth),
                    depth_gate=config.silhouette_depth_gate,
                    depth_gate_occlusion_only=(
                        config.silhouette_depth_gate_occlusion_only
                    ),
                )

            if eroded_mask is not None and config.sam_overlay_output_dir is not None:
                overlay_path = (
                    Path(config.sam_overlay_output_dir)
                    / f"{frame_id:06d}_obj_{object_id:06d}.png"
                ).resolve()
                write_rgb_image(
                    overlay_path,
                    green_mask_overlay(
                        np.asarray(pose.rgb),
                        eroded_mask,
                        alpha=config.sam_overlay_alpha,
                    ),
                )

            record["cam_R_m2c"] = result.R.reshape(-1).tolist()
            record["cam_t_m2c"] = (result.t * 1000.0).tolist()
            record["icp"] = {
                "status": "refined",
                "mode": (
                    "silhouette-constrained"
                    if silhouette_diagnostics is not None
                    else config.mode.value
                ),
                "base_mode": config.mode.value,
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
                    "target_neighborhood_distance_mm": (
                        None
                        if config.target_neighborhood_distance is None
                        else config.target_neighborhood_distance * 1000.0
                    ),
                    "max_iteration": config.max_iteration,
                    "relative_fitness": config.relative_fitness,
                    "relative_rmse": config.relative_rmse,
                    "normal_radius_mm": config.normal_radius * 1000.0,
                    "normal_max_nn": config.normal_max_nn,
                    "colored_icp_lambda_geometric": (
                        config.colored_icp_lambda_geometric
                    ),
                    "visibility_radius_factor": config.visibility_radius_factor,
                    "random_seed": config.random_seed,
                },
            }
            if silhouette_diagnostics is not None:
                record["icp"]["silhouette"] = {
                    "method": "signed-distance-contour-gauss-newton",
                    "initial_candidate": silhouette_diagnostics.initial_candidate,
                    "initial_objective": silhouette_diagnostics.initial_objective,
                    "final_objective": silhouette_diagnostics.final_objective,
                    "accepted_iterations": (
                        silhouette_diagnostics.accepted_iterations
                    ),
                    "attempted_iterations": (
                        silhouette_diagnostics.attempted_iterations
                    ),
                    "geometric_correspondence_count": (
                        silhouette_diagnostics.geometric_correspondence_count
                    ),
                    "silhouette_point_count": (
                        silhouette_diagnostics.silhouette_point_count
                    ),
                    "silhouette_depth_rejected_point_count": (
                        silhouette_diagnostics.silhouette_depth_rejected_point_count
                    ),
                    "rotation_delta_deg_from_rough": (
                        silhouette_diagnostics.rotation_delta_deg
                    ),
                    "translation_delta_mm_from_rough": (
                        silhouette_diagnostics.translation_delta_m * 1000.0
                    ),
                    "parameters": {
                        "weight": config.silhouette_weight,
                        "max_iteration": config.silhouette_max_iteration,
                        "image_scale": config.silhouette_image_scale,
                        "distance_scale_px": (
                            config.silhouette_distance_scale_px
                        ),
                        "splat_radius_px": config.silhouette_splat_radius_px,
                        "huber_delta_px": config.silhouette_huber_delta_px,
                        "pose_prior_weight": (
                            config.silhouette_pose_prior_weight
                        ),
                        "damping": config.silhouette_damping,
                        "max_rotation_deg": (
                            config.silhouette_max_rotation_deg
                        ),
                        "max_translation_mm": (
                            config.silhouette_max_translation * 1000.0
                        ),
                        "max_step_rotation_deg": (
                            config.silhouette_max_step_rotation_deg
                        ),
                        "max_step_translation_mm": (
                            config.silhouette_max_step_translation * 1000.0
                        ),
                        "depth_gate_mm": (
                            None
                            if config.silhouette_depth_gate is None
                            else config.silhouette_depth_gate * 1000.0
                        ),
                        "depth_gate_occlusion_only": (
                            config.silhouette_depth_gate_occlusion_only
                        ),
                    },
                }
            if config.mode is ICPMode.ProjectivePointToPlane:
                record["icp"]["projective"] = {
                    "correspondence_method": (
                        "frontmost sampled-model z-buffer point to organized "
                        "sensor-depth point at the same rounded pixel"
                    ),
                    "occlusion_rule": (
                        "reject missing/unstable depth or absolute depth and 3D "
                        "disagreement above max_correspondence_distance"
                    ),
                    "normal_depth_delta_mm": (
                        config.projective_normal_depth_delta * 1000.0
                    ),
                    "huber_delta_mm": config.projective_huber_delta * 1000.0,
                    "max_step_rotation_deg": (
                        config.projective_max_step_rotation_deg
                    ),
                    "max_step_translation_mm": (
                        config.projective_max_step_translation * 1000.0
                    ),
                }
            if segmentation is not None and eroded_mask is not None:
                record["icp"]["segmentation"] = {
                    "method": "sam2-projected-box",
                    "model_id": config.sam_model_id,
                    "device": getattr(segmenter, "device", config.sam_device),
                    "selected_candidate_index": segmentation.selected_index,
                    "selected_score": segmentation.selected_score,
                    "prompt_box_xyxy": segmentation.prompt_box_xyxy.tolist(),
                    "box_padding_fraction": config.sam_box_padding_fraction,
                    "erosion_radius_px": config.sam_erosion_radius_px,
                    "mask_pixel_count": int(np.count_nonzero(segmentation.mask)),
                    "target_mask_pixel_count": int(
                        np.count_nonzero(
                            segmentation.mask
                            if segmentation.target_mask is None
                            else segmentation.target_mask
                        )
                    ),
                    "eroded_mask_pixel_count": int(np.count_nonzero(eroded_mask)),
                }
                if overlay_path is not None:
                    try:
                        overlay_reference = overlay_path.relative_to(dataset.root)
                    except ValueError:
                        overlay_reference = overlay_path
                    record["icp"]["segmentation"].update(
                        {
                            "overlay_image": str(overlay_reference),
                            "overlay_mask": "eroded",
                            "overlay_color_rgb": [0, 255, 0],
                            "overlay_alpha": config.sam_overlay_alpha,
                        }
                    )
                if selected_mask_path is not None and eroded_mask_path is not None:
                    record["icp"]["segmentation"].update(
                        {
                            "selected_mask_image": str(selected_mask_path),
                            "eroded_mask_image": str(eroded_mask_path),
                            "mask_encoding": "uint8 PNG, foreground=255, background=0",
                        }
                    )
            solved_count += 1
            print(
                f"frame={frame_id} obj={object_id} candidates={len(candidates.points)} "
                f"fitness={result.fitness:.4f} "
                f"inlier_rmse={result.inlier_rmse * 1000.0:.3f} mm"
            )

    if not solved_count:
        raise ValueError("Input annotation JSON contains no solved poses.")
    return output


def _record_icp_failure(
    record: dict[str, Any],
    *,
    config: ICPConfig,
    rough_rotation: np.ndarray,
    rough_translation_m: np.ndarray,
    reason: str,
    candidate_point_count: int,
    target_point_count: int,
    segmentation: SAM2SegmentationResult | None,
    eroded_mask: np.ndarray | None,
    segmenter: Any | None,
) -> None:
    """Record a per-pose failure while preserving the rough input pose."""
    record["icp"] = {
        "status": "failed",
        "mode": config.mode.value,
        "reason": reason,
        "fallback": "rough_pose",
        "candidate_point_count": int(candidate_point_count),
        "target_point_count": int(target_point_count),
        "rough_cam_R_m2c": rough_rotation.reshape(-1).tolist(),
        "rough_cam_t_m2c": (rough_translation_m * 1000.0).tolist(),
        "parameters": {
            "max_correspondence_distance_mm": (
                config.max_correspondence_distance * 1000.0
            ),
            "target_neighborhood_distance_mm": (
                None
                if config.target_neighborhood_distance is None
                else config.target_neighborhood_distance * 1000.0
            ),
            "sam_erosion_radius_px": config.sam_erosion_radius_px,
            "silhouette_weight": config.silhouette_weight,
        },
    }
    if segmentation is not None and eroded_mask is not None:
        record["icp"]["segmentation"] = {
            "method": "sam2-projected-box",
            "model_id": config.sam_model_id,
            "device": getattr(segmenter, "device", config.sam_device),
            "selected_candidate_index": segmentation.selected_index,
            "selected_score": segmentation.selected_score,
            "prompt_box_xyxy": segmentation.prompt_box_xyxy.tolist(),
            "box_padding_fraction": config.sam_box_padding_fraction,
            "erosion_radius_px": config.sam_erosion_radius_px,
            "mask_pixel_count": int(np.count_nonzero(segmentation.mask)),
            "target_mask_pixel_count": int(
                np.count_nonzero(
                    segmentation.mask
                    if segmentation.target_mask is None
                    else segmentation.target_mask
                )
            ),
            "eroded_mask_pixel_count": int(np.count_nonzero(eroded_mask)),
        }


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
    parser.add_argument(
        "--target-neighborhood-distance-mm",
        type=_positive_float,
        default=None,
        help=(
            "Before ICP, retain only measured target points within this 3D "
            "distance of the rough transformed model (default: disabled)"
        ),
    )
    parser.add_argument("--max-iteration", type=int, default=50)
    parser.add_argument("--relative-fitness", type=float, default=1e-6)
    parser.add_argument("--relative-rmse", type=float, default=1e-6)
    parser.add_argument("--normal-radius-mm", type=_positive_float, default=10.0)
    parser.add_argument("--normal-max-nn", type=int, default=30)
    parser.add_argument(
        "--colored-lambda-geometric",
        type=float,
        default=0.999,
        help=(
            "Colored ICP geometric weight in [0, 1]; the remaining weight is "
            "photometric (default: 0.999)"
        ),
    )
    parser.add_argument("--visibility-radius-factor", type=float, default=100.0)
    parser.add_argument("--projective-normal-depth-delta-mm", type=_positive_float, default=20.0)
    parser.add_argument("--projective-huber-delta-mm", type=_positive_float, default=3.0)
    parser.add_argument("--projective-max-step-rotation-deg", type=_positive_float, default=1.0)
    parser.add_argument("--projective-max-step-translation-mm", type=_positive_float, default=2.0)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument(
        "--sam-model",
        default=None,
        help=(
            "Enable projected-box SAM masking with this Hugging Face model ID "
            "(for example facebook/sam2.1-hiera-small)"
        ),
    )
    parser.add_argument(
        "--sam-device",
        default="auto",
        help="SAM device: auto, cpu, cuda, or a device such as cuda:0",
    )
    parser.add_argument(
        "--sam-box-padding",
        type=float,
        default=0.25,
        help="Fractional padding around the projected model box (default: 0.25)",
    )
    parser.add_argument(
        "--sam-erode-px",
        type=int,
        default=0,
        help="Pixel radius used to erode the selected SAM mask (default: 0)",
    )
    parser.add_argument(
        "--sam-overlay-dir",
        type=Path,
        default=None,
        help="Save the eroded SAM mask over each RGB image in this directory",
    )
    parser.add_argument(
        "--sam-mask-dir",
        type=Path,
        default=None,
        help="Save exact selected and eroded binary SAM masks in this directory",
    )
    parser.add_argument(
        "--sam-overlay-alpha",
        type=float,
        default=0.45,
        help="Opacity of the green SAM overlay (default: 0.45)",
    )
    parser.add_argument(
        "--silhouette-weight",
        type=float,
        default=0.0,
        help=(
            "Enable bounded silhouette-constrained ICP with this non-negative "
            "contour-loss weight (default: 0, disabled)"
        ),
    )
    parser.add_argument("--silhouette-max-iteration", type=int, default=12)
    parser.add_argument("--silhouette-image-scale", type=float, default=0.5)
    parser.add_argument("--silhouette-distance-scale-px", type=float, default=5.0)
    parser.add_argument("--silhouette-splat-radius-px", type=int, default=2)
    parser.add_argument("--silhouette-huber-delta-px", type=float, default=4.0)
    parser.add_argument("--silhouette-pose-prior-weight", type=float, default=0.05)
    parser.add_argument("--silhouette-damping", type=float, default=1e-3)
    parser.add_argument("--silhouette-max-rotation-deg", type=float, default=8.0)
    parser.add_argument("--silhouette-max-translation-mm", type=float, default=12.0)
    parser.add_argument(
        "--silhouette-max-step-rotation-deg", type=float, default=1.0
    )
    parser.add_argument(
        "--silhouette-max-step-translation-mm", type=float, default=2.0
    )
    parser.add_argument(
        "--silhouette-depth-gate-mm",
        type=_positive_float,
        default=None,
        help=(
            "Use a contour sample only when its rendered and measured depths "
            "at the same pixel differ by no more than this value (default: disabled)"
        ),
    )
    parser.add_argument(
        "--silhouette-depth-gate-occlusion-only",
        action="store_true",
        help=(
            "Reject only measured depth that is in front of the rendered "
            "contour by more than the depth gate"
        ),
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    config = ICPConfig(
        mode=ICPMode(args.mode),
        model_sample_count=args.model_sample_count,
        model_scale=args.model_scale,
        target_voxel_size=args.target_voxel_size_mm / 1000.0,
        max_correspondence_distance=args.max_correspondence_distance_mm / 1000.0,
        target_neighborhood_distance=(
            None
            if args.target_neighborhood_distance_mm is None
            else args.target_neighborhood_distance_mm / 1000.0
        ),
        max_iteration=args.max_iteration,
        relative_fitness=args.relative_fitness,
        relative_rmse=args.relative_rmse,
        normal_radius=args.normal_radius_mm / 1000.0,
        normal_max_nn=args.normal_max_nn,
        colored_icp_lambda_geometric=args.colored_lambda_geometric,
        visibility_radius_factor=args.visibility_radius_factor,
        projective_normal_depth_delta=(
            args.projective_normal_depth_delta_mm / 1000.0
        ),
        projective_huber_delta=args.projective_huber_delta_mm / 1000.0,
        projective_max_step_rotation_deg=(
            args.projective_max_step_rotation_deg
        ),
        projective_max_step_translation=(
            args.projective_max_step_translation_mm / 1000.0
        ),
        random_seed=args.random_seed,
        sam_model_id=args.sam_model,
        sam_device=args.sam_device,
        sam_box_padding_fraction=args.sam_box_padding,
        sam_erosion_radius_px=args.sam_erode_px,
        sam_overlay_output_dir=args.sam_overlay_dir,
        sam_mask_output_dir=args.sam_mask_dir,
        sam_overlay_alpha=args.sam_overlay_alpha,
        silhouette_weight=args.silhouette_weight,
        silhouette_max_iteration=args.silhouette_max_iteration,
        silhouette_image_scale=args.silhouette_image_scale,
        silhouette_distance_scale_px=args.silhouette_distance_scale_px,
        silhouette_splat_radius_px=args.silhouette_splat_radius_px,
        silhouette_huber_delta_px=args.silhouette_huber_delta_px,
        silhouette_pose_prior_weight=args.silhouette_pose_prior_weight,
        silhouette_damping=args.silhouette_damping,
        silhouette_max_rotation_deg=args.silhouette_max_rotation_deg,
        silhouette_max_translation=args.silhouette_max_translation_mm / 1000.0,
        silhouette_max_step_rotation_deg=(
            args.silhouette_max_step_rotation_deg
        ),
        silhouette_max_step_translation=(
            args.silhouette_max_step_translation_mm / 1000.0
        ),
        silhouette_depth_gate=(
            None
            if args.silhouette_depth_gate_mm is None
            else args.silhouette_depth_gate_mm / 1000.0
        ),
        silhouette_depth_gate_occlusion_only=(
            args.silhouette_depth_gate_occlusion_only
        ),
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
