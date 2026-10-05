"""SAM V2 prompting and reviewed-mask silhouette ICP for one annotation."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import cv2
import numpy as np
import open3d as o3d

from .icp import (
    ICPConfig,
    ICPMode,
    erode_binary_mask,
    load_sampled_model,
    projected_model_box,
    refine_annotations,
)
from .loader import DataSet

SAM_MODEL_ID = "facebook/sam2.1-hiera-large"
SILHOUETTE_WEIGHT = 0.10
EROSION_RADIUS_PX = 4


def rough_silhouette(
    points_m: np.ndarray,
    rotation: np.ndarray,
    translation_m: np.ndarray,
    intrinsics: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Rasterize a solid approximate CAD silhouette for SAM's mask prior."""
    height, width = image_shape
    camera = np.asarray(points_m) @ rotation.T + translation_m
    camera = camera[np.isfinite(camera).all(axis=1) & (camera[:, 2] > 0)]
    if len(camera) < 3:
        raise ValueError("Manual pose projects too few model points in front of the camera.")
    u = np.rint(intrinsics[0, 0] * camera[:, 0] / camera[:, 2] + intrinsics[0, 2]).astype(int)
    v = np.rint(intrinsics[1, 1] * camera[:, 1] / camera[:, 2] + intrinsics[1, 2]).astype(int)
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    occupancy = np.zeros((height, width), np.uint8)
    occupancy[v[valid], u[valid]] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    occupancy = cv2.morphologyEx(cv2.dilate(occupancy, kernel), cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(occupancy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) >= 4]
    if not contours:
        raise ValueError("Manual pose did not form a usable projected CAD silhouette.")
    solid = np.zeros_like(occupancy)
    cv2.drawContours(solid, contours, -1, 255, thickness=cv2.FILLED)
    return solid > 0


def low_resolution_logits(mask: np.ndarray) -> np.ndarray:
    resized = cv2.resize(np.asarray(mask, np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)
    return np.where(resized > 0, 8.0, -8.0).astype(np.float32)[None, :, :]


def generate_sam_mask(
    dataset: DataSet,
    frame_id: int,
    object_id: int,
    rotation: np.ndarray,
    translation_m: np.ndarray,
    manual_clicks_xy: np.ndarray,
    extra_positive_xy: np.ndarray,
    extra_negative_xy: np.ndarray,
    segmenter: Any,
    *,
    box_padding: float = 0.25,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Prompt SAM 2.1 Large with CAD box, clicks, and rough silhouette prior."""
    pose = dataset[frame_id]
    clicks = np.asarray(manual_clicks_xy, dtype=np.float32).reshape(-1, 2)
    positive = np.asarray(extra_positive_xy, dtype=np.float32).reshape(-1, 2)
    negative = np.asarray(extra_negative_xy, dtype=np.float32).reshape(-1, 2)
    if len(clicks) < 3:
        raise ValueError("At least three confirmed Procrustes clicks are required for SAM.")
    o3d.utility.random.seed(0)
    cloud = load_sampled_model(dataset.models[object_id], sample_count=50_000, model_scale=0.001)
    points = np.asarray(cloud.points)
    box = projected_model_box(points, rotation, translation_m, pose.camera_intrinsics, pose.rgb.shape, padding_fraction=box_padding)
    prior = rough_silhouette(points, rotation, translation_m, pose.camera_intrinsics, pose.rgb.shape[:2])
    point_coords = np.concatenate((clicks, positive, negative), axis=0)
    point_labels = np.concatenate((np.ones(len(clicks) + len(positive), np.int32), np.zeros(len(negative), np.int32)))
    torch_module = segmenter._torch
    autocast = (
        torch_module.autocast(device_type="cuda", dtype=torch_module.bfloat16)
        if str(segmenter.device).startswith("cuda") else nullcontext()
    )
    with torch_module.inference_mode(), autocast:
        segmenter.predictor.set_image(np.ascontiguousarray(pose.rgb[:, :, :3]))
        masks, scores, _ = segmenter.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=np.asarray(box, np.float32),
            mask_input=low_resolution_logits(prior),
            multimask_output=False,
            return_logits=False,
        )
    masks = np.asarray(masks)
    scores = np.asarray(scores, np.float64).reshape(-1)
    if masks.shape != (1, *pose.rgb.shape[:2]) or len(scores) != 1:
        raise ValueError("SAM returned an unexpected mask shape.")
    if not np.isfinite(scores[0]):
        raise ValueError("SAM returned a non-finite mask score.")
    mask = np.asarray(masks[0] > 0, dtype=bool)
    if not np.any(mask):
        raise ValueError("SAM returned an empty mask; add prompts and retry.")
    if not np.any(erode_binary_mask(mask, EROSION_RADIUS_PX)):
        raise ValueError("SAM mask vanishes after the required 4-pixel erosion.")
    return mask, float(scores[0]), box


class ApprovedMaskSegmenter:
    """Supply a human-approved binary mask to the existing V1 ICP backend."""

    model_id = SAM_MODEL_ID + " (human-approved mask)"
    device = "precomputed"

    def __init__(self, mask: np.ndarray):
        self.mask = np.asarray(mask, dtype=bool)

    def predict_masks(self, image: np.ndarray, box_xyxy: np.ndarray, *, multimask_output: bool) -> tuple[np.ndarray, np.ndarray]:
        if self.mask.shape != image.shape[:2]:
            raise ValueError("Approved mask dimensions do not match this RGB frame.")
        return self.mask[None].copy(), np.array([1.0])


def propose_silhouette_icp(
    dataset: DataSet,
    frame_id: int,
    object_id: int,
    rotation: np.ndarray,
    translation_m: np.ndarray,
    manual_rmse_m: float,
    mask: np.ndarray,
    *,
    method: str = "silhouette",
    silhouette_weight: float = SILHOUETTE_WEIGHT,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return SAM-masked silhouette or colored ICP without mutating the raw pose."""
    if method not in ("silhouette", "colored"):
        raise ValueError("Method must be 'silhouette' or 'colored'.")
    if not np.isfinite(silhouette_weight) or silhouette_weight < 0:
        raise ValueError("Silhouette weight must be finite and non-negative.")
    record = {
        "obj_id": object_id,
        "annotation_status": "solved",
        "cam_R_m2c": np.asarray(rotation, np.float64).reshape(-1).tolist(),
        "cam_t_m2c": (np.asarray(translation_m, np.float64) * 1000.0).tolist(),
        "annotation_rmsd_mm": manual_rmse_m * 1000.0,
    }
    config = ICPConfig(
        mode=ICPMode.Colored if method == "colored" else ICPMode.PointToPlane,
        model_sample_count=50_000,
        max_correspondence_distance=0.010,
        max_iteration=50,
        normal_radius=0.010,
        normal_max_nn=30,
        random_seed=0,
        sam_model_id=SAM_MODEL_ID,
        sam_device="precomputed",
        sam_erosion_radius_px=EROSION_RADIUS_PX,
        silhouette_weight=silhouette_weight if method == "silhouette" else 0.0,
    )
    refined = refine_annotations(dataset, {str(frame_id): [record]}, config=config, segmenter=ApprovedMaskSegmenter(mask))
    result = refined[str(frame_id)][0]
    info = result.get("icp", {})
    if info.get("status") != "refined":
        raise ValueError(f"Silhouette ICP could not refine this pose: {info.get('reason', 'no valid correspondences')}.")
    info["review_method"] = method
    info["review_silhouette_weight"] = silhouette_weight if method == "silhouette" else None
    return (
        np.asarray(result["cam_R_m2c"], np.float64).reshape(3, 3),
        np.asarray(result["cam_t_m2c"], np.float64) / 1000.0,
        info,
    )
