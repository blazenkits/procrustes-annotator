"""Portable SAM + silhouette-ICP batch for the manual annotation handoff.

No GUI or CUDA import is required to create a job. The GPU workstation runs
this module on the archive and returns a single JSON results file.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d

from .icp import load_sam2_predictor
from .loader import DataSet
from .review_pipeline import SAM_MODEL_ID, generate_sam_mask, propose_silhouette_icp

RAW_FORMAT = "procrustes-raw-batch-v1"
RESULT_FORMAT = "procrustes-icp-batch-v1"


def record_hash(record: dict[str, Any]) -> str:
    """Bind imported results to the exact manual pose and prompts."""
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def encode_mask(mask: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", np.asarray(mask, np.uint8) * 255)
    if not ok:
        raise ValueError("Could not encode the SAM mask.")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def decode_mask(encoded: str, shape: tuple[int, int]) -> np.ndarray:
    raw = base64.b64decode(encoded, validate=True)
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != shape or not np.any(image):
        raise ValueError("Result mask is empty or does not match the RGB frame.")
    return image > 0


def write_json(path: Path, document: dict[str, Any]) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as file:
        temporary = Path(file.name)
        try:
            json.dump(document, file, indent=2, allow_nan=False)
            file.write("\n")
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def make_job_archive(dataset_root: Path, raw_path: Path, archive_path: Path) -> None:
    """Package the small dataset and raw poses for an SSH file transfer."""
    dataset_root = Path(dataset_root).resolve()
    raw_path = Path(raw_path).resolve()
    archive_path = Path(archive_path).resolve()
    with raw_path.open(encoding="utf-8") as file:
        raw = json.load(file)
    if raw.get("format") != RAW_FORMAT:
        raise ValueError("Raw pose file has the wrong format.")
    capture = dataset_root / "train" if (dataset_root / "train" / "scene_camera.json").is_file() else dataset_root
    depth_dir = capture / "depth" / "1" if (capture / "depth" / "1").is_dir() else capture / "depth"
    files = {capture / "scene_camera.json"}
    if (capture / "scene_gt.json").is_file():
        files.add(capture / "scene_gt.json")
    for frame, record in _records(raw):
        files.add(capture / "rgb" / f"{frame:06d}.png")
        files.add(depth_dir / f"{frame:06d}.png")
        files.add(dataset_root / "models" / f"obj_{int(record['obj_id']):06d}.ply")
    missing = sorted(path for path in files if not path.is_file() or path.is_symlink())
    if missing:
        raise FileNotFoundError(f"GPU job is missing required dataset file {missing[0]}")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
        archive.write(raw_path, "raw_poses.json")
        for path in sorted(files):
            archive.write(path, "dataset/" + path.relative_to(dataset_root).as_posix())


def _records(raw: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    if raw.get("format") != RAW_FORMAT or not isinstance(raw.get("poses"), dict):
        raise ValueError("Expected a procrustes-raw-batch-v1 document.")
    pairs = []
    for frame_text, records in raw["poses"].items():
        if not isinstance(records, list):
            raise TypeError(f"Frame {frame_text} must contain a list of poses.")
        for record in records:
            pairs.append((int(frame_text), record))
    keys = [(frame, int(record["obj_id"])) for frame, record in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("Raw pose file contains duplicate frame/object pairs.")
    return pairs


def depth_rmse_mm(dataset: DataSet, frame_id: int, object_id: int, rotation: np.ndarray, translation_m: np.ndarray) -> float | None:
    """Rendered CAD versus captured depth, excluding |residual| >= 50 mm.

    This is sensor agreement, *not* pose ground truth. The ray direction has
    z=1, so Open3D's ray parameter is camera-space depth in metres.
    """
    frame = dataset[frame_id]
    measured = np.asarray(frame.depth, np.float32)
    height, width = measured.shape
    mesh = o3d.io.read_triangle_mesh(str(dataset.models[object_id].mesh_path))
    if not mesh.has_triangles():
        raise ValueError(f"Object {object_id} has no renderable mesh triangles.")
    mesh.scale(0.001, center=(0, 0, 0))
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(rotation, np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation_m, np.float64).reshape(3)
    mesh.transform(transform)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    yy, xx = np.indices((height, width), dtype=np.float32)
    k = frame.camera_intrinsics
    rays = np.zeros((height, width, 6), np.float32)
    rays[..., 3] = (xx - float(k[0, 2])) / float(k[0, 0])
    rays[..., 4] = (yy - float(k[1, 2])) / float(k[1, 1])
    rays[..., 5] = 1.0
    rendered = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    valid = np.isfinite(rendered) & np.isfinite(measured) & (measured > 0)
    residual = (rendered[valid] - measured[valid]) * 1000.0
    residual = residual[np.abs(residual) < 50.0]
    return float(np.sqrt(np.mean(np.square(residual, dtype=np.float64)))) if len(residual) else None


def run_batch(
    dataset: DataSet,
    raw: dict[str, Any],
    *,
    segmenter: Any | None = None,
    progress: Callable[[int, int, int, int], None] | None = None,
) -> dict[str, Any]:
    """Generate every SAM mask and bounded ICP proposal; never accept poses."""
    pairs = _records(raw)
    if not pairs:
        raise ValueError("No manually solved poses were supplied.")
    method = str(raw.get("method", "silhouette"))
    weight = float(raw.get("silhouette_weight", 0.10))
    if method not in ("silhouette", "colored") or not np.isfinite(weight) or weight < 0:
        raise ValueError("Batch refinement settings are invalid.")
    if segmenter is None:
        segmenter = load_sam2_predictor(SAM_MODEL_ID, device="auto")
    results: dict[str, dict[str, Any]] = {}
    for index, (frame_id, record) in enumerate(pairs, 1):
        object_id = int(record["obj_id"])
        if progress:
            progress(index, len(pairs), frame_id, object_id)
        entry: dict[str, Any] = {
            "obj_id": object_id,
            "raw_sha256": record_hash(record),
            "status": "failed",
        }
        try:
            rotation = np.asarray(record["cam_R_m2c"], np.float64).reshape(3, 3)
            translation = np.asarray(record["cam_t_m2c"], np.float64).reshape(3) / 1000.0
            clicks = np.asarray(record["manual_clicks_xy"], np.float32).reshape(-1, 2)
            positive = np.asarray(record.get("extra_positive_xy", []), np.float32).reshape(-1, 2)
            negative = np.asarray(record.get("extra_negative_xy", []), np.float32).reshape(-1, 2)
            mask, score, box = generate_sam_mask(
                dataset, frame_id, object_id, rotation, translation, clicks,
                positive, negative, segmenter,
                box_padding=float(record.get("box_padding", 0.25)),
            )
            entry.update({
                "mask_png_base64": encode_mask(mask),
                "sam_score": score,
                "sam_box_xyxy": np.asarray(box, np.float64).reshape(4).tolist(),
                "raw_depth_rmse_mm": depth_rmse_mm(dataset, frame_id, object_id, rotation, translation),
            })
            candidate_r, candidate_t, icp_info = propose_silhouette_icp(
                dataset, frame_id, object_id, rotation, translation,
                float(record.get("annotation_rmsd_mm", 0.0)) / 1000.0, mask,
                method=method, silhouette_weight=weight,
            )
            entry.update({
                "status": "ready",
                "candidate_rotation_m2c": candidate_r.reshape(-1).tolist(),
                "candidate_translation_m2c_mm": (candidate_t * 1000.0).tolist(),
                "icp_depth_rmse_mm": depth_rmse_mm(dataset, frame_id, object_id, candidate_r, candidate_t),
                "icp_info": icp_info,
                "method": method,
                "silhouette_weight": weight if method == "silhouette" else None,
            })
        except Exception as error:  # noqa: BLE001 - isolate failures per pose
            entry["error"] = str(error)
        results.setdefault(str(frame_id), {})[str(object_id)] = entry
    return {"format": RESULT_FORMAT, "model_id": SAM_MODEL_ID,
            "method": method, "silhouette_weight": weight if method == "silhouette" else None,
            "results": results}


def run_job_archive(archive_path: Path, output_path: Path) -> dict[str, Any]:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable on this machine. Run the GPU job on the GPU workstation.")
    with tempfile.TemporaryDirectory(prefix="procrustes-gpu-job-") as temporary:
        root = Path(temporary)
        with zipfile.ZipFile(archive_path) as archive:
            for item in archive.infolist():
                target = (root / item.filename).resolve()
                if not target.is_relative_to(root) or item.file_size > 2_000_000_000:
                    raise ValueError("GPU job contains an unsafe archive member.")
            archive.extractall(root)
        with (root / "raw_poses.json").open(encoding="utf-8") as file:
            raw = json.load(file)
        dataset = DataSet.load(root / "dataset", depth_preset=1)
        result = run_batch(dataset, raw, progress=lambda i, n, f, o: print(f"[{i}/{n}] frame {f}, object {o}", flush=True))
        write_json(output_path, result)
        return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a portable Procrustes SAM+ICP GPU job.")
    parser.add_argument("--job", type=Path, required=True, help="Archive produced by Proceed to ICP")
    parser.add_argument("--output", type=Path, required=True, help="JSON file to return to the annotator")
    args = parser.parse_args(argv)
    result = run_job_archive(args.job, args.output)
    statuses = [entry["status"] for frame in result["results"].values() for entry in frame.values()]
    print(f"Saved {args.output}: {statuses.count('ready')} ready, {statuses.count('failed')} failed")


if __name__ == "__main__":
    main()
