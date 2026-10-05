"""Interactive viewer for comparing silhouette-ICP weight sweeps."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QSlider,
)

try:
    from ..backend.loader import DataSet, ReferenceInstance
    from .icp_viewer import (
        ICPViewerWindow,
        ReviewPose,
        ViewerMode,
        _dataset_has_frame,
        load_reference_instances,
        parse_review_poses,
    )
except ImportError:  # pragma: no cover - direct ``python src/tools/...`` use.
    repository_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repository_root))
    from src.backend.loader import DataSet, ReferenceInstance
    from src.tools.icp_viewer import (
        ICPViewerWindow,
        ReviewPose,
        ViewerMode,
        _dataset_has_frame,
        load_reference_instances,
        parse_review_poses,
    )


@dataclass(frozen=True)
class SilhouetteSweep:
    """One refined pose file and its single silhouette-loss weight."""

    weight: float
    path: Path
    review_poses: tuple[ReviewPose, ...]


def parse_silhouette_sweep(
    document: dict[str, Any],
    *,
    source: str | Path = "<memory>",
) -> SilhouetteSweep:
    """Parse one output and verify every refined record uses one weight."""
    path = Path(source)
    review_poses = tuple(parse_review_poses(document))
    weights: list[float] = []
    for records in document.values():
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            metadata = record.get("icp")
            if not isinstance(metadata, dict) or metadata.get("status") != "refined":
                continue
            silhouette = metadata.get("silhouette")
            parameters = (
                silhouette.get("parameters")
                if isinstance(silhouette, dict)
                else None
            )
            if not isinstance(parameters, dict) or "weight" not in parameters:
                raise ValueError(
                    f"{path} contains a refined pose without silhouette weight metadata."
                )
            weight = float(parameters["weight"])
            if not np.isfinite(weight) or weight < 0:
                raise ValueError(f"{path} contains an invalid silhouette weight.")
            weights.append(weight)
    if not weights:
        raise ValueError(f"{path} contains no silhouette-constrained poses.")
    reference = weights[0]
    if not all(np.isclose(weight, reference, rtol=0.0, atol=1e-12) for weight in weights):
        raise ValueError(f"{path} contains more than one silhouette weight.")
    return SilhouetteSweep(
        weight=reference,
        path=path.expanduser().resolve() if path != Path("<memory>") else path,
        review_poses=review_poses,
    )


def load_silhouette_sweep(path: str | Path) -> SilhouetteSweep:
    input_path = Path(path).expanduser().resolve()
    with input_path.open(encoding="utf-8") as file:
        document = json.load(file)
    if not isinstance(document, dict):
        raise ValueError(f"Expected a JSON object in {input_path}.")
    return parse_silhouette_sweep(document, source=input_path)


def load_silhouette_sweeps(paths: Iterable[str | Path]) -> list[SilhouetteSweep]:
    """Load, sort, and validate compatible sweep outputs."""
    sweeps = sorted(
        (load_silhouette_sweep(path) for path in paths),
        key=lambda sweep: sweep.weight,
    )
    if not sweeps:
        raise ValueError("At least one swept pose JSON is required.")
    for previous, current in zip(sweeps, sweeps[1:]):
        if np.isclose(previous.weight, current.weight, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"Duplicate silhouette weight {current.weight:g}: "
                f"{previous.path} and {current.path}"
            )
    reference_keys = _pose_keys(sweeps[0].review_poses)
    for sweep in sweeps[1:]:
        keys = _pose_keys(sweep.review_poses)
        if keys != reference_keys:
            missing = sorted(reference_keys - keys)
            extra = sorted(keys - reference_keys)
            raise ValueError(
                f"{sweep.path} does not contain the same frame/object records "
                f"as {sweeps[0].path}; missing={missing}, extra={extra}."
            )
    return sweeps


def _pose_keys(review_poses: Iterable[ReviewPose]) -> set[tuple[int, int]]:
    return {(pose.frame_id, pose.object_id) for pose in review_poses}


class SilhouetteSweepViewerWindow(ICPViewerWindow):
    """The standard ICP viewer with a discrete silhouette-weight slider."""

    def __init__(
        self,
        dataset: DataSet,
        sweeps: list[SilhouetteSweep],
        *,
        reference_instances: dict[int, tuple[ReferenceInstance, ...]] | None = None,
        model_scale: float = 0.001,
    ) -> None:
        if not sweeps:
            raise ValueError("At least one silhouette sweep is required.")
        self.sweeps = sweeps
        self._active_sweep_index = 0
        filtered_sweeps: list[SilhouetteSweep] = []
        for sweep in sweeps:
            filtered = tuple(
                pose
                for pose in sweep.review_poses
                if pose.object_id in dataset.models
                and _dataset_has_frame(dataset, pose.frame_id)
            )
            filtered_sweeps.append(
                SilhouetteSweep(sweep.weight, sweep.path, filtered)
            )
        reference_keys = [
            (pose.frame_id, pose.object_id)
            for pose in filtered_sweeps[0].review_poses
        ]
        if not reference_keys:
            raise ValueError("No sweep records match the loaded dataset.")
        for sweep in filtered_sweeps[1:]:
            if [
                (pose.frame_id, pose.object_id) for pose in sweep.review_poses
            ] != reference_keys:
                raise ValueError("Sweep records differ after dataset filtering.")
        self.sweeps = filtered_sweeps
        super().__init__(
            dataset,
            list(self.sweeps[0].review_poses),
            reference_instances=reference_instances,
            model_scale=model_scale,
        )
        self.setWindowTitle("Silhouette-ICP Weight Sweep Viewer")
        self._mode_buttons[ViewerMode.ICP].setText("3  Silhouette ICP (green)")
        self._mode_buttons[ViewerMode.RoughAndICP].setText(
            "4  Rough + Silhouette ICP"
        )
        self._update_weight_label()
        self._update_details()

    @property
    def current_sweep(self) -> SilhouetteSweep:
        return self.sweeps[self._active_sweep_index]

    def _build_ui(self) -> None:
        super()._build_ui()
        row = QHBoxLayout()
        row.addWidget(QLabel("Silhouette weight λs:"))
        self.weight_slider = QSlider(Qt.Orientation.Horizontal)
        self.weight_slider.setRange(0, len(self.sweeps) - 1)
        self.weight_slider.setSingleStep(1)
        self.weight_slider.setPageStep(1)
        self.weight_slider.setTickInterval(1)
        self.weight_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.weight_slider.setMinimumWidth(420)
        self.weight_slider.setValue(self._active_sweep_index)
        self.weight_slider.valueChanged.connect(self._select_sweep)
        row.addWidget(self.weight_slider, stretch=1)
        self.weight_label = QLabel()
        self.weight_label.setMinimumWidth(190)
        row.addWidget(self.weight_label)
        self.centralWidget().layout().insertLayout(1, row)
        self._update_weight_label()

    def _select_sweep(self, index: int) -> None:
        if index == self._active_sweep_index:
            return
        selected_pose_index = self.pose_selector.currentIndex()
        self._active_sweep_index = index
        self.review_poses = list(self.current_sweep.review_poses)
        self._update_weight_label()
        if selected_pose_index >= 0:
            self.pose_selector.setCurrentIndex(selected_pose_index)
            review_pose = self.current_review_pose
            self.scene_view.set_refined_pose(
                self.dataset.models[review_pose.object_id].mesh_path,
                review_pose.icp_rotation_m2c,
                review_pose.icp_translation_m2c_m,
                model_scale=self.model_scale,
            )
            self._refresh_overlay()
            self._update_details()

    def _update_weight_label(self) -> None:
        if not hasattr(self, "weight_label"):
            return
        sweep = self.current_sweep
        self.weight_label.setText(
            f"λs = {sweep.weight:g}   ({self._active_sweep_index + 1}/{len(self.sweeps)})"
        )
        self.weight_slider.setToolTip(str(sweep.path))

    def _update_details(self) -> None:
        super()._update_details()
        self.details_label.setText(
            f"λs {self.current_sweep.weight:g}    |    "
            f"{self.details_label.text()}    |    {self.current_sweep.path.name}"
        )


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review multiple silhouette-ICP sweep outputs with a weight slider."
    )
    parser.add_argument("--dataset", type=Path, required=True, help="BOP dataset root")
    parser.add_argument(
        "--poses",
        "--inputs",
        dest="poses",
        type=Path,
        nargs="+",
        required=True,
        help="Two or more silhouette-ICP pose JSON files",
    )
    parser.add_argument(
        "--model-scale",
        type=float,
        default=0.001,
        help="PLY coordinate-to-metre scale (default: 0.001)",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        help=(
            "Optional full BOP scene_gt JSON used only for GT overlays and ADD; "
            "defaults to the dataset scene_gt.json"
        ),
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    application = QApplication.instance() or QApplication([sys.argv[0]])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataset = DataSet.load(args.dataset)
        sweeps = load_silhouette_sweeps(args.poses)
        reference_instances = (
            None
            if args.ground_truth is None
            else load_reference_instances(args.ground_truth)
        )
        window = SilhouetteSweepViewerWindow(
            dataset,
            sweeps,
            reference_instances=reference_instances,
            model_scale=args.model_scale,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        QMessageBox.critical(None, "Cannot start silhouette sweep viewer", str(error))
        return 1
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
