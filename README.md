GT annotation tool for 6D Pose Estimation RGBD datasets written in Qt/PySide6.

Uses Procrustes point cloud matching via Kabsch algorithm.

# Usage

1. Install [uv](https://github.com/astral-sh/uv) 

2. To run: 
```bash
uv run procrustes-annotator
```
- To open a project:
  
```bash
uv run procrustes-annotator main.project
```

You may override `DataSet.load()` in loader.py to comply with a custom dataset structure.

## ICP refinement

Refine an exported pose file with single-scale Open3D ICP:

```bash
uv run python src/backend/icp.py \
  --dataset dataset \
  --input out/main_poses.json \
  --output out/main_poses_icp.json \
  --mode point-to-plane \
  --model-sample-count 50000 \
  --max-correspondence-distance-mm 10
```

The backend works in metres, while preserving BOP-style `cam_t_m2c` output in
millimetres. It uniformly samples the mesh, keeps points visible from the
camera under the rough pose, unprojects valid RGB-D pixels into camera space,
and uses the rough model-to-camera transform to initialize ICP. Raw depth 0
and 65535 are treated as invalid by the dataset loader. Deferred records are
copied unchanged. Solved records retain their annotation fields, replace the
pose with the ICP result, and add an `icp` object containing the rough pose,
fitness, inlier RMSE, correspondence count, parameters, and refined 4x4
transformation. Object/background masking is intentionally left as a TODO.

Review rough and refined poses with the PySide6 overlay viewer:

```bash
uv run procrustes-icp-viewer \
  --dataset dataset \
  --poses out/main_poses_icp.json
```

Press `1` for RGB only, `2` for the rough pose in red, `3` for the ICP pose in
green, or `4` for both. In the combined 2D view, matching pixels appear yellow.
Use the top tabs to switch between the 2D RGB overlay and a 3D RGB-D scene. The
3D tab shows the measured RGB-colored point cloud with translucent, outlined
rough and ICP meshes using the same `1`–`4` controls.
