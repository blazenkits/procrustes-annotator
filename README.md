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

ICP refinement 스크립트입니다. 수작업 annotation JSON 파일을 넣으면 ICP refinement 된 JSON이 출력됩니다.:

```bash
uv run python src/backend/icp.py \
  --dataset dataset \
  --input out/main_poses.json \
  --output out/main_poses_icp.json \
  --mode point-to-plane \
  --model-sample-count 50000 \
  --max-correspondence-distance-mm 10
```

## ICP vs Original Viewer

ICP / Original Pose를 놓고 비교할 수 있는 도구입니다.
```bash
uv run procrustes-icp-viewer \
  --dataset dataset \
  --poses out/main_poses_icp.json
```
