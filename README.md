# Multi-object Procrustes annotator

# 사용법

```bash
uv sync
uv run procrustes-annotator /absolute/path/to/dataset
```

The dataset directory should contain `models/obj_*.ply`,
`train/scene_camera.json`, `train/rgb/`, and `train/depth/`. Depth may either be
directly under `train/depth/` or under numbered presets such as
`train/depth/1/` and `train/depth/2/`. `train/scene_gt.json` is optional and is
used only for reference overlays. Local annotation does **not** need CUDA or
SAM; install the optional `segmentation` extra on a CUDA workstation to run the
SAM + ICP batch locally.

## Workflow

1. For each well-visible frame/object pair, pick matching CAD and RGB-D points
   and click **Solve Procrustes**. A solved manual pose is automatically a
   setup target. There is no separate target checkbox or confirmation step.
   Use `C`/`X` to move through objects and frames, and hold `Tab` to inspect
   the pose overlay. **Setup overview…** lists progress.
2. Click **Proceed to ICP…** in the footer. Choose bounded silhouette ICP and
   its weight (default 0.10), or SAM-masked color ICP. Choose a path for a
   normal raw-pose JSON export. The app saves that JSON **before** processing
   and also creates a `*_batch_input.json` manifest and `*_gpu_job.zip`
   containing only the required RGB, depth, camera, and CAD files.
3. If CUDA is available locally, the app runs SAM 2.1 Hiera Large and bounded
   silhouette ICP for every manually solved pair. Otherwise, move the job ZIP
   to the GPU machine, run the command below there, then download the single
   result JSON and click **Import GPU results…**. No SSH credentials are stored
   in the app.
4. **Review SAM + ICP…** opens one window with frame/object dropdowns and
   Previous/Next controls. The blue boundary on the left is the SAM mask;
   left/right click adds positive/negative prompts. Both images zoom with the
   wheel. The ICP image on the right is draggable; check **Drag/pan mask image**
   to drag the left one. Hold **Tab** to see the raw Procrustes pose on the right.
   ICP is kept by default: check **Reject ICP — use raw pose** only where needed,
   then click **Confirm all selections** once to save the decisions. To change
   prompts, method, or silhouette weight for a pair, click **Regenerate SAM +
   ICP…**. This creates a one-pair GPU job when the local machine lacks CUDA.
   The displayed raw/ICP RMSE is rendered CAD depth versus sensor
   depth, omitting absolute residuals ≥50 mm; it measures sensor agreement,
   **not** pose ground-truth accuracy.
5. Click **Solve setup (all objects)**. It jointly fits accepted manual/ICP
   target poses and infers the missing frame/object poses. The solver requires
   a connected observation graph: each frame needs an observed object, and
   every object must have been observed in at least one connected frame.
   Review the reported fit discrepancies before exporting.

All frames initially belong to `setup_0`. If the camera or object arrangement
changes, assign affected frames a different Setup ID. The setup solver assumes
one physical instance per CAD model ID. Save the editable `.project` via the
File menu; export final poses there too. The final export keeps raw poses and
ICP review decisions, and saves approved SAM masks in a `*_masks/` folder.

## GPU workstation command

From this `procrustes-annotator` directory on the GPU workstation:

```bash
uv sync --extra segmentation
uv run --extra segmentation python -m src.backend.batch_pipeline \
  --job /path/to/uploaded_gpu_job.zip \
  --output /path/to/gpu_results.json
```

The command requires CUDA, extracts the self-contained job temporarily, and
writes one result file. The first run may download the SAM 2.1 Hiera Large
checkpoint. Transfer `gpu_results.json` back to the annotator machine. Result
imports are checked against a hash of each raw pose and its SAM prompts, so a
stale result cannot silently replace a later manual edit.

Project files from versions 3–5 remain readable; new saves use version 6.

Changing CAD landmarks automatically invalidates the current Procrustes pose
without a confirmation popup. If RGB-D points have been selected but no pose
has been solved, the app warns before leaving that pair or proceeding to ICP.
