# Procrustes Annotator

Launch with a BOP-style dataset directory:

```bash
uv run procrustes-annotator dataset
```

Projects are versioned JSON documents stored with a `.project` extension.
Use **File → New project**, **Save project**, or **Load project** in the app. A
saved project can also be reopened directly from the command line:

```bash
uv run procrustes-annotator annotation.project
```

The project stores the dataset location (absolute and project-relative),
per-frame model landmarks and RGBD observations, solved Procrustes transforms,
the selected object/frame, interaction mode, model camera, RGB zoom/pan,
splitter sizes, and window geometry.

For an untouched frame, the app previews the most recently used model-marker
list for that object. The first model edit or successful RGB click copies that
list into independent per-frame state, allowing pose-specific corrections
without requiring the markers to be recreated from scratch.

Use **File → Export solved poses as JSON** to produce a BOP `scene_gt`-style
mapping. Rotations use `cam_R_m2c`; translations use `cam_t_m2c` in
millimetres. Each exported pose also includes its annotation RMSD and
per-landmark residuals in millimetres. Press **R** or use the **Defer pose**
button to mark the current object/frame unannotatable. Deferred export records
use `"annotation_status": "deferred"` and omit the unavailable transform.

The RGB view marks invalid depth pixels with a light-red tint. RGB clicks use
the mean of valid depth samples within a two-pixel circular neighborhood, so
isolated missing-depth speckles do not prevent annotation.
