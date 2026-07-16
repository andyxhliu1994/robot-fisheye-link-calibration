# Output files

Every generated report, intermediate, overlay, and export belongs under
`outputs/`. These files are local run artifacts and are ignored by Git.

## Preflight reports

### `outputs/integrity_report.json`

Reports frame alignment, camera configuration readability, and Unity GT
transform consistency for the full supported Unity recorder bundle.

### `outputs/dataset_sanity_report.json`

Reports whether camera streams are distinct, whether each stream changes over
time, whether sampled camera transforms differ, and whether old detection files
look suspiciously identical. It catches multi-camera export bugs early.

## Detection and board pose intermediates

### `outputs/detections/<camera_name>.jsonl`

One ChArUco detection record per sampled frame, including invalid frames and
their reasons. Records contain marker/corner counts and detected image points.

### `outputs/debug_overlays/<camera_name>/frame_xxxxxx.jpg`

Visual ChArUco detection overlays. Use these to diagnose the board definition,
visibility, blur, and false detections.

### `outputs/board_poses/<camera_name>.jsonl`

Per-frame ray-based `T_camera_board` estimates and quality/evaluation metadata.
This is the direct input to link calibration.

### `outputs/pose_overlays/<camera_name>/frame_xxxxxx.jpg`

Pose-quality overlays with detected corners and ray-error visualization. Use
them to catch camera-model or frame-adapter mismatches.

## Link calibration intermediates

### `outputs/link_calibration/link_calibration_summary.json`

Independent candidate-link rankings and static `T_link_camera` estimates for
each camera, including score margins, consistency, observability, confidence,
and warnings. Motion-limited estimates may retain an unobservable gauge and are
not automatically safe for deployment.

The same directory may contain per-camera result JSONs.

### `outputs/shared_board_recovery/shared_board_recovery_summary.json`

Classifies observable anchors and motion-limited cameras, estimates the shared
fixed-board pose, and records recovered mounts where supported. It documents
anchor agreement, recovery consistency, confidence, and insufficient-anchor
warnings.

## Final calibration and validation

### `outputs/final_calibration/final_calibration.json`

This is the main deployment artifact. Keep it with the robot/camera setup. For
each camera it contains:

- camera name and selected attached link;
- static `T_link_camera` in matrix and translation/quaternion forms;
- calibration source, confidence, warnings, and observability;
- transform, camera-frame, link-frame, board, camera-model, and adapter metadata;
- lightweight references indicating whether evaluation reports exist.

It does not embed per-camera `gt_*` error metrics. It stores static
`T_link_camera`, not runtime `T_base_cam`.

### `outputs/final_calibration/final_static_calibration_validation.json`

Optional Unity-only evaluation of final static `T_link_camera` and attached-link
correctness. It is not needed for deployment and is never used to select final
transforms or confidence.

### `outputs/final_calibration/final_camera_pose_validation.json`

Optional Unity-only evaluation of per-frame absolute `T_base_cam` produced by
the final static calibration and recorded link poses.

### `outputs/final_calibration/camera_poses_base/<camera_name>.jsonl`

Offline per-frame `T_base_cam` records composed as
`T_base_link @ T_link_camera`. These are useful for validation or downstream
data preparation, but runtime systems should compose current FK in the same way.

### `outputs/final_calibration/README.md`

A concise deployment reminder generated beside `final_calibration.json`.

## Depth-model compatibility

`outputs/depth_model_compat/` contains integration artifacts that reproduce the
depth model's expected transform input style:

- `transforms/<camera_name>/frame_xxxxxx.json`: per-frame `T_base_cam`, including
  both `T_base_cam_rowmajor` and translation/quaternion parser fallback fields;
- `camera_poses_base/<camera_name>.jsonl`: JSONL mirror;
- `depth_model_samples.json`: multi-view sample manifest pointing to original
  RGB/depth files and calibrated transform JSONs;
- `depth_model_compatibility_report.json`: schema, parser, frame-count, relative
  pose, and adapter compatibility checks;
- `depth_model_pose_validation.json`: optional Unity-only absolute and relative
  GT metrics.

If depth images are absent, manifest `depth_path` values are null and pose export
still succeeds.

## What to deploy

Deploy `final_calibration.json` plus the software/configuration needed to obtain
current `T_base_link` for each selected attached link. At each timestamp compute:

```text
T_base_cam(t) = T_base_link(t) @ T_link_camera
```

Validation reports, debug overlays, detections, sampled board poses, and Unity GT
files are not required in a real deployment.
