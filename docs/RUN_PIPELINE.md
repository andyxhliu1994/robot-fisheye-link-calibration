# Run the calibration pipeline

Run every command from the repository root. Raw inputs stay under `dataset/`;
all commands write generated artifacts only under `outputs/`.

## Environment

Python 3.10–3.12 is supported:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Before a new recording, complete the
[dataset checklist](NEW_DATASET_CHECKLIST.md). The commands below use Unity GT
evaluation when it is available. For a real/no-GT recording, replace every
`--evaluate-gt` with `--no-evaluate-gt`.

There are two ways to run the static mount stages after board poses exist:

- **Debug / step-by-step mode** runs link calibration, shared-board recovery,
  and final export separately. Use it for a new setup, wrong-link diagnostics,
  recovery inspection, motion-limited cameras, or threshold tuning.
- **Routine wrapper mode** invokes those same three CLIs in order. Use it for
  repeated runs after the setup and board poses have already been checked.

## Standard command sequence

### 1. Run tests

```bash
python -m pytest
```

### 2. Check the Unity dataset bundle

```bash
python -m calibration_pipeline.run_integrity_check \
  --dataset ./dataset \
  --output ./outputs/integrity_report.json
```

The current integrity tool verifies alignment across the full Unity recorder
bundle, including CSV and GT sources. It should pass for supported Unity
recordings. A core-only real-robot dataset without those optional sources cannot
use this GT-oriented check unchanged; validate its required files and frame IDs
with the checklist, then use the no-GT calibration path below.

### 3. Check camera-stream distinctness

```bash
python -m calibration_pipeline.run_dataset_sanity_check \
  --dataset ./dataset \
  --output ./outputs/dataset_sanity_report.json \
  --max-frames 50 \
  --frame-stride 10
```

This catches duplicated render/export streams before they contaminate later
stages.

### 4. Detect ChArUco features

Start with the documented debug sample:

```bash
python -m calibration_pipeline.run_charuco_detection \
  --dataset ./dataset \
  --output ./outputs \
  --max-frames 100 \
  --frame-stride 5 \
  --save-overlays
```

Inspect the overlays and valid ratios. Omit `--max-frames` and use
`--frame-stride 1` to process every frame for a production run. `--camera NAME`
restricts a diagnostic run to one camera.

### 5. Estimate `T_camera_board`

```bash
python -m calibration_pipeline.run_pose_from_charuco \
  --dataset ./dataset \
  --output ./outputs \
  --max-frames 100 \
  --frame-stride 5 \
  --save-overlays \
  --evaluate-gt
```

This stage detects ChArUco corners from the selected RGB frames, maps pixels to
calibrated rays, applies the configured ray-frame adapter, and estimates board
pose. For a full run, omit `--max-frames` and use `--frame-stride 1`.

### 6. Associate links and estimate independent mounts (debug mode)

```bash
python -m calibration_pipeline.run_link_calibration \
  --dataset ./dataset \
  --board-poses ./outputs/board_poses \
  --output ./outputs \
  --evaluate-gt
```

This tests all candidate links and independently estimates static
`T_link_camera`. Unity GT is consulted only after the ranking and estimate are
complete.

### 7. Recover motion-limited cameras (debug mode)

```bash
python -m calibration_pipeline.run_shared_board_recovery \
  --dataset ./dataset \
  --link-calibration ./outputs/link_calibration/link_calibration_summary.json \
  --board-poses ./outputs/board_poses \
  --output ./outputs \
  --evaluate-gt
```

The default requires at least two fully observable anchor cameras. Do not use
`--allow-single-anchor` merely to suppress an insufficient-anchor warning; it is
an explicit reduced-redundancy mode.

### 8. Export the final static calibration (debug mode)

```bash
python -m calibration_pipeline.run_final_calibration_export \
  --dataset ./dataset \
  --link-calibration ./outputs/link_calibration/link_calibration_summary.json \
  --shared-board-recovery ./outputs/shared_board_recovery/shared_board_recovery_summary.json \
  --output ./outputs \
  --evaluate-gt
```

Keep `outputs/final_calibration/final_calibration.json` for deployment. The
static GT report, when available, is separate and evaluation-only.

### Routine alternative for steps 6–8

After `outputs/board_poses/` exists, replace the three debug commands above with:

```bash
python -m calibration_pipeline.run_static_calibration_pipeline \
  --dataset ./dataset \
  --board-poses ./outputs/board_poses \
  --output ./outputs \
  --evaluate-gt
```

For real/no-GT data:

```bash
python -m calibration_pipeline.run_static_calibration_pipeline \
  --dataset ./dataset \
  --board-poses ./outputs/board_poses \
  --output ./outputs \
  --no-evaluate-gt
```

The wrapper only orchestrates the existing independent link calibration,
shared-board recovery, and final static export. It does not run ChArUco
detection, estimate `T_camera_board`, or export depth-model compatibility data.
It produces the same standard summaries and final deployment artifact. Use
`--dry-run` to inspect the planned lower-level commands without executing them.

### 9. Optionally export offline absolute camera poses

```bash
python -m calibration_pipeline.run_export_final_camera_poses \
  --dataset ./dataset \
  --calibration ./outputs/final_calibration/final_calibration.json \
  --output ./outputs/final_calibration \
  --evaluate-gt
```

This composes recorded link poses with the final static mounts and writes
per-frame `T_base_cam` JSONL files. It is useful for offline validation and
depth-model preparation; it does not change static calibration.

### 10. Validate depth-model input compatibility

```bash
python -m calibration_pipeline.run_depth_model_compat_export \
  --dataset ./dataset \
  --calibration ./outputs/final_calibration/final_calibration.json \
  --output ./outputs/depth_model_compat \
  --evaluate-gt
```

By default this exports all aligned frames, per-frame transform JSONs, JSONL
mirrors, and a sample manifest. Missing depth images are represented by null
`depth_path` values rather than failing the pose export.

### 11. Summarize one completed experiment

```bash
python -m calibration_pipeline.run_calibration_evaluation \
  --dataset ./dataset \
  --outputs ./outputs \
  --output ./outputs/evaluation \
  --evaluate-gt
```

This read-only evaluator summarizes the available Milestones 2–6 outputs into
an aggregation-ready JSON, per-camera/ranking/pairwise CSV tables, simple PNG
plots, and `report.md`. It does not change `final_calibration.json` or other
calibration artifacts. GT metrics appear only when evaluation is enabled and
the separate validation reports/Unity transforms exist; GT-free detection,
ray-error, score-margin, observability, recovery, confidence, and compatibility
metrics remain available for real robots.

The candidate-link score heatmap uses `log10` residual scores for color contrast
while annotating each cell with its raw score. ★ marks the predicted/best link;
✓ marks the GT link when GT evaluation is available. Static `T_link_camera`
translation and runtime translation plots display millimetres, and static
translation/rotation bars include numeric labels. The relative-pose translation
heatmap also displays millimetres. These are display-only conversions: CSV and
JSON translation fields remain in metres for compatibility.

For a real/no-GT run:

```bash
python -m calibration_pipeline.run_calibration_evaluation \
  --dataset ./dataset \
  --outputs ./outputs \
  --output ./outputs/evaluation \
  --no-evaluate-gt
```

Use `--camera NAME` to report one camera, `--no-save-plots` for tables only,
`--max-pairs N` to cap evaluated frames per ordered source-target pair, and
`--strict` when missing expected inputs should fail instead of becoming report
warnings. This command reports one experiment only; future benchmark tooling may
aggregate its `calibration_metrics_summary.json` outputs across runs.

## Real robot / no-GT mode

Use the same estimation stages but disable evaluation:

```bash
python -m calibration_pipeline.run_pose_from_charuco \
  --dataset ./dataset --output ./outputs \
  --max-frames 100 --frame-stride 5 --save-overlays \
  --no-evaluate-gt

python -m calibration_pipeline.run_link_calibration \
  --dataset ./dataset --board-poses ./outputs/board_poses \
  --output ./outputs --no-evaluate-gt

python -m calibration_pipeline.run_shared_board_recovery \
  --dataset ./dataset \
  --link-calibration ./outputs/link_calibration/link_calibration_summary.json \
  --board-poses ./outputs/board_poses \
  --output ./outputs --no-evaluate-gt

python -m calibration_pipeline.run_final_calibration_export \
  --dataset ./dataset \
  --link-calibration ./outputs/link_calibration/link_calibration_summary.json \
  --shared-board-recovery ./outputs/shared_board_recovery/shared_board_recovery_summary.json \
  --output ./outputs --no-evaluate-gt

python -m calibration_pipeline.run_depth_model_compat_export \
  --dataset ./dataset \
  --calibration ./outputs/final_calibration/final_calibration.json \
  --output ./outputs/depth_model_compat --no-evaluate-gt

python -m calibration_pipeline.run_calibration_evaluation \
  --dataset ./dataset --outputs ./outputs \
  --output ./outputs/evaluation --no-evaluate-gt
```

The final calibration still exports successfully. Unity GT validation reports
will be absent. Confidence is based on calibration evidence such as board-pose
consistency, link score margin, observability rank, anchor agreement, and ray
error—not on GT.

Directly obtaining `T_base_link` from ROS, URDF, DH parameters, or a robot SDK
is outside the current pipeline. Convert those sources to synchronized
`link_poses/frame_xxxxxx.json` first.

## Interpreting each stage

### Integrity

For a full Unity recording, the overall result should be `PASS`; expected frame
counts and IDs should align. Resolve missing, duplicate, or unexpected frames
before calibration.

### Dataset sanity

All-camera byte-identical same-frame counts should be zero or demonstrably rare.
Streams and camera transforms should differ across cameras. Identical streams
usually indicate an export/render-target bug.

### ChArUco detection

The valid ratio depends on visibility, but a very low ratio means there may not
be enough board poses for link calibration. Inspect invalid reasons and visual
overlays. Partial board visibility is supported; at least four markers and eight
ChArUco corners are the approximate validity threshold.

### Board pose

Mean ray angular error should be small and pose overlays should align with the
board. Invalid records should be explainable, commonly by insufficient corners.
Large or systematic errors suggest a board geometry, camera model, FOV, or
adapter mismatch.

### Link calibration

The best link should have a meaningful score margin over the runner-up. Rich
motion should give full observability rank. Inspect every camera's ranking,
confidence, consistency statistics, and warnings rather than accepting only the
best-link name.

### Motion-limited recovery

Shoulder or single-axis link motion can leave mount degrees of freedom
unobservable. Shared-board recovery can resolve such cameras when enough fully
observable anchor cameras see the same fixed board. If anchors are insufficient,
the correct result is a warning and low confidence—not fabricated certainty.

### Final calibration

Every camera should have an `attached_link`, `T_link_camera`, confidence, and
observability record. Investigate warnings. Per-camera `gt_*` error fields must
not appear in this deployment file; validation is referenced separately.

### Unity GT reports

GT metrics are available only when Unity truth files exist and evaluation is
enabled. They assess completed outputs but are never used to pick a link,
transform, or confidence.

## Troubleshooting

### All camera RGB streams are identical

This commonly indicates a Unity RenderTexture/export bug. Run dataset sanity,
then fix the recorder and recollect data. Do not continue with duplicated images.

### ChArUco detection is too low

Check board size and distance, edge distortion, focus/motion blur, lighting,
occlusion, the ArUco dictionary, and metric board dimensions. Collect more
board-visible frames with varied robot poses.

### Camera model or FOV is wrong

An 180/210/240 mismatch, wrong OCamCalib JSON, resolution mismatch, or incorrect
ray adapter causes systematic ray/pose errors. Use the calibration created for
the actual FOV and camera settings.

### The board moved

The link calibration assumes a fixed board. If it moved, split the recording
into fixed-board segments or use a future method that models known board motion.

### Observability is low

Single-axis motion cannot independently constrain all six mount degrees of
freedom. Collect more diverse joint motion or use shared-board recovery with at
least two suitable anchor cameras.

### There are not enough anchors

Without fully observable anchors, complete shared-board recovery may be
impossible. Add observable cameras or recollect data with richer motion.

### Transform directions look wrong

Use `T_A_B: p_A = T_A_B @ p_B`. Runtime composition is
`T_base_cam = T_base_link @ T_link_camera`. Invert an input only when its
documented semantics are the opposite direction.

### The depth model pose is confusing

The depth model consumes per-frame `T_base_cam`, not static `T_link_camera`.
Obtain current robot FK for the attached link and compose the two transforms.
Do not apply the camera-ray adapter a second time.
