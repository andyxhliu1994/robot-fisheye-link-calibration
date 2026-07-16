# Quickstart: calibrate a new dataset

This guide covers a replacement Unity recording or a real-robot recording that
has already been converted to the current file interface. See
[DATASET_FORMAT.md](DATASET_FORMAT.md) for exact schemas and
[RUN_PIPELINE.md](RUN_PIPELINE.md) for every command.

## 1. Prepare the repository

Clone or pull the repository, then create an isolated Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest
```

Keep the current folder as the session root. `calibration_pipeline/`, `dataset/`,
`outputs/`, and `tests/` are siblings.

## 2. Place the new recording

Create or replace local `./dataset/` outside the calibration workflow. Never use
pipeline code to rewrite raw data. At minimum provide:

```text
dataset/cameras/<camera>/rgb/frame_xxxxxx.jpg
dataset/link_poses/frame_xxxxxx.json
dataset/candidate_links.json
dataset/charuco_board_config.json
dataset/camera_model_config.json
dataset/camera_calibration/<matching-file>.json
dataset/session_summary.json
```

Camera images and link-pose files must share synchronized frame IDs. Every
link-pose file must include `T_base_link` for all candidate links with semantics
`p_base = T_base_link @ p_link`.

## 3. Match physical/configured geometry

1. Set `charuco_board_config.json` to the printed board dictionary and measured
   square/marker dimensions.
2. Set `camera_model_config.json` to the actual projection model, calibration
   file, raw ray frame, pose camera frame, and adapter.
3. Put the matching intrinsic/fisheye file under `camera_calibration/`.
4. Ensure `candidate_links.json` uses exactly the link paths exported in every
   `link_poses/frame_xxxxxx.json`.
5. Confirm the board stayed fixed throughout the sequence.

For `setup_allround_8` or any other multi-camera setup, use consistent camera
names in `cameras/`, `session_summary.json`, and all downstream consumers.
Synchronize every camera and include every plausible attached link.

## 4. Unity FOV checks

Use a camera calibration created for the recording's actual FOV, resolution, and
Unity physical camera settings. The depth-model manifest normally maps:

- 180° to `FisheyeConversions_2`;
- 210° to `FisheyeConversions_3`;
- 240° to `FisheyeConversions_4`.

For 210°/240° data, update the OCamCalib path, confirm the generated
`conversion_name`, and verify the ray adapter. Do not reuse the 180° intrinsic
file by accident. Current Unity OCamCalib data generally uses
`diag(1, -1, 1)`; confirm this against the actual exporter and calibration.

## 5. Validate before estimating

For a complete Unity recorder bundle, run:

```bash
python -m calibration_pipeline.run_integrity_check \
  --dataset ./dataset \
  --output ./outputs/integrity_report.json

python -m calibration_pipeline.run_dataset_sanity_check \
  --dataset ./dataset \
  --output ./outputs/dataset_sanity_report.json \
  --max-frames 50 \
  --frame-stride 10
```

Do not proceed if frames are misaligned or all camera streams are identical.
The current integrity CLI also expects the optional Unity CSV/GT bundle. On a
core-only real-robot dataset, manually/externally validate the required file
alignment using [NEW_DATASET_CHECKLIST.md](NEW_DATASET_CHECKLIST.md); absence of
Unity GT does not prevent calibration.

## 6. Run a small visual debug sample

```bash
python -m calibration_pipeline.run_charuco_detection \
  --dataset ./dataset --output ./outputs \
  --max-frames 100 --frame-stride 5 --save-overlays

python -m calibration_pipeline.run_pose_from_charuco \
  --dataset ./dataset --output ./outputs \
  --max-frames 100 --frame-stride 5 --save-overlays \
  --evaluate-gt
```

For real/no-GT data, use `--no-evaluate-gt` on pose estimation. Inspect
`outputs/debug_overlays/` and `outputs/pose_overlays/`. Resolve low visibility,
bad detections, or systematic pose errors before processing all desired frames.

## 7. Run calibration and export

For a first run or a setup that needs diagnosis, continue with the separate link
calibration, shared-board recovery, and final-export commands in
[RUN_PIPELINE.md](RUN_PIPELINE.md). Separate commands make link rankings,
motion-limited recovery, and threshold behavior easier to inspect.

For a routine run after `outputs/board_poses/` has been verified, use:

```bash
python -m calibration_pipeline.run_static_calibration_pipeline \
  --dataset ./dataset \
  --board-poses ./outputs/board_poses \
  --output ./outputs \
  --evaluate-gt
```

For real/no-GT data, replace the final flag with `--no-evaluate-gt`. This wrapper
does not rerun detection or board-pose estimation; it calls the existing link
calibration, shared recovery, and final export CLIs in sequence.

Inspect:

```text
outputs/link_calibration/link_calibration_summary.json
outputs/shared_board_recovery/shared_board_recovery_summary.json
outputs/final_calibration/final_calibration.json
```

Pay attention to score margins, observability, confidence, anchor agreement, and
warnings. Unity validation reports, when available, are diagnostic only.

## 8. Deploy the static result

Keep `outputs/final_calibration/final_calibration.json`. At runtime, obtain the
current FK pose of each camera's selected attached link and compute:

```text
T_base_cam(t) = T_base_link_attached(t) @ T_link_camera
```

Feed per-frame `T_base_cam` to the depth model. Do not feed `T_link_camera`
directly and do not apply the ray adapter again.

## Real-robot capture requirements

A real recording needs synchronized RGB images, matching camera intrinsics,
ChArUco geometry, candidate link names, and per-frame `T_base_link` for every
candidate. Produce link poses using a robot SDK, ROS `/tf`, URDF FK, DH FK, or
vendor FK before invoking this pipeline.

The robot base may remain stationary, but the links must execute sufficiently
diverse motion. Cameras on motion-limited links can sometimes be recovered from
the shared fixed board when at least two fully observable anchor cameras are
available.

The current pipeline does not yet ingest ROS/URDF/SDK data directly; that
adapter is future work and is not part of this milestone.
