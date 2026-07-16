# Project milestones

## Milestone 1 — Dataset integrity check

Status: completed

- `outputs/integrity_report.json`: PASS
- pytest: 15 passed
- Maximum translation error: 2.02e-7 m
- Maximum rotation error: 2.70e-5 deg
- Maximum matrix element error: 6.35e-7

## Milestone 2 — ChArUco detection and debug overlays

Status: completed

- ChArUco detection implemented.
- Five detection JSONL files generated locally.
- 100 sampled records generated per camera.
- 500 debug overlays generated locally.
- Each camera: 62/100 valid frames.
- Mean marker count: 16.88.
- Mean ChArUco corner count: 20.80.
- Failures per camera: 30 no markers, 5 insufficient ChArUco corners, and
  3 insufficient markers.
- pytest: 19 passed.
- Milestone 1 integrity check: PASS.
- Important: sampled RGB files across the five cameras appear byte-identical and
  should be investigated before pose estimation.

## Milestone 2.5 — Camera stream sanity check

Status: completed — dataset check FAIL (`DATASET_IMAGE_EXPORT_SUSPECT`)

- All 50 sampled same-frame RGB sets were byte-identical across all five cameras.
- All ten camera-pair pixel comparisons had zero mean absolute difference, zero
  maximum absolute difference, and zero RMSE.
- Each camera had 49 unique hashes across 50 sampled frames and changed in 48/49
  adjacent sampled transitions. `frame_000000` and `frame_000010` were repeated.
- Exported camera transforms were distinct: sampled pairwise base-camera
  translation differences ranged from 0.1181 m to 0.9306 m and rotation
  differences ranged from 6.3569 deg to 179.7069 deg.
- Each camera reported a different `gt_link_path_rel`.
- Milestone 2 detection JSONLs were identical across cameras when ignoring
  `camera_name`, consistent with the duplicated RGB streams.
- Diagnosis: the Unity image export likely saved the same camera or render texture
  into every camera folder. Do not proceed to pose estimation until the export
  pipeline is fixed or independently verified.
- pytest: 24 passed.
- Milestone 1 integrity check: PASS.

### Corrected multi-camera dataset validation

Status: PASS (`CAMERA_STREAMS_AND_TRANSFORMS_DISTINCT`)

- The Unity recorder RenderTexture issue was fixed by forcing a unique RGB
  RenderTexture for each sub-camera.
- The corrected dataset has 1,022 aligned frames.
- Dataset sanity check: PASS.
- All-camera byte-identical frames: 0/50 sampled frames.
- Camera transforms distinct across cameras: true.
- Detection JSONLs identical ignoring `camera_name`: false.
- Integrity check: PASS.
- ChArUco detection rerun on the corrected dataset (100 sampled frames per
  camera):
  - Cam1: 65/100 valid.
  - Cam2: 53/100 valid.
  - Cam3: 27/100 valid.
  - Cam4: 31/100 valid.
  - Cam5: 47/100 valid.
- pytest: 24 passed.

## Milestone 3 — Ray-based ChArUco board pose estimation

Status: completed

- Implemented `T_camera_board` estimation from ChArUco correspondences and
  calibrated unit rays using robust `scipy.optimize.least_squares` refinement.
- The estimation core uses no robot kinematics or Unity ground truth; optional GT
  metrics are computed only after each pose has been estimated.
- Generated 500 local board-pose JSONL records and 500 pose overlays from 100
  stride-sampled frames per camera.
- Valid poses matched frames with at least eight ChArUco corners:
  - Cam1: 65/100; mean ray error 0.0254 deg.
  - Cam2: 53/100; mean ray error 0.0224 deg.
  - Cam3: 27/100; mean ray error 0.0226 deg.
  - Cam4: 31/100; mean ray error 0.0249 deg.
  - Cam5: 47/100; mean ray error 0.0278 deg.
- Overall: 223/500 valid poses, 0.0248 deg mean ray error, 0.0242 deg
  median ray error, and 0.0611 deg maximum per-frame mean ray error.
- No optimized pose had board points behind the camera; all invalid records were
  caused by insufficient detected ChArUco corners.
- Evaluation-only GT comparison used an explicit adapter from the ChArUco
  outer-corner/+Y-down frame to Unity's centered/+Y-up board frame. Across valid
  poses, mean GT error was 0.0124 m and 1.1460 deg (median 0.0108 m and
  0.8631 deg).
- pytest: 29 passed.
- Milestone 1 integrity check: PASS.
- Corrected-dataset sanity check: PASS (`CAMERA_STREAMS_AND_TRANSFORMS_DISTINCT`).

## Milestone 4 — Link association and camera mount estimation

Status: completed

- Implemented six-candidate link ranking and static `T_link_camera` estimation
  from board-pose consistency using hand-eye initialization and robust joint
  SE(3) refinement. Ground truth is excluded from estimation, scoring, ranking,
  and initialization.
- Generated `outputs/link_calibration/link_calibration_summary.json` and one
  local result JSON per camera, including the complete six-link ranking.
- The GT-attached link ranked first for all five cameras:
  - Cam1: `shoulder_link`; score 0.009114; margin 0.350257; 65 poses.
  - Cam2: `upper_arm_link`; score 0.009987; margin 0.551230; 53 poses.
  - Cam3: `forearm_link`; score 0.010626; margin 0.633190; 27 poses.
  - Cam4: `wrist_1_link`; score 0.004911; margin 0.439152; 31 poses.
  - Cam5: `wrist_2_link`; score 0.006057; margin 0.430774; 47 poses.
- Mean best-link board consistency ranged from 0.0031 m to 0.0076 m and
  0.36 deg to 0.99 deg.
- Cam2–Cam5 mounts were fully observable (Jacobian rank 12/12). Their optional
  post-estimation GT errors ranged from 0.0026 m to 0.0101 m and 0.68 deg to
  0.84 deg.
- Cam1 shoulder motion was rank deficient (10/12): translation along and
  rotation about its single motion axis are not identifiable from this dataset.
  Its output uses an explicit minimum-norm gauge and must not be interpreted as
  a fully recovered mount; optional GT error was 0.1229 m and 175.98 deg.
- pytest: 34 passed.
- Milestone 1 integrity check: PASS.
- Corrected-dataset sanity check: PASS (`CAMERA_STREAMS_AND_TRANSFORMS_DISTINCT`).

## Milestone 4.5 — Shared-board recovery for motion-limited cameras

Status: completed

- Implemented observability-based camera classification without assuming a
  particular link name: Cam2–Cam5 were selected as fully observable anchors and
  Cam1 was selected as motion-limited.
- Estimated one robust shared `T_base_board` from 158 valid anchor-camera poses.
  Anchor agreement was 0.006516 m mean translation and 0.6468 deg mean rotation.
- Recovered Cam1's full `T_link_camera` from 65 valid poses using the shared
  fixed board, with high confidence. Recovery consistency was 0.023125 m mean
  translation and 1.0389 deg mean rotation.
- Optional post-recovery GT evaluation measured 0.014094 m translation error and
  0.6628 deg rotation error for Cam1, replacing the unobservable independent
  gauge result from Milestone 4.
- Added explicit handling for multiple motion-limited cameras, no-recovery-needed
  datasets, one-anchor opt-in recovery, and zero/insufficient-anchor datasets.
- Ground truth is excluded from anchor selection, shared-board estimation,
  recovery, and confidence assignment; it is loaded only after recovery is
  finalized.
- Generated `outputs/shared_board_recovery/shared_board_recovery_summary.json`
  locally.
- pytest: 41 passed.
- Milestone 1 integrity check: PASS.
- Corrected-dataset sanity check: PASS (`CAMERA_STREAMS_AND_TRANSFORMS_DISTINCT`).

## Milestone 5 — Final calibration export and pose validation

Status: completed

- Generated `outputs/final_calibration/final_calibration.json` with explicit
  `T_A_B` semantics, deployment frame definitions, board geometry, link-frame
  metadata, and the actual OCamCalib-to-pose ray adapter matrix
  `diag(1, -1, 1)`.
- Final transform selection used no GT: Cam1 uses its high-confidence
  shared-board recovery, while Cam2–Cam5 retain their fully observable
  independent link calibrations. All five final cameras have high confidence.
- Optional post-selection mount validation ranged from 0.002597 m to 0.014094 m
  translation error and 0.6628 deg to 0.8430 deg rotation error.
- Exported 1,022 depth-model-compatible `T_base_cam` records for each of five
  cameras (5,110 total) using only
  `T_base_cam = T_base_link @ T_link_camera`.
- Offline absolute-pose validation evaluated all 1,022 frames per camera. All
  cameras passed the 0.05 m mean-translation and 3 deg mean-rotation thresholds;
  mean errors matched the static mount validation ranges above.
- Generated `outputs/final_calibration/final_camera_pose_validation.json` and a
  concise deployment `README.md` locally. The notes explicitly state that the
  depth model consumes per-frame `T_base_cam`, not static `T_link_camera`, and
  that the camera-ray adapter must not be applied twice.
- Ground truth is excluded from transform selection, adapter metadata,
  confidence assignment, and exported pose computation; it is used only after
  finalization for validation.
- pytest: 48 passed.
- Milestone 1 integrity check: PASS.
- Corrected-dataset sanity check: PASS (`CAMERA_STREAMS_AND_TRANSFORMS_DISTINCT`).

## Milestone 5.1 — Deployment/validation metadata separation

Status: completed

- Removed all per-camera GT-derived fields from the deployment-critical
  `final_calibration.json` camera entries while preserving static transforms,
  confidence, warnings, observability, transform conventions, and adapter/frame
  metadata.
- Added a lightweight evaluation-only `validation` reference block to
  `final_calibration.json`.
- Moved optional `T_link_camera` GT metrics into
  `outputs/final_calibration/final_static_calibration_validation.json`; all five
  cameras were evaluated with the same 0.002597–0.014094 m translation and
  0.6628–0.8430 deg rotation error ranges reported in Milestone 5.
- Kept per-frame absolute-pose metrics separate in
  `outputs/final_calibration/final_camera_pose_validation.json`; all 5,110
  exported `T_base_cam` poses still pass validation.
- Export with missing GT or `--no-evaluate-gt` remains successful and emits
  false/null validation references without embedding GT fields.
- pytest: 50 passed.

## Milestone 6 — Depth-model pose compatibility validation

Status: completed

- Completed integration/compatibility validation only; no calibration algorithm,
  final static calibration, depth model, or dataset files were changed.
- Exported 1,022 depth-model-compatible per-frame transforms for each of five
  cameras (5,110 JSON files total), plus five JSONL mirrors. Every record
  contains both `T_base_cam_rowmajor` and the `t_base_cam` /
  `q_base_cam_xyzw` fallback representation, composed as
  `T_base_cam = T_base_link @ T_link_camera`.
- Generated a 1,022-sample manifest using the reference depth-model structure
  and the 180-degree conversion key `FisheyeConversions_2`. The dataset has no
  depth files, so all 5,110 manifest views retain their RGB and calibrated
  transform paths with `depth_path` set to null.
- Parser compatibility and relative-pose computation smoke tests passed. The
  exported adapter metadata preserves the calibrated OCamCalib-to-camera pose
  Y-axis flip and records that it was already applied during calibration.
- Evaluation-only Unity GT absolute validation covered all 5,110 poses. Per-camera
  mean errors ranged from 0.002597 m to 0.014094 m translation and 0.6628 deg to
  0.8430 deg rotation, consistent with the Milestone 5 validation.
- Evaluation-only relative validation covered 20,440 ordered source-target pairs
  in each direction. For `T_src_tgt`, translation error was 0.012391 m mean,
  0.011850 m median, and 0.026965 m max; rotation error was 0.9389 deg mean,
  1.0419 deg median, and 1.6382 deg max. The inverse direction produced the same
  aggregate statistics. GT was not used to generate or modify any transform.
- Generated `outputs/depth_model_compat/depth_model_samples.json`,
  `depth_model_compatibility_report.json`, and
  `depth_model_pose_validation.json` locally.
- pytest: 60 passed.
- Milestone 1 integrity check: PASS.
- Corrected-dataset sanity check: PASS (`CAMERA_STREAMS_AND_TRANSFORMS_DISTINCT`).

## Milestone 6.5 — Reproducibility documentation

Status: completed

- Added a concise root `README.md` and detailed documentation for dataset inputs,
  the complete command sequence, output meanings, transform/frame conventions,
  new Unity and real-robot datasets, troubleshooting, and a reproducibility
  checklist.
- Documented `link_poses/*.json` as the calibration-core kinematic input and
  distinguished it from optional Unity GT/debug sources. Also recorded the
  current Unity-specific integrity checker's full-bundle expectations.
- Documented Unity 180/210/240 FOV conversion mappings, matching OCamCalib and
  ray-adapter requirements, deployment-safe validation separation, and runtime
  depth-model composition with `T_base_cam = T_base_link @ T_link_camera`.
- No calibration algorithms, raw dataset files, or generated output schemas were
  changed.
- Documentation relative-link check: PASS (14 links across seven Markdown files).
- pytest: 60 passed.
- Milestone 1 integrity check: PASS.
- Corrected-dataset sanity check: PASS (`CAMERA_STREAMS_AND_TRANSFORMS_DISTINCT`;
  0/50 all-camera byte-identical sampled frames).

## Milestone 6.6 — Static calibration wrapper and input examples

Status: completed

- Added `python -m calibration_pipeline.run_static_calibration_pipeline` as a
  convenience wrapper for the existing independent link calibration,
  shared-board recovery, and final static calibration export CLIs. It starts
  from existing board-pose JSONLs and does not rerun detection, board-pose
  estimation, or depth-model compatibility export.
- Added GT/no-GT, camera, minimum-pose, anchor-count, single-anchor, and dry-run
  pass-through controls, early missing-board-pose validation, standard output
  checks, and a per-camera terminal summary.
- Kept all lower-level commands available and documented routine wrapper mode
  separately from first-run/research debugging mode.
- Added schema-faithful documentation templates for candidate links, per-frame
  link poses, ChArUco geometry, Unity and real/OpenCV-style camera models,
  session metadata, optional Unity camera GT, joint states, and flattened link
  poses. All seven JSON and two CSV templates validated successfully.
- Documented Unity versus real-robot inputs, frame/adapter differences, no-GT
  operation, and the precomputed `T_base_link` boundary.
- No core calibration algorithms, raw dataset files, or output schemas changed.
- Current-dataset wrapper dry run: PASS; all three planned lower-level commands
  and standard paths were printed without overwriting calibration outputs.
- Documentation relative-link check: PASS (25 links across eight Markdown files).
- pytest: 67 passed.
- Milestone 1 integrity check: PASS.
- Corrected-dataset sanity check: PASS (`CAMERA_STREAMS_AND_TRANSFORMS_DISTINCT`;
  0/50 all-camera byte-identical sampled frames).

## Milestone 6.7 — Single-experiment calibration evaluation

Status: completed

- Added `python -m calibration_pipeline.run_calibration_evaluation` and reusable
  single-run metric, plotting, and Markdown-report modules. Existing calibration
  and deployment artifacts are read-only inputs; all new artifacts are written
  under `outputs/evaluation/`.
- Generated an aggregation-ready summary JSON, five-row per-camera and GT-free
  CSVs, a 30-row camera/link ranking table, a 20-row ordered source-target table,
  and a concise `report.md` for the current experiment.
- Generated all 13 available GT-free and evaluation-only plots with matplotlib,
  including link-score and relative-pose heatmaps. Plotting is robust to missing
  metrics and uses no interactive backend.
- Current GT-free summary: five high-confidence cameras, 44.6% mean detection
  valid ratio, 0.024615 deg mean per-camera board-pose ray error, 0.480921 mean
  link score margin, one motion-limited camera recovered from four anchors, and
  both depth compatibility smoke tests passing.
- Optional Unity/offline GT summary: 100% attached-link top-1 accuracy; mean
  static error 0.007879 m and 0.7578 deg; mean runtime error 0.007879 m and
  0.7578 deg; mean relative error 0.012391 m and 0.9389 deg across 20,440 ordered
  source-target frame samples.
- Documented GT/no-GT invocation, report interpretation, output meanings, and
  the single-experiment boundary. Multi-run/cross-FOV aggregation was not added.
- No core calibration algorithms, raw dataset files, deployment calibration, or
  existing calibration-output schemas changed.
- pytest: 79 passed.
- Milestone 1 integrity check: PASS.
- Corrected-dataset sanity check: PASS (`CAMERA_STREAMS_AND_TRANSFORMS_DISTINCT`;
  0/50 all-camera byte-identical sampled frames).
