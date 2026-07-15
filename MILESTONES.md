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
