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
