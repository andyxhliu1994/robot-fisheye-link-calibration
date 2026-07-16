# Input-file samples

The files under `docs/sample_inputs/` are small documentation templates, not a
recording and not calibration data. Copy their structure—not their numeric
values—when preparing a new local `dataset/`.

| Template | Purpose | Required? |
| --- | --- | --- |
| [`candidate_links.json`](sample_inputs/candidate_links.json) | Candidate robot links and base frame | Yes |
| [`link_poses/frame_000000.json`](sample_inputs/link_poses/frame_000000.json) | Per-frame `T_base_link` for every candidate | Yes |
| [`charuco_board_config.json`](sample_inputs/charuco_board_config.json) | Printed board geometry | Yes |
| [`camera_model_config_unity.json`](sample_inputs/camera_model_config_unity.json) | Unity OCamCalib rays and Y-flip adapter | Choose/configure one |
| [`camera_model_config_opencv.json`](sample_inputs/camera_model_config_opencv.json) | Real camera whose calibrated rays already use its pose frame | Choose/configure one |
| [`session_summary.json`](sample_inputs/session_summary.json) | Dataset/camera metadata | Yes |
| [`optional_unity_transform.json`](sample_inputs/optional_unity_transform.json) | Unity GT `T_base_cam` for validation | No |
| [`joint_states.csv`](sample_inputs/joint_states.csv) | Raw joint state/debug input | No |
| [`link_poses.csv`](sample_inputs/link_poses.csv) | Flattened link-pose/debug table | No |

## Required kinematic direction

The per-frame JSON is the current pipeline's kinematic input. It must include
every candidate link, with:

```text
p_base = T_base_link @ p_link
```

It is not `T_link_base`. Invert an SDK's `T_link_base` before exporting it.
`link_path_rel` values must match `candidate_links.json` exactly.

`joint_states.csv` cannot replace these transforms unless a validated FK
provider converts joint states to `T_base_link`. `link_poses.csv` is useful for
manual inspection, but the current `UnityLinkPoseProvider` reads the per-frame
JSON files.

## Camera configuration choices

The current pose estimator supports `ocamcalib`. The Unity example maps raw
OCamCalib rays into a right-handed Unity pose frame with
`diag(1, -1, 1)`. That adapter is applied during `T_camera_board` estimation and
must not be applied again to final `T_link_camera`.

The real/OpenCV-style example uses identity only for a camera model whose
calibrated rays already use the same optical pose frame (+X right, +Y down, +Z
forward). Verify the actual model/exporter convention; do not assume identity
merely because the camera is physical.

## Optional Unity transform

The optional transform template shows redundant matrix and translation/
quaternion representations plus frame/adapter metadata. Files of this kind are
Unity GT validation inputs only. They are not required for real data and are
not used to estimate attached links or `T_link_camera`.
