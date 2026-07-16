# Dataset format

The calibration pipeline reads local raw data from `./dataset/`. Do not modify,
move, or rename files in that directory during a run. Filenames use a shared
`frame_xxxxxx` identifier to align cameras and robot kinematics.

Copyable structural examples are available in
[INPUT_FILE_SAMPLES.md](INPUT_FILE_SAMPLES.md).

## Calibration-core minimum

```text
dataset/
  cameras/
    <camera_name>/
      rgb/
        frame_000000.jpg
        frame_000001.jpg
        ...
  link_poses/
    frame_000000.json
    frame_000001.json
    ...
  candidate_links.json
  charuco_board_config.json
  camera_model_config.json
  camera_calibration/
    <camera_calibration_file>.json
  session_summary.json
```

This is the minimum data model for the calibration stages. The current
Milestone 1 integrity CLI was built for a full Unity recorder export and also
checks several validation/debug files listed later. A real-robot dataset may be
calibratable with the core inputs even when that Unity-specific integrity report
cannot run in full.

## Camera RGB frames

`cameras/<camera_name>/rgb/frame_xxxxxx.jpg` contains calibration images. Every
camera and `link_poses/` should use the same frame IDs. The board need not be
fully visible in every image, but enough synchronized frames must show at least
four markers and eight ChArUco corners for pose estimation.

Use stable, unique camera names across `cameras/`, metadata, calibration outputs,
and downstream configuration. Camera streams must be genuinely distinct; run
the dataset sanity check before calibration.

## Per-frame candidate-link poses

`link_poses/frame_xxxxxx.json` is a required core input. Each file provides the
pose of every candidate robot link at that image frame. The current provider
reads entries shaped like:

```json
{
  "frame_id": "frame_000000",
  "base_frame_name": "base",
  "links": [
    {
      "link_name": "wrist_1_link",
      "link_path_rel": "base/.../wrist_1_link",
      "valid": true,
      "T_base_link_rowmajor": [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
      ]
    }
  ]
}
```

`T_base_link` has the exact meaning:

```text
p_base = T_base_link @ p_link
```

It is a link pose relative to the robot base, **not** `T_link_base`. If an SDK
or exporter provides `T_link_base`, invert it before writing
`T_base_link_rowmajor`.

These poses are needed because the camera's attached link is initially unknown.
The pipeline tests every candidate. The correct link should make the fixed-board
relationship most consistent over time:

```text
T_base_board ~= T_base_link(t) @ T_link_camera @ T_camera_board(t)
```

During calibration, provide `T_base_link` for every candidate link at every
aligned frame. After calibration, runtime code only needs the link selected for
each camera.

## Candidate links

`candidate_links.json` lists the robot links to test and names the base frame.
Each `link_path_rel` should match a key exported in every link-pose file. Typical
candidates include `shoulder_link`, `upper_arm_link`, `forearm_link`, and wrist
links. Omitting the true attached link prevents correct association; adding many
nearly identical or unmoving candidates can weaken score margins.

## ChArUco board configuration

`charuco_board_config.json` describes the physical board:

- `type`: `charuco`
- `dictionary`: the printed ArUco dictionary, currently `DICT_4X4_1000`
- `squares_x`, `squares_y`: square counts
- `square_length_m`, `marker_length_m`: measured metric dimensions
- `first_marker_id` and optional `marker_count`
- optional human-readable board-frame definition

The values must match the board that appears in the images. The OpenCV board
object frame starts at an outer board corner, with +X across columns, +Y across
rows, and Z normal to the board.

## Camera model configuration

`camera_model_config.json` selects the projection model, intrinsic calibration,
raw ray frame, pose camera frame, and ray-frame adapter. The current estimator
uses `ocamcalib` and expects `default_calibration_file` to be relative to
`dataset/`, for example:

```json
{
  "default_camera_model": "ocamcalib",
  "default_calibration_file": "camera_calibration/my_fisheye.json",
  "ray_frame": "ocamcalib_raw",
  "pose_camera_frame": "unity_camera",
  "ray_frame_adapter": {
    "type": "matrix_3x3",
    "name": "flip_y_to_unity_camera_frame",
    "matrix": [[1, 0, 0], [0, -1, 0], [0, 0, 1]]
  }
}
```

The model, calibration file, image resolution, physical camera configuration,
FOV, and adapter must describe the images being processed.

## Intrinsic/fisheye calibration

`camera_calibration/*.json` contains camera intrinsics, such as OCamCalib's
`distortion_center`, `stretch_matrix`, and `taylor_coefficient`. It must match
the FOV, resolution, lens or Unity physical camera settings used for capture.
Do not reuse another FOV's calibration unless deliberately testing
generalization.

The current configuration selects one default file for all cameras. Therefore,
the cameras must share the matching model or the configuration/provider must be
extended before mixing different intrinsics.

## Session summary

`session_summary.json` records dataset identity and sanity metadata such as
`setup_id`, `session_name`, `frame_count_so_far`, `camera_count`, and
`camera_names`. The depth-model manifest uses `setup_id` and `session_name`.
Keep names and frame counts consistent with the actual folders.

## Optional Unity validation and debug data

These files are useful but are not inputs to the calibration estimate:

```text
dataset/
  cameras/<camera_name>/transform/frame_xxxxxx.json
  board_pose_base.csv
  setup_used.json
  joint_states.csv
  link_poses.csv
```

### Camera transforms

`cameras/<camera_name>/transform/frame_xxxxxx.json` stores Unity ground-truth
`T_base_cam`. It is used only after estimation for absolute and relative pose
validation. It is not required to generate `T_link_camera` or runtime poses.

### Board pose

`board_pose_base.csv` stores Unity ground-truth board poses. It supports
evaluation of estimated `T_camera_board`; it is not required on a real robot.

### Setup truth

`setup_used.json` records Unity's true camera-link assignments and static camera
mounts. It is evaluation-only and must not influence link selection, transform
selection, or confidence.

### Joint states

`joint_states.csv` is useful for motion debugging and future FK providers. Joint
angles alone cannot replace link poses. They become usable kinematic input only
through a model/provider:

```text
joint_states.csv + FK (URDF, DH, robot SDK, or equivalent) -> T_base_link
```

### Flattened link poses

`link_poses.csv` is a table representation of the same kind of information in
`link_poses/*.json`. It is convenient for plots and spot checks, but the current
`UnityLinkPoseProvider` reads the per-frame JSON files.

The current integrity report treats the Unity CSV and GT sources as an aligned
bundle. Their absence can make that report unavailable even though the
calibration-core inputs above are present.

## Real-robot sources for `T_base_link`

For a real robot, per-frame candidate link poses can come from:

1. a robot SDK that directly returns link poses;
2. synchronized ROS `/tf` transforms;
3. joint states evaluated through URDF forward kinematics;
4. joint states evaluated through DH kinematics; or
5. vendor-specific forward kinematics.

The robot base itself does not need to move. The links must move enough, and in
sufficiently varied directions, to make camera mounts observable. A camera on a
single-axis or otherwise motion-limited link may require shared-board recovery
from fully observable anchor cameras.

At runtime, only each camera's selected attached link is required:

```text
T_base_cam(t) = T_base_link_attached(t) @ T_link_camera
```

Direct ROS/URDF/SDK ingestion is future work; the current pipeline boundary is
the per-frame JSON representation above.

## Unity FOV mapping

For the existing Unity depth-model conventions:

| Fisheye FOV | Manifest `conversion_name` |
| --- | --- |
| 180° | `FisheyeConversions_2` |
| 210° | `FisheyeConversions_3` |
| 240° | `FisheyeConversions_4` |

This key selects the corresponding depth-model ray-map/OCam lookup family. The
camera model configuration must point to the matching OCamCalib JSON. If Unity
physical camera settings change, recalibrate or select their matching file.

Current Unity data typically uses `diag(1, -1, 1)` to map OCamCalib raw rays to
the Unity camera pose frame. A real OpenCV optical camera may use identity,
depending on its documented pose convention. Confirm the convention rather
than choosing an adapter from the platform name alone.

## Unity dataset vs real robot dataset

A Unity recording usually includes synchronized RGB, `candidate_links.json`,
camera/board configuration, matching camera calibration, and `T_base_link`
exported directly from Unity transforms. It may additionally include GT camera
transforms, `setup_used.json`, `board_pose_base.csv`, `joint_states.csv`, and
`link_poses.csv` for evaluation and debugging.

A real-robot recording usually includes synchronized RGB, physical-camera
intrinsics, camera/board configuration, a candidate link list, and precomputed
`T_base_link` for every candidate and frame. Those link poses may come from an
SDK link-pose API, ROS `/tf`, joint states plus URDF FK, joint states plus DH
kinematics, or vendor SDK FK. Real recordings normally have no camera GT,
`setup_used.json`, or `board_pose_base.csv`.

Use `--no-evaluate-gt` for real/no-GT data; `final_calibration.json` is still
generated. Confidence comes from internal board consistency, link score margin,
observability, ray error, and shared-anchor agreement rather than GT errors.
Future Milestone 7 work may add direct FK/ROS/URDF adapters, but the current
pipeline already consumes correctly precomputed `T_base_link` JSONs.

Unity's right-handed camera pose frame is commonly +X right, +Y up, +Z forward,
with raw OCamCalib rays adapted by `diag(1, -1, 1)`. A real OpenCV optical frame
is commonly +X right, +Y down, +Z forward and may use identity when its ray model
already outputs that exact pose frame. Record the actual convention in
`camera_model_config.json`; platform labels alone are not sufficient.
