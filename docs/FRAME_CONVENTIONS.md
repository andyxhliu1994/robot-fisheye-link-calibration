# Frame and transform conventions

Transform direction must be explicit at every boundary. This project uses one
rule everywhere:

```text
T_A_B means p_A = T_A_B @ p_B
```

`T_A_B` maps coordinates expressed in frame B into frame A. Matrix transforms
are homogeneous 4×4 row-major arrays in JSON; translations are in meters and
quaternions use `xyzw` order.

## Calibration transforms

### `T_camera_board`

```text
p_camera = T_camera_board @ p_board
```

This is estimated independently for each valid ChArUco image from board object
points and calibrated camera rays. The board frame is OpenCV ChArUco's object
frame: origin at an outer board corner, +X along columns, +Y along rows, and Z
normal to the board.

### `T_base_link`

```text
p_base = T_base_link @ p_link
```

This is a required per-frame kinematic input for all candidate links during
calibration. It comes from `link_poses/*.json` and may be produced by Unity, a
robot SDK, ROS `/tf`, URDF FK, DH FK, or another validated kinematics source.
It is not `T_link_base`.

### `T_link_camera`

```text
p_link = T_link_camera @ p_camera
```

This is the final static calibration. It maps camera-frame points into the
selected robot link frame and should remain constant while the mount is
unchanged.

For a fixed board, a correct link/mount makes:

```text
T_base_board ~= T_base_link(t) @ T_link_camera @ T_camera_board(t)
```

consistent over the recording.

## Runtime and depth-model transforms

### `T_base_cam`

```text
p_base = T_base_cam @ p_camera
T_base_cam(t) = T_base_link(t) @ T_link_camera
```

This is the absolute camera pose needed by the depth model. It changes as the
attached robot link moves. `final_calibration.json` stores `T_link_camera`; it
does not store a single runtime `T_base_cam` because current FK is required.

### Target-to-source relative pose

```text
T_src_tgt = inv(T_base_src) @ T_base_tgt
```

This maps a target-camera 3D point into the source-camera frame:

```text
p_src = T_src_tgt @ p_tgt
```

It is the direction used by the referenced depth-model warping path.

### Source-to-target relative pose

```text
T_tgt_src = inv(T_base_tgt) @ T_base_src
```

This maps a source-camera 3D point into the target-camera frame:

```text
p_tgt = T_tgt_src @ p_src
```

The two relative transforms are opposites. Name the source and target before
composing rather than relying on an ambiguous word such as "extrinsics."

## Camera frames and ray adapter

The camera projection model returns a unit ray in its calibrated raw ray frame.
The pose estimator needs that ray in the camera pose frame used by
`T_camera_board`, `T_link_camera`, and `T_base_cam`. The configured adapter has
semantics:

```text
ray_camera_pose = R_adapter @ ray_camera_model_raw
```

For the current Unity OCamCalib data it is typically:

```text
R_adapter = diag(1, -1, 1)
```

This flips raw OCamCalib Y into the right-handed Unity camera pose frame (+X
image right, +Y image up, +Z forward). A real OpenCV optical setup may use an
identity adapter, but only if its ray and pose frames are already identical.

The camera-ray adapter is applied while estimating `T_camera_board`, so its
effect is already present in final `T_link_camera`. **Do not apply it again** to
`T_link_camera` or runtime `T_base_cam`.

## Board GT adapter

Unity's board GameObject truth can use a different origin and axes from OpenCV's
ChArUco object frame. The board GT adapter used by optional pose evaluation
aligns those board frames only for error computation. It does not change link
association or final `T_link_camera`, and it is not a runtime adapter.

## Link frame source

The base and link frames are defined by the source that generated
`T_base_link`: the Unity hierarchy, robot SDK, ROS tree, or FK model. Their names,
origins, and axes must stay consistent between calibration and deployment.
Changing the robot base or link-frame definition invalidates direct reuse of the
stored mount unless the calibration is transformed accordingly.

## Deployment-safe validation separation

After Milestone 5.1, `final_calibration.json` contains static calibration and
adapter/frame metadata but no per-camera GT error fields. Its `validation` block
only indicates whether separate reports exist. Unity-only metrics live in:

```text
outputs/final_calibration/final_static_calibration_validation.json
outputs/final_calibration/final_camera_pose_validation.json
```

GT is evaluation-only and is never used to select attached links, mount
transforms, confidence, or runtime poses.
