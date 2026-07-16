# New dataset checklist

Complete this checklist before calibration.

## Recording and synchronization

- [ ] Every camera has `rgb/frame_xxxxxx.jpg` images.
- [ ] Camera names are unique and consistent across directories and metadata.
- [ ] RGB and `link_poses/` frame IDs are synchronized.
- [ ] There are no missing, duplicated, or unexpected frame IDs.
- [ ] Each camera stream changes over time and differs from the other cameras.
- [ ] The ChArUco board is fixed and visible in enough varied robot poses.
- [ ] Robot/link motion is sufficiently diverse for mount observability.

## Kinematics

- [ ] Every frame has `link_poses/frame_xxxxxx.json`.
- [ ] Every file contains all candidate links and valid 4×4 transforms.
- [ ] `T_base_link` semantics are confirmed as
      `p_base = T_base_link @ p_link`.
- [ ] Any source `T_link_base` values were inverted before export.
- [ ] `candidate_links.json` paths match link-pose keys exactly.
- [ ] The true attached link for every camera is included among the candidates.
- [ ] Base/link frame definitions will remain identical at deployment time.

## Board and camera geometry

- [ ] ChArUco dictionary and square counts match the printed board.
- [ ] Square and marker dimensions are measured in meters.
- [ ] Camera model and calibration file match lens/FOV/resolution/settings.
- [ ] Unity 180/210/240 data uses the matching OCamCalib file and conversion key.
- [ ] Raw ray and camera pose frames are documented.
- [ ] The ray adapter matrix is confirmed; it was not guessed from camera type.

## Validation mode

- [ ] Unity GT files are present and `--evaluate-gt` is intended, **or**
- [ ] This is real/no-GT data and `--no-evaluate-gt` is selected.
- [ ] `joint_states.csv` is not being treated as a substitute for `T_base_link`
      unless a validated FK provider converts it.
- [ ] Any integrity-check limitation for a core-only real dataset is understood.

## Run hygiene and review

- [ ] Existing outputs are intentionally retained or cleared outside the raw
      dataset; new generated files will stay under `outputs/`.
- [ ] `python -m pytest` passes.
- [ ] Integrity and sanity reports were reviewed where applicable.
- [ ] Detection and pose overlays were visually inspected.
- [ ] Link score margins, observability, anchor agreement, confidence, and
      warnings will be reviewed before deployment.
- [ ] Runtime code will compose
      `T_base_cam = T_base_link @ T_link_camera`.
- [ ] Runtime code will not apply the camera-ray adapter twice.
